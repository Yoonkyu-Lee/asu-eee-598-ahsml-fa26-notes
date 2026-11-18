"""How boxes keep each other's sets equal (docs/protocol.md; the model's rules are sim/sync.py, kept separate):
ids-first push (CP28), one fetch per id (CP30 K), set reconciliation on every contact (CP29 S2, sim/recon.py) with
its triggers (a) came on, (b) a peer appeared, (c) an id did not come, (e) a peer went down, (f) the slow floor
(CP30 F'), windows and expiry, and the test-only faults."""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import ingest
from box_state import Node, log
from common import msg_from_json, msg_to_json
from sim.box import Msg, ref_target
from sim.recon import FP, Session

REQ_TIMEOUT_S = 1.0      # CP28 lazy push: ask the next announcer after this (sim.exchange Scene.req_timeout_s)
OFFER_TIMEOUT_S = 30.0   # CP30 K: an offered id that has not come by then is released (and its session runs again)
TICK_S, JUMP_S = 5.0, 30.0   # local clock check: a jump over JUMP_S means the computer slept


def targets(node: Node, topic: int) -> list[int]:
    """The topic's mesh peers (rule 2, graft): only peers known to be on, filled up to D from the topic's other
    holders that are on, most recently heard from first."""
    mesh = [int(p) for p in node.mesh.get(topic, {}).get(node.bid, ())]
    out = [p for p in mesh if p in node.up]
    D = node.cfg.get("D", 2)
    if len(out) < D:
        more = [int(p) for p in node.holders.get(topic, ()) if int(p) != node.bid and int(p) not in out
                and int(p) in node.up]
        more.sort(key=lambda p: -node.peer_seen.get(p, 0.0))
        out += more[:D - len(out)]
    return out


# ---- ids first (CP28): announce, the peer wants it from the first announcer only
def send(node: Node, peer: int, m: Msg, direct: bool = False) -> None:
    """Push m to peer: its id first (CP28), or the envelope when the peer already asked for it (direct)."""
    node.count["send"] += 1
    node.inc()
    payload = {"msg": msg_to_json(m), "frm": node.bid}
    lazy = not direct

    # ids first: no retries. A peer that missed an id hears it from another announcer (REQ_TIMEOUT_S) or gets it
    # when it comes on (catch-up); retrying an off box only lands late after it restarts (CP28 D, run 3)
    tries = 1 if lazy else 3

    def go():
        try:
            for i in range(tries):
                try:
                    if lazy:
                        node.cut(peer)
                        node.count["ihave_sent"] += 1
                        req = urllib.request.Request(node.url(peer, "/ihave1"),
                                                     data=json.dumps({"mid": str(m.mid), "frm": node.bid}).encode(),
                                                     headers={"Content-Type": "application/json"})
                        want = json.loads(urllib.request.urlopen(req, timeout=60).read())["want"]
                        if want:
                            node.post(peer, "/gossip", payload)
                    else:
                        node.post(peer, "/gossip", payload)
                    node.up.add(peer)
                    return
                except Exception as e:
                    if i == tries - 1:
                        log(f"gossip to box {peer} ({node.addr[peer]}) failed:", e)
                        node.up.discard(peer)
                        mark_down(node, peer)
                        return
                    time.sleep(0.5 * (i + 1))
        finally:
            node.inc(-1)
    node.pool.submit(go)


def ihave1(node: Node, mid: int, frm: int) -> bool:
    """An id was announced. Want it from the first announcer only; check again after REQ_TIMEOUT_S."""
    with node.lock:
        node.count["ihave_rx"] += 1
        node.up.add(frm)
        node.peer_seen[frm] = time.time()
        if mid in node.got or node.forgotten(mid):
            return False
        if mid in node.fetching:                           # CP30 K: a session is fetching it
            node.skipped.setdefault(mid, set()).add(frm)
            return False
        ann = node.announcers.setdefault(mid, [])
        ann.append(frm)
        if len(ann) > 1:
            return False
        node.lazy_pending.add(mid)
    threading.Timer(REQ_TIMEOUT_S, _lazy_check, (node, mid, 1)).start()
    return True


def _lazy_check(node: Node, mid: int, i: int) -> None:
    with node.lock:
        if mid in node.got or mid in node.silent_mids:
            node.lazy_pending.discard(mid)
            return
        ann = node.announcers.get(mid, [])
        exhausted = i >= len(ann)
        if exhausted:
            node.lazy_pending.discard(mid)
        else:
            frm = ann[i]
    if exhausted:                                          # CP30 K: the push never came; skipped sessions
        _release(node, mid)                                # run again
        return
    try:
        node.cut(frm)
        d = json.loads(urllib.request.urlopen(node.url(frm, f"/env?mid={mid}&frm={node.bid}"), timeout=60).read())
        if d.get("msg"):
            node.count["env_rx"] += 1
            ingest.deliver(node, msg_from_json(d["msg"]), frm)
            return
    except Exception as e:
        log(f"envelope {mid} from box {frm} failed:", type(e).__name__)
    if i + 1 >= len(node.announcers.get(mid, [])):
        with node.lock:                                    # (c) announced, never obtained: reconcile
            node.dirty.update(node.announcers.get(mid, []))
        recon_bg(node, frm, "c")
        with node.lock:
            node.lazy_pending.discard(mid)
        _release(node, mid)
    threading.Timer(REQ_TIMEOUT_S, _lazy_check, (node, mid, i + 1)).start()


# ---- one fetch per id (CP30 K)
def offer(node: Node, frm: int, ids: list) -> list[str]:
    """CP30 K: a peer's session found these missing here. Want what is not held and not on its way; what is
    on its way from someone else is noted (if that fails, a session with frm runs again)."""
    want = []
    with node.lock:
        node.up.add(frm)
        for x in ids:
            i = int(x)
            if i in node.got or i in node.gone:
                continue
            if i in node.fetching or (i in node.lazy_pending and i not in node.silent_mids):
                node.skipped.setdefault(i, set()).add(frm)
                continue
            node.fetching[i] = frm
            want.append(x)
    if want:
        threading.Timer(OFFER_TIMEOUT_S, _offer_timeout, (node, frm, [int(x) for x in want])).start()
    return want


def _offer_timeout(node: Node, frm: int, ids: list) -> None:
    with node.lock:
        done = [i for i in ids if node.fetching.get(i) == frm]
        for i in done:
            del node.fetching[i]
    for i in done:
        if i not in node.got:
            with node.lock:
                node.dirty.add(frm)
            recon_bg(node, frm, "c")
        _release(node, i)


def _release(node: Node, mid: int) -> None:
    """CP30 K: an id stopped being in flight. If it is still missing, the sessions that skipped it run again
    (c); one that is running gets a second session when it ends."""
    with node.lock:
        peers = node.skipped.pop(mid, ())
        if mid in node.got or mid in node.gone or mid in node.silent_mids:
            return
        now_busy = [q for q in peers if q in node.recon_busy]
        for q in peers:
            node.dirty.add(q)
        node.again.update(now_busy)
        go = [q for q in peers if q not in now_busy]
    for q in go:
        recon_bg(node, q, "c")


# ---- hello
def hello_info(node: Node) -> dict:
    return {"box": node.bid, "proto": node.cfg["proto"], "n": node.n, "t": time.time()}


def _hello(node: Node, peer: int) -> dict | None:
    try:
        node.cut(peer)
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(node.url(peer, f"/hello?frm={node.bid}"), timeout=10).read())
        t1 = time.time()
    except Exception as e:
        log(f"box {peer} ({node.addr[peer]}) not reachable:", type(e).__name__)
        node.up.discard(peer)
        return None
    node.up.add(peer)
    node.peer_seen[peer] = time.time()
    node.count["hello"] += 1
    if node.st:
        node.st.ev("peer", peer=peer, rtt_ms=(t1 - t0) * 1e3, offset_ms=(r.get("t", (t0 + t1) / 2) - (t0 + t1) / 2) * 1e3)
    return r


# ---- CP29 "recon": set reconciliation on every contact (sim/recon.py S2, the model's rules in sim/sync.py)
def _enc(msg) -> list:
    k = lambda x: [x[0], str(x[1])]
    out = []
    for lo, hi, mode, pay in msg:
        out.append([k(lo), k(hi), mode, [str(pay[0]), pay[1]] if mode == FP else [k(x) for x in pay]])
    return out


def _dec(rows) -> list:
    k = lambda x: (int(x[0]), int(x[1]))
    return [(k(lo), k(hi), mode, (int(pay[0]), int(pay[1])) if mode == FP else [k(x) for x in pay])
            for lo, hi, mode, pay in rows]


def common(node: Node, p: int) -> set:
    """CP33: the subs this box and peer p both hold (the holders table: the config here, a sub's member list in a
    real deployment)."""
    return {t for t in node.box.topics if p in node.holders.get(t, ())}


def recon_serve(node: Node, frm: int, rows: list, topics: list | None = None) -> dict:
    """Answer one round of a peer's session (stateless: what I need here goes back with the reply)."""
    with node.lock:
        node.count["recon_rx"] += 1
        node.up.add(frm)
        node.peer_seen[frm] = time.time()
        cm = common(node, frm) if topics is None else {int(x) for x in topics} & set(node.box.topics)
        s = Session(node.keys.view(cm))
        out = s.process(_dec(rows))
        need = sorted(s.need)
    return {"reply": _enc(out), "need": [[t, str(i)] for t, i in need]}


def envs(node: Node, mids: list) -> list:
    with node.lock:
        return [msg_to_json(node.msgs[int(x)]) for x in mids if int(x) in node.msgs]


def recon_bg(node: Node, p: int, why: str) -> None:
    threading.Thread(target=recon_with, args=(node, p, why), daemon=True).start()


def recon_with(node: Node, p: int, why: str) -> bool:
    """One session with peer p: find both differences, fetch mine, push theirs. False if the peer failed."""
    with node.lock:
        if p in node.recon_busy or p == node.bid:
            return True
        cm = common(node, p)
        if not cm:                                             # CP33: no sub in common, nothing to reconcile
            return True
        node.recon_busy.add(p)
    t_start = time.time()
    take: list = []
    try:
        full, theirs, rounds, nbytes, ver0 = _rounds(node, p, cm)
        take = _claim(node, p, full)
        got, b = _fetch(node, p, take)
        nbytes += b
        push, b = _give(node, p, theirs)
        nbytes += b
        with node.lock:
            node.dirty.discard(p)
            node.probe.pop(p, None)
            node.up.add(p)
            node.last_recon[p] = time.time()
            if (node.cfg.get("floor_skip") and not full and not theirs
                    and (node.ver == ver0 or node.fault.get("no_fence"))):
                node.clean.add(p)                              # CP30 F': no difference, no change since I began
            elif not full and not theirs:
                node.count["fence_held"] += 1
            node.count["recon_sessions"] += 1
            node.count["recon_" + why] += 1
            node.count["recon_bytes"] += nbytes
            node.count["recon_fetched"] += got
            node.count["recon_pushed"] += len(push)
        if node.st:
            node.st.ev("recon", peer=p, why=why, rounds=rounds, need=len(take), gave=len(push), bytes=nbytes,
                       ms=(time.time() - t_start) * 1e3)
        if take or push:
            log(f"recon ({why}) with box {p}: got {got}, gave {len(push)}, {rounds} rounds, {nbytes} B")
        return True
    except Exception as e:
        log(f"recon ({why}) with box {p} failed:", type(e).__name__)
        mark_down(node, p)
        return False
    finally:
        with node.lock:
            done = [i for _, i in take if node.fetching.get(i) == p]
            for i in done:
                del node.fetching[i]
            node.recon_busy.discard(p)
            rerun = p in node.again
            node.again.discard(p)
        for i in done:
            _release(node, i)
        if rerun:
            recon_bg(node, p, "c")


def _rounds(node: Node, p: int, cm: set) -> tuple:
    """The session's rounds over the subs in common: (what I lack, what p lacks, rounds, bytes, my set's version
    when it began)."""
    now = node.now()
    lo, hi = (int(now - node.R + 2 * node.sigma), -1), (int(now + 2 * node.sigma) + 1, -1)
    with node.lock:
        s = Session(node.keys.view(cm))
        msg = s.initiate(lo, hi)
        ver0 = node.ver                                    # CP30 F: the fence
    theirs, rounds, nbytes = set(), 0, 0
    while msg and rounds < 64:
        node.cut(p)
        body = json.dumps({"frm": node.bid, "msg": _enc(msg), "topics": sorted(cm)}).encode()
        req = urllib.request.Request(node.url(p, "/recon"), data=body, headers={"Content-Type": "application/json"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        nbytes += len(body) + len(raw)
        r = json.loads(raw)
        theirs |= {(int(t), int(i)) for t, i in r["need"]}
        reply = _dec(r["reply"])
        rounds += 1
        st = node.fault.get("stall")
        if st and int(st["peer"]) == p and rounds == 1:   # test-only: hold the session after round 1
            node.fault["stall"] = None
            time.sleep(float(st["s"]))
        if not reply:
            break
        with node.lock:
            msg = s.process(reply)
    return sorted(s.need), theirs, rounds, nbytes, ver0


def _claim(node: Node, p: int, full: list) -> list:
    """CP30 K: of what I lack, take from p only what is not held and not in flight; note the rest as skipped."""
    take = []
    with node.lock:
        for t, i in full:
            if i in node.got or i in node.gone:
                continue
            pushing = i in node.lazy_pending and i not in node.silent_mids
            if i in node.fetching or pushing:
                node.skipped.setdefault(i, set()).add(p)
                continue
            node.fetching[i] = p
            take.append((t, i))
    return take


def _fetch(node: Node, p: int, take: list) -> tuple[int, int]:
    """Fetch the envelopes I lack from p (deletes first) and take them in: (count, bytes)."""
    if not take:
        return 0, 0
    node.cut(p)
    body = json.dumps({"frm": node.bid, "mids": [str(i) for _, i in take]}).encode()
    req = urllib.request.Request(node.url(p, "/envs"), data=body, headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    rows = json.loads(raw)
    rows.sort(key=lambda d: d["kind"] != "delete")     # deletes first
    for d in rows:
        ingest.deliver(node, msg_from_json(d), p, via="recon")
    return len(rows), len(body) + len(raw)


def _give(node: Node, p: int, theirs: set) -> tuple[list, int]:
    """CP30 K: offer p the ids it lacks; it takes what is not already on its way, and gets those envelopes directly:
    (the envelopes pushed, bytes)."""
    with node.lock:
        offered = [i for _, i in sorted(theirs) if i in node.msgs]
    if not offered:
        return [], 0
    node.cut(p)
    body = json.dumps({"frm": node.bid, "ids": [str(i) for i in offered]}).encode()
    req = urllib.request.Request(node.url(p, "/offer"), data=body, headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    nbytes = len(body) + len(raw)
    want = {int(x) for x in json.loads(raw)["want"]}
    with node.lock:
        push = [node.msgs[i] for i in offered if i in want and i in node.msgs]
    for m in push:                                         # they lack these: the envelopes, directly
        node.post(p, "/gossip", {"msg": msg_to_json(m), "frm": node.bid})
        nbytes += 1500
    return push, nbytes


# ---- triggers (a), (b), (e) and the loops: probes of down peers, the slow floor (f), the local clock
def mark_down(node: Node, p: int) -> None:
    """A transfer to p failed: probe it with backoff until it answers, then reconcile (c); and (e) reconcile with
    the peers that are still up, since what p was relaying may now come from them."""
    with node.lock:
        was = p in node.up
        node.up.discard(p)
        node.clean.discard(p)
        node.dirty.add(p)
        if p not in node.probe:
            node.probe[p] = [5.0, time.time() + 5.0]
        others = [q for q in node.up if q != p]
    if was:
        for q in others:
            recon_bg(node, q, "e")


def peer_appeared(node: Node, p: int) -> None:
    """(b) a peer said hello: if it was down, unknown or dirty here, reconcile with it."""
    with node.lock:
        new = p not in node.up or p in node.dirty or p in node.probe
        if new:
            node.clean.discard(p)
        node.up.add(p)
        node.probe.pop(p, None)
        node.peer_seen[p] = time.time()
    if new:
        recon_bg(node, p, "b")


def come_on(node: Node) -> None:
    """(a) this box came on (start, or the laptop woke): hello every listed box, reconcile with each that answers."""
    t_come_on = time.time()
    with node.lock:
        node.clean.clear()
    others = [p for p in range(node.n) if p != node.bid]
    with ThreadPoolExecutor(8) as ex:
        alive = [p for p, r in zip(others, ex.map(lambda q: _hello(node, q), others)) if r is not None]
        if node.st:                   # measurement only (events.jsonl): more round trips for the clock offset
            list(ex.map(lambda q: _hello(node, q), [q for q in alive for _ in range(4)]))
        list(ex.map(lambda q: recon_with(node, q, "a"), alive))
    with node.lock:
        for p in others:
            if p not in alive and p not in node.probe:
                node.probe[p] = [5.0, time.time() + 5.0]
    if node.st:
        node.st.ev("catch_done", ms=(time.time() - t_come_on) * 1e3, asked=alive)
    log(f"came on: reconciled with {alive}")


def alive_loop(node: Node) -> None:
    """Local only: treat a clock jump (the computer slept) as coming on again."""
    last = time.time()
    while True:
        time.sleep(TICK_S)
        now = time.time()
        if now - last > JUMP_S:
            log(f"the clock jumped {now - last:.0f} s (asleep?): catching up")
            come_on(node)
        last = now


def recon_loop(node: Node) -> None:
    """Probes of down peers (backoff 5 s .. 300 s) and the slow floor: a quiet pair is checked every P_AE."""
    rng = random.Random()
    while True:
        time.sleep(1.0)
        now = time.time()
        with node.lock:
            due = [p for p, (b, t) in node.probe.items() if t <= now]
            for p in due:
                b = node.probe[p][0]
                node.probe[p] = [min(2 * b, node.pmax), now + min(2 * b, node.pmax)]
            floor = []
            for p in list(node.up):
                nxt = node.next_floor.setdefault(p, node.last_recon.get(p, now) + rng.uniform(*node.pae))
                heard = max(node.peer_seen.get(p, 0.0), node.last_recon.get(p, 0.0))
                if now >= nxt and p in node.clean and now - heard <= node.pae[1]:   # CP30 F': nothing changed
                                                               # since a clean session, and p spoke lately
                    node.count["floor_skip"] += 1
                    node.next_floor[p] = now + rng.uniform(*node.pae)
                    continue
                if now >= nxt:
                    if now - node.last_recon.get(p, 0.0) >= node.pae[0]:
                        floor.append(p)
                        node.next_floor[p] = now + rng.uniform(*node.pae)
                    else:
                        node.next_floor[p] = node.last_recon[p] + rng.uniform(*node.pae)
        for p in due:
            if _hello(node, p) is not None:
                peer_appeared(node, p)
        for p in floor:
            recon_bg(node, p, "f")


def expiry_loop(node: Node) -> None:
    """Windows (cp/CP29.md): a message is dropped when t_key < now - R; a delete and a forgotten id when
    t_key < now - R - 2 sigma."""
    while True:
        time.sleep(float(node.cfg.get("expiry_every_s", 60.0)))
        now = node.now()
        cut, cut2 = now - node.R, now - node.R - 2 * node.sigma
        drop, drop_del = [], []
        with node.lock:
            for mid, m in list(node.msgs.items()):
                if m.kind == "delete":
                    rt = ref_target(m)
                    if rt is not None and rt[1] < cut2:
                        drop_del.append((mid, rt))
                elif m.t0 < cut:
                    drop.append(mid)
            for mid in drop:
                m = node.msgs.pop(mid)
                if mid not in node.gone:
                    node.keys.discard((int(m.t0), mid))
                node.got.discard(mid)
                node.full.pop(mid, None)
                node.count["expired"] += 1
            for mid, (tgt, t0) in drop_del:
                node.msgs.pop(mid, None)
                node.got.discard(mid)
                node.keys.discard((int(t0), mid))
                node.keys.discard((int(t0), tgt))
                node.gone.discard(tgt)
                node.tomb.pop(tgt, None)
                node.count["expired_tomb"] += 1
        if node.st:
            for mid in drop + [m for m, _ in drop_del]:
                node.st.forget(mid)


# ---- test and debug (cfg "faults", /api/holds)
def set_fault(node: Node, d: dict) -> dict:
    """Test-only (cfg "faults"): cut links to listed peers, shift this box's clock, swallow the next envelopes."""
    with node.lock:
        if "drop" in d:
            node.fault["drop"] = {int(x) for x in d["drop"]}
        if "offset" in d:
            node.fault["offset"] = float(d["offset"])
        if "silent" in d:
            node.fault["silent"] = int(d["silent"])
        if "stall" in d:                                       # CP30: {"peer": p, "s": seconds}, one session
            node.fault["stall"] = d["stall"]
        if "no_fence" in d:                                    # CP30: the fence off (to show it matters)
            node.fault["no_fence"] = bool(d["no_fence"])
    if "recon_now" in d:                                       # CP30: start a session with this peer now
        recon_bg(node, int(d["recon_now"]), "t")
    with node.lock:
        return {"drop": sorted(node.fault["drop"]), "offset": node.fault["offset"], "silent": node.fault["silent"]}


def holds(node: Node) -> dict:
    """Test/debug: the reconciled set, the held envelopes and the forgotten ids (ids only)."""
    with node.lock:
        return {"keys": [[t, str(i)] for t, i in node.keys.keys], "msgs": [str(x) for x in node.msgs],
                "keys_by_sub": {str(s): [[t, str(i)] for t, i in ks.keys] for s, ks in node.keys.by.items()},
                "subs": sorted(node.box.topics),
                "gone": [str(x) for x in node.gone], "count": dict(node.count),
                "clean": sorted(node.clean), "fetching": len(node.fetching), "busy": sorted(node.recon_busy)}
