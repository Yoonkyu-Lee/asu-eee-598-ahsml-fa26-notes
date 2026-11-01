"""CP21: the same small scenario on five real box processes (E1) and on the simulator (E2), compared box by box and
operation by operation (CP15 rule X2), then the demo function checks D1-D6 on the real boxes.

  .venv/Scripts/python.exe src/demo/x2.py          -> results/cp21/cp21.json   (box logs in logs/cp21_box*.log)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import A, B, IMG_BYTES, MEMB, ROOM, Coco, community, msg_to_json, save_cfg  # noqa: E402
from bench_cp1 import load_posts  # noqa: E402
from node import Encoder  # noqa: E402
from sim.box import Shared, World, h, sub_key  # noqa: E402
from sim.stack import Accel, Cost, DayNet  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "results" / "cp21"
LOGS = ROOT / "logs"
OPS = ("ed25519", "proof", "rln", "decrypt", "text", "image", "insert", "tally", "delete", "query", "send", "pull", "dup")
N_PRELOAD = 100
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ scenario

def scenario(cfg: dict) -> tuple[World, list, dict]:
    n = cfg["n"]
    W = World(n, seed=cfg["seed"])
    coco = Coco()
    tok = Encoder("int8").tok
    room_key = bytes.fromhex(cfg["room_key"])
    hold = {int(t): v for t, v in cfg["holders"].items()}
    acts: list = []
    tags: dict = defaultdict(list)
    title_of: dict = {}
    clock = [0.0]

    def add(user, kind, topic, body, tag=None, **kw):
        clock[0] += 30.0
        text = ((body.get("title") or "") + "\n\n" + (body.get("body") or "")).strip()
        key = room_key if topic == ROOM else sub_key(topic)
        m = W.make(user, kind, topic, clock[0], body, 400 + len(json.dumps(body)), key,
                   tokens=len(tok.encode(text).ids) if text else 0, **kw)
        acts.append(("msg", m.mid, user))
        title_of[m.mid] = body.get("title") or body.get("body") or ""
        if tag:
            tags[tag].append(m.mid)
        return m

    for j in range(N_PRELOAD):                       # photos already in the sub (D2)
        i = 300 + j
        add(j % n, "post", A, {"title": coco.caps[i][0], "body": ""}, "preload", image=i, pull_bytes=IMG_BYTES)
    rows = load_posts(20)   # 0-11 posts, 12-19 comments; room notes reuse 17-19 titles (never post titles)
    posts = []
    for j in range(12):
        u, topic = (j % n, A) if j % 3 else (hold[B][j % len(hold[B])], B)
        if j < 4:
            i = 400 + j
            m = add(u, "post", topic, {"title": coco.caps[i][0], "body": ""}, "post", image=i, pull_bytes=IMG_BYTES)
        else:
            r = rows[j]
            m = add(u, "post", topic, {"title": r["title"], "body": r["body"][:1500]}, "post")
        posts.append(m)
    for j in range(8):
        tgt = posts[j % len(posts)]
        u = hold[tgt.topic][(j + 1) % len(hold[tgt.topic])]
        add(u, "comment", tgt.topic, {"body": rows[12 + j]["body"][:180] or rows[12 + j]["title"], "parent": tgt.mid}, "comment")
    voters, pairs = Counter(), set()
    j = 0
    while sum(voters.values()) < 10:                 # ten votes, no (voter, post) pair twice
        tgt = posts[(2 * j) % len(posts)]
        u = hold[tgt.topic][(j + 2) % len(hold[tgt.topic])]
        j += 1
        if (u, tgt.mid) in pairs:
            continue
        pairs.add((u, tgt.mid))
        add(u, "vote", tgt.topic, {"target": tgt.mid, "dir": 1}, "vote", target=tgt.mid)
        voters[tgt.mid] += 1
    first_vote = next(m for m in W.msgs if m.kind == "vote")
    add(first_vote.truth["user"], "vote", first_vote.topic, {"target": first_vote.truth["target"], "dir": 1}, "double_vote",
        target=first_vote.truth["target"])
    for m in (posts[5], posts[9]):
        add(m.truth["user"], "delete", m.topic, {"target": m.mid}, "delete", target=m.mid)
    for j in range(3):
        add(hold[ROOM][j], "room", ROOM, {"body": f"room note {j}: {rows[17 + j]['title']} (said in the room)"}, "room")
    add(1, "comment", A, {"body": "this one carries a proof that does not verify"}, "bad_proof", proof_ok=False)
    # rate-limit breach: user 3 reuses the message index it just used in this epoch -> its secret is revealed
    m1 = add(3, "comment", A, {"body": "an ordinary comment"}, "before_reuse")
    k = W.counter[(3, m1.epoch)] - 1
    clock[0] -= 29.0                                  # same epoch as m1
    add(3, "comment", A, {"body": "a second comment on the same share"}, "reuse", reuse_k=k)
    W.members.discard(W.ids[3].commit)               # the group drops the revealed identity
    add(3, "post", A, {"title": "posted after being slashed", "body": "should be refused"}, "after_slash")
    for j, m in enumerate(posts[:5]):
        acts.append(("search", j % n, title_of[m.mid]))
    exp_votes = {mid: c for mid, c in voters.items()}
    return W, acts, {"tags": dict(tags), "votes": exp_votes, "posts": [m.mid for m in posts], "titles": title_of}


# ------------------------------------------------------------------ E1: processes

def get(port: int, path: str):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=300).read())


def post(port: int, path: str, obj: dict):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def quiet(ports: list[int]) -> None:
    calm, last = 0, None
    while calm < 3:
        st = [get(p, "/api/stats") for p in ports]
        snap = [(s["pending"], sum(s["count"].values())) for s in st]
        calm = calm + 1 if all(x[0] == 0 for x in snap) and snap == last else 0
        last = snap
        time.sleep(0.03)


def e1(cfg: dict, W: World, acts: list) -> dict:
    LOGS.mkdir(exist_ok=True)
    cfg_path = LOGS / "cp21_cfg.json"          # holds the demo room key: kept out of git
    save_cfg(cfg, cfg_path)
    py = sys.executable
    procs = [subprocess.Popen([py, str(HERE / "box_server.py"), "--cfg", str(cfg_path), "--id", str(i)],
                              stdout=open(LOGS / f"cp21_box{i}.log", "w"), stderr=subprocess.STDOUT)
             for i in range(cfg["n"])]
    ports = cfg["ports"]
    try:
        for p in ports:
            for _ in range(600):
                try:
                    get(p, "/api/stats")
                    break
                except Exception:
                    time.sleep(0.5)
        log("E1 boxes up")
        wall, searches = {}, []
        for a in acts:
            if a[0] == "msg":
                t = time.time()
                post(ports[a[2]], "/inject", {"msg": msg_to_json(W.msgs[a[1]])})
                quiet(ports)
                wall[a[1]] = time.time() - t
            else:
                searches.append({"box": a[1], "q": a[2], "top": get(ports[a[1]], "/api/search?q=" + quote(a[2]))[:3]})
                quiet(ports)
        stats = [get(p, "/api/stats") for p in ports]
        debug = [get(p, "/api/debug") for p in ports]
        log("E1 scenario done")
        return {"stats": stats, "debug": debug, "wall_s": wall, "searches": searches, "ports": ports, "procs": procs}
    except Exception:
        for p in procs:
            p.terminate()
        raise


# ------------------------------------------------------------------ E2: simulator

class CountAccel(Accel):
    def __init__(self, M, **kw):
        super().__init__(M, **kw)
        self.names = defaultdict(Counter)

    def run(self, box, unit, t, dur, name=""):
        self.names[box][name] += 1
        return super().run(box, unit, t, dur, name)


class CountNet(DayNet):
    def _send(self, frm, to, nbytes, t_ready, fn, *args):
        self.a.names[frm]["pull" if getattr(fn, "__name__", "") == "got_full" else "send"] += 1
        return super()._send(frm, to, nbytes, t_ready, fn, *args)

    def deliver(self, box, mid, frm):
        if self.got[mid][box]:
            self.a.names[box]["dup"] += 1
        return super().deliver(box, mid, frm)


def e2(cfg: dict, W: World, acts: list) -> dict:
    n = cfg["n"]
    cost = Cost(**json.loads((ROOT / "results" / "cp17" / "cp17.json").read_text(encoding="utf-8"))["cost_box_s"])
    members = {int(t): np.array(v) for t, v in cfg["holders"].items()}
    acc = CountAccel(n, hours=24)
    net = CountNet(members, n, W, Shared(), {}, cost, acc, uplink_bps=20e6, hop_s=0.05, membership_topic=MEMB,
                   D=cfg["D"], seed=cfg["seed"])
    net.mesh = {int(t): {int(b): np.array(p, dtype=int) for b, p in m.items()} for t, m in cfg["mesh"].items()}
    for i, a in enumerate(acts):
        t = 30.0 * (i + 1)
        if a[0] == "msg":
            net.eng.at(t, net.publish, W.msgs[a[1]], a[2])
        else:
            net.eng.at(t, net.search, a[1], 12)
    net.eng.run()
    return {"names": {b: dict(acc.names[b]) for b in range(n)}, "events": net.eng.events}


# ------------------------------------------------------------------ checks

def checks(cfg, W, info, r1) -> dict:
    ports, hold = r1["ports"], {int(t): v for t, v in cfg["holders"].items()}
    tags, dbg = info["tags"], r1["debug"]
    deleted = {W.msgs[m].truth["target"] for m in tags["delete"]}
    out = {}
    # D1: every live post is first for its title at every holder; and settled within 5 s
    d1, miss = 0, []
    for mid in tags["post"] + tags["preload"][:10]:
        if mid in deleted:
            continue
        m = W.msgs[mid]
        for b in hold[m.topic]:
            q = r1["titles"][mid]
            top = get(ports[b], "/api/search?q=" + quote(q))
            ok = bool(top) and top[0]["mid"] == str(mid)
            d1 += ok
            if not ok:
                miss.append((mid, b, top[0]["mid"] if top else None))
    out["D1"] = {"hits": d1, "misses": miss, "max_settle_s": max(r1["wall_s"].values()),
                 "pass": not miss and max(r1["wall_s"].values()) <= 5.0}
    # D2: photo search equals an offline ranking with the same models; Recall@10 for second captions
    from bench_cp18 import ClipText
    from sim import ml
    coco = Coco()
    img_mids = [mid for mid in tags["preload"] + tags["post"] if W.msgs[mid].image is not None and mid not in deleted]
    ienc, ct = ml.ImageEncoder("fp32", threads=1), ClipText(threads=1)
    E = ienc.encode(np.stack([coco.pixels(W.msgs[m].image) for m in img_mids]))
    same, rec = 0, 0
    qs = tags["preload"][:20]
    for mid in qs:
        q = coco.caps[W.msgs[mid].image][1]
        got = [int(x["mid"]) for x in get(ports[0], "/api/search_image?q=" + quote(q))]
        off = [img_mids[i] for i in np.argsort(-(E @ ct.encode([q])[0]))[:10]]
        same += got == off
        rec += mid in got
    out["D2"] = {"identical_rankings": same, "queries": len(qs), "recall@10": rec / len(qs), "pass": same == len(qs)}
    # D3: deleted posts are gone from search and feed at every holder
    d3 = []
    for mid in deleted:
        m = W.msgs[mid]
        for b in hold[m.topic]:
            top = [x["mid"] for x in get(ports[b], "/api/search?q=" + quote(r1["titles"][mid]))]
            feed = [x["mid"] for x in get(ports[b], "/api/feed")]
            d3.append(str(mid) not in top and str(mid) not in feed and str(mid) in dbg[b]["deleted"])
    out["D3"] = {"checks": len(d3), "pass": all(d3)}
    # D4: room messages readable only inside the room
    d4 = []
    for b in range(cfg["n"]):
        r = get(ports[b], "/api/room")
        inside = b in hold[ROOM]
        texts = [x["body"] for x in r["items"]]
        d4.append(len(texts) == len(tags["room"]) if inside else (not r["member"] and not texts))
        hits = [x["mid"] for x in get(ports[b], "/api/search?q=" + quote("said in the room"))]
        d4.append(any(str(mid) in hits for mid in tags["room"]) == inside)
    out["D4"] = {"pass": all(d4)}
    # D5: one vote per identity, slashing, refusal after slashing
    tgt = W.msgs[tags["double_vote"][0]].truth["target"]
    scores = [dbg[b]["scores"].get(str(tgt)) for b in hold[W.msgs[tgt].topic]]
    banned = [str(h("id", W.ids[3].a0)) in dbg[b]["banned"] for b in range(cfg["n"])]
    after = tags["after_slash"][0]
    refused = [str(after) not in dbg[b]["scores"] for b in range(cfg["n"])]
    out["D5"] = {"scores_of_double_voted_post": scores, "expected": info["votes"][tgt], "banned_everywhere": banned,
                 "after_slash_refused": refused,
                 "pass": all(s == info["votes"][tgt] for s in scores) and all(banned) and all(refused)}
    # D6: the bad proof is verified only by its author and the author's mesh neighbours
    bad = tags["bad_proof"][0]
    m = W.msgs[bad]
    author = m.truth["user"]
    allowed = {author, *cfg["mesh"][str(m.topic)][str(author)]}
    who = [b for b in range(cfg["n"]) if str(bad) in dbg[b]["verified"]]
    out["D6"] = {"verified_by": who, "allowed": sorted(allowed), "pass": set(who) <= allowed}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = community()
    W, acts, info = scenario(cfg)
    log("scenario", len(acts), "actions,", len(W.msgs), "messages")
    r1 = e1(cfg, W, acts)
    r1["titles"] = info["titles"]
    try:
        res = {"community": {k: v for k, v in cfg.items() if k != "room_key"}, "actions": len(acts),
               "messages": len(W.msgs)}
        r2 = e2(cfg, W, acts)
        log("E2 done", r2["events"], "events")
        table, diff = {}, []
        for b in range(cfg["n"]):
            c1, c2 = r1["stats"][b]["count"], r2["names"][b]
            table[b] = {op: [c1.get(op, 0), c2.get(op, 0)] for op in OPS}
            diff += [(b, op, *v) for op, v in table[b].items() if v[0] != v[1]]
        res["x2"] = {"table": table, "differences": diff, "pass": not diff,
                     "e1_slash_published": [s["count"].get("slash_published", 0) for s in r1["stats"]]}
        log("X2", "pass" if not diff else diff)
        res["e1_wall_s"] = {"max": max(r1["wall_s"].values()), "mean": float(np.mean(list(r1["wall_s"].values())))}
        res["searches"] = r1["searches"]
        res["checks"] = checks(cfg, W, info, r1)
        res["chip_virtual_ms"] = [s["chip_virtual_ms"] for s in r1["stats"]]
        for k, v in res["checks"].items():
            log(k, v)
        (OUT / "cp21.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        log("wrote")
    finally:
        for p in r1["procs"]:
            p.terminate()


if __name__ == "__main__":
    main()
