"""The ingest path (mirrors sim.proto24.DayNet24.deliver, same operation names): checks, the set, the disk, forwarding,
decryption, the encoder queue; slashing notices; picture pulls (I2); taking the stored messages back at start (CP25)."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

import disk
import encoder
import peer_sync
from box_state import Node, log
from common import MEMB, msg_from_json, msg_to_json
from encoder import CHIP
from sim.box import Msg, content_id, h, ref_target
from sim import proto24 as p24


def deliver(node: Node, m: Msg, frm: int, via: str = "gossip") -> None:
    """CP29: check everything a receiver can check BEFORE the id enters the set, the disk or any served list."""
    with node.lock:
        if m.mid in node.gone:                                 # deleted by its author: a stale copy (CP25)
            node.count["forgotten"] += 1
            return
        if m.mid in node.got:
            node.count["dup"] += 1
            return
        if content_id(m) != m.mid:                             # someone else's id, or altered on the way
            node.count["rej_id"] += 1
            return
        if m.topic not in node.box.topics:                     # CP33: not my sub (header), before verifying
            node.count["rej_sub"] += 1
            return
        tk = m.t0
        if m.kind == "delete":
            rt = ref_target(m)
            if rt is None:                                     # the target header is not bound to this key
                node.count["rej_ref"] += 1
                return
            tk = rt[1]
        now = node.now()
        if not (now - node.R <= tk <= now + 2 * node.sigma):
            node.count["refuse_window"] += 1
            return
        if m.epub.hex() in node.tomb.get(m.mid, ()):           # its delete came first
            node.count["forgotten"] += 1
            node.gone.add(m.mid)
            return
        v = node.box.validate(m)
        if v[0] != "ok":                                       # counted and refused; nothing is stored or served
            _refuse(node, m, frm, via, v)
            return
        node.got.add(m.mid)
        node.msgs[m.mid] = m
        node.keys.add((int(tk), m.mid), m.topic)
        node.ver += 1                                          # CP30 F: my set changed; only the sender
        node.clean = (node.clean & {frm}) if frm >= 0 else set()   # still matches me
        node.fetching.pop(m.mid, None)
        node.skipped.pop(m.mid, None)
        node.lazy_pending.discard(m.mid)
        if node.st and not node.replaying:
            node.st.add_message(msg_to_json(m), frm)
        if frm >= 0 and not node.replaying:
            node.peer_seen[frm] = time.time()
            node.up.add(frm)
        _take(node, m, frm, via, v)


def _refuse(node: Node, m: Msg, frm: int, via: str, v: tuple) -> None:
    """A message that failed validation (called with the lock held): counted, a reused share is published as a
    slashing notice, and nothing is stored or served."""
    node.count["rej_" + v[0]] += 1
    if node.st and not node.replaying:
        node.st.ev("recv", mid=str(m.mid), frm=frm, kind=m.kind, via=via, verdict=v[0])
    if v[0] == "slash" and frm >= 0 and not node.replaying:
        publish_slash(node, v[1])
    if frm >= 0:
        node.box.apply(m, time.time(), v)
    elif not node.replaying:                           # my own (the /api/misbehave demo): it goes out
        for peer in peer_sync.targets(node, m.topic):  # anyway, as an envelope (not stored, so no id)
            peer_sync.send(node, peer, m, direct=True)


def _take(node: Node, m: Msg, frm: int, via: str, v: tuple) -> None:
    """The rest of ingest of a valid message (called with the lock held): stages, forwarding, decryption,
    encoding, apply."""
    t = time.time()
    for op in p24.stages(m, v[0]):
        node.count[op] += 1
        if op == "proof":
            node.verified.append(m.mid)
            node.chip_s += CHIP["proof_s"]
    if node.st and not node.replaying:
        node.st.ev("recv", mid=str(m.mid), frm=frm, kind=m.kind, via=via, verdict=v[0])
    for peer in peer_sync.targets(node, m.topic):
        if peer != frm and not node.replaying:
            peer_sync.send(node, peer, m)
    if m.kind == "slash":
        node.box.apply(m, t, v)
        _notice(node, m)
        return
    node.count["decrypt"] += 1
    _ingest24(node, m, frm, t, v)


def _ingest24(node: Node, m: Msg, frm: int, t: float, v: tuple) -> None:
    """After decryption (CP24): count check, then the encoder queue (text first, pictures last; old group first)."""
    if m.kind in p24.TEXT_KINDS:
        txt = text_of(node, m)
        if txt is None:
            node.box.apply(m, t, v)
            return
        feed = node.text_enc.tokenize([txt])
        ntok = int(feed["attention_mask"].sum())
        if node.box.apply(m, t, v, tokens=ntok) != "ok":     # underweight: not indexed
            return
        feed = {k: a[:, :p24.F1_TOKENS] for k, a in feed.items()}
        encoder.enqueue(node, 1 if m.tier == "old" else 3, ("text", m.mid, feed))
        if m.image is not None and m.kind == "post":
            node.full_ev.setdefault(m.mid, threading.Event())
            if frm < 0 and m.mid not in node.full and node.coco is not None:
                disk.have_picture(node, m.mid, node.coco.pixels(m.image).tobytes())
            if m.mid in node.full:                          # mine, or stored (CP25)
                node.full_ev[m.mid].set()
                encoder.enqueue(node, 4 if m.tier == "old" else 5, ("image", m.mid))
            elif frm >= 0:
                node.parent[m.mid] = frm
                _request(node, m.mid)
        return
    node.box.apply(m, t, v)
    if m.kind == "vote":
        node.count["tally"] += 1
    elif m.kind == "delete":
        node.count["delete"] += 1
        if node.st:
            disk.after_delete(node, m)


def text_of(node: Node, m: Msg) -> str | None:
    key = node.box.key_for(m.topic)
    body = node.S.open(m, key) if key else None
    if body is None:
        return None
    return ((body.get("title") or "") + "\n\n" + (body.get("body") or "")).strip()


def _notice(node: Node, m: Msg) -> None:
    a0 = int(json.loads(m.ct)["a0"])
    c = h("id", a0)
    mine = c == node.commit
    node.notices.append({"t": time.time(), "identity": hex(c)[2:10], "mine": mine})
    node.W.members.discard(c)
    if mine:
        node.slashed = True


def publish_slash(node: Node, a0: int) -> None:
    """A reused RLN share gives the identity secret a0: publish it once (content-addressed, so the notices of
    boxes that found the same reuse are one message)."""
    mid = (1 << 40) + h("slash-id", a0) % (1 << 30)
    if a0 in node.slash_sent or mid in node.got:           # already published, or already got someone's notice
        return
    node.slash_sent.add(a0)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    # CP29: ids are content hashes, so the notice is made the same on every finder (key, nonce and the hour are
    # drawn from a0; Ed25519 signatures are deterministic): finders in the same hour publish one message
    import hashlib
    seed = hashlib.blake2b(b"slash|" + str(a0).encode(), digest_size=44).digest()
    sk, nonce = Ed25519PrivateKey.from_private_bytes(seed[:32]), seed[32:]
    t0 = float(int(node.now() // 3600) * 3600)
    s = Msg(mid, "slash", MEMB, t0, 400.0, sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
            b"", nonce, b"", 0, 0, 0, 0, None, True)
    s.ct = json.dumps({"a0": str(a0)}).encode()
    s.sig = sk.sign(s.header() + nonce + s.ct)
    s.mid = content_id(s)
    if s.mid in node.got:                                      # someone's identical notice is already here
        return
    node.count["slash_published"] += 1
    deliver(node, s, -1)


# ---- picture pulls (I2): the envelope came without its picture; fetch it from the sender first, then the others
def _request(node: Node, mid: int) -> None:
    if mid in node.asked:
        return
    node.asked.add(mid)
    node.inc()

    def go():
        try:
            order = [node.parent[mid]] + [p for p in range(node.n) if p not in (node.parent[mid], node.bid)]
            for peer in order:
                try:
                    data = urllib.request.urlopen(node.url(peer, f"/pull?mid={mid}&frm={node.bid}"),
                                                  timeout=300).read()
                except Exception as e:
                    log(f"pull {mid} from box {peer} failed:", e)
                    continue
                with node.lock:
                    if node.forgotten(mid):
                        return
                    disk.have_picture(node, mid, data)
                    node.full_ev[mid].set()
                    encoder.enqueue(node, 4 if node.msgs[mid].tier == "old" else 5, ("image", mid))
                return
        finally:
            node.inc(-1)
    node.pool.submit(go)


def serve_pull(node: Node, mid: int) -> bytes | None:
    with node.lock:
        if node.forgotten(mid):
            return None
        ev = node.full_ev.setdefault(mid, threading.Event())
        if mid not in node.full and mid in node.parent:
            _request(node, mid)
    ev.wait(300)
    with node.lock:
        node.count["pull"] += 1
        return node.full.get(mid)


def replay(node: Node) -> int:
    """Take the stored messages again, in arrival order: same checks, same index; nothing is sent."""
    t0 = time.perf_counter()
    rows = node.st.messages()
    node.replaying = True
    try:
        for d in rows:
            m = msg_from_json(d)
            pic = node.st.picture(m.mid)
            if pic is not None:
                node.full[m.mid] = pic
            node.replayed.add(m.mid)
            deliver(node, m, int(d.get("_frm", -1)))
    finally:
        node.replaying = False
    ms = (time.perf_counter() - t0) * 1e3
    node.st.ev("start", restored=len(rows), ms=ms, identity=hex(node.commit)[2:10], tier=node.tier)
    if rows:
        log(f"restored {len(rows)} messages from {node.st.dir} in {ms / 1e3:.1f} s (encoding continues in the background)")
    return len(rows)
