"""Box software of the design (CP24), on top of sim.box (CP17), for the demo boxes and DayNet.

- O-b (CP19): table lookup (same epoch, nullifier and x = a replay) -> proof -> Ed25519 -> nullifiers.
- I5 + F1 (CP23): a text message spends 1 or 2 quarter-unit text indices (live text is encoded up to 128 tokens and
  weighed the same way), a picture post one more index in a separate picture space. Every index has its own RLN share
  and nullifier, so spending any index twice reveals the secret. Votes and deletes spend one index of an action space
  (CP17's limit, 20 per 10 minutes; outside the CP23 model).
- T3 (CP22): the age group is public (it picks the group's tree and limits). The old group's window is 10 minutes,
  the new group's 1 hour.
- The box checks after decryption that the text index count covers the tokens (outside the circuit, CP23).

The pairing check is still not run: `proof_ok` stands for it (CP17), and is false once the author's identity has
been removed from the group.
"""
from __future__ import annotations

import json
import os

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from . import ml
from .box import R, Box, Msg, World, h, recover
from .stack import DayNet

F1_TOKENS = 128          # live text is encoded up to here (CP23 F1)
TEXT_SPLIT = 82          # 1 text index up to 81 tokens, 2 from 82 (sim_cp19.EncTable(8, 1024): w(81) <= 1/4 < w(82))
LIMITS = {"old": {"epoch": 600.0, "t": 7, "p": 1}, "new": {"epoch": 3600.0, "t": 18, "p": 2}}   # CP23 I5 + F1
ACT = {"epoch": 600.0, "a": 20}
TEXT_KINDS = ("post", "comment", "room")


def text_indices(tokens: int) -> int:
    return 1 if tokens < TEXT_SPLIT else 2


def need(kind: str, tokens: int, picture: bool) -> dict:
    """Indices a message spends, per space (main space first)."""
    if kind in TEXT_KINDS:
        return {"t": text_indices(tokens), **({"p": 1} if picture else {})}
    return {"a": 1}


def window(tier: str, space: str) -> float:
    return ACT["epoch"] if space == "a" else LIMITS[tier]["epoch"]


def limit(tier: str, space: str) -> int:
    return ACT["a"] if space == "a" else LIMITS[tier][space]


def share(a0: int, epoch: int, space: str, k: int, x: int) -> tuple[int, int]:
    """RLN with an index space: a1 = H(a0, epoch, space, k), y = a0 + a1 x, nullifier = H(a1)."""
    a1 = h("a1", a0, epoch, space, k)
    return (a0 + a1 * x) % R, h("nul", a1)


class Over(Exception):
    """The honest client has no index left in this window: wait until `until`."""

    def __init__(self, space: str, until: float):
        super().__init__(f"no {space} index left until t = {until:.0f}")
        self.space, self.until = space, until


class World24(World):
    def __init__(self, n_users: int, seed: int = 17, tiers: list | None = None):
        super().__init__(n_users, seed=seed)
        self.tier = list(tiers) if tiers else ["new"] * n_users
        self.used: dict[tuple, int] = {}          # (user, space, epoch) -> next free index
        self.waits = 0                            # honest messages that had to wait for the next window

    def budget(self, user: int, t: float) -> dict:
        tier = self.tier[user]
        out = {}
        for sp in ("t", "p", "a"):
            w = window(tier, sp)
            e = int(t // w)
            out[sp] = {"left": limit(tier, sp) - self.used.get((user, sp, e), 0), "limit": limit(tier, sp),
                       "window_s": w, "resets_in_s": (e + 1) * w - t}
        return out

    def make(self, user: int, kind: str, topic: int, t: float, body: dict, size: float, key: bytes, *,
             target: int | None = None, reuse: bool = False, proof_ok: bool = True, image: int | None = None,
             tokens: int = 0, pull_bytes: float = 0.0, truth: dict | None = None) -> Msg:
        ident, tier = self.ids[user], self.tier[user]
        spend = need(kind, tokens, image is not None and kind == "post")
        plan = []
        for i, (sp, n) in enumerate(spend.items()):
            e = int(t // window(tier, sp))
            used = self.used.get((user, sp, e), 0)
            start = max(used - 1, 0) if (reuse and i == 0) else used    # misbehaving: the main space's last index again
            if start + n > limit(tier, sp) and not reuse:
                raise Over(sp, (e + 1) * window(tier, sp))
            plan.append((sp, e, start, n))
        for sp, e, start, n in plan:
            self.used[(user, sp, e)] = max(self.used.get((user, sp, e), 0), start + n)
        sk = self.keys[target] if kind == "delete" else Ed25519PrivateKey.generate()
        nonce = os.urandom(12)
        plain = json.dumps(body).encode()
        x = h("x", nonce, plain)
        ys = [share(ident.a0, e, sp, k, x) for sp, e, start, n in plan for k in range(start, start + n)]
        epoch = plan[0][1]
        mid = len(self.msgs)
        vnul = ident.vote_nullifier(target) if kind == "vote" else None
        m = Msg(mid, kind, topic, t, size, sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw), b"", nonce,
                b"", epoch, x, ys[0][0], ys[0][1], vnul, proof_ok and ident.commit in self.members, image, tokens,
                pull_bytes, {"user": user, "commit": ident.commit, "target": target, **(truth or {})},
                tier, spend.get("t", 0), spend.get("p", 0), [[nul, y] for y, nul in ys[1:]])
        m.ct = ChaCha20Poly1305(key).encrypt(nonce, plain, m.header())
        m.sig = sk.sign(m.header() + nonce + m.ct)
        self.keys[mid] = sk
        self.msgs.append(m)
        return m


def stages(m: Msg, verdict: str) -> tuple:
    """Operations a box runs on a message before it stops (O-b order)."""
    if m.kind == "slash":
        return ("ed25519",)
    if verdict == "dup_lookup":
        return ("lookup",)
    if verdict == "bad_proof":
        return ("lookup", "proof")
    if verdict == "bad_sig":
        return ("lookup", "proof", "ed25519")
    return ("lookup", "proof", "ed25519", "rln")


class Box24(Box):
    def validate(self, m: Msg) -> tuple[str, int | None]:
        if m.kind == "slash":
            return super().validate(m)
        seen = self.rln.get((m.epoch, m.nul))                       # (1) lookup
        if seen is not None and seen[0] == m.x:
            return "dup_lookup", None
        n_shares = (m.n_t + m.n_p) if m.kind in TEXT_KINDS else 1     # (2) proof, with the key picked by (n_t, n_p)
        picture = m.image is not None and m.kind == "post"
        if (not m.proof_ok or len(m.shares) != n_shares - 1 or (picture and m.n_p != 1)
                or (m.kind in TEXT_KINDS and m.n_t not in (1, 2))):
            return "bad_proof", None
        if not self.S.sig_ok(m):                                      # (3) signature
            return "bad_sig", None
        pts = [(m.nul, m.y)] + [(int(a), int(b)) for a, b in m.shares]
        for nul, y in pts:                                            # (4) nullifiers
            prev = self.rln.get((m.epoch, nul))
            if prev is not None and prev[0] != m.x:
                return "slash", recover(prev[0], prev[1], m.x, y)
            if prev is not None:
                return "dup", None
        for nul, y in pts:
            self.rln[(m.epoch, nul)] = (m.x, y)
        if m.kind == "vote":
            if m.vnul in self.vnul:
                return "double_vote", None
            self.vnul.add(m.vnul)
        return "ok", None

    def apply(self, m: Msg, t: float, verdict: tuple, tokens: int | None = None) -> str:
        """Box.apply, after the index count is checked against the decrypted text (outside the circuit)."""
        if verdict[0] == "ok" and m.kind in TEXT_KINDS and m.n_t < text_indices(m.tokens if tokens is None else tokens):
            self.log.append((t, m.mid, "underweight"))
            return "underweight"
        return super().apply(m, t, verdict)


class DayNet24(DayNet):
    """DayNet with the CP24 box: O-b order and operations, F1 encoding, I5 shares. Units stay FIFO (CP24 N-a)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ref = Box24(-1, (), {}, self.S, {"text": 0})

    def deliver(self, box: int, mid: int, frm: int) -> None:
        m = self.W.msgs[mid]
        g = self.got[mid]
        if g[box]:
            self.dup_bytes[box] += m.size
            return
        g[box] = True
        t, a, c = self.eng.t, self.a, self.c
        self.n[mid] += 1
        self.tmax[mid] = t
        if mid in self.arr:
            self.arr[mid][box] = t
        v = self.verdict[mid]
        t1 = t
        for op in stages(m, v[0]):
            if op == "proof":
                t1 = a.run(box, "verify", t1, c.verify, "proof")
            else:
                dur = {"lookup": c.nullifier, "ed25519": c.sig, "rln": c.nullifier * (1 + len(m.shares))}[op]
                t1 = a.run(box, "host", t1, dur, op)
        sb = self.sample.get(box)
        sv = None
        if sb is not None:
            sv = sb.validate(m)
            if sv[0] != v[0]:
                self.mismatch += 1
        if v[0] != "ok" and frm >= 0:
            if v[0] == "slash":
                self._slash(box, v[1], t1)
            if sb is not None:
                sb.apply(m, t1, sv)
            return
        for peer in self.mesh[m.topic].get(box, ()):
            if peer != frm:
                self._send(box, int(peer), m.size, t1, self.deliver, int(peer), mid, box)
        if v[0] != "ok":
            return
        if m.kind == "slash":
            if sb is not None:
                sb.apply(m, t1, sv)
            return
        t2 = a.run(box, "host", t1, c.decrypt(m.size), "decrypt")
        if m.kind in TEXT_KINDS:
            if m.n_t < text_indices(m.tokens):
                return
            t3 = a.run(box, "enc_text", t2, ml.chip_text_s(min(m.tokens, F1_TOKENS)), "text")
            t4 = a.run(box, "host", t3, c.insert, "insert")
            self.index_n[box] += 1
            if sb is not None:
                self.searchable[(box, mid)] = t4
            if m.image is not None:
                if frm < 0:
                    self._index_image(box, mid, t2)
                else:
                    self.pull[mid]["parent"][box] = frm
                    self._request(box, mid, t2)
        elif m.kind == "vote":
            a.run(box, "host", t2, c.tally, "tally")
        elif m.kind == "delete":
            a.run(box, "host", t2, c.insert, "delete")
            self.index_n[box] -= 1
        if sb is not None:
            sb.apply(m, t1, sv)
