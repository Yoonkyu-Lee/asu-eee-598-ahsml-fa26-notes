"""A day of the community with the box stack (CP17): gossip over topic meshes, every box charged for its work on
a virtual accelerator, sample boxes running the full box software (box.Box).

Per arriving message, at every box: Ed25519 (host) -> proof (verify unit) -> RLN / nullifier check (host) ->
forward if valid -> decrypt (host) -> text: encode (enc_text) + index insert (host); image post: pull the picture
from the gossip parent (I2), encode (enc_image) + insert; vote: tally (host); delete: remove (host).
Units are FIFO servers per box. Their busy time is kept per box per hour, for all boxes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import ml
from .box import Box, Msg, Shared, World
from .engine import Engine
from .network import meshes
from .trace import Trace

UNITS = ("verify", "enc_text", "enc_image", "host")


@dataclass
class Cost:
    """Seconds on the box. Host numbers: laptop measurement x Pi 5 factor. Chip numbers: CP12 / CP14 models."""
    sig: float
    nullifier: float
    decrypt_fixed: float
    decrypt_per_b: float
    insert: float
    tally: float
    search_fixed: float
    search_per_vec: float
    feed_fixed: float
    feed_per_item: float
    verify: float
    image_enc: float

    def decrypt(self, nbytes: float) -> float:
        return self.decrypt_fixed + self.decrypt_per_b * nbytes


class Accel:
    def __init__(self, M: int, hours: int = 24, watch=(), window=(0.0, 0.0)):
        self.free = {u: [0.0] * M for u in UNITS}
        self.busy = {u: np.zeros((M, hours)) for u in UNITS}
        self.ops = {u: np.zeros(M) for u in UNITS}
        self.H = hours
        self.watch, self.window = set(watch), window
        self.trace = Trace()

    def run(self, box: int, unit: str, t: float, dur: float, name: str = "") -> float:
        f = self.free[unit]
        start = t if t > f[box] else f[box]
        f[box] = start + dur
        h = int(start // 3600)
        self.busy[unit][box, h if 0 <= h < self.H else (0 if h < 0 else self.H - 1)] += dur
        self.ops[unit][box] += 1
        if box in self.watch and self.window[0] <= start < self.window[1]:
            self.trace.span(f"box {box}", unit, name, start, dur)
        return start + dur


class DayNet:
    def __init__(self, members: dict, M: int, world: World, shared: Shared, sample: dict, cost: Cost, accel: Accel,
                 *, uplink_bps: float, hop_s: float, membership_topic: int, D: int = 6, seed: int = 17):
        self.W, self.S, self.sample, self.c, self.a = world, shared, sample, cost, accel
        self.rng = np.random.default_rng(seed)
        self.members = members
        self.mesh = meshes(members, D, self.rng)
        self.M, self.up, self.hop = M, uplink_bps, hop_s
        self.memb_topic = membership_topic
        self.eng = Engine()
        self.uplink_free = [0.0] * M
        self.ref = Box(-1, (), {}, shared, {"text": 0})   # validates every message in publish order
        self.verdict: dict[int, tuple] = {}
        self.got: dict[int, np.ndarray] = {}
        self.arr: dict[int, np.ndarray] = {}               # arrival times, for messages we time (deletes, posts)
        self.n = defaultdict(int)
        self.tmax = defaultdict(float)
        self.dup_bytes = np.zeros(M)
        self.index_n = np.zeros(M)
        self.searchable: dict[tuple, float] = {}           # (sample box, mid) -> time searchable
        self.mismatch = 0                                  # sample box verdict != reference verdict
        self.slashed: set[int] = set()
        self.pull = {}                                     # image mid -> {full, parent, asked, want}
        self.search_lat: list[float] = []
        self.on_publish = None                             # hook (driver bookkeeping)

    # ---- publishing
    def publish(self, m: Msg, author: int) -> None:
        self.verdict[m.mid] = self.ref.validate(m)
        if self.on_publish:
            self.on_publish(m)
        self.got[m.mid] = np.zeros(self.M, bool)
        if m.kind in ("delete", "post", "slash") or m.image is not None:
            self.arr[m.mid] = np.full(self.M, np.nan)
        if m.image is not None:
            self.pull[m.mid] = {"full": {author: self.eng.t}, "parent": {}, "asked": set(), "want": defaultdict(list)}
        self.deliver(author, m.mid, -1)

    def _send(self, frm: int, to: int, nbytes: float, t_ready: float, fn, *args) -> None:
        start = max(t_ready, self.uplink_free[frm])
        self.uplink_free[frm] = start + nbytes * 8 / self.up
        self.eng.at(self.uplink_free[frm] + self.hop, fn, *args)

    # ---- receiving
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
        t1 = a.run(box, "host", t, c.sig, "ed25519")
        if m.kind != "slash":
            t1 = a.run(box, "verify", t1, c.verify, "proof")
            t1 = a.run(box, "host", t1, c.nullifier, "rln")
        v = self.verdict[mid]
        sb = self.sample.get(box)
        sv = None
        if sb is not None:
            sv = sb.validate(m)
            if sv[0] != v[0]:
                self.mismatch += 1
        if v[0] != "ok" and frm >= 0:                         # a relay drops it; the author's own box sends anyway
            if v[0] == "slash" and mid not in self.slashed:   # the first box to see the reuse publishes the secret
                self.slashed.add(mid)
                s = self.W.make_slash(t1, v[1], self.memb_topic)
                self.eng.at(t1, self.publish, s, box)
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
        if m.kind in ("post", "comment", "room"):
            t3 = a.run(box, "enc_text", t2, ml.chip_text_s(m.tokens), "text")
            t4 = a.run(box, "host", t3, c.insert, "insert")
            self.index_n[box] += 1
            if sb is not None:
                self.searchable[(box, mid)] = t4
            if m.image is not None:
                if frm < 0:                      # the author's box has the picture already
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

    # ---- image pulls (I2): ask the gossip parent, which serves once it has the picture
    def _request(self, box: int, mid: int, t_ready: float) -> None:
        P = self.pull[mid]
        if box in P["asked"]:
            return
        P["asked"].add(box)
        self.eng.at(t_ready + self.hop, self._serve_or_wait, P["parent"][box], box, mid)

    def _serve_or_wait(self, holder: int, box: int, mid: int) -> None:
        P = self.pull[mid]
        if holder in P["full"]:
            self._send(holder, box, self.W.msgs[mid].pull_bytes, self.eng.t, self.got_full, box, mid)
        else:
            P["want"][holder].append(box)
            self._request(holder, mid, self.eng.t)

    def got_full(self, box: int, mid: int) -> None:
        P, t = self.pull[mid], self.eng.t
        P["full"][box] = t
        for b in P["want"].pop(box, []):
            self._send(box, b, self.W.msgs[mid].pull_bytes, t, self.got_full, b, mid)
        self._index_image(box, mid, t)

    def _index_image(self, box: int, mid: int, t: float) -> None:
        t1 = self.a.run(box, "enc_image", t, self.c.image_enc, "image")
        t2 = self.a.run(box, "host", t1, self.c.insert, "insert")
        self.index_n[box] += 1
        sb = self.sample.get(box)
        if sb is not None and mid in sb.items and mid not in sb.deleted:
            sb.index("image", mid)
            self.searchable[(box, mid, "image")] = t2

    # ---- the owner's own queries
    def search(self, box: int, q_tokens: int) -> None:
        t, a, c = self.eng.t, self.a, self.c
        t1 = a.run(box, "enc_text", t, ml.chip_text_s(q_tokens), "query")
        t2 = a.run(box, "host", t1, c.search_fixed + c.search_per_vec * self.index_n[box], "search")
        self.search_lat.append(t2 - t)

    def feed(self, box: int) -> None:
        self.a.run(box, "host", self.eng.t, self.c.feed_fixed + self.c.feed_per_item * self.index_n[box], "feed")
