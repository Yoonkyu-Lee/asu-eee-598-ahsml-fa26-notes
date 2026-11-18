"""Oracle for the CP29 sync model: knows the ground truth, checks every property on every event.

Properties (cp/CP29.md, written before results):
  S1 no honest box holds / serves / applies an invalid message (receiver-verifiable validity)
  S2 no resurrection: a box that applied a delete never again holds the target while the target is inside its window
  S3 no foreign delete applied
  S4 no squatting: a valid message is never refused because an invalid envelope took its id
  L1/L2 two honest boxes in contact for T_bound end up with every item that one of them held at the start and that
        stays inside both boxes' reconciliation band (L2 = the same for deletes)
  F1 storage: no held message older than now - R, no tombstone / forgotten id older than now - R - 2 sigma
Physical limit: boxes that never meet a holder are not counted (L1 is about pairs in contact).
Also recorded, separately: dishonest stamps (not a violation), injected clock-bound violations.
"""
from __future__ import annotations

from collections import Counter, defaultdict


class Oracle:
    def __init__(self, sim):
        self.sim = sim
        self.viol = Counter()
        self.examples: dict[str, list] = defaultdict(list)
        self.epoch: dict[tuple, int] = defaultdict(int)       # contact epoch per honest pair
        self.contact_since: dict[tuple, float] = {}
        self.l1_checks = 0
        self.l1_items = 0
        self.conv_s: list[float] = []                         # time from a difference being seen to it closing
        self.cells = Counter()                                # functional coverage

    def bad(self, prop: str, **info):
        self.viol[prop] += 1
        if len(self.examples[prop]) < 5:
            self.examples[prop].append({"t": round(self.sim.eng.t, 3), **info})

    # ---- safety, on events
    def on_hold(self, box, msg, claimed: int):
        if not box.honest:
            return
        if claimed != msg.mid or not msg.valid or (msg.kind == "delete" and not msg.target_sig_ok):
            self.bad("S1", box=box.i, mid=claimed, kind=msg.kind)
        if msg.kind != "delete" and msg.mid in box.applied_del and box.now() - self.sim.sc.R <= msg.t0:
            self.bad("S2", box=box.i, mid=msg.mid)

    def on_apply_delete(self, box, d):
        if box.honest and not d.target_sig_ok:
            self.bad("S3", box=box.i, mid=d.mid)

    def on_refuse(self, box, msg, claimed: int, reason: str):
        if not box.honest:
            return
        if reason == "dup" and claimed == msg.mid and msg.valid and claimed in box.invalid_held:
            self.bad("S4", box=box.i, mid=claimed)

    def f1_sweep(self):
        R, s2 = self.sim.sc.R, 2 * self.sim.sc.sigma
        slack = 601.0                                    # the expiry sweep runs every 600 s
        for b in self.sim.boxes:
            if not b.honest or not b.on:
                continue
            now = b.now()
            if b.keys.keys and b.keys.keys[0][0] < now - R - s2 - slack:
                self.bad("F1", box=b.i, oldest=b.keys.keys[0][0], now=round(now))
            old = sum(1 for m in b.held.values() if m.kind != "delete" and m.tkey < now - R - slack)
            if old:
                self.bad("F1", box=b.i, old_held=old)

    # ---- liveness: contact intervals
    def pair(self, a, b):
        return (a, b) if a < b else (b, a)

    def contact_changed(self, a: int, b: int):
        p = self.pair(a, b)
        self.epoch[p] += 1
        if self.sim.in_contact(*p):
            self.contact_since[p] = self.sim.eng.t
            self.checkpoint(p)
        else:
            self.contact_since.pop(p, None)

    def checkpoint(self, p):
        """Snapshot the difference of an honest pair in contact; check it T_bound later."""
        sim, sc = self.sim, self.sim.sc
        a, b = sim.boxes[p[0]], sim.boxes[p[1]]
        T = sc.t_bound
        lo = max(a.now(), b.now()) + T - sc.R + 2 * sc.sigma          # stays inside both B for the whole check
        hi = min(a.now(), b.now()) + 2 * sc.sigma
        diff = [k for k in set(a.keys.keys) ^ set(b.keys.keys) if lo <= k[0] <= hi]
        ep = self.epoch[p]
        sim.eng.after(T, self._check, p, ep, diff, sim.eng.t)
        if sim.in_contact(*p):
            sim.eng.after(sc.l1_every_s, self._recheck, p, ep)

    def _recheck(self, p, ep):
        if self.epoch[p] == ep and self.sim.in_contact(*p):
            self.checkpoint(p)

    def _check(self, p, ep, diff, t0):
        if self.epoch[p] != ep:
            return                                   # contact broke in between: L1 makes no claim
        a, b = self.sim.boxes[p[0]], self.sim.boxes[p[1]]
        self.l1_checks += 1
        self.l1_items += len(diff)
        for k in diff:
            if not (k in a.keys and k in b.keys):
                kind = self.sim.kind_of.get(k[1], "?")
                self.bad("L2" if kind == "delete" else "L1", pair=p, key=k, kind=kind, since=round(t0, 1))

    # ---- coverage
    def cell(self, *key):
        self.cells[key] += 1

    def report(self) -> dict:
        return {"violations": dict(self.viol), "examples": {k: v for k, v in self.examples.items()},
                "l1_checks": self.l1_checks, "l1_items": self.l1_items}
