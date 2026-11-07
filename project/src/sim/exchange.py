"""Box-to-box exchange with boxes that come and go (CP26): when do boxes talk, to whom, and what do they send.

One topic held by all n boxes. Boxes switch on and off (always-on home boxes with short outages, intermittent laptop
boxes with short sessions). The mesh links only boxes that are on: a box that comes on grafts D peers, a box whose
degree drops below D_LO grafts more after a heartbeat. Envelopes are pushed to the mesh (eager push, CP17). What the
protocols add:

  X0  push only.
  X1  push + every POLL_S each box asks every other box that is on for its whole list (the CP24 demo).
  X1m push + every POLL_S each box asks its mesh peers for their whole lists.
  X2  push + when a box comes on: catch up from one box ("everything published since I went off, minus a margin";
      the answer carries the responder's coverage start, a gap is asked of another box) and offer the ids it took in
      its last W seconds (the other side asks only for what it lacks). Event-driven only.
  X3  X2 + every T_D each box sends its mesh peers the ids it took in the last W seconds; they ask for what they lack.

Bytes per box are split into: needed (each envelope it must hold, once), push duplicates, overhead (requests,
id lists, envelopes it already had). Coverage start: the earliest publish time from which a box holds everything.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .engine import Engine

DAY = 86400.0
ENV = 1000.0            # bytes per envelope (추정, CP16: header + proof + text)
ID = 8.0                # bytes per message id
HDR = 40.0              # bytes per request / answer header (추정)


@dataclass(frozen=True)
class Scene:
    n: int = 100
    always: float = 1.0                 # share of always-on boxes
    always_n: int | None = None         # or an exact count (C-b: 1)
    msgs_per_day: float = 1.0           # per box (추정, CP16 Params)
    on_out_per_day: float = 1.0         # always-on: outages per day (추정)
    on_out_s: float = 600.0             # always-on: mean outage (추정)
    lap_on_s: float = 1800.0            # intermittent: mean session (추정)
    lap_off_s: float = DAY              # intermittent: mean time off (추정)
    p_loss: float = 1e-3                # a push silently lost (추정)
    D: int = 6
    D_LO: int = 4
    D_HI: int = 12
    hop_s: float = 0.05
    uplink_bps: float = 20e6
    days: float = 2.0
    grace_s: float = 600.0              # at the end every box comes on for this long
    poll_s: float = 20.0                # X1, X1m
    t_d: float = 10.0                   # X3
    w_s: float = 60.0                   # X2 offer window, X3 digest window, catch-up margin
    hb_s: float = 1.0                   # heartbeat: graft delay after losing a peer
    seed: int = 26


class Exchange:
    def __init__(self, sc: Scene, proto: str, script: list | None = None):
        """script: [(t, "on" | "off" | "pub", box)] instead of random churn and writing (to replay a demo scene)."""
        self.sc, self.proto, self.script = sc, proto, script
        self.rng = np.random.default_rng(sc.seed)
        n = sc.n
        k = sc.always_n if sc.always_n is not None else int(round(sc.always * n))
        self.always = np.zeros(n, bool)
        self.always[:k] = True
        self.eng = Engine()
        self.on = np.zeros(n, bool)
        self.mesh = [set() for _ in range(n)]
        self.up_free = np.zeros(n)
        cap = int(n * sc.msgs_per_day * (sc.days + 1) * 1.5 + 1000)
        self.t0 = np.zeros(cap)
        self.author = np.zeros(cap, np.int64)
        self.got = np.zeros((cap, n), bool)
        self.arr = np.full((cap, n), np.nan)
        self.M = 0
        self.recent = [deque() for _ in range(n)]          # (arrival t, mid)
        self.session = np.full(n, np.nan)                  # start of the current session
        self.last_off = np.full(n, np.nan)                 # end of the last session
        self.cov = np.full(n, np.nan)                      # coverage start (nan: holds nothing yet)
        self.needed = np.zeros(n)
        self.dup = np.zeros(n)
        self.over = np.zeros(n)
        self.reqs = np.zeros(n)
        self.live: list[float] = []
        self.catch: list[float] = []
        self.catch_fail = 0
        self.lost_offline = 0
        self.end = sc.days * DAY
        self.sid = np.zeros(n, np.int64)                   # session number (events of an older session are void)
        self.rejoin_t = np.full(n, np.nan)
        self.prev_session = np.full(n, np.nan)             # start of the last finished session
        self.ready = np.zeros(n, bool)                     # caught up in this session (may answer for old times)
        self.ops = [Counter() for _ in range(n)]           # per box: catch_req, catch_env, offer_ids, want_ids
        self.lo = np.zeros(n)                              # the start of the gap a box that came on must fill
        self.offered = np.zeros(n, np.int64)               # session in which the box made its offer
        self.stalled = np.zeros(n, bool)                   # came on, nobody to ask: waits for a box to appear
        self.inflight = np.zeros(n, bool)                  # a catch-up request is out
        self.waited = np.zeros(n, bool)                    # this session's catch-up had to wait for a box to appear
        self.catch_waited: list[float] = []

    # ---- churn
    def start(self) -> None:
        sc, rng = self.sc, self.rng
        if self.script:
            for t, k, b in self.script:
                if k == "on":                       # like a real box, the first one does not know it is the first
                    self.eng.at(t, self._come_on, b, False)
                elif k == "off":
                    self.eng.at(t, self._go_off, b, None)
                else:
                    self.eng.at(t, self._publish, b, None)
        for b in range(sc.n if not self.script else 0):
            if self.always[b]:
                self._come_on(b, first=True)
            elif rng.random() < sc.lap_on_s / (sc.lap_on_s + sc.lap_off_s):
                self._come_on(b, first=True)
            else:
                self.eng.at(rng.exponential(sc.lap_off_s), self._come_on, b)
        if self.proto in ("X1", "X1m"):
            for b in range(sc.n):
                self.eng.at(rng.random() * sc.poll_s, self._poll, b)
        if self.proto == "X3":
            self.eng.at(sc.t_d, self._digest_tick)
        self.eng.at(self.end, self._grace)

    def _online_s(self, b: int) -> float:
        return DAY / self.sc.on_out_per_day if self.always[b] else self.sc.lap_on_s

    def _come_on(self, b: int, first: bool = False) -> None:
        if self.on[b]:
            return
        sc, t = self.sc, self.eng.t
        self.on[b] = True
        self.prev_session[b] = self.session[b]
        self.session[b] = t
        self.sid[b] += 1
        self.inflight[b] = self.stalled[b] = self.waited[b] = False
        self.ready[b] = self.proto not in ("X2", "X3") or (first and t == 0.0)
        if first and t == 0.0:
            self.cov[b] = 0.0
        self._graft(b, sc.D)
        if t < self.end and not self.script:
            self.eng.at(t + self.rng.exponential(self._online_s(b)), self._go_off, b, self.sid[b])
            self.eng.at(t + self.rng.exponential(DAY / sc.msgs_per_day * self._avail(b)), self._publish, b, self.sid[b])
        if self.proto in ("X2", "X3") and not (first and t == 0.0):
            self.rejoin_t[b] = t
            self.lo[b] = 0.0 if np.isnan(self.cov[b]) else max(self.last_off[b] - sc.w_s, 0.0)
            self._catch_up(b, self.lo[b], np.inf, frozenset(), self.sid[b])
            for q in np.flatnonzero(self.on & self.stalled):  # a box appeared: those that had nobody to ask try
                q = int(q)                                    # again (in the demo: on that box's /hello)
                if q != b:
                    self.stalled[q] = False
                    self.eng.at(t + sc.hop_s, self._catch_up, q, self.lo[q], np.inf, frozenset(), self.sid[q])

    def _avail(self, b: int) -> float:
        sc = self.sc
        if self.always[b]:
            return 1.0
        return sc.lap_on_s / (sc.lap_on_s + sc.lap_off_s)

    def _go_off(self, b: int, sid: int | None) -> None:
        if not self.on[b] or (sid is not None and sid != self.sid[b]) or self.eng.t >= self.end:
            return
        sc, t = self.sc, self.eng.t
        self.on[b] = False
        self.last_off[b] = t
        for p in list(self.mesh[b]):
            self.mesh[p].discard(b)
            if len(self.mesh[p]) < sc.D_LO:
                self.eng.at(t + sc.hb_s, self._graft, p, sc.D)
        self.mesh[b].clear()
        if self.script:
            return
        off = self.rng.exponential(sc.on_out_s if self.always[b] else sc.lap_off_s)
        self.eng.at(min(t + off, self.end), self._come_on, b)

    def _graft(self, b: int, want: int) -> None:
        if not self.on[b]:
            return
        cand = np.flatnonzero(self.on)
        self.rng.shuffle(cand)
        for p in cand:
            if len(self.mesh[b]) >= want:
                break
            p = int(p)
            if p != b and p not in self.mesh[b] and len(self.mesh[p]) < self.sc.D_HI:
                self.mesh[b].add(p)
                self.mesh[p].add(b)

    def _grace(self) -> None:
        for b in range(self.sc.n):
            if not self.on[b]:
                self._come_on(b)

    # ---- push (rule 1)
    def _publish(self, b: int, sid: int | None) -> None:
        t = self.eng.t
        if not self.on[b] or (sid is not None and sid != self.sid[b]) or t >= self.end:
            return
        m = self.M
        self.M += 1
        self.t0[m], self.author[m] = t, b
        self._take(b, m, "author")
        self._push(b, m, -1)
        if not self.script:
            self.eng.at(t + self.rng.exponential(DAY / self.sc.msgs_per_day * self._avail(b)), self._publish, b, sid)

    def _take(self, b: int, m: int, how: str) -> bool:
        if self.got[m, b]:
            return False
        t = self.eng.t
        self.got[m, b] = True
        self.arr[m, b] = t
        if how != "author":
            self.needed[b] += ENV
        self.recent[b].append((t, m))
        if how == "push" and self.session[b] <= self.t0[m]:
            self.live.append(t - self.t0[m])
        return True

    def _send(self, frm: int, nbytes: float, fn, *args) -> None:
        start = max(self.eng.t, self.up_free[frm])
        self.up_free[frm] = start + nbytes * 8 / self.sc.uplink_bps
        self.eng.at(self.up_free[frm] + self.sc.hop_s, fn, *args)

    def _push(self, b: int, m: int, frm: int) -> None:
        for p in self.mesh[b]:
            if p != frm:
                if self.rng.random() < self.sc.p_loss:
                    continue
                self._send(b, ENV, self._recv_push, p, m, b)

    def _recv_push(self, b: int, m: int, frm: int) -> None:
        if not self.on[b]:
            self.lost_offline += 1
            return
        if self._take(b, m, "push"):
            self._push(b, m, frm)
        else:
            self.dup[b] += ENV

    # ---- whole-list polling (X1, X1m)
    def _poll(self, b: int) -> None:
        t, sc = self.eng.t, self.sc
        if t >= self.end + sc.grace_s:
            return
        if self.on[b]:
            peers = np.flatnonzero(self.on) if self.proto == "X1" else np.array(sorted(self.mesh[b]), np.int64)
            M = self.M
            mine = self.got[:M, b]
            for p in peers:
                p = int(p)
                if p == b:
                    continue
                theirs = self.got[:M, p]
                k = int(theirs.sum())
                self.reqs[b] += 1
                self.over[b] += 2 * HDR
                new = np.flatnonzero(theirs & ~mine)
                self.over[b] += (k - len(new)) * ENV
                for m in new:
                    if self._take(b, int(m), "pull"):
                        self._push(b, int(m), p)
                mine = self.got[:M, b]
        self.eng.at(t + sc.poll_s, self._poll, b)

    # ---- catching up and offering (X2, X3)
    def _catch_up(self, b: int, lo: float, hi: float, tried: frozenset, sid: int) -> None:
        """Ask one box that is on for everything published in [lo, hi). Prefer a mesh peer whose coverage starts at
        or before lo, else any box that is on with such coverage, else the one with the earliest coverage (its answer
        then leaves the gap [lo, its coverage) to ask of another box)."""
        if not self.on[b] or sid != self.sid[b] or self.ready[b] or (not tried and self.inflight[b]):
            return
        self.inflight[b] = True
        on = [int(p) for p in np.flatnonzero(self.on) if p != b and p not in tried]
        if not on:                  # nobody to ask: stay not ready (claim this session only), try when a box appears
            self.stalled[b] = self.waited[b] = True
            self.inflight[b] = False
            return
        good = [p for p in on if p in self.mesh[b] and self._cov(p) <= lo] or [p for p in on if self._cov(p) <= lo]
        p = good[0] if good else min(on, key=self._cov)
        if self.offered[b] != sid:
            self.offered[b] = sid
            self._offer(b, p)
        self.reqs[b] += 1
        self.ops[b]["catch_req"] += 1
        self.over[b] += HDR
        self.eng.at(self.eng.t + self.sc.hop_s, self._answer_catch_up, p, b, lo, hi, tried | {p}, sid)

    def _answer_catch_up(self, p: int, b: int, lo: float, hi: float, tried: frozenset, sid: int) -> None:
        if not self.on[p]:                                        # gone meanwhile: ask someone else
            return self._catch_up(b, lo, hi, tried, sid)
        M = self.M
        sel = np.flatnonzero(self.got[:M, p] & (self.t0[:M] >= lo) & (self.t0[:M] < hi))
        self._send(p, HDR + len(sel) * ENV, self._got_catch_up, b, sel, lo, hi, self._cov(p), tried, sid)

    def _got_catch_up(self, b: int, sel: np.ndarray, lo: float, hi: float, cov_p: float, tried: frozenset,
                      sid: int) -> None:
        if not self.on[b] or sid != self.sid[b]:
            self.catch_fail += 1
            self.inflight[b] = False
            return
        self.over[b] += HDR
        self.ops[b]["catch_env"] += len(sel)
        for m in sel:
            if not self._take(b, int(m), "catch"):
                self.over[b] += ENV
        if cov_p > lo:                                            # the responder lacks [lo, cov_p)
            if any(q != b and q not in tried and self._cov(q) < cov_p for q in np.flatnonzero(self.on)):
                return self._catch_up(b, lo, cov_p, tried, sid)
            gap_end = cov_p                                       # nobody that is on has it: b holds from cov_p
        else:
            gap_end = None
        if gap_end is None:
            if np.isnan(self.cov[b]):
                self.cov[b] = lo
        else:                                                     # [lo, gap_end) is nowhere on: claim what is held
            self.cov[b] = min(gap_end, self.session[b])
        self.ready[b] = True
        self.inflight[b] = False
        if self.rejoin_t[b] < self.end:          # the closing grace (every box on at once) is not a catch-up sample
            (self.catch_waited if self.waited[b] else self.catch).append(self.eng.t - self.rejoin_t[b])

    def _cov(self, p: int) -> float:
        """What p may claim: its coverage start once it has caught up in this session, else only this session (it has
        everything pushed since it came on). A box that cannot fill its gap stays at that claim."""
        return float(self.cov[p]) if self.ready[p] and not np.isnan(self.cov[p]) else float(self.session[p])

    def _offer(self, b: int, p: int) -> None:
        """Ids this box took in its last session, to the box it catches up from (that box asks for what it lacks).
        Covers what was written while the other side was off (by this box or by boxes that have gone off since)."""
        if np.isnan(self.last_off[b]) or np.isnan(self.prev_session[b]):
            return
        M = self.M
        a = self.arr[:M, b]
        ids = np.flatnonzero((a >= self.prev_session[b]) & (a <= self.last_off[b])).tolist()
        if ids:
            self._ihave(b, p, ids)

    def _ihave(self, b: int, p: int, ids: list) -> None:
        sc = self.sc
        self.over[p] += HDR + ID * len(ids)
        self.ops[b]["offer_ids"] += len(ids)
        want = [m for m in ids if not self.got[m, p]]
        if not want:
            return
        self.over[b] += HDR + ID * len(want)
        self.ops[p]["want_ids"] += len(want)
        self.reqs[p] += 1
        self._send(b, HDR + ENV * len(want), self._recv_iwant, p, want, b)

    def _recv_iwant(self, p: int, want: list, frm: int) -> None:
        if not self.on[p]:
            return
        for m in want:
            if self._take(p, m, "repair"):
                self._push(p, m, frm)
            else:
                self.over[p] += ENV

    def _digest_tick(self) -> None:
        sc, t = self.sc, self.eng.t
        if t >= self.end + sc.grace_s:
            return
        for b in np.flatnonzero(self.on):
            b = int(b)
            r = self.recent[b]
            while r and r[0][0] < t - sc.w_s:
                r.popleft()
            if not r:
                continue
            ids = [m for _, m in r]
            for p in list(self.mesh[b]):
                self._ihave(b, p, ids)
        self.eng.at(t + sc.t_d, self._digest_tick)

    # ---- run and measure
    def run(self) -> dict:
        self.start()
        self.eng.run(self.end + self.sc.grace_s)
        sc, M, n = self.sc, self.M, self.sc.n
        should = self.t0[:M] <= self.end - sc.w_s
        miss = int((~self.got[:M][should]).sum())
        needed = should.sum() * ENV
        lv = np.array(self.live) if self.live else np.array([np.nan])
        ca = np.array(self.catch) if self.catch else np.array([np.nan])
        per_box = lambda x: float(x.mean())
        return {"proto": self.proto, "n": n, "always_boxes": int(self.always.sum()), "messages": int(M),
                "missing_pairs": miss, "missing_frac": miss / max(should.sum() * n, 1),
                "live_p50_s": float(np.nanpercentile(lv, 50)), "live_p99_s": float(np.nanpercentile(lv, 99)),
                "catch_up_n": len(self.catch), "catch_up_p99_s": float(np.nanpercentile(ca, 99)),
                "catch_up_all_p99_s": float(np.nanpercentile(np.array(self.catch + self.catch_waited or [np.nan]), 99)),
                "catch_up_waited_n": len(self.catch_waited),
                "catch_up_waited_p50_s": float(np.median(self.catch_waited)) if self.catch_waited else None,
                "catch_up_waited_max_s": float(max(self.catch_waited)) if self.catch_waited else None,
                "catch_up_fail": self.catch_fail, "lost_to_offline": self.lost_offline,
                "per_box_needed_B": per_box(self.needed), "per_box_dup_B": per_box(self.dup),
                "per_box_over_B": per_box(self.over), "per_box_reqs": per_box(self.reqs),
                "over_over_needed": float(self.over.sum() / max(self.needed.sum(), 1)),
                "dup_over_needed": float(self.dup.sum() / max(self.needed.sum(), 1)),
                "should_bytes_per_box": float(needed), "events": self.eng.events}


def x1_formula(n: int, sc: Scene, M_total: float) -> dict:
    """X1 at large n: every POLL_S each box that is on asks every other box that is on for its whole list. With
    messages growing linearly to M_total over the run and availability a, a box's overhead over the run is about
    (run / POLL_S) x a x (a n - 1) x (mean list ~ M_total / 2) x ENV (추정, checked against the event model at small n)."""
    a = 1.0 if sc.always >= 1 else sc.always + (1 - sc.always) * sc.lap_on_s / (sc.lap_on_s + sc.lap_off_s)
    ticks = sc.days * DAY / sc.poll_s
    over = ticks * a * max(a * n - 1, 0) * (M_total / 2) * ENV
    return {"per_box_over_B": over, "per_box_reqs": ticks * a * max(a * n - 1, 0),
            "over_over_needed": over / max(M_total * ENV, 1)}
