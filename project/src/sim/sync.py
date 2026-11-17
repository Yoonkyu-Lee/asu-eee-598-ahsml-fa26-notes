"""Box-to-box sync with boxes that come and go, links that drop, clocks that differ and boxes that lie (CP29).

Message-level model. Each box holds signed immutable messages (post / comment / vote / delete). Variants:

  S0   the CP26-CP28 demo box (X2): push ids first (no retries, only to peers known up); when a box comes on it asks ONE
       peer for "everything since I went off minus 60 s" by author stamp, trusts the peer's coverage claim, asks a gap
       of one more peer, and offers its last session's ids to one peer. Stores an envelope before validating it (H8),
       ids are not content-checked, tombstones are kept forever, nothing expires.
  S1   push ids first + set reconciliation with fixed 1 h buckets (sim/recon.py), content ids, validate first,
       windows A/B and tombstone expiry, triggers (a)-(e); with the slow anti-entropy floor (f) unless floor=False.
  S2   the same with range-based reconciliation (RBSR).

Faults: boxes on/off (always-on with outages, laptops, a correlated outage), link drops, partitions, transfers that
fail observably (p_fail) and envelopes acked then lost (p_silent, the only loss nobody sees), per-box clock offsets
(|offset| <= sigma), invalid messages, liars (echo / withhold / squat). The oracle (sim/oracle.py) checks S1-S4, L1,
L2, F1 on every event; there is no final all-on phase.

A script [(t, action, *args)] replaces random churn and writing (layer 2 directed scenes); see `run_script`.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from .engine import Engine
from .oracle import Oracle
from .recon import KeySet, reconcile

DAY = 86400.0
ENV, HDR, IDB = 1000, 40, 12          # bytes: envelope, request/answer header, one key on the wire (추정, CP26/recon)


@dataclass
class SyncScene:
    n: int = 10
    variant: str = "S2"                 # S0 / S1 / S2
    floor: bool = True                  # anti-entropy floor (S1/S2)
    always: float = 1.0
    always_n: int | None = None
    msgs_per_day: float = 1.0           # per box (추정, CP16)
    kinds: tuple = (("post", 0.6), ("comment", 0.25), ("vote", 0.1), ("delete", 0.05))
    on_out_per_day: float = 1.0
    on_out_s: float = 600.0
    lap_on_s: float = 1800.0
    lap_off_s: float = DAY
    days: float = 3.0
    R: float = DAY                      # retention (test placeholder, cp/CP29.md)
    sigma: float = 120.0                # clock bound
    D: int = 6
    D_HI: int = 12
    hop_s: float = 0.05
    timeout_s: float = 2.0
    p_fail: float = 1e-2                # a transfer fails observably
    p_silent: float = 1e-3              # an envelope is acked and then lost
    link_drops_per_day: float = 2.0     # per box
    link_drop_s: float = 600.0
    partitions_per_day: float = 0.5
    partition_s: float = 1800.0
    corr_at: float | None = None        # C-x: at this time a share of the boxes go off together
    corr_frac: float = 0.5
    corr_s: float = 7200.0
    invalid_frac: float = 0.02          # published messages with a bad signature (adversarial authors)
    liars: float = 0.0                  # share of boxes that lie
    liar_mode: str = "mixed"            # echo / withhold / squat / mixed
    foreign_del_frac: float = 0.0       # liars publish deletes of other people's posts
    backdate_frac: float = 0.0          # dishonest stamps (valid, reported separately)
    backdate_s: float = 6 * 3600.0
    skew_violation: float = 0.0         # injected: one box's offset beyond sigma (reported separately)
    pae: tuple = (600.0, 1800.0)        # floor period, uniform
    probe0: float = 5.0
    p_max: float = 300.0
    w_s: float = 60.0                   # S0 catch-up margin
    l1_every_s: float = 1800.0
    seed: int = 29

    @property
    def t_bound(self) -> float:          # L1 deadline: floor period max + probe cap + slack (cp/CP29.md L1)
        return self.pae[1] + self.p_max + 120.0


@dataclass(eq=False)
class Msg:
    mid: int
    kind: str
    t0: int                              # author stamp (int s)
    true_t: float
    author: int
    target: int | None = None
    target_t0: int | None = None
    valid: bool = True
    target_sig_ok: bool = True
    honest_stamp: bool = True

    @property
    def tkey(self) -> int:
        return self.target_t0 if self.kind == "delete" else self.t0


class Box:
    def __init__(self, sim, i: int, honest: bool, liar_mode: str | None, offset: float, always: bool):
        self.sim, self.i, self.honest, self.liar_mode, self.off, self.always = sim, i, honest, liar_mode, offset, always
        self.on = False
        self.sid = 0
        self.held: dict[int, Msg] = {}          # claimed id -> message (S0 also stores invalid ones here)
        self.keys = KeySet()
        self.forgotten: dict[int, int] = {}     # target id -> tkey
        self.applied_del: set[int] = set()
        self.pending: dict[int, tuple] = {}     # comment/vote waiting for its target: claimed -> (msg, frm)
        self.invalid_held: set[int] = set()
        self.mesh: set[int] = set()
        self.up: dict[int, bool] = {}
        self.dirty: set[int] = set()
        self.backoff: dict[int, float] = {}
        self.probing: set[int] = set()
        self.requested: dict[int, int] = {}
        self.inflight: set[int] = set()
        self.cand: list[int] = []
        self.refill_t = -1e18
        self.down_since: dict[int, float] = {}
        self.prev_taken: list[int] = []
        self.asked: set[int] = set()
        self.last_recon: dict[int, float] = {}
        self.orphan: set[int] = set()
        self.cand_down: set[int] = set()
        self.session = 0.0
        self.last_off = None
        self.taken: list[int] = []              # S0: ids taken this session (offer)
        self.cov = None
        self.stalled = False
        self.c = Counter()                      # bytes and work counters

    def now(self) -> float:
        return self.sim.eng.t + self.off

    def has(self, mid: int) -> bool:
        return mid in self.held or mid in self.forgotten or mid in self.pending


class Sync:
    def __init__(self, sc: SyncScene, script: list | None = None):
        self.sc, self.script = sc, script
        self.rng = random.Random(sc.seed)
        self.eng = Engine()
        self.oracle = Oracle(self)
        n = sc.n
        k = sc.always_n if sc.always_n is not None else int(round(sc.always * n))
        nl = int(round(sc.liars * n))
        modes = ["echo", "withhold", "squat"]
        self.boxes: list[Box] = []
        for i in range(n):
            liar = (n - nl) <= i                      # liars are the last boxes (never box 0)
            mode = (modes[i % 3] if sc.liar_mode == "mixed" else sc.liar_mode) if liar else None
            off = self.rng.uniform(-sc.sigma, sc.sigma)
            self.boxes.append(Box(self, i, not liar, mode, off, i < k))
        if script is not None:                          # directed scenes set clocks explicitly
            for B in self.boxes:
                B.off = 0.0
        if sc.skew_violation and n > 1:
            self.boxes[1].off = sc.skew_violation
        self.link_down: set[tuple] = set()
        self.group: dict[int, int] = {}                # partition: box -> side
        self.kind_of: dict[int, str] = {}
        self.msgs: list[Msg] = []
        self.forced: dict[tuple, list] = defaultdict(list)   # script: (a, b) -> ["fail"/"silent"] for next transfers
        self.tag: dict[str, Msg] = {}
        self.live: list[float] = []
        self.ev = Counter()
        self.end = sc.days * DAY

    # ------------------------------------------------------------------ helpers
    def pair(self, a, b):
        return (a, b) if a < b else (b, a)

    def link_ok(self, a: int, b: int) -> bool:
        if self.pair(a, b) in self.link_down:
            return False
        ga, gb = self.group.get(a), self.group.get(b)
        return ga is None or gb is None or ga == gb

    def in_contact(self, a: int, b: int) -> bool:
        A, B = self.boxes[a], self.boxes[b]
        return (A.on and B.on and A.honest and B.honest and self.link_ok(a, b)
                and b in A.mesh and a in B.mesh)

    def _contact(self, a: int, b: int):
        if self.boxes[a].honest and self.boxes[b].honest:
            self.oracle.contact_changed(a, b)

    def _exp(self, rate: float) -> float:
        return self.rng.expovariate(rate) if rate > 0 else float("inf")

    def new_id(self) -> int:
        return self.rng.getrandbits(63) + 1

    # ------------------------------------------------------------------ transfers
    def xfer(self, a: int, b: int, nbytes: int, cat: str, handler, *args, fail=None):
        """a sends to b. handler(*args) runs at b on arrival; fail() runs at a after a timeout on failure."""
        A, B = self.boxes[a], self.boxes[b]
        if not A.on:
            return
        A.c["tx_" + cat] += nbytes
        forced = self.forced.get((a, b))
        f = forced.pop(0) if forced else None
        if f == "pass":
            f = None
        if f == "fail" or not B.on or not self.link_ok(a, b) or (f is None and self.rng.random() < self.sc.p_fail):
            self.ev["xfer_fail"] += 1
            if fail:
                self.eng.after(self.sc.timeout_s, self._failed, a, A.sid, fail)
            return
        self.eng.after(self.sc.hop_s, self._arrive, b, B.sid, nbytes, cat, f, handler, args)

    def _failed(self, a, sid, fail):
        if self.boxes[a].on and self.boxes[a].sid == sid:
            fail()

    def _arrive(self, b, sid, nbytes, cat, f, handler, args):
        B = self.boxes[b]
        if not B.on or B.sid != sid:
            return
        B.c["rx_" + cat] += nbytes
        handler(*args, silent=(f == "silent"))

    # ------------------------------------------------------------------ peers
    def mark_up(self, a: int, b: int):
        A = self.boxes[a]
        was = A.up.get(b)
        A.up[b] = True
        A.backoff.pop(b, None)
        A.down_since.pop(b, None)
        A.cand_down.discard(b)
        if b not in A.mesh:
            A.mesh.add(b)
            self._contact(a, b)
            ups = [p for p in A.mesh if A.up.get(p) and p != b]
            if len(ups) + 1 > self.sc.D_HI:                # prune back to D, both sides drop the link
                self.rng.shuffle(ups)
                for p in ups[: len(ups) + 1 - self.sc.D]:
                    A.mesh.discard(p)
                    self.boxes[p].mesh.discard(a)
                    self._contact(a, p)
        if self.sc.variant != "S0" and A.honest and was is not True:
            if was is False or b in A.dirty:
                self.recon(a, b, "b")                          # (b) peer became up after down
            self._floor_start(a, b)

    def mark_down(self, a: int, b: int):
        A = self.boxes[a]
        was = A.up.get(b)
        A.up[b] = False
        A.down_since.setdefault(b, self.eng.t)
        if self.sc.variant == "S0" or not A.honest:
            self._graft(a)
            return
        A.dirty.add(b)
        self._probe_later(a, b)
        if was:
            for p in list(A.mesh):                             # (e) a peer went down: reconcile with the rest
                if p != b and A.up.get(p):
                    self.recon(a, p, "e")
        self._graft(a)

    def _probe_later(self, a, b):
        A = self.boxes[a]
        if b in A.probing:
            return
        d = A.backoff.get(b, self.sc.probe0 / 2) * 2
        d = min(d, self.sc.p_max)
        A.backoff[b] = d
        A.probing.add(b)
        self.eng.after(d, self._probe, a, A.sid, b)

    def _probe(self, a, sid, b):
        A = self.boxes[a]
        A.probing.discard(b)
        if not A.on or A.sid != sid or A.up.get(b):
            return
        if b not in A.mesh and b not in A.dirty and b not in A.cand_down:
            return
        if self.eng.t - A.down_since.get(b, self.eng.t) > 3600.0:      # down for an hour: stop probing, drop it
            A.mesh.discard(b)
            A.dirty.discard(b)
            A.cand_down.discard(b)
            self._contact(a, b)
            return
        self.xfer(a, b, HDR, "probe", self._hello_at, b, a, fail=lambda: self._probe_later(a, b))

    def _hello_at(self, b, a, silent=False):
        """b got a hello / probe from a: both now know each other is up (mutual graft)."""
        B = self.boxes[b]
        self.mark_up(b, a)
        if self.sc.variant == "S0" and B.stalled and B.honest:
            B.stalled = False
            self._s0_catch_up(b, a)
        self.xfer(b, a, HDR, "probe", self._hello_back, a, b)

    def _hello_back(self, a, b, silent=False):
        A = self.boxes[a]
        first = A.up.get(b) is not True
        self.mark_up(a, b)
        if self.sc.variant == "S0":
            if first and A.cov is None and not A.stalled and A.inflight == set():
                self._s0_catch_up(a, b)
            elif A.stalled:
                A.stalled = False
                self._s0_catch_up(a, b)
        elif first and A.honest:
            self.recon(a, b, "a")                              # (a) I came on: reconcile with each up peer

    def _graft(self, a: int):
        A = self.boxes[a]
        upn = sum(1 for p in A.mesh if A.up.get(p))
        need = self.sc.D - upn
        if need > 0 and not A.cand and self.eng.t - A.refill_t > 600.0:
            A.refill_t = self.eng.t
            A.cand = [c for c in range(self.sc.n) if c != a and c not in A.mesh]
            self.rng.shuffle(A.cand)
        while need > 0 and A.cand:
            c = A.cand.pop()
            if c in A.mesh:
                continue
            need -= 1
            self.xfer(a, c, HDR, "probe", self._hello_at, c, a, fail=lambda c=c: self._cand_failed(a, c))

    def _cand_failed(self, a, c):
        A = self.boxes[a]
        if A.up.get(c):                                   # it reached us meanwhile: a stale failure, ignore
            return
        A.up[c] = False
        A.down_since.setdefault(c, self.eng.t)
        if self.sc.variant != "S0" and A.honest and len(A.cand_down) < self.sc.D_HI:
            A.cand_down.add(c)
            self._probe_later(a, c)
        self._graft(a)

    # ------------------------------------------------------------------ on / off
    def come_on(self, b: int):
        B = self.boxes[b]
        if B.on:
            return
        B.on = True
        B.sid += 1
        B.session = self.eng.t
        B.up, B.mesh, B.dirty, B.backoff, B.probing, B.down_since = {}, set(), set(), {}, set(), {}
        B.refill_t = self.eng.t
        B.cand_down = set()
        B.requested, B.inflight, B.taken, B.cov, B.stalled = {}, set(), [], None, False
        self.ev["come_on"] += 1
        if self.sc.variant != "S0":
            self._expire(b)
            self.eng.after(600.0, self._expiry_tick, b, B.sid)
        else:
            self.eng.after(self.sc.timeout_s * 3, self._s0_stall_check, b, B.sid)
        others = [c for c in range(self.sc.n) if c != b]
        self.rng.shuffle(others)
        B.cand = others
        self._graft(b)

    def go_off(self, b: int):
        B = self.boxes[b]
        if not B.on:
            return
        B.on = False
        B.sid += 1
        B.last_off = self.eng.t
        for p in list(B.mesh):
            self._contact(b, p)
        B.mesh = set()                                  # peers' meshes keep b until they notice
        B.prev_taken = B.taken

    # ------------------------------------------------------------------ expiry (S1/S2)
    def _expiry_tick(self, b, sid):
        B = self.boxes[b]
        if B.on and B.sid == sid:
            self._expire(b)
            self.eng.after(600.0, self._expiry_tick, b, sid)

    def _expire(self, b):
        B, sc = self.boxes[b], self.sc
        now = B.now()
        for k in B.keys.drop_before(int(now - sc.R - 2 * sc.sigma)):
            B.held.pop(k[1], None)
            B.forgotten.pop(k[1], None)
        cut = now - sc.R
        for mid, m in list(B.held.items()):
            if m.kind != "delete" and m.tkey < cut:
                del B.held[mid]
                if mid not in B.forgotten:
                    B.keys.discard((m.tkey, mid))
        for mid, (m, f) in list(B.pending.items()):
            if m.tkey < cut:
                del B.pending[mid]

    # ------------------------------------------------------------------ accept path
    def accept(self, b: int, claimed: int, m: Msg, frm: int | None, how: str = "push") -> bool:
        B, sc = self.boxes[b], self.sc
        if not B.honest:                                  # liars keep what they get (they may serve it or not)
            if claimed not in B.held:
                B.held[claimed] = m
                B.keys.add((m.tkey, claimed))
                if B.liar_mode == "squat":
                    self._announce(b, claimed, frm)
            return True
        if sc.variant == "S0":
            return self._accept_s0(b, claimed, m, frm, how)
        if B.has(claimed):
            B.c["dup_env"] += 1
            self.oracle.on_refuse(B, m, claimed, "dup")
            return False
        if claimed != m.mid:                              # content id check (cheap hash), before any verification
            B.c["rej_id"] += 1
            return False
        B.c["verify"] += 1
        if not m.valid or (m.kind == "delete" and not m.target_sig_ok):
            B.c["rej_sig"] += 1
            B.c["verify_invalid"] += 1
            return False
        now = B.now()
        if not (now - sc.R <= m.tkey <= now + 2 * sc.sigma):
            B.c["refuse_window"] += 1
            return False
        if m.kind in ("comment", "vote") and m.target not in B.held and m.target not in B.forgotten:
            B.orphan.add(claimed)                         # held and passed on, not shown until its target arrives
            B.c["orphan"] += 1
            if frm is not None:
                self.recon(b, frm, "d")                   # (d) unknown reference
        self._hold(b, claimed, m, frm, how)
        return True

    def _hold(self, b, claimed, m, frm, how):
        B = self.boxes[b]
        B.held[claimed] = m
        B.keys.add((m.tkey, claimed))
        B.c["needed_env"] += 1
        if m.kind == "post":
            B.c["encode"] += 1
        if m.author != b and m.true_t <= self.eng.t and B.on:
            if B.session <= m.true_t:
                self.live.append(self.eng.t - m.true_t)
        self.oracle.on_hold(B, m, claimed)
        if m.kind == "delete":
            self._apply_delete(b, m)
        if B.orphan:
            B.orphan -= {o for o in B.orphan if B.held.get(o) is not None and B.held[o].target == claimed}
        self._announce(b, claimed, frm)

    def _apply_delete(self, b, d: Msg):
        B = self.boxes[b]
        self.oracle.on_apply_delete(B, d)
        B.applied_del.add(d.target)
        tgt = B.held.pop(d.target, None)
        if self.sc.variant == "S0":
            B.forgotten[d.target] = d.target_t0
            return
        B.forgotten[d.target] = d.target_t0
        B.keys.add((d.target_t0, d.target))               # the target's key stays (or appears) as a forgotten id
        if tgt is not None and tgt.kind != "post":
            pass

    def _accept_s0(self, b, claimed, m, frm, how):
        """The demo box before CP29: store, then validate (H8); ids not content-checked; no windows."""
        B = self.boxes[b]
        if claimed in B.held or claimed in B.forgotten:
            B.c["dup_env"] += 1
            self.oracle.on_refuse(B, m, claimed, "dup")
            return False
        B.held[claimed] = m
        B.keys.add((m.tkey, claimed))
        B.taken.append(claimed)
        B.c["needed_env"] += 1
        self.oracle.on_hold(B, m, claimed)
        B.c["verify"] += 1
        ok = claimed == m.mid and m.valid and (m.kind != "delete" or m.target_sig_ok)
        if not ok:
            B.invalid_held.add(claimed)                   # stays stored and served (H8)
            B.c["rej_sig"] += 1
            return False
        if m.kind == "post":
            B.c["encode"] += 1
        if m.author != b and B.on and B.session <= m.true_t:
            self.live.append(self.eng.t - m.true_t)
        if m.kind == "delete":
            self._apply_delete(b, m)
        self._announce(b, claimed, frm)
        return True

    # ------------------------------------------------------------------ push: ids first (P2)
    def _announce(self, b, mid, frm):
        B = self.boxes[b]
        if B.liar_mode in ("echo", "withhold"):
            return
        for p in list(B.mesh):
            if p != frm and B.up.get(p):
                self.xfer(b, p, HDR + IDB, "ihave", self._ihave_at, p, b, mid,
                          fail=lambda p=p: self.mark_down(b, p))

    def _ihave_at(self, p, frm, mid, silent=False):
        P = self.boxes[p]
        if P.has(mid) or mid in P.requested:
            return
        if not P.honest and P.liar_mode != "squat":
            return
        P.requested[mid] = frm
        self.xfer(p, frm, HDR + IDB, "iwant", self._iwant_at, frm, p, mid,
                  fail=lambda: self._want_failed(p, frm, mid))
        if self.sc.variant != "S0" and P.honest:
            self.eng.after(4 * self.sc.timeout_s, self._want_check, p, P.sid, frm, mid)

    def _iwant_at(self, frm, p, mid, silent=False):
        F = self.boxes[frm]
        if F.liar_mode == "squat":                        # serves a forged envelope under someone else's id
            fake = Msg(self.new_id(), "post", int(F.now()), self.eng.t, frm, valid=False)
            self.xfer(frm, p, ENV + HDR, "env", self._env_at, p, frm, mid, fake)
            return
        if not F.honest and F.liar_mode == "withhold":
            return
        m = F.held.get(mid)
        if m is not None:
            self.xfer(frm, p, ENV + HDR, "env", self._env_at, p, frm, mid, m)

    def _want_failed(self, p, frm, mid):
        P = self.boxes[p]
        if P.requested.get(mid) == frm:
            del P.requested[mid]
        self.mark_down(p, frm)

    def _want_check(self, p, sid, frm, mid):
        """(c) an IHAVE was answered but the envelope never came: mark the pair dirty."""
        P = self.boxes[p]
        if P.on and P.sid == sid and P.requested.get(mid) == frm:
            del P.requested[mid]
            P.dirty.add(frm)
            self.ev["want_timeout"] += 1
            if P.up.get(frm):
                self.recon(p, frm, "c")

    def _env_at(self, p, frm, mid, m, silent=False):
        P = self.boxes[p]
        P.requested.pop(mid, None)
        if silent or (self.sc.p_silent and self.rng.random() < self.sc.p_silent):
            self.ev["silent_loss"] += 1                   # acked, then lost: nobody sees it
            return
        ok = self.accept(p, mid, m, frm, "push")
        if not ok and P.honest and self.sc.variant != "S0" and mid != m.mid:
            P.dirty.add(frm)                              # (c) a forged envelope is an observable failure

    # ------------------------------------------------------------------ reconciliation (S1/S2)
    def recon(self, a: int, b: int, why: str):
        A, sc = self.boxes[a], self.sc
        if sc.variant == "S0" or not A.on or not A.honest or b in A.inflight:
            return
        A.inflight.add(b)
        A.c["recon_" + why] += 1
        now = A.now()
        lo = (int(now - sc.R + 2 * sc.sigma), -1)
        hi = (int(now + 2 * sc.sigma) + 1, -1)
        self.xfer(a, b, 0, "recon", self._recon_at, b, a, lo, hi,          # bytes counted by reconcile()
                  fail=lambda: self._recon_failed(a, b))

    def _recon_failed(self, a, b):
        A = self.boxes[a]
        A.inflight.discard(b)
        A.dirty.add(b)
        self.mark_down(a, b)

    def _recon_at(self, b, a, lo, hi, silent=False):
        A, B, sc = self.boxes[a], self.boxes[b], self.sc
        liar = B.liar_mode if B.liar_mode in ("echo", "withhold") else None
        r = reconcile(A.keys, B.keys, lo, hi, sc.variant, liar_b=liar)
        A.c["tx_recon"] += r["bytes"] // 2
        B.c["tx_recon"] += r["bytes"] - r["bytes"] // 2
        dt = r["rounds"] * sc.hop_s
        self.eng.after(dt, self._recon_done, a, A.sid, b, B.sid, r["need_a"], r["need_b"])

    def _recon_done(self, a, sida, b, sidb, need_a, need_b):
        A, B = self.boxes[a], self.boxes[b]
        if not (A.on and A.sid == sida and B.on and B.sid == sidb):
            A.inflight.discard(b)
            return
        A.c["recon_sessions"] += 1
        state = {"a": a, "b": b, "left": 2 if B.honest else 1, "ok": True}
        self._fetch(a, b, sorted(need_a), state)
        if B.honest:
            self._fetch(b, a, sorted(need_b), state)

    def _fetch(self, x, y, keys, state):
        """x fetches the envelopes for `keys` from y."""
        X = self.boxes[x]
        X.c["fetch_keys"] += len(keys)
        if not keys:
            self._fetch_end(x, y, state, True)
            return
        leg = state.setdefault("legs", {})
        leg[x] = False
        self.xfer(x, y, HDR + IDB * len(keys), "fetch", self._serve_at, y, x, keys, state,
                  fail=lambda: self._leg_end(x, y, state, False))
        self.eng.after(4 * self.sc.timeout_s, self._leg_end, x, y, state, False)   # no answer: a failure

    def _serve_at(self, y, x, keys, state, silent=False):
        Y = self.boxes[y]
        out = []
        if Y.honest or Y.liar_mode == "squat":
            for k in keys:
                m = Y.held.get(k[1])
                if m is not None:
                    out.append((k[1], m))
        out.sort(key=lambda e: e[1].kind != "delete")      # deletes first
        self.xfer(y, x, HDR + ENV * len(out), "fetched", self._fetched_at, x, y, out, state,
                  fail=lambda: None)

    def _fetched_at(self, x, y, out, state, silent=False):
        if state.get("legs", {}).get(x) is not False:
            return                                         # this leg already timed out
        if not silent:
            for mid, m in out:
                self.accept(x, mid, m, y, "recon")
        else:
            self.ev["silent_loss"] += 1
        self._leg_end(x, y, state, True)

    def _leg_end(self, x, y, state, ok):
        legs = state.setdefault("legs", {})
        if legs.get(x) is not False:
            return
        legs[x] = True
        self._fetch_end(x, y, state, ok)

    def _fetch_end(self, x, y, state, ok):
        state["left"] -= 1
        state["ok"] &= ok
        if state["left"] == 0:
            a, b = state["a"], state["b"]
            A = self.boxes[a]
            A.inflight.discard(b)
            if state["ok"]:
                A.dirty.discard(b)
                self.boxes[b].dirty.discard(a)
                A.last_recon[b] = self.boxes[b].last_recon[a] = self.eng.t
            else:
                A.dirty.add(b)
                self._probe_later(a, b)

    # ------------------------------------------------------------------ floor (f)
    def _floor_start(self, a, b):
        if not self.sc.floor or self.sc.variant == "S0":
            return
        A = self.boxes[a]
        self.eng.after(self.rng.uniform(*self.sc.pae), self._floor_tick, a, A.sid, b)

    def _floor_tick(self, a, sid, b):
        A = self.boxes[a]
        if not A.on or A.sid != sid or not A.up.get(b) or b not in A.mesh:
            return
        last = A.last_recon.get(b, -1e18)
        if self.eng.t - last >= self.sc.pae[0]:            # the pair was quiet: check it
            self.recon(a, b, "f")
            nxt = self.rng.uniform(*self.sc.pae)
        else:                                               # a session just ran: next check P_AE after it
            nxt = max(last + self.rng.uniform(*self.sc.pae) - self.eng.t, 1.0)
        self.eng.after(nxt, self._floor_tick, a, sid, b)

    # ------------------------------------------------------------------ S0 catch-up and offer
    def _s0_stall_check(self, b, sid):
        B = self.boxes[b]
        if B.on and B.sid == sid and B.cov is None and not B.inflight and not any(B.up.values()):
            B.stalled = True

    def _s0_catch_up(self, a, b, lo=None):
        A = self.boxes[a]
        if lo is None:
            lo = (A.last_off - self.sc.w_s + A.off) if A.last_off is not None else -1e18
        A.inflight.add(b)
        self.xfer(a, b, HDR, "since", self._s0_since_at, b, a, lo, fail=lambda: self._s0_ask_next(a, b, lo))

    def _s0_since_at(self, b, a, lo, silent=False):
        B = self.boxes[b]
        if B.honest:
            out = [(mid, m) for mid, m in B.held.items() if m.t0 >= lo]
            cov = B.cov if B.cov is not None else B.session + B.off
        else:
            out, cov = [], lo                             # a liar claims it covers everything and sends nothing
        self.xfer(b, a, HDR + ENV * len(out), "since_env", self._s0_since_back, a, b, lo, out, cov)

    def _s0_since_back(self, a, b, lo, out, cov, silent=False):
        A = self.boxes[a]
        A.inflight.discard(b)
        for mid, m in out:
            self.accept(a, mid, m, b, "since")
        if cov > lo and not A.inflight:
            asked = getattr(A, "asked", set()) | {b}
            A.asked = asked
            nxt = [p for p in A.mesh if A.up.get(p) and p not in asked]
            if nxt:
                self._s0_catch_up(a, nxt[0], lo)
                return
        A.cov = lo if cov <= lo else cov                  # unfilled gap is kept, never retried (H2)
        A.asked = set()
        offer = list(A.prev_taken)
        if offer:
            self.xfer(a, b, HDR + IDB * len(offer), "offer", self._s0_offer_at, b, a, offer)

    def _s0_ask_next(self, a, b, lo):
        A = self.boxes[a]
        A.inflight.discard(b)
        nxt = [p for p in A.mesh if A.up.get(p) and p != b]
        if nxt:
            self._s0_catch_up(a, nxt[0], lo)
        else:
            A.stalled = True

    def _s0_offer_at(self, b, a, offer, silent=False):
        B = self.boxes[b]
        want = [mid for mid in offer if not B.has(mid)]
        if want and B.honest:
            self.xfer(b, a, HDR + IDB * len(want), "want", self._s0_want_at, a, b, want)

    def _s0_want_at(self, a, b, want, silent=False):
        A = self.boxes[a]
        out = [(mid, A.held[mid]) for mid in want if mid in A.held]
        self.xfer(a, b, HDR + ENV * len(out), "since_env", self._s0_offer_back, b, a, out)

    def _s0_offer_back(self, b, a, out, silent=False):
        for mid, m in out:
            self.accept(b, mid, m, a, "offer")

    # ------------------------------------------------------------------ writing
    def publish(self, b: int, kind: str = None, tag: str = None, stamp_dt: float = 0.0, target: str | int = None,
                valid: bool = True, foreign: bool = False):
        B, sc = self.boxes[b], self.sc
        if not B.on:
            return None
        if kind is None:
            r, acc = self.rng.random(), 0.0
            for k, p in sc.kinds:
                acc += p
                if r < acc:
                    kind = k
                    break
        honest_stamp = True
        if stamp_dt == 0.0 and sc.backdate_frac and self.rng.random() < sc.backdate_frac:
            stamp_dt, honest_stamp = -self.rng.uniform(0, sc.backdate_s), False
        elif stamp_dt:
            honest_stamp = abs(stamp_dt) <= sc.sigma
        t0 = int(B.now() + stamp_dt)
        tgt = None
        if kind in ("comment", "vote", "delete"):
            if isinstance(target, str):
                tgt = self.tag[target]
            elif target is not None:
                tgt = B.held.get(target)
            else:
                pool = [m for m in B.held.values() if m.kind == "post" and m.valid
                        and (kind != "delete" or m.author == b or foreign)]
                tgt = self.rng.choice(pool) if pool else None
            if tgt is None:
                kind = "post"
        if valid and sc.invalid_frac and self.rng.random() < sc.invalid_frac and self.script is None:
            valid = False
        m = Msg(self.new_id(), kind, t0, self.eng.t, b, valid=valid, honest_stamp=honest_stamp)
        if kind != "post":
            m.target = tgt.mid
            if kind == "delete":
                m.target_t0 = tgt.t0
                m.target_sig_ok = (tgt.author == b) and not foreign
        self.msgs.append(m)
        self.kind_of[m.mid] = kind
        if tag:
            self.tag[tag] = m
        self.ev["pub_" + kind] += 1
        self._cover(b, m)
        if not B.honest:                                  # liars' own messages go out as they are
            self._hold_liar(b, m)
        elif m.valid and m.target_sig_ok:
            self.accept(b, m.mid, m, None, "own")
        else:                                             # an adversarial author pushes its invalid message
            self._announce_raw(b, m)
        return m

    def _hold_liar(self, b, m):
        B = self.boxes[b]
        B.held[m.mid] = m
        B.keys.add((m.tkey, m.mid))
        for p in list(B.mesh):
            if B.up.get(p):
                self.xfer(b, p, HDR + IDB, "ihave", self._ihave_at, p, b, m.mid)

    def _announce_raw(self, b, m):
        B = self.boxes[b]
        B.held[m.mid] = m                                 # only so that its IWANT can be served
        for p in list(B.mesh):
            if B.up.get(p):
                self.xfer(b, p, HDR + IDB, "ihave", self._ihave_at, p, b, m.mid)

    def _cover(self, b, m):
        """Functional coverage cell for a writing event (cp/CP29.md layer 1)."""
        on = [x for x in self.boxes if x.on and x.honest and x.i != b]
        if m.kind == "post":
            who = "others on" if on else "author only"
        else:
            hold = [x for x in on if m.target in x.held]
            who = "holders on" if hold else ("non-holders only" if on else "author only")
        faults = []
        if self.link_down:
            faults.append("link drop")
        if self.group:
            faults.append("partition")
        if any(not x.honest for x in self.boxes):
            faults.append("liar")
        if not m.valid:
            faults.append("invalid")
        faults += ["failed transfer", "silent loss", "skew"]
        for f in faults:
            self.oracle.cell(m.kind, who, f)

    # ------------------------------------------------------------------ random world
    def _churn(self, b: int):
        sc, B = self.sc, self.boxes[b]
        if B.always:
            self.come_on(b)
            self.eng.after(self._exp(sc.on_out_per_day / DAY), self._outage, b)
        elif self.rng.random() < sc.lap_on_s / (sc.lap_on_s + sc.lap_off_s):
            self.come_on(b)
            self.eng.after(self._exp(1 / sc.lap_on_s), self._lap_off, b)
        else:
            self.eng.after(self._exp(1 / sc.lap_off_s), self._lap_on, b)

    def _outage(self, b):
        sc = self.sc
        self.go_off(b)
        self.eng.after(self._exp(1 / sc.on_out_s), self._back, b)

    def _back(self, b):
        self.come_on(b)
        self.eng.after(self._exp(self.sc.on_out_per_day / DAY), self._outage, b)

    def _lap_on(self, b):
        self.come_on(b)
        self.eng.after(self._exp(1 / self.sc.lap_on_s), self._lap_off, b)

    def _lap_off(self, b):
        self.go_off(b)
        self.eng.after(self._exp(1 / self.sc.lap_off_s), self._lap_on, b)

    def _writer(self, b):
        sc = self.sc
        B = self.boxes[b]
        if B.on:
            foreign = (not B.honest) and sc.foreign_del_frac and self.rng.random() < sc.foreign_del_frac
            self.publish(b, "delete" if foreign else None, foreign=bool(foreign))
        self.eng.after(self._exp(sc.msgs_per_day / DAY * (DAY / max(self._duty(b), 1.0))), self._writer, b)

    def _duty(self, b) -> float:
        sc = self.sc
        if self.boxes[b].always:
            return DAY
        return DAY * sc.lap_on_s / (sc.lap_on_s + sc.lap_off_s)

    def _link_drop(self, b):
        sc = self.sc
        B = self.boxes[b]
        peers = [p for p in B.mesh] or [p for p in range(sc.n) if p != b]
        p = self.rng.choice(peers)
        pr = self.pair(b, p)
        if pr not in self.link_down:
            self.link_down.add(pr)
            self.ev["link_drop"] += 1
            self._contact(*pr)
            self.eng.after(self._exp(1 / sc.link_drop_s), self._link_up, pr)
        self.eng.after(self._exp(sc.link_drops_per_day / DAY), self._link_drop, b)

    def _link_up(self, pr):
        self.link_down.discard(pr)
        self._contact(*pr)

    def _partition(self):
        sc = self.sc
        if not self.group:
            ids = list(range(sc.n))
            self.rng.shuffle(ids)
            half = len(ids) // 2
            self.group = {b: (0 if j < half else 1) for j, b in enumerate(ids)}
            self.ev["partition"] += 1
            self._contacts_all()
            self.eng.after(self._exp(1 / sc.partition_s), self._heal)
        self.eng.after(self._exp(sc.partitions_per_day / DAY), self._partition)

    def _heal(self):
        self.group = {}
        self._contacts_all()

    def _contacts_all(self):
        for A in self.boxes:
            for p in A.mesh:
                if A.i < p:
                    self._contact(A.i, p)

    def _corr(self):
        sc = self.sc
        on = [b.i for b in self.boxes if b.on]
        self.rng.shuffle(on)
        for b in on[: int(len(on) * sc.corr_frac)]:
            self.go_off(b)
            self.eng.after(sc.corr_s, self.come_on, b)
        self.ev["corr_outage"] += 1

    def _f1_tick(self):
        self.oracle.f1_sweep()
        self.eng.after(3600.0, self._f1_tick)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        sc = self.sc
        if self.script is not None:
            return self.run_script()
        for b in range(sc.n):
            self._churn(b)
            self.eng.after(self._exp(sc.msgs_per_day / DAY), self._writer, b)
            if sc.link_drops_per_day:
                self.eng.after(self._exp(sc.link_drops_per_day / DAY), self._link_drop, b)
        if sc.partitions_per_day:
            self.eng.after(self._exp(sc.partitions_per_day / DAY), self._partition)
        if sc.corr_at is not None:
            self.eng.at(sc.corr_at, self._corr)
        self.eng.after(3600.0, self._f1_tick)
        self.eng.run(self.end)
        return self.result()

    def run_script(self) -> dict:
        """Script actions: on b | off b | pub b kind tag [stamp_dt] [target] [valid] | link a b up/down |
        force a b fail/silent | part [group-a ids] | heal | offset b s | liar b mode | until t (end)."""
        end = self.end
        for row in self.script:
            t, act, *args = row
            if act == "until":
                end = t
                continue
            self.eng.at(t, self._do, act, args)
        self.eng.after(3600.0, self._f1_tick)
        self.eng.run(end)
        return self.result()

    def _do(self, act, args):
        if act == "on":
            self.come_on(args[0])
        elif act == "off":
            self.go_off(args[0])
        elif act == "pub":
            b, kind, tag = args[:3]
            stamp = args[3] if len(args) > 3 else 0.0
            tgt = args[4] if len(args) > 4 else None
            valid = args[5] if len(args) > 5 else True
            foreign = args[6] if len(args) > 6 else False
            self.publish(b, kind, tag, stamp, tgt, valid, foreign)
        elif act == "link":
            a, b, state = args
            pr = self.pair(a, b)
            (self.link_down.discard if state == "up" else self.link_down.add)(pr)
            self._contact(a, b)
        elif act == "force":
            a, b, kind = args[:3]
            self.forced[(a, b)].extend([kind] * (args[3] if len(args) > 3 else 1))
        elif act == "part":
            side = set(args[0])
            self.group = {b: (0 if b in side else 1) for b in range(self.sc.n)}
            self._contacts_all()
        elif act == "heal":
            self.group = {}
            self._contacts_all()
        elif act == "offset":
            self.boxes[args[0]].off = args[1]
        elif act == "liar":
            B = self.boxes[args[0]]
            B.honest, B.liar_mode = False, args[1]
        elif act == "call":
            args[0](self)

    # ------------------------------------------------------------------ results
    def holds(self, b: int, tag: str) -> bool:
        m = self.tag[tag]
        return m.mid in self.boxes[b].held

    def forgot(self, b: int, tag: str) -> bool:
        m = self.tag[tag]
        return m.mid in self.boxes[b].forgotten

    def result(self) -> dict:
        sc = self.sc
        tot = Counter()
        for B in self.boxes:
            if B.honest:
                tot.update(B.c)
        need_b = tot["needed_env"] * ENV
        sent = sum(v for k, v in tot.items() if k.startswith("tx_"))
        over = sent - need_b
        live = sorted(self.live)
        p99 = live[int(0.99 * (len(live) - 1))] if live else None
        honest = [B for B in self.boxes if B.honest]
        n_days = sc.days
        idle = sum(v for k, v in tot.items() if k in ("tx_recon", "tx_probe")) / max(len(honest), 1) / n_days
        return {
            "n": sc.n, "variant": sc.variant, "floor": sc.floor, "seed": sc.seed,
            "messages": len(self.msgs), "events": self.eng.events,
            "violations": dict(self.oracle.viol), "examples": self.oracle.examples,
            "l1_checks": self.oracle.l1_checks, "l1_items": self.oracle.l1_items,
            "needed_bytes": need_b, "over_bytes": over, "over_over_needed": over / need_b if need_b else None,
            "sent_bytes": sent, "live_p99_s": p99,
            "recon_tx_per_box_day": idle,
            "tx": {k[3:]: v for k, v in tot.items() if k.startswith("tx_")},
            "work": {k: tot[k] for k in ("verify", "verify_invalid", "encode", "needed_env", "dup_env", "rej_id",
                                         "rej_sig", "refuse_window", "recon_sessions", "fetch_keys")},
            "triggers": {k: tot[k] for k in tot if k.startswith("recon_") and k != "recon_sessions"},
            "ev": dict(self.ev),
            "cells": {"|".join(k): v for k, v in self.oracle.cells.items()},
        }
