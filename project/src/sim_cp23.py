"""CP23: enforcing the weight limit (T1) in the proof. The CP22 box (T2 pictures last + T3 age groups on) with T1 as
an implementable structure instead of a real-valued weight (cp/CP23.md):
  I0      CP22's real-valued weight (reference)
  I1(q)   a message spends n = ceil(w q) consecutive RLN message indices (q per unit), one share + nullifier each
  I1p     I1(1) as one padded circuit; I4 the post split into n messages (same limits and attack as I1(1))
  I3a/b   size classes counted separately (text / picture post; short <= 64 tokens / long text / picture post)
Attack strategies: S-p 256 tokens + picture (CP22), S-t 256-token text only, S-c every class at its limit (I3).
Fixes if S-t breaks rule S2 everywhere: F1 live text = first 128 tokens (box only), F2 index priority (the proof
reveals whether the message lies in the window's first m units; the box serves those first), F1 + F2.
Rule S3: slashing holds, S2 under the worst strategy, the 100 Mbps invalid-proof flood still fits the 6x6 verify unit,
fewest changes (fix first, then implementation), K_crit_old at least half of the best passer; then a window-start burst.

  .venv/Scripts/python.exe src/sim_cp23.py          -> results/cp23/cp23.json  (reads results/cp23/rln_n.json)
Every box number is a model (추정) on CP16 behaviour and CP18-CP20 chip numbers; rln_n.json is a CPU 예비 측정.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_cp19 as c19  # noqa: E402
import sim_cp20 as c20  # noqa: E402
import sim_cp22 as c22  # noqa: E402
from sim_cp16 import BASE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp23"
T0 = time.time()
OVER_MAX = c22.OVER_MAX
LATE_MAX = 5e-3                    # F2: honest messages pushed past the first m units (half of K1's 1% tail)
CAP = 128                          # F1
SHORT = 64                         # I3b short class
O3_BASE_KB = 522                   # CP14 w = 8 fixed-base table per public-input base
CHIP_PER_INPUT = 352 / 26_785      # CP17 W3 count: 32 windows x 11 multiplications per extra public input
FLOOD_MSG_B = 1024                 # CP13: 100 Mbps = 12,207 messages/s of 1 KB


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ honest side

class Honest23(c22.Honest):
    """CP22's honest sample, keeping each message's tokens and picture flag so that every implementation can weigh it."""

    def __init__(self, et, lens, peak, seed: int = 22):
        super().__init__(et, lens, peak, seed)
        rng = np.random.default_rng(seed)            # same draws as c22.Honest
        n = 200_000
        is_post = rng.random(n) < self.post_share
        self.img = is_post & (rng.random(n) < BASE.image_frac)
        self.L = np.minimum(np.where(is_post, rng.choice(lens, n), c22.COMMENT_TOKENS), 256)
        self.tw = np.array([et.text1(int(x))[0] for x in range(1, 257)]) / self.unit
        assert np.allclose(self.weights(None), self.w)

    def text_w(self, L, cap):
        return self.tw[np.minimum(L, cap or 256) - 1]

    def weights(self, cap) -> np.ndarray:
        return self.text_w(self.L, cap) + self.img * self.img_w

    def refused(self, w: np.ndarray, W: float, epoch: float, draws: int = 400, seed: int = 7) -> float:
        """c22.Honest.refused_weight for any weight array."""
        keep = self.w
        self.w = w
        try:
            return self.refused_weight(W, epoch, draws, seed)
        finally:
            self.w = keep

    def late(self, w: np.ndarray, m: float, epoch: float, draws: int = 400, seed: int = 8) -> float:
        """Share of honest messages that do not lie in the window's first m units (indices are spent in order)."""
        from scipy.stats import poisson
        rng = np.random.default_rng(seed)
        nmax = 120
        exp_late = np.zeros(nmax + 1)
        for k in range(1, nmax + 1):
            cum = np.cumsum(rng.choice(w, (draws, k)), axis=1)
            exp_late[k] = (cum > m + 1e-9).sum(axis=1).mean()
        lam = self.rate * epoch
        pm = poisson.pmf(np.arange(nmax + 1)[:, None], lam[None, :])
        return float((exp_late[:, None] * pm).sum() / lam.sum())

    def class_share(self, classes) -> list[float]:
        return [float(c["sel"](self).mean()) for c in classes]

    def refused_classes(self, classes, limits, epoch) -> float:
        shares = self.class_share(classes)
        keep = self.rate
        out = 0.0
        try:
            for p, L in zip(shares, limits):
                self.rate = keep * p
                out += p * self.refused_count(int(L), epoch)
        finally:
            self.rate = keep
        return out


# ------------------------------------------------------------------ implementations

CLASSES = {
    "I3a": [{"name": "text", "sel": lambda H: ~H.img, "L": 256, "image": False},
            {"name": "picture", "sel": lambda H: H.img, "L": 256, "image": True}],
    "I3b": [{"name": "short", "sel": lambda H: ~H.img & (H.L <= SHORT), "L": SHORT, "image": False},
            {"name": "long", "sel": lambda H: ~H.img & (H.L > SHORT), "L": 256, "image": False},
            {"name": "picture", "sel": lambda H: H.img, "L": 256, "image": True}],
}
IMPL = {"I0": {"kind": "cont"}, "I1q1": {"kind": "quant", "q": 1}, "I1q2": {"kind": "quant", "q": 2},
        "I1q4": {"kind": "quant", "q": 4}, "I3a": {"kind": "class"}, "I3b": {"kind": "class"},
        "I5": {"kind": "dual", "q": 4}}     # added after the first full run (cp/CP23.md): text units + picture count
SAME_AS = {"I1p": "I1q1", "I4": "I1q1"}          # same limits and attack; only the proof structure differs
ORDER_IMPL = ["I3a", "I3b", "I1p", "I1q1", "I1q2", "I1q4", "I5", "I4"]
ORDER_FIX = ["none", "F1", "F2", "F1+F2"]


def honest_w(H, impl, cap):
    """Honest message weights in the implementation's units (I5: the text part only, pictures are counted apart)."""
    w = H.text_w(H.L, cap) if IMPL[impl]["kind"] == "dual" else H.weights(cap)
    if IMPL[impl]["kind"] in ("quant", "dual"):
        q = IMPL[impl]["q"]
        w = np.ceil(w * q - 1e-9) / q
    return w


def pick(H, impl, cap, s_new):
    """Old group: 10-minute window, the tightest limit meeting 1e-3 on its own. New group: windows of 10 min and 1 h,
    the limit giving the lowest attacker encoder rate within the leftover refusal budget (CP22 pick_limits).
    Returns per group {limit(s), epoch, refused}."""
    kind = IMPL[impl]["kind"]
    if kind == "dual":
        return pick_dual(H, impl, cap, s_new)
    if kind == "class":
        cls = CLASSES[impl]
        cost = [H.text_w(np.array([c["L"]]), cap)[0] + c["image"] * H.img_w for c in cls]

        def best(epoch, budget):
            grids = [range(1, 41)] * len(cls)
            # per class, the refusal of each limit (classes are independent)
            shares = H.class_share(cls)
            keep = H.rate
            tab = []
            try:
                for p in shares:
                    H.rate = keep * p
                    tab.append([p * H.refused_count(L, epoch) for L in range(1, 41)])
            finally:
                H.rate = keep
            top = None
            import itertools
            for Ls in itertools.product(*grids):
                r = sum(tab[i][L - 1] for i, L in enumerate(Ls))
                if r <= budget:
                    a = sum(L * c for L, c in zip(Ls, cost)) / epoch
                    if top is None or a < top[0]:
                        top = (a, list(Ls), r)
            return top
        a, Ls, r_old = best(c22.EPOCH, OVER_MAX)
        old = {"limits": Ls, "epoch": c22.EPOCH, "refused": r_old, "units_per_s": a}
        budget = (OVER_MAX - (1 - s_new) * r_old) / s_new
        cand = [(E, best(E, budget)) for E in c22.NEW_EPOCHS]
        E, (a, Ls, r) = min(((E, b) for E, b in cand if b), key=lambda x: x[1][0])
        new = {"limits": Ls, "epoch": E, "refused": r, "units_per_s": a}
        return {"old": old, "new": new, "refused_total": (1 - s_new) * r_old + s_new * r}
    w = honest_w(H, impl, cap)
    step = 1 / IMPL[impl]["q"] if kind == "quant" else 0.5
    grid = np.arange(step, 120.0, step)
    Wo = next(W for W in grid if H.refused(w, W, c22.EPOCH) <= OVER_MAX)
    r_old = H.refused(w, Wo, c22.EPOCH)
    old = {"W": float(Wo), "epoch": c22.EPOCH, "refused": r_old}
    budget = (OVER_MAX - (1 - s_new) * r_old) / s_new
    top = None
    for E in c22.NEW_EPOCHS:
        W = next((g for g in grid if H.refused(w, g, E) <= budget), None)
        if W is not None and (top is None or W / E < top[0] / top[1]):
            top = (float(W), E, H.refused(w, W, E))
    new = {"W": top[0], "epoch": top[1], "refused": top[2]}
    return {"old": old, "new": new, "refused_total": (1 - s_new) * r_old + s_new * top[2]}


def pick_dual(H, impl, cap, s_new):
    """I5: W text units (q per unit, every message's text part) and L picture posts per window. A message is refused
    when its text does not fit or, for a picture post, the picture count is used up (refusals added, an upper bound).
    The attacker's live work is the text units, so the smallest W is chosen first, then the smallest L."""
    w = honest_w(H, impl, cap)
    q = IMPL[impl]["q"]
    p_pic = float(H.img.mean())
    keep = H.rate

    def r_pic(L, E):
        try:
            H.rate = keep * p_pic
            return p_pic * H.refused_count(L, E)
        finally:
            H.rate = keep

    def best(E, budget):
        rp = [r_pic(L, E) for L in range(1, 21)]
        for W in np.arange(1 / q, 30.0, 1 / q):
            rt = H.refused(w, W, E)
            if rt + rp[-1] > budget:
                continue
            L = next(i + 1 for i, r in enumerate(rp) if rt + r <= budget)
            return {"W": float(W), "L_pic": L, "epoch": E, "refused": rt + rp[L - 1]}
        return None
    old = best(c22.EPOCH, OVER_MAX)
    budget = (OVER_MAX - (1 - s_new) * old["refused"]) / s_new
    cand = [b for b in (best(E, budget) for E in c22.NEW_EPOCHS) if b]
    new = min(cand, key=lambda b: (b["W"] / b["epoch"], b["L_pic"] / b["epoch"]))
    return {"old": old, "new": new, "refused_total": (1 - s_new) * old["refused"] + s_new * new["refused"]}


def pick_m(H, impl, cap, epoch):
    """F2: smallest m (units; for classes, messages per class) with honest late share <= LATE_MAX in this window."""
    if IMPL[impl]["kind"] == "class":
        shares = H.class_share(CLASSES[impl])
        keep = H.rate
        try:
            def late(m):
                out = 0.0
                for p in shares:
                    H.rate = keep * p
                    out += p * H.late(np.ones(1), m, epoch)
                return out
            for m in range(1, 41):
                s = late(m)
                if s <= LATE_MAX:
                    return float(m), s
        finally:
            H.rate = keep
        return None, None
    w = honest_w(H, impl, cap)
    step = 1 / IMPL[impl]["q"] if IMPL[impl]["kind"] == "quant" else 0.5
    for m in np.arange(step, 60.0, step):
        s = H.late(w, m, epoch)
        if s <= LATE_MAX:
            return float(m), s
    return None, None


def streams(H, impl, cap, lim, strategy, m=None):
    """Per identity per window: list of (count, L, image, late). Integer counts except I0 (CP22's real rate)."""
    kind = IMPL[impl]["kind"]
    tw256 = H.text_w(np.array([256]), cap)[0]
    if kind == "dual":
        q = IMPL[impl]["q"]
        wt = math.ceil(tw256 * q - 1e-9) / q
        n_text = math.floor(lim["W"] / wt + 1e-9)
        if strategy == "S-t":
            out = [[n_text, 256, False]]
        else:                                        # every picture post also spends its text units
            n_pic = min(lim["L_pic"], n_text)
            out = [[n_pic, 256, True]] + ([[n_text - n_pic, 256, False]] if n_text > n_pic else [])
        return [(n, L, img, False) for n, L, img in out]
    if kind == "class":
        out = []
        for c, L in zip(CLASSES[impl], lim["limits"]):
            if strategy == "S-t" and c["image"]:
                continue
            out.append([L, c["L"], c["image"]])
    else:
        W = lim["W"]
        q = IMPL[impl].get("q")
        wq = (lambda x: math.ceil(x * q - 1e-9) / q) if kind == "quant" else (lambda x: x)
        wp, wt = wq(tw256 + H.img_w), wq(tw256)
        if strategy == "S-p":
            n = W / wp if kind == "cont" else math.floor(W / wp + 1e-9)
            out = [[n, 256, True]]
            left = W - n * wp
            if kind != "cont" and left >= wt - 1e-9:
                out.append([math.floor(left / wt + 1e-9), 256, False])
        else:
            n = W / wt if kind == "cont" else math.floor(W / wt + 1e-9)
            out = [[n, 256, False]]
    if m is None:
        return [(n, L, img, False) for n, L, img in out]
    # F2: units are spent in list order and the part beyond the first m units is late (classes: per class)
    res, used = [], 0.0
    for n, L, img in out:
        if kind == "class":
            early = min(n, int(m))
        else:
            w = wp if img else wt
            early = max(0, min(n, math.floor((m - used) / w + 1e-9)))
            used += n * w
        res += [(early, L, img, False), (n - early, L, img, True)]
    return [s for s in res if s[0] > 0]


# ------------------------------------------------------------------ the box

class Box23(c22.Box22):
    """Box22 with F1 (texts encoded up to CAP tokens) and F2 (early-index messages first). Encoder classes:
    query 0, old early text 1, new early text 3, old late text 4, new late text 5, then pictures (T2) 6 / 7."""

    def __init__(self, et, cost, *, cap=None, p_late=None, **kw):
        self.cap, self.p_late = cap, p_late or {}
        super().__init__(et, cost, **kw)

    def message(self, t, kind, *, tier=None, late=None, **kw):
        """Box22.message plus the late flag (drawn for honest messages only when F2 is on, so that without F2 the
        random stream is CP22's)."""
        L = kw.get("L", 0)
        if self.cap and L:
            L = min(L, self.cap)
        if tier is None:
            tier = "new" if (self.t3 and self.rng.random() < self.s_new) else "old"
        if late is None:
            late = bool(self.p_late) and self.rng.random() < self.p_late.get(tier, 0.0)
        if kw.get("record", True) and kw.get("cls", 1) != 2 and kind in ("post", "comment"):
            for k in (f"text_{tier}", "text"):
                self.sub[k].append(t)
            if kw.get("image"):
                for k in (f"image_{tier}", "image"):
                    self.sub[k].append(t)
        self.eng.at(t, self._arrive, {"kind": kind, "L": L, "image": kw.get("image", False), "cls": kw.get("cls", 1),
                                      "rec": kw.get("record", True), "valid": True, "replay": False, "t_box": t,
                                      "tier": tier, "late": late})

    def _enc_submit(self, job):
        if job["kind"] != "query" and job["cls"] != 2:
            m = job["m"]
            tier, late = m.get("tier", "old"), m.get("late", False)
            if job["kind"] == "text" or not self.t2:
                job["cls"] = {("old", False): 1, ("new", False): 3, ("old", True): 4, ("new", True): 5}[(tier, late)]
            else:
                job["cls"] = 6 if tier == "old" else 7
        super(c22.Box22, self)._enc_submit(job)       # past Box22's own class mapping


def attack(et, inp, lens, *, K, tier, plan, epoch, s_new, vm, cap=None, p_late=None, burst=False, dur=3600.0,
           seed=221):
    """K identities of one group, each sending `plan` [(count per window, L, image, late)] every window."""
    box = Box23(et, inp["cost"], t2=True, t3=True, vkind="chip", vm=vm, sched="Q2", seed=seed, cap=cap, p_late=p_late)
    box.s_new = s_new
    rng = np.random.default_rng(seed)
    for n, L, img, late in plan:
        if burst:
            for w0 in np.arange(0.0, dur, epoch):
                cnt = int(round(K * n))
                for t in np.sort(rng.uniform(w0, w0 + 1.0, cnt)):
                    box.message(float(t), "post", L=L, image=img, record=False, tier=tier, late=late)
        else:
            k = rng.poisson(K * n / epoch * dur)
            for t in np.sort(rng.uniform(0, dur, k)):
                box.message(float(t), "post", L=L, image=img, record=False, tier=tier, late=late)
    c19.legit(box, inp, lens, 0, dur, lambda t: 20, query_dt=5, rng=rng)
    for t in np.arange(2.5, dur, 5.0):
        box.message(float(t), "comment", L=int(rng.choice(lens)), tier="old")
        box.message(float(t) + 1.0, "comment", L=int(rng.choice(lens)), tier="new")
    box.run(dur + 300)
    p = lambda k: float(np.percentile(c19.lat(box, k), 99, method="higher")) if box.sub[k] else None
    r = {"K": K, "tier": tier, "text_old_p99_s": p("text_old"), "text_new_p99_s": p("text_new"),
         "image_old_p99_s": p("image_old"), "query_p99_ms": (p("query") or 0) * 1e3,
         "encoder_busy": box.e_s / box.eng.t}
    lim = c19.LIMITS["K1_text_box_s"]
    r["K1_old"] = r["text_old_p99_s"] is not None and r["text_old_p99_s"] <= lim
    r["K1_new"] = r["text_new_p99_s"] is not None and r["text_new_p99_s"] <= lim
    r["K2"] = r["query_p99_ms"] <= c19.LIMITS["K2_query_s"] * 1e3
    r["ok"] = r["K1_old"] and r["K1_new"] and r["K2"]
    return r


# ------------------------------------------------------------------ costs

def costs(impl, rln, n_hon, n_max, f2):
    """Proof-side costs. n_hon: honest messages' index counts (array), n_max: the heaviest message's count.
    Chip: CP17 per-input count on the CP14 O3 verifier (추정). CPU: rln_n.json (F2's extra input not measured)."""
    v = {r["n"]: r for r in rln}
    ns = np.array(sorted(v))
    cpu = lambda key, n: float(np.interp(n, ns, [v[k][key] for k in ns]))
    chip = lambda inputs: 1 + CHIP_PER_INPUT * (inputs - 5)
    x = 1 if f2 else 0
    if impl in ("I3a", "I3b", "I4"):              # one RLN-1 circuit; I4 spends n_max proofs on the heaviest post
        per_msg, circuits, n_heavy = np.ones(len(n_hon)), [1], 1
        proofs = n_max if impl == "I4" else 1
    elif impl == "I1p":                            # one circuit with n_max slots for every message
        per_msg, circuits, n_heavy, proofs = np.full(len(n_hon), float(n_max)), [n_max], n_max, 1
    elif impl == "I5":                             # text indices 1..n_max-1, with or without one picture index
        per_msg, n_heavy, proofs = n_hon.astype(float), n_max, 1
        circuits = [n + p for n in range(1, n_max) for p in (0, 1)]
    else:                                          # one circuit (and verification key) per n
        per_msg, circuits, n_heavy, proofs = n_hon.astype(float), list(range(1, n_max + 1)), n_max, 1
    inputs = lambda n: 3 + 2 * n + x
    per_byte = max(chip(inputs(n)) / ((FLOOD_MSG_B + 32 * (inputs(n) - 5)) / FLOOD_MSG_B) for n in circuits)
    return {"verification_keys": len(circuits), "o3_tables_MB": sum(inputs(n) for n in circuits) * O3_BASE_KB / 1024,
            "chip_verify_honest_mean": float(np.mean([chip(inputs(n)) for n in per_msg])),
            "chip_verify_heaviest": chip(inputs(n_heavy)) * proofs,
            "flood_cost_per_byte_vs_now": per_byte,
            "cpu_verify_us_honest_mean": float(np.mean([cpu("verify_us", n) for n in per_msg])),
            "cpu_verify_us_heaviest": cpu("verify_us", n_heavy) * proofs,
            "prove_ms_honest_mean": float(np.mean([cpu("prove_us", n) for n in per_msg])) / 1e3,
            "prove_ms_heaviest": cpu("prove_us", n_heavy) / 1e3 * proofs,
            "constraints_heaviest": cpu("constraints", n_heavy),
            "bytes_extra_heaviest": 32 * (inputs(n_heavy) - 5) + (128 + 32 * 5) * (proofs - 1)}


# ------------------------------------------------------------------ main

class Ctx:
    def __init__(self):
        self.inp = c19.inputs()
        self.lens = np.minimum(c19.token_lengths(), 256)
        self.et = c20.Fp16Image(c19.EncTable(8, 1024), 12 * 4 + 1)
        self.vm = c20.machines()["6x6"]
        self.H = Honest23(self.et, self.lens, float(self.inp["diurnal"][20]))
        self.s_new = c22.S_NEW["base"]
        self.rln = json.loads((OUT / "rln_n.json").read_text(encoding="utf-8"))["rows"]

    def attack(self, **kw):
        kw.setdefault("s_new", self.s_new)
        return attack(self.et, self.inp, self.lens, vm=self.vm, **kw)


def strategies_of(impl):
    return ["S-c", "S-t"] if IMPL[impl]["kind"] == "class" else ["S-p", "S-t"]


def evaluate(ctx, impl, fix):
    H, et = ctx.H, ctx.et
    cap = CAP if "F1" in fix else None
    lim = pick(H, impl, cap, ctx.s_new)
    p_late, ms = {}, {}
    if "F2" in fix:
        for g in ("old", "new"):
            ms[g], p_late[g] = pick_m(H, impl, cap, lim[g]["epoch"])
    row = {"limits": lim, "m": ms or None, "p_late": p_late or None, "strategies": {}}
    for st in strategies_of(impl):
        plan_new = streams(H, impl, cap, lim["new"], st, ms.get("new"))
        plan_old = streams(H, impl, cap, lim["old"], st, ms.get("old"))
        go = lambda K, tier, plan, E: ctx.attack(K=K, tier=tier, plan=plan, epoch=E, cap=cap, p_late=p_late)
        ref = go(c22.K_REF, "new", plan_new, lim["new"]["epoch"])
        live = sum(n * et.text1(L)[0] for n, L, img, late in plan_old if not late) / lim["old"]["epoch"]
        k0 = max(1, int(0.9 / max(live, 1e-9)))              # CP22's starting point
        kc, runs = c22.k_crit(lambda K: go(K, "old", plan_old, lim["old"]["epoch"]), max(1, k0 // 2))
        live_new = sum(n * et.text1(L)[0] for n, L, img, late in plan_new if not late) / lim["new"]["epoch"]
        kn, _ = c22.k_crit(lambda K: go(K, "new", plan_new, lim["new"]["epoch"]),
                           max(1, int(0.9 / max(live_new, 1e-9)) // 2))
        row["strategies"][st] = {"plan_new": plan_new, "plan_old": plan_old, "at_K_ref": ref, "K_crit_old": kc,
                                 "K_crit_new": kn,
                                 "runs": {str(k): {kk: v[kk] for kk in ("K1_old", "K1_new", "K2", "text_old_p99_s",
                                                                         "text_new_p99_s")}
                                          for k, v in runs.items()}}
        log(impl, fix, st, "ref ok", ref["ok"], {k: ref[k] for k in ("text_old_p99_s", "text_new_p99_s",
                                                                      "query_p99_ms")}, "K_crit_old", kc,
            "K_crit_new", kn)
    worst = min(row["strategies"].values(), key=lambda s: (s["at_K_ref"]["ok"], s["K_crit_old"]))
    row["worst"] = next(k for k, v in row["strategies"].items() if v is worst)
    row["S2_pass"] = lim["refused_total"] <= OVER_MAX and all(v["at_K_ref"]["ok"] for v in row["strategies"].values())
    row["K_crit_old"] = min(v["K_crit_old"] for v in row["strategies"].values())
    row["K_crit_new"] = min(v["K_crit_new"] for v in row["strategies"].values())
    return row


def save(res, name="cp23.json"):
    (OUT / name).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def decide(ctx, res):
    """Rule S3 at the fewest-change fix level where something passes (else the last level run), then the s_new
    sensitivity and the window-start burst."""
    H = ctx.H
    run_at = lambda impl, fix: res["runs"].get(f"{SAME_AS.get(impl, impl)}|{fix}")
    levels = [f for f in ORDER_FIX if any(k.endswith("|" + f) for k in res["runs"])]
    tables = {}
    for fix in levels:
        cap = CAP if "F1" in fix else None
        table = {}
        for impl in ORDER_IMPL:
            r = run_at(impl, fix)
            if r is None:
                continue
            q = IMPL[SAME_AS.get(impl, impl)].get("q", 1)
            if impl == "I5":
                n_hon = (np.ceil(H.text_w(H.L, cap) * q - 1e-9) + H.img)[:20_000].astype(int)
                n_max = math.ceil(H.text_w(np.array([256]), cap)[0] * q - 1e-9) + 1
            else:
                n_hon = np.ceil(H.weights(cap)[:20_000] * q - 1e-9).astype(int)
                n_max = math.ceil((H.text_w(np.array([256]), cap)[0] + H.img_w) * q - 1e-9)
            c = costs(impl, ctx.rln, n_hon, n_max, "F2" in fix)
            c["flood_ok"] = c["flood_cost_per_byte_vs_now"] <= 1.0 + 1e-9
            table[impl] = {"S2_pass": r["S2_pass"], "K_crit_old": r["K_crit_old"], "K_crit_new": r["K_crit_new"],
                           "worst": r["worst"], "refused_total": r["limits"]["refused_total"], "slashing": True,
                           "costs": c}
        tables[fix] = table
    fix = next((f for f in levels if any(t["S2_pass"] and t["costs"]["flood_ok"] for t in tables[f].values())),
               levels[-1])
    table = tables[fix]
    passers = [i for i in ORDER_IMPL if i in table and table[i]["S2_pass"] and table[i]["costs"]["flood_ok"]]
    best_k = max((table[i]["K_crit_old"] for i in passers), default=0)
    choice = next((i for i in passers if table[i]["K_crit_old"] >= best_k / 2), None)
    res["rule_S3"] = {"fix": fix, "tables": tables, "passers": passers, "pick": choice,
                      "I2": "rejected in design: ancestor tags with shares slash honest sibling blocks; without "
                            "shares an overlap is refused but not slashed"}
    log("rule S3", fix, passers, "pick", choice)
    save(res)

    target = SAME_AS.get(choice, choice) if choice else "I1q1"
    cap = CAP if "F1" in fix else None
    sens = {}
    for k in ("low", "high"):
        v = c22.S_NEW[k]
        lim = pick(H, target, cap, v)
        ms = {g: pick_m(H, target, cap, lim[g]["epoch"]) for g in ("old", "new")} if "F2" in fix else {}
        out = {}
        for st in strategies_of(target):
            plan = streams(H, target, cap, lim["new"], st, ms["new"][0] if ms else None)
            a = ctx.attack(K=c22.K_REF, tier="new", plan=plan, epoch=lim["new"]["epoch"], s_new=v, cap=cap,
                           p_late={g: ms[g][1] for g in ms} if ms else None)
            out[st] = {k2: a[k2] for k2 in ("ok", "text_old_p99_s", "text_new_p99_s", "query_p99_ms")}
        sens[k] = {"s_new": v, "target": target, "fix": fix, "limits": lim, "at_K_ref": out,
                   "pass": lim["refused_total"] <= OVER_MAX and all(o["ok"] for o in out.values())}
        log("s_new", v, target, fix, lim["new"], {s2: o["ok"] for s2, o in out.items()})
    res["s_new_sensitivity"] = sens
    save(res)

    res.pop("burst", None)
    if choice:
        r = run_at(choice, fix)
        st = r["worst"]
        b = ctx.attack(K=c22.K_REF, tier="new", plan=r["strategies"][st]["plan_new"],
                       epoch=r["limits"]["new"]["epoch"], cap=cap, p_late=r["p_late"], burst=True)
        res["burst"] = {"strategy": st, "at_K_ref": b}
        log("burst", {k: b[k] for k in ("ok", "text_old_p99_s", "text_new_p99_s", "query_p99_ms")})
    save(res)
    log("wrote")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    H = ctx.H
    res = {"inputs": {"unit_s": H.unit, "image_units": H.img_w, "text128_units": float(H.text_w(np.array([128]), None)[0]),
                      "late_max": LATE_MAX, "s_new": ctx.s_new}, "runs": {}}
    log("inputs", res["inputs"])
    for fix in ORDER_FIX:
        for impl in IMPL:
            if impl == "I5":
                continue                            # see i5()
            res["runs"][f"{impl}|{fix}"] = evaluate(ctx, impl, fix)
            save(res)
        if fix == "none":
            res["runs"]["I0|none"]["note"] = "CP22 reference (S-p reproduces CP22's T1+T2+T3)"
        if any(res["runs"].get(f"{SAME_AS.get(i, i)}|{fix}", {}).get("S2_pass") for i in ORDER_IMPL):
            break                                   # a fix level that passes ends the search (fewest changes)
    decide(ctx, res)


def i5() -> None:
    """I5 (added after the first full run): text units and picture count apart, without and with F1; then rule S3
    again over everything."""
    ctx = Ctx()
    res = json.loads((OUT / "cp23.json").read_text(encoding="utf-8"))
    for fix in ("none", "F1"):
        res["runs"][f"I5|{fix}"] = evaluate(ctx, "I5", fix)
        res["runs"][f"I5|{fix}"]["note"] = "added after the first full run"
        save(res)
    decide(ctx, res)


def need() -> None:
    """When no implementation holds S-t: how tight must the new group's 1-hour limit be for K_REF text-only
    identities, and what does that cost honest new users? I0 and I1(1), without and with F1.
    -> results/cp23/need.json"""
    inp = c19.inputs()
    lens = np.minimum(c19.token_lengths(), 256)
    et = c20.Fp16Image(c19.EncTable(8, 1024), 12 * 4 + 1)
    vm = c20.machines()["6x6"]
    H = Honest23(et, lens, float(inp["diurnal"][20]))
    E = 3600.0
    out = {}
    for impl in ("I0", "I1q1"):
        for fix in ("none", "F1"):
            cap = CAP if fix == "F1" else None
            w = honest_w(H, impl, cap)
            Wo = next(W for W in np.arange(0.5, 60, 0.5) if H.refused(w, W, c22.EPOCH) <= OVER_MAX)
            r_old = H.refused(w, Wo, c22.EPOCH)
            step = 0.5 if impl == "I0" else 1.0
            rows = []
            for Wn in np.arange(step, 8.0 + 1e-9, step):
                plan = streams(H, impl, cap, {"W": float(Wn)}, "S-t")
                a = attack(et, inp, lens, K=c22.K_REF, tier="new", plan=plan, epoch=E, s_new=c22.S_NEW["base"], vm=vm,
                           cap=cap)
                r_new = H.refused(w, Wn, E)
                s_max = (OVER_MAX - r_old) / (r_new - r_old) if r_new > r_old else 1.0
                rows.append({"W_new": float(Wn), "ok": a["ok"], "text_new_p99_s": a["text_new_p99_s"],
                             "refused_new": r_new, "s_new_max_for_1e-3": s_max})
                log("need", impl, fix, Wn, a["ok"], round(r_new, 4), round(s_max, 4))
            okW = [r["W_new"] for r in rows if r["ok"]]
            out[f"{impl}|{fix}"] = {"W_old": float(Wo), "refused_old": r_old, "rows": rows,
                                    "W_new_max_ok": max(okW) if okW else None}
            (OUT / "need.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log("wrote need")


if __name__ == "__main__":
    need() if "--need" in sys.argv else i5() if "--i5" in sys.argv else main()
