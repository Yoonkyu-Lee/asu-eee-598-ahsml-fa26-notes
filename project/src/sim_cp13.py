"""CP13: batch verification under a flood that mixes in invalid proofs. Rules are pre-registered in cp/CP13.md.

Run from project/:  .venv/Scripts/python.exe src/sim_cp13.py
Writes results/cp13/cp13_flood.json.

1. Trace (src/bn254_trace.py, same formulas as CP11) the cost of each kind of test, on proofs already decoded:
   batch test of k proofs, single test, building a cached Miller value per proof, a test from cached values.
2. Turn base-field operation counts into cycles on the verification unit (M multipliers, A adders):
   lock-step microcode over independent tests reaches the marginal utilization of a CP11 batch (measured here
   by scheduling CP11 batches); sensitivity: one test at a time (each test scheduled alone).
3. Monte Carlo over proof streams with a fraction f invalid, for strategies A-D and batch sizes n.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bn254_trace as bn  # noqa: E402
import sim_cp11 as s11  # noqa: E402
from bn254_trace import C, Fp, Fp2, Fp6, Fp12, P, R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp13"
MHZ = s11.MHZ
FLOOD = 12_207
MUL_MM2 = 0.283293       # CP10 S=1 multiplier, synthesized
ADD_MM2 = 0.011456 / 4   # CP12: 2 x 256-bit Aladdin adders per modular adder
MUL_LEAK_MW, ADD_LEAK_MW = 1.4514, 0.0384
NS = (8, 16, 32, 64, 128)
FS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def fresh12(f):
    """An Fp12 input (a cached value read from memory)."""
    fp2 = lambda x: Fp2(Fp(x.c0.v), Fp(x.c1.v))
    f6 = lambda x: Fp6(fp2(x.c0), fp2(x.c1), fp2(x.c2))
    return Fp12(f6(f.c0), f6(f.c1))


def counts(trace) -> dict:
    m = sum(1 for t in trace if t[0] == "m")
    return {"mul": m, "add": len(trace) - m}


class Setup:
    """Decoded proofs, verifying key and prepared lines, made without tracing (decoding is costed separately)."""

    def __init__(self, n: int = 16, seed: int = 598):
        rng = random.Random(seed)
        self.rng = rng
        bn.tracing(False)
        self.proofs = [(s11.rand_g1(rng), s11.rand_g2(rng), s11.rand_g1(rng)) for _ in range(n)]
        self.alpha = s11.rand_g1(rng)
        beta, gamma, delta = (s11.rand_g2(rng) for _ in range(3))
        self.ic = [s11.rand_g1(rng) for _ in range(6)]
        self.publics = [[rng.randrange(R) for _ in range(5)] for _ in range(n)]
        self.lines = {k: bn.prepare(q) for k, q in (("beta", beta), ("gamma", gamma), ("delta", delta))}
        self.r = [rng.getrandbits(128) | 1 for _ in range(n)]

    def pts(self, k):
        """Decoded points of the first k proofs as fresh inputs (read from the 256 B per-proof store)."""
        g1 = lambda p: (s11.fresh(p[0]), s11.fresh(p[1]))
        g2 = lambda q: (s11.fresh(q[0]), s11.fresh(q[1]))
        return [(g1(a), g2(b), g1(c)) for a, b, c in self.proofs[:k]]

    def weights_tail(self, k, cs):
        """sum r_i C_i, public-input weighting, sum r L, (sum r) alpha; returns the three affine G1 points."""
        r = self.r[:k]
        sC = s11.straus(cs, r)
        coeff = [sum(ri * x[j] for ri, x in zip(r, self.publics)) % R for j in range(5)]
        for _ in range(k):  # scalar-field weighting of public inputs, counted as base-field work (as CP11)
            for _ in range(5):
                C(self.rng.randrange(P)) * C(self.rng.randrange(P))
        sL = s11.straus(self.ic, [sum(r) % R] + coeff)
        sA = bn.scalar_mul(self.alpha, sum(r) % R)
        return sC, sL, sA


def trace_of(fn) -> list:
    bn.TRACE.clear()
    bn.tracing(True)
    fn()
    bn.tracing(False)
    return list(bn.TRACE)


def t_decode(su: Setup, k: int):
    return trace_of(lambda: s11.ls_decode(su.proofs[:k]))


def t_batch(su: Setup, k: int):
    """Batch test of k decoded proofs (CP11's batch without decoding)."""
    def run():
        d = su.pts(k)
        rA = s11.ls_scalar_mul([a for a, _, _ in d], su.r[:k])
        sC, sL, sA = su.weights_tail(k, [c for _, _, c in d])
        aff = s11.batch_affine(rA + [sC, sL, sA])
        pre = [((aff[k][0], -aff[k][1]), su.lines["delta"]), ((aff[k + 1][0], -aff[k + 1][1]), su.lines["gamma"]),
               ((aff[k + 2][0], -aff[k + 2][1]), su.lines["beta"])]
        bn.final_exp(bn.miller_loop_otf([(p, b) for p, (_, b, _) in zip(aff[:k], d)], pre))
    return trace_of(run)


def t_single(su: Setup):
    """Single verification of a decoded proof (CP11's single without decoding)."""
    def run():
        (A, B, Cp), x = su.pts(1)[0], su.publics[0]
        L = s11.batch_affine([s11.straus(su.ic, [1] + x)])[0]
        pre = [((L[0], -L[1]), su.lines["gamma"]), ((Cp[0], -Cp[1]), su.lines["delta"])]
        bn.final_exp(bn.miller_loop_otf([(A, B)], pre))
    return trace_of(run)


def t_cache_build(su: Setup, k: int):
    """After a failed batch: one Miller value per proof, g_i = ML(r_i A_i, B_i), from the affine r_i A_i the batch
    test already made (kept, 64 B each). Each value gets its own squarings (no sharing)."""
    def run():
        d = su.pts(k)
        bn.tracing(False)
        rA = s11.batch_affine(s11.ls_scalar_mul([a for a, _, _ in su.proofs[:k]], su.r[:k]))
        bn.tracing(True)
        for (x, y), (_, b, _) in zip(rA, d):
            bn.miller_loop_otf([((s11.fresh(x), s11.fresh(y)), b)], [])
    return trace_of(run)


def t_cached_test(su: Setup, j: int):
    """Test of a subset of j proofs from cached Miller values: product of the g_i, the three vk pairs for this
    subset (MSMs redone over the subset), one final exponentiation."""
    bn.tracing(False)
    d0 = su.proofs[0]
    g = bn.miller_loop_otf([(d0[0], d0[1])], [])
    def run():
        d = su.pts(j)
        prod = fresh12(g)
        for _ in range(j - 1):
            prod = prod * fresh12(g)
        sC, sL, sA = su.weights_tail(j, [c for _, _, c in d])
        aff = s11.batch_affine([sC, sL, sA])
        pre = [((aff[0][0], -aff[0][1]), su.lines["delta"]), ((aff[1][0], -aff[1][1]), su.lines["gamma"]),
               ((aff[2][0], -aff[2][1]), su.lines["beta"])]
        bn.final_exp(prod * bn.miller_loop_otf([], pre))
    return trace_of(run)


def linear(f, ks=(4, 8)):
    """Fixed + per-proof op counts from two sizes."""
    a, b = (counts(f(k)) for k in ks)
    per = {x: (b[x] - a[x]) / (ks[1] - ks[0]) for x in ("mul", "add")}
    fixed = {x: a[x] - ks[0] * per[x] for x in ("mul", "add")}
    return fixed, per


def lockstep_util(mults: int, adders: int, built) -> float:
    """Marginal utilization of a CP11 batch (decode included) on this machine, relative to the op-count bound."""
    r8 = s11.schedule(built[8]["trace"], mults, adders, 256)
    r4 = s11.schedule(built[4]["trace"], mults, adders, 256)
    cyc = (r8["cycles"] - r4["cycles"]) / 4
    mm = (r8["mul_ops"] - r4["mul_ops"]) / 4
    aa = (r8["add_ops"] - r4["add_ops"]) / 4
    return max(mm * s11.MUL_OCC / mults, aa / adders) / cyc


class Machine:
    def __init__(self, mults, adders, util, costs):
        self.M, self.A, self.u, self.c = mults, adders, util, costs

    def cyc(self, ops: dict) -> float:
        return max(ops["mul"] * s11.MUL_OCC / self.M, ops["add"] / self.A) / self.u


def lin(c, k):
    fixed, per = c
    return {x: fixed[x] + k * per[x] for x in ("mul", "add")}


def add(acc, ops, k=1.0):
    acc["mul"] += k * ops["mul"]
    acc["add"] += k * ops["add"]


def resolve(bad: np.ndarray, strat: str, co: dict, tests: list) -> dict:
    """Op counts to classify one batch (decoding excluded). bad: bool per proof. tests collects test sizes."""
    ops = {"mul": 0.0, "add": 0.0}
    n = len(bad)
    if n == 1:
        add(ops, co["single"])
        tests.append(1)
        return ops
    add(ops, lin(co["batch"], n))
    tests.append(n)
    if not bad.any():
        return ops
    if strat == "A":
        add(ops, co["single"], n)
        tests.extend([1] * n)
        return ops
    if strat == "C":
        add(ops, co["cache_build"][1], n)  # per-proof Miller values, made once
        test = lambda k: co["single_cached"] if k == 1 else lin(co["cached"], k)
    else:
        test = lambda k: co["single"] if k == 1 else lin(co["batch"], k)

    def split(lo, hi):  # bad[lo:hi] holds at least one invalid proof and has already been tested
        if hi - lo == 1:
            return
        mid = (lo + hi) // 2
        add(ops, test(mid - lo))
        tests.append(mid - lo)
        if bad[lo:mid].any():
            split(lo, mid)
            if bad[mid:hi].any():
                add(ops, test(hi - mid))
                tests.append(hi - mid)
                if hi - mid > 1:
                    split(mid, hi)
        else:  # left passed, so the right half is known to fail: split it without testing it whole
            split(mid, hi)
    split(0, n)
    return ops


def run_strategy(strat, n, f, co, mach, rng, batches=400, spread=False):
    tot = {"mul": 0.0, "add": 0.0}
    lat = []
    proofs = 0
    for _ in range(batches):
        if spread:  # adversary spreads invalid proofs evenly (no shuffling at the unit)
            k = int(round(f * n))
            bad = np.zeros(n, bool)
            bad[:k] = True
            rng.shuffle(bad) if k else None
        else:
            bad = rng.random(n) < f
        tests = []
        ops = resolve(bad, strat, co, tests)
        add(ops, co["decode"][1], n)
        add(tot, ops)
        proofs += n
        lat.append(mach.cyc(ops))
    cyc_pp = mach.cyc(tot) / proofs
    return cyc_pp, tot, proofs, lat


def energy_mJ(ops: dict) -> float:
    return (ops["mul"] * s11.NJ_MUL * 1e3 + ops["add"] * s11.PJ_ADD + (ops["mul"] + ops["add"]) * s11.PJ_TRAFFIC) * 1e-9


CP12_CHIP_ORDINARY_W = 0.0918   # results/cp12/cp12_chip.json chip_A.ordinary_W (encoder at its ordinary rate)
CP12_VERIFY_MM2, CP12_VERIFY_LEAK_MW = 1.8283, 10.888


def summarize(rows, fine) -> dict:
    """Strategy D = best batch size per f (oracle), from B or C plus singles; worst case over f; D2 check."""
    out = {}
    for m, a in sorted({(r["M"], r["A"]) for r in rows}):
        mine = [r for r in rows + [x for x in fine if (x["M"], x["A"]) == (m, a)] if (r["M"], r["A"]) == (m, a)]
        fs = sorted({r["f"] for r in mine})
        env = {}
        single = [r for r in mine if r["strategy"] == "single" and r["f"] == 1.0]
        for s in ("B", "C"):
            env[s] = []
            for f in fs:
                cand = [r for r in mine if r["f"] == f and r["strategy"] == s] + single  # singles do not depend on f
                if not cand:
                    continue
                b = max(cand, key=lambda r: r["proofs_per_s"])
                rate = min(b["proofs_per_s"], FLOOD)
                env[s].append({"f": f, "proofs_per_s": b["proofs_per_s"], "pick": f"{b['strategy']} n={b['n']}",
                               "mJ_per_proof": b["mJ_per_proof"], "verify_W": rate * b["mJ_per_proof"] * 1e-3})
        worst = min(e["proofs_per_s"] for e in env["B"])
        ok_f = [e["f"] for e in env["B"] if e["proofs_per_s"] >= FLOOD]
        f_max = max(f for f in ok_f if all(e["proofs_per_s"] >= FLOOD for e in env["B"] if e["f"] <= f)) if ok_f else 0.0
        area = CP12_VERIFY_MM2 + (m - 5) * MUL_MM2 + (a - 4) * ADD_MM2
        leak = CP12_VERIFY_LEAK_MW + (m - 5) * MUL_LEAK_MW + (a - 4) * ADD_LEAK_MW
        chip_W = CP12_CHIP_ORDINARY_W + max(e["verify_W"] for e in env["B"]) + leak * 1e-3
        out[f"{m}x{a}"] = {"envelope_B": env["B"], "envelope_C_worst": min(e["proofs_per_s"] for e in env["C"]),
                           "worst_proofs_per_s": worst, "f_max": f_max, "verify_mm2": area, "verify_leak_mW": leak,
                           "chip_flood_W_worst_f": chip_W,
                           "D2_ok": worst >= FLOOD and area <= 2 * CP12_VERIFY_MM2 and chip_W <= 2.0}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    su = Setup()
    co = {"decode": linear(lambda k: t_decode(su, k)),
          "batch": linear(lambda k: t_batch(su, k)),
          "single": counts(t_single(su)),
          "cache_build": linear(lambda k: t_cache_build(su, k)),
          "cached": linear(lambda k: t_cached_test(su, k)),
          "single_cached": counts(t_cached_test(su, 1))}
    # check linearity of the batch test at a third size
    b16 = counts(t_batch(su, 16))
    co["batch_linearity_16"] = {"traced": b16, "fit": lin(co["batch"], 16)}
    print(json.dumps(co, indent=1), flush=True)

    built = {n: s11.build(n) for n in (4, 8)}
    machines = [(5, 4), (5, 6), (5, 8), (5, 10), (6, 6), (6, 8), (7, 7), (7, 10), (8, 8), (8, 10), (10, 10)]
    utils = {}
    for m, a in machines:
        utils[f"{m}x{a}"] = lockstep_util(m, a, built)
        print("lock-step util", m, a, round(utils[f"{m}x{a}"], 3), flush=True)
    # one test at a time: each kind of test scheduled alone on M=5, A=4
    alone = {}
    for name, tr in (("single", t_single(su)), ("batch8", t_batch(su, 8)), ("cached1", t_cached_test(su, 1)),
                     ("cached8", t_cached_test(su, 8)), ("cache_build1", t_cache_build(su, 1))):
        r = s11.schedule(tr, 5, 4, 256)
        alone[name] = {**r, "util_vs_bound": max(r["mul_ops"] * s11.MUL_OCC / 5, r["add_ops"] / 4) / r["cycles"]}
        print("alone", name, r["cycles"], round(alone[name]["util_vs_bound"], 3), flush=True)

    res = {"machine": {"MHz": MHZ, "mul_occ": s11.MUL_OCC, "nJ_mul": s11.NJ_MUL, "pJ_add": s11.PJ_ADD,
                       "pJ_traffic": s11.PJ_TRAFFIC},
           "op_counts": co, "lockstep_util": utils, "alone_M5A4": alone}

    rng = np.random.default_rng(13)
    rows = []
    for m, a in machines:
        mach = Machine(m, a, utils[f"{m}x{a}"], co)
        for strat in ("A", "B", "C"):
            for n in NS:
                for f in FS:
                    cpp, tot, np_, lat = run_strategy(strat, n, f, co, mach, rng)
                    row = {"M": m, "A": a, "strategy": strat, "n": n, "f": f, "proofs_per_s": MHZ * 1e6 / cpp,
                           "mJ_per_proof": energy_mJ({k: v / np_ for k, v in tot.items()}),
                           "batch_work_ms_p99": float(np.percentile(lat, 99)) / (MHZ * 1e3),
                           "fill_ms": n / FLOOD * 1e3}
                    rows.append(row)
        # n = 1: singles only (also strategy D's choice when f is large)
        for f in FS:
            cpp, tot, np_, lat = run_strategy("A", 1, f, co, mach, rng, batches=50)
            rows.append({"M": m, "A": a, "strategy": "single", "n": 1, "f": f, "proofs_per_s": MHZ * 1e6 / cpp,
                         "mJ_per_proof": energy_mJ({k: v / np_ for k, v in tot.items()}),
                         "batch_work_ms_p99": float(np.percentile(lat, 99)) / (MHZ * 1e3), "fill_ms": 0.0})
        print("machine", m, a, "done", flush=True)
    res["rows"] = rows
    # adversary spreading invalid proofs evenly, M=5 (sensitivity for unshuffled batches)
    mach = Machine(5, 4, utils["5x4"], co)
    spread = []
    for strat in ("B", "C"):
        for n in NS:
            for f in (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8):
                cpp, tot, np_, _ = run_strategy(strat, n, f, co, mach, rng, spread=True)
                spread.append({"strategy": strat, "n": n, "f": f, "proofs_per_s": MHZ * 1e6 / cpp})
    res["spread_M5"] = spread
    # finer f around the crossing, for f_max
    fine = []
    for m, a in ((5, 4), (8, 8)):
        mach = Machine(m, a, utils[f"{m}x{a}"], co)
        for f in np.round(np.arange(0.02, 0.21, 0.01), 2):
            for n in (8, 16, 32):
                cpp, tot, np_, _ = run_strategy("B", n, float(f), co, mach, rng, batches=1000)
                fine.append({"M": m, "A": a, "strategy": "B", "n": n, "f": float(f), "proofs_per_s": MHZ * 1e6 / cpp,
                             "mJ_per_proof": energy_mJ({k: v / np_ for k, v in tot.items()})})
    res["fine"] = fine
    res["summary"] = summarize(rows, fine)
    print(json.dumps(res["summary"], indent=1))
    (OUT / "cp13_flood.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT / "cp13_flood.json")


if __name__ == "__main__":
    main()
