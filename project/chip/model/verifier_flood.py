"""CP13: batch verification under a flood that mixes in invalid proofs. Rules are pre-registered in cp/CP13.md.

Run from project/:  .venv/Scripts/python.exe chip/model/verifier_flood.py   (CP13: was src/sim_cp13.py)
Writes results/chip/cp13_flood.json.

1. Trace (verifier_cycles.py, same tests as CP11) the cost of each kind of test, on proofs already decoded:
   batch test of k proofs, single test, building a cached Miller value per proof, a test from cached values.
2. Turn base-field operation counts into cycles on the verification unit (M multipliers, A adders):
   lock-step microcode over independent tests reaches the marginal utilization of a CP11 batch (measured here
   by scheduling CP11 batches); sensitivity: one test at a time (each test scheduled alone).
3. Monte Carlo over proof streams with a fraction f invalid, for strategies A-D (`STRATEGIES`) and batch sizes n.
   CP14 reuses steps 2-3 (`flood`, `cycles`, `unit_size`, `flood_chip_W`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bn254_trace as bn  # noqa: E402
import verifier_cycles as s11  # noqa: E402
from bn254_trace import Fp, Fp2, Fp6, Fp12  # noqa: E402

OUT = s11.OUT
MHZ, FLOOD = s11.MHZ, s11.FLOOD
MUL_MM2 = 0.283293       # CP10 S=1 multiplier, synthesized
ADD_MM2 = 0.011456 / 4   # CP12: 2 x 256-bit Aladdin adders per modular adder
MUL_LEAK_MW, ADD_LEAK_MW = 1.4514, 0.0384
CP12_CHIP_ORDINARY_W = 0.0918   # results/chip/cp12_chip.json chip_A.ordinary_W (encoder at its ordinary rate)
CP12_VERIFY_MM2, CP12_VERIFY_LEAK_MW = 1.8283, 10.888
NS = (8, 16, 32, 64, 128)
FS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
MACHINES = [(5, 4), (5, 6), (5, 8), (5, 10), (6, 6), (6, 8), (7, 7), (7, 10), (8, 8), (8, 10), (10, 10)]

# How a failed batch is resolved (cp/CP13.md alternatives; D = the best n per f, in `summarize`).
#   split: halve and re-test (False: every proof alone); prep: per-proof work done once after the failure;
#   test1 / test: op-count keys of a test of one proof / of k proofs.
STRATEGIES = {
    "A": {"split": False, "test1": "single"},                                                # one by one
    "B": {"split": True, "test1": "single", "test": "batch"},                                # halves, recomputed
    "C": {"split": True, "prep": "cache_build", "test1": "single_cached", "test": "cached"},  # halves, cached Miller
}


def fresh12(f):
    """An Fp12 input (a cached value read from memory)."""
    fp2 = lambda x: Fp2(Fp(x.c0.v), Fp(x.c1.v))
    f6 = lambda x: Fp6(fp2(x.c0), fp2(x.c1), fp2(x.c2))
    return Fp12(f6(f.c0), f6(f.c1))


def counts(trace) -> dict:
    m = sum(1 for t in trace if t[0] == "m")
    return {"mul": m, "add": len(trace) - m}


def t_decode(su: s11.Setup, k: int):
    return s11.trace_of(lambda: s11.ls_decode(su.proofs[:k]))


def t_batch(su: s11.Setup, k: int):
    """Batch test of k decoded proofs (CP11's batch without decoding)."""
    return s11.trace_of(lambda: su.batch_test(su.pts(k)))


def t_single(su: s11.Setup):
    """Single verification of a decoded proof (CP11's single without decoding)."""
    return s11.trace_of(lambda: su.single_test(su.pts(1)[0], su.publics[0]))


def t_cache_build(su: s11.Setup, k: int):
    """After a failed batch: one Miller value per proof, g_i = ML(r_i A_i, B_i), from the affine r_i A_i the batch
    test already made (kept, 64 B each). Each value gets its own squarings (no sharing)."""
    def run():
        d = su.pts(k)
        bn.tracing(False)
        rA = s11.batch_affine(s11.ls_scalar_mul([a for a, _, _ in su.proofs[:k]], su.r[:k]))
        bn.tracing(True)
        for (x, y), (_, b, _) in zip(rA, d):
            bn.miller_loop_otf([((s11.fresh(x), s11.fresh(y)), b)], [])
    return s11.trace_of(run)


def t_cached_test(su: s11.Setup, j: int):
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
        aff = s11.batch_affine(su.weights_tail(j, [c for _, _, c in d]))
        bn.final_exp(prod * bn.miller_loop_otf([], su.vk_pairs(aff)))
    return s11.trace_of(run)


def linear(f, ks=(4, 8)):
    """Fixed + per-proof op counts from two sizes."""
    a, b = (counts(f(k)) for k in ks)
    per = {x: (b[x] - a[x]) / (ks[1] - ks[0]) for x in ("mul", "add")}
    fixed = {x: a[x] - ks[0] * per[x] for x in ("mul", "add")}
    return fixed, per


def lockstep_util(mults: int, adders: int, built) -> float:
    """Marginal utilization of a CP11 batch (decode included) on this machine, relative to the op-count bound."""
    r8, r4, cyc = s11.per_proof({n: built[n]["trace"] for n in (4, 8)}, mults, adders, 256)
    mm = (r8["mul_ops"] - r4["mul_ops"]) / 4
    aa = (r8["add_ops"] - r4["add_ops"]) / 4
    return max(mm * s11.MUL_OCC / mults, aa / adders) / cyc


def cycles(mul, add, m: int, a: int, util: float):
    """Unit cycles for op counts (numbers or arrays): the busier unit kind, at lock-step utilization."""
    return np.maximum(mul * s11.MUL_OCC / m, add / a) / util


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
    s = STRATEGIES[strat]
    if not s["split"]:
        add(ops, co[s["test1"]], n)
        tests.extend([1] * n)
        return ops
    if "prep" in s:
        add(ops, co[s["prep"]][1], n)

    def test(k):
        add(ops, co[s["test1"]] if k == 1 else lin(co[s["test"]], k))
        tests.append(k)

    def split(lo, hi):  # bad[lo:hi] holds at least one invalid proof and has already been tested
        if hi - lo == 1:
            return
        mid = (lo + hi) // 2
        test(mid - lo)
        if bad[lo:mid].any():
            split(lo, mid)
            if bad[mid:hi].any():
                test(hi - mid)
                split(mid, hi)
        else:  # left passed, so the right half is known to fail: split it without testing it whole
            split(mid, hi)
    split(0, n)
    return ops


def flood(strat: str, n: int, f: float, co: dict, rng, batches: int, spread=False, defer=False):
    """Classify `batches` batches of n with a share f invalid: (op totals, per-batch ops [(mul, add)], table reads).
    spread: the adversary spreads invalid proofs evenly (no shuffling at the unit). defer (CP14 O4): the G2 subgroup
    check runs after the test, on the accepted proofs only. Reads count fixed-base table entries (CP14 O3)."""
    tot = {"mul": 0.0, "add": 0.0}
    per, reads = [], 0
    for _ in range(batches):
        if spread:
            k = int(round(f * n))
            bad = np.zeros(n, bool)
            bad[:k] = True
            rng.shuffle(bad) if k else None
        else:
            bad = rng.random(n) < f
        tests = []
        ops = resolve(bad, strat, co, tests)
        add(ops, co["decode"][1], n)
        if defer:
            add(ops, co["subgroup"][1], int(n - bad.sum()))
            add(ops, co["subgroup"][0], 1.0 if (n - bad.sum()) else 0.0)
        reads += sum(co.get("reads_single", 0) if t == 1 else co.get("reads_batch", 0) for t in tests)
        add(tot, ops)
        per.append((ops["mul"], ops["add"]))
    return tot, np.array(per), reads


def unit_size(m: int, a: int):
    """Verification unit area (mm2) and leakage (mW) with m multipliers and a adders, from CP12's 5 + 4."""
    return (CP12_VERIFY_MM2 + (m - 5) * MUL_MM2 + (a - 4) * ADD_MM2,
            CP12_VERIFY_LEAK_MW + (m - 5) * MUL_LEAK_MW + (a - 4) * ADD_LEAK_MW)


def flood_chip_W(env: list, leak_mW: float) -> float:
    """Chip power in the worst flood of an envelope: ordinary encoder + verification at min(capacity, flood) + leak."""
    return CP12_CHIP_ORDINARY_W + max(min(e["proofs_per_s"], FLOOD) * e["mJ_per_proof"] * 1e-3 for e in env) \
        + leak_mW * 1e-3


def measure(strat, n, f, co, m, a, u, rng, batches=400, spread=False) -> dict:
    """One Monte Carlo point on one machine: capacity, energy per proof, P99 work per batch."""
    tot, per, _ = flood(strat, n, f, co, rng, batches, spread)
    proofs = batches * n
    return {"proofs_per_s": MHZ * 1e6 / (float(cycles(tot["mul"], tot["add"], m, a, u)) / proofs),
            "mJ_per_proof": s11.energy_mJ({k: v / proofs for k, v in tot.items()}),
            "batch_work_ms_p99": float(np.percentile(cycles(per[:, 0], per[:, 1], m, a, u), 99)) / (MHZ * 1e3)}


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
        area, leak = unit_size(m, a)
        chip_W = flood_chip_W(env["B"], leak)
        out[f"{m}x{a}"] = {"envelope_B": env["B"], "envelope_C_worst": min(e["proofs_per_s"] for e in env["C"]),
                           "worst_proofs_per_s": worst, "f_max": f_max, "verify_mm2": area, "verify_leak_mW": leak,
                           "chip_flood_W_worst_f": chip_W,
                           "D2_ok": worst >= FLOOD and area <= 2 * CP12_VERIFY_MM2 and chip_W <= 2.0}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    su = s11.Setup(16)
    co = {"decode": linear(lambda k: t_decode(su, k)),
          "batch": linear(lambda k: t_batch(su, k)),
          "single": counts(t_single(su)),
          "cache_build": linear(lambda k: t_cache_build(su, k)),
          "cached": linear(lambda k: t_cached_test(su, k)),
          "single_cached": counts(t_cached_test(su, 1))}
    # check linearity of the batch test at a third size
    co["batch_linearity_16"] = {"traced": counts(t_batch(su, 16)), "fit": lin(co["batch"], 16)}
    print(json.dumps(co, indent=1), flush=True)

    built = {n: s11.build(n) for n in (4, 8)}
    utils = {}
    for m, a in MACHINES:
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
    for m, a in MACHINES:
        u = utils[f"{m}x{a}"]
        for strat in ("A", "B", "C"):
            for n in NS:
                for f in FS:
                    rows.append({"M": m, "A": a, "strategy": strat, "n": n, "f": f,
                                 **measure(strat, n, f, co, m, a, u, rng), "fill_ms": n / FLOOD * 1e3})
        # n = 1: singles only (also strategy D's choice when f is large)
        for f in FS:
            rows.append({"M": m, "A": a, "strategy": "single", "n": 1, "f": f,
                         **measure("A", 1, f, co, m, a, u, rng, batches=50), "fill_ms": 0.0})
        print("machine", m, a, "done", flush=True)
    res["rows"] = rows
    # adversary spreading invalid proofs evenly, M=5 (sensitivity for unshuffled batches)
    res["spread_M5"] = [{"strategy": strat, "n": n, "f": f,
                         "proofs_per_s": measure(strat, n, f, co, 5, 4, utils["5x4"], rng, spread=True)["proofs_per_s"]}
                        for strat in ("B", "C") for n in NS for f in (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8)]
    # finer f around the crossing, for f_max
    fine = []
    for m, a in ((5, 4), (8, 8)):
        for f in np.round(np.arange(0.02, 0.21, 0.01), 2):
            for n in (8, 16, 32):
                r = measure("B", n, float(f), co, m, a, utils[f"{m}x{a}"], rng, batches=1000)
                fine.append({"M": m, "A": a, "strategy": "B", "n": n, "f": float(f),
                             "proofs_per_s": r["proofs_per_s"], "mJ_per_proof": r["mJ_per_proof"]})
    res["fine"] = fine
    res["summary"] = summarize(rows, fine)
    print(json.dumps(res["summary"], indent=1))
    (OUT / "cp13_flood.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT / "cp13_flood.json")


if __name__ == "__main__":
    main()
