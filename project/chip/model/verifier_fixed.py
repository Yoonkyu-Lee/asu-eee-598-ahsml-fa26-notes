"""CP14: shrink the fixed cost of a single check, then re-size the verification unit. Rules in cp/CP14.md.

Run from project/:  .venv/Scripts/python.exe chip/model/verifier_fixed.py   (CP14: was src/sim_cp14.py)
Writes results/chip/cp14_fixed.json.

Optimizations (each is checked against the CP11 formulas before it is used):
  O1 cyclotomic squaring (Granger-Scott) in the final exponentiation
  O2 NAF exponent x in the final exponentiation
  O3 fixed-base tables for the verifying key's G1 points (IC_j, alpha): window w bits, read from host DRAM
  O4 G2 subgroup check only for proofs whose test passed
  O5 uncompressed proofs on the wire (on-curve checks instead of square roots)
Costs per kind of test are traced with verifier_cycles.py's tests; the flood model is CP13's (strategy B with the
best batch size per invalid share f, lock-step utilization per machine).
"""
from __future__ import annotations

import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bn254_trace as bn  # noqa: E402
import verifier_cycles as s11  # noqa: E402
import verifier_flood as s13  # noqa: E402
from bn254_trace import C, Fp12, R  # noqa: E402

OUT = s11.OUT
MHZ, FLOOD = s11.MHZ, s11.FLOOD
DRAM_PJ_B = 64.0                 # CP12: Accelergy LPDDR4, per byte
DRAM_LIMIT_GBS = 0.85            # rule D1: 10% of LPDDR4 8.5 GB/s
MACHINES = [(m, a) for m in (5, 6, 7, 8) for a in (4, 5, 6, 8)]
NS = (1, 8, 16, 32, 64, 128)
FS = (0.0, 0.001, 0.003, 0.01) + tuple(round(0.02 + 0.01 * i, 2) for i in range(19)) + (0.3, 0.5, 0.7, 1.0)
VARIANTS = {  # cumulative
    "V0 CP13": {},
    "V1 +O1": {"cyc": 1},
    "V2 +O2": {"cyc": 1, "naf": 1},
    "V3 +O3 w4": {"cyc": 1, "naf": 1, "w": 4},
    "V3 +O3 w8": {"cyc": 1, "naf": 1, "w": 8},
    "V4 +O4 (w8)": {"cyc": 1, "naf": 1, "w": 8, "defer": 1},
    "V4 +O4 (w4)": {"cyc": 1, "naf": 1, "w": 4, "defer": 1},
    "V5 +O5 (w8)": {"cyc": 1, "naf": 1, "w": 8, "defer": 1, "raw": 1},
}


# ------------------------------------------------------------------ O3: fixed-base tables

@lru_cache(maxsize=None)
def table_entry(base_key: str, j: int, w: int, i: int, d: int):
    """Affine d * 2^(w i) * P, made off-line (not traced). base_key selects the setup's point list."""
    pt = TABLE_BASES[base_key][j]
    was = bn._on[0]
    bn.tracing(False)
    q = bn.scalar_mul(pt, d << (w * i)).to_affine()
    bn.tracing(was)
    return q


TABLE_BASES: dict = {}


def fixed_msm(base_key: str, scalars: list[int], w: int, acc=None):
    """sum k_j P_j from tables: one mixed addition per non-zero window digit, no doublings. Returns (Jac, reads)."""
    mask, reads = (1 << w) - 1, 0
    for j, k in enumerate(scalars):
        i = 0
        while k >> (w * i):
            d = (k >> (w * i)) & mask
            if d:
                x, y = table_entry(base_key, j, w, i, d)
                x, y = s11.fresh(x), s11.fresh(y)
                acc = bn.Jac(x, y, bn.one_like(x)) if acc is None else acc.add_affine(x, y)
                reads += 1
            i += 1
    return acc, reads


# ------------------------------------------------------------------ op counts of each test, by variant

def tests(su: s11.Setup, v: dict):
    """The tests of a variant, each a function of k proofs returning its trace; and the table reads (O3) they
    record as they run."""
    TABLE_BASES["ic"] = su.ic
    TABLE_BASES["alpha"] = [su.alpha]
    w, reads = v.get("w"), {}
    final = bn.final_exp
    if v.get("cyc") or v.get("naf"):
        def final(f):
            return bn.final_exp_opt(f, cyc=bool(v.get("cyc")), use_naf=bool(v.get("naf")))
    vk_msm = ic_msm = None
    if w:
        def vk_msm(scalars):
            (sL, n1), (sA, n2) = fixed_msm("ic", scalars, w), fixed_msm("alpha", scalars[:1], w)
            reads["batch"] = n1 + n2
            return sL, sA

        def ic_msm(x):
            ic0 = (s11.fresh(su.ic[0][0]), s11.fresh(su.ic[0][1]))
            acc, n = fixed_msm("ic", [0] + x, w, bn.Jac(ic0[0], ic0[1], bn.one_like(ic0[0])))
            reads["single"] = n + 1
            return acc

    def decode(k):  # per batch: point recovery, and the subgroup check unless deferred (O4)
        if not v.get("raw"):
            return (s11.ls_recover if v.get("defer") else s11.ls_decode)(su.proofs[:k])
        for a, b, c in su.proofs[:k]:  # O5: on-curve checks of uncompressed points
            for p, coef in ((a, bn.B1), (c, bn.B1), (b, bn.B2)):
                x, y = s11.fresh(p[0]), s11.fresh(p[1])
                y.sq() - (x.sq() * x + coef)

    def subgroup(k):  # O4: the G2 subgroup check alone, over k accepted proofs
        s11.ls_subgroup([(s11.fresh(b[0]), s11.fresh(b[1])) for _, b, _ in su.proofs[:k]])

    run = {"decode": decode, "subgroup": subgroup,
           "batch": lambda k: su.batch_test(su.pts(k), final, vk_msm),
           "single": lambda k: su.single_test(su.pts(1)[0], su.publics[0], final, ic_msm)}
    traced = lambda fn: lambda k=1: s11.trace_of(lambda: fn(k))
    return {name: traced(fn) for name, fn in run.items()}, reads


def costs(su: s11.Setup, v: dict) -> dict:
    """Op counts (CP13's form): decoding (point recovery, plus the subgroup check unless deferred), batch test,
    single test, the deferred subgroup check, table reads per test, and the cost of one invalid proof."""
    t, reads = tests(su, v)
    co = {"decode": s13.linear(t["decode"]), "batch": s13.linear(t["batch"]), "single": s13.counts(t["single"]())}
    if v.get("defer"):
        co["subgroup"] = s13.linear(t["subgroup"])
    co["reads_single"] = reads.get("single", 0)
    co["reads_batch"] = reads.get("batch", 0)
    co["invalid_proof"] = {x: co["decode"][1][x] + co["single"][x] for x in ("mul", "add")}
    return co


# ------------------------------------------------------------------ self-tests (rule D1a / S3)

def selftest(su: s11.Setup) -> dict:
    bn.tracing(False)
    rng = random.Random(14)
    out = {}
    a, b, _ = su.proofs[0]
    f = bn.miller_loop([(a, bn.prepare(b))])
    easy = f.conj() * f.inv()
    r = easy.frob(2) * easy                      # in the cyclotomic subgroup
    out["O1 cyc_sq == sq"] = r.cyc_sq() == r.sq()
    out["O2 naf exp == binary exp"] = r.exp_cyc_naf(bn.X) == r.exp_cyc(bn.X)
    ref = bn.final_exp(f)
    out["O1+O2 final_exp"] = all(bn.final_exp_opt(f, c, n) == ref for c in (False, True) for n in (False, True))
    p2, q2 = bn.scalar_mul(bn.G1_GEN, 2).to_affine(), bn.scalar_mul(bn.G2_GEN, 2).to_affine()
    fe = lambda p, q: bn.final_exp_opt(bn.miller_loop([(p, bn.prepare(q))]))
    e = fe(bn.G1_GEN, bn.G2_GEN)
    out["O1+O2 bilinear"] = (not (e == Fp12.one())) and fe(p2, bn.G2_GEN) == fe(bn.G1_GEN, q2) == e * e
    TABLE_BASES["ic"] = su.ic
    ks = [rng.randrange(R) for _ in range(6)]
    ref_pt = s11.batch_affine([s11.straus(su.ic, ks)])[0]
    ok = True
    for w in (4, 8):
        acc, _ = fixed_msm("ic", ks, w)
        ok &= s11.batch_affine([acc])[0][0] == ref_pt[0]
    out["O3 fixed-base == Straus"] = bool(ok)
    x, y = su.proofs[0][0]
    out["O5 on-curve accepts / rejects"] = (y.sq() == x.sq() * x + bn.B1) and not ((y + C(1)).sq() == x.sq() * x + bn.B1)
    return {k: bool(v) for k, v in out.items()}


# ------------------------------------------------------------------ flood model (CP13's)

def simulate(co: dict, defer: bool, rng, batches=300) -> dict:
    """Op totals and per-batch ops for every (n, f), strategy B: independent of the machine."""
    out = {}
    for n in NS:
        for f in FS:
            b = batches if n > 1 else 50
            tot, per, reads = s13.flood("B", n, f, co, rng, b, defer=defer)
            out[(n, f)] = {"tot": {x: v / (b * n) for x, v in tot.items()}, "per": per, "reads": reads / (b * n)}
    return out


def evaluate(sim: dict, m: int, a: int, util: float) -> dict:
    """Strategy D on one machine: the best n per f, the worst f, area and flood power."""
    env = []
    for f in FS:
        rate = {n: MHZ * 1e6 / float(s13.cycles(sim[(n, f)]["tot"]["mul"], sim[(n, f)]["tot"]["add"], m, a, util))
                for n in NS}
        n = max(NS, key=rate.get)
        s = sim[(n, f)]
        lat = s13.cycles(s["per"][:, 0], s["per"][:, 1], m, a, util) / (MHZ * 1e3)
        env.append({"f": f, "n": n, "proofs_per_s": rate[n],
                    "mJ_per_proof": s11.energy_mJ(s["tot"]) + s["reads"] * 64 * DRAM_PJ_B * 1e-9,
                    "dram_GBs_at_flood": s["reads"] * 64 * FLOOD / 1e9, "batch_work_ms_p99": float(np.percentile(lat, 99))})
    worst = min(env, key=lambda e: e["proofs_per_s"])
    area, leak = s13.unit_size(m, a)
    chip_W = s13.flood_chip_W(env, leak)
    return {"M": m, "A": a, "util": util, "worst": worst, "envelope": env, "verify_mm2": area, "chip_flood_W": chip_W,
            "max_dram_GBs": max(e["dram_GBs_at_flood"] for e in env),
            "ok": worst["proofs_per_s"] >= FLOOD and chip_W <= 2.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    su = s11.Setup(16)
    res = {"selftest": selftest(su)}
    print(res["selftest"], flush=True)
    assert all(res["selftest"].values()), "an optimization changes the result (rule S3)"

    costs_ = {}
    for name, v in VARIANTS.items():
        co = costs(su, v)
        costs_[name] = co
        print(name, "single", co["single"], "decode/proof", co["decode"][1], "invalid proof", co["invalid_proof"],
              flush=True)

    built = {n: s11.build(n) for n in (4, 8)}
    utils = {f"{m}x{a}": s13.lockstep_util(m, a, built) for m, a in MACHINES}
    print("util", {k: round(u, 3) for k, u in utils.items()}, flush=True)
    # one test at a time (sensitivity): the fastest single trace on 6 x 6
    r = s11.schedule(tests(su, VARIANTS["V4 +O4 (w8)"])[0]["single"](), 6, 6, 256)
    print("alone single V4 6x6", r["cycles"], flush=True)
    res.update({"op_counts": costs_, "lockstep_util": utils, "alone_single_V4_M6A6": r, "machines": {}})

    rng = np.random.default_rng(14)
    for name, v in VARIANTS.items():
        sim = simulate(costs_[name], bool(v.get("defer")), rng)
        rows = [evaluate(sim, m, a, utils[f"{m}x{a}"]) for m, a in MACHINES]
        res["machines"][name] = rows
        ok = [r for r in rows if r["ok"]]
        pick = min(ok, key=lambda r: (r["verify_mm2"], r["M"])) if ok else None
        print(name, " ".join(f"{r['M']}x{r['A']}:{r['worst']['proofs_per_s']:.0f}" for r in rows),
              "pick", (pick["M"], pick["A"]) if pick else None, flush=True)
    # O4 worst case: proofs that pass the test but fail the subgroup check cost what valid proofs cost (f = 0 row)
    (OUT / "cp14_fixed.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT / "cp14_fixed.json")


if __name__ == "__main__":
    main()
