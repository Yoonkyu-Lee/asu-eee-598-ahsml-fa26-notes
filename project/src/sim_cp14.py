"""CP14: shrink the fixed cost of a single check, then re-size the verification unit. Rules in cp/CP14.md.

Run from project/:  .venv/Scripts/python.exe src/sim_cp14.py
Writes results/cp14/cp14_fixed.json.

Optimizations (each is checked against the CP11 formulas before it is used):
  O1 cyclotomic squaring (Granger-Scott) in the final exponentiation
  O2 NAF exponent x in the final exponentiation
  O3 fixed-base tables for the verifying key's G1 points (IC_j, alpha): window w bits, read from host DRAM
  O4 G2 subgroup check only for proofs whose test passed
  O5 uncompressed proofs on the wire (on-curve checks instead of square roots)
Costs per kind of test are traced with src/bn254_trace.py; the flood model is CP13's (strategy B with the best
batch size per invalid share f, lock-step utilization per machine).
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
import sim_cp11 as s11  # noqa: E402
import sim_cp13 as s13  # noqa: E402
from bn254_trace import C, Fp12, R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp14"
MHZ, FLOOD = s11.MHZ, s13.FLOOD
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


# ------------------------------------------------------------------ traced tests, by variant

class Costs:
    def __init__(self, su: s13.Setup, v: dict):
        self.su, self.v = su, v
        TABLE_BASES["ic"] = su.ic
        TABLE_BASES["alpha"] = [su.alpha]
        self.reads = {}

    def final(self, f):
        if self.v.get("cyc") or self.v.get("naf"):
            return bn.final_exp_opt(f, cyc=bool(self.v.get("cyc")), use_naf=bool(self.v.get("naf")))
        return bn.final_exp(f)

    def decode(self, k):
        """Per batch: point recovery (and the subgroup check unless deferred)."""
        su = self.su
        if self.v.get("raw"):  # O5: on-curve checks of uncompressed points
            def run():
                for a, b, c in su.proofs[:k]:
                    for p in (a, c):
                        x, y = s11.fresh(p[0]), s11.fresh(p[1])
                        y.sq() - (x.sq() * x + bn.B1)
                    x, y = s11.fresh(b[0]), s11.fresh(b[1])
                    y.sq() - (x.sq() * x + bn.B2)
        elif self.v.get("defer"):
            def run():
                ax = [s11.fresh(a[0]) for a, _, _ in su.proofs[:k]]
                cx = [s11.fresh(c[0]) for _, _, c in su.proofs[:k]]
                bx = [s11.fresh(b[0]) for _, b, _ in su.proofs[:k]]
                s11.ls_pow([x.sq() * x + bn.B1 for x in ax + cx], (bn.P + 1) // 4)
                s11.ls_fp2_sqrt([x.sq() * x + bn.B2 for x in bx])
        else:
            return s13.t_decode(su, k)
        return s13.trace_of(run)

    def subgroup(self, k):
        """O4: the G2 subgroup check alone, in lock-step over k accepted proofs."""
        su = self.su

        def run():
            bs = [(s11.fresh(b[0]), s11.fresh(b[1])) for _, b, _ in su.proofs[:k]]
            m = s11.ls_scalar_mul(bs, [6 * bn.X * bn.X] * len(bs))
            zi = s11.ls_inv([j.z for j in m])
            for j, z, b in zip(m, zi, bs):
                z2 = z.sq()
                (j.x * z2, j.y * z2 * z), bn._frob_point(b)
        return s13.trace_of(run)

    def batch(self, k):
        su, w = self.su, self.v.get("w")

        def run():
            d = su.pts(k)
            r = su.r[:k]
            rA = s11.ls_scalar_mul([a for a, _, _ in d], r)
            sC = s11.straus([c for _, _, c in d], r)
            coeff = [sum(ri * x[j] for ri, x in zip(r, su.publics)) % R for j in range(5)]
            for _ in range(k):
                for _ in range(5):
                    C(su.rng.randrange(bn.P)) * C(su.rng.randrange(bn.P))
            if w:
                sL, n1 = fixed_msm("ic", [sum(r) % R] + coeff, w)
                sA, n2 = fixed_msm("alpha", [sum(r) % R], w)
                self.reads[("batch", k)] = n1 + n2
            else:
                sL = s11.straus(su.ic, [sum(r) % R] + coeff)
                sA = bn.scalar_mul(su.alpha, sum(r) % R)
            aff = s11.batch_affine(rA + [sC, sL, sA])
            pre = [((aff[k][0], -aff[k][1]), su.lines["delta"]), ((aff[k + 1][0], -aff[k + 1][1]), su.lines["gamma"]),
                   ((aff[k + 2][0], -aff[k + 2][1]), su.lines["beta"])]
            self.final(bn.miller_loop_otf([(p, b) for p, (_, b, _) in zip(aff[:k], d)], pre))
        return s13.trace_of(run)

    def single(self):
        su, w = self.su, self.v.get("w")

        def run():
            (A, B, Cp), x = su.pts(1)[0], su.publics[0]
            if w:
                ic0 = (s11.fresh(su.ic[0][0]), s11.fresh(su.ic[0][1]))
                acc, n = fixed_msm("ic", [0] + x, w, bn.Jac(ic0[0], ic0[1], bn.one_like(ic0[0])))
                self.reads["single"] = n + 1
                L = s11.batch_affine([acc])[0]
            else:
                L = s11.batch_affine([s11.straus(su.ic, [1] + x)])[0]
            pre = [((L[0], -L[1]), su.lines["gamma"]), ((Cp[0], -Cp[1]), su.lines["delta"])]
            self.final(bn.miller_loop_otf([(A, B)], pre))
        return s13.trace_of(run)


# ------------------------------------------------------------------ self-tests (rule D1a / S3)

def selftest(su: s13.Setup) -> dict:
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


# ------------------------------------------------------------------ flood model

def simulate(co: dict, defer: bool, rng, batches=300) -> dict:
    """Op totals and per-batch ops for every (n, f): independent of the machine."""
    out = {}
    for n in NS:
        for f in FS:
            tot = {"mul": 0.0, "add": 0.0}
            per, reads = [], 0
            for _ in range(batches if n > 1 else 50):
                bad = rng.random(n) < f
                tests = []
                ops = s13.resolve(bad, "B", co, tests)
                s13.add(ops, co["decode"][1], n)
                if defer:
                    s13.add(ops, co["subgroup"][1], int(n - bad.sum()))
                    s13.add(ops, co["subgroup"][0], 1.0 if (n - bad.sum()) else 0.0)
                reads += sum(co["reads_single"] if t == 1 else co["reads_batch"] for t in tests)
                s13.add(tot, ops)
                per.append((ops["mul"], ops["add"]))
            k = (batches if n > 1 else 50) * n
            out[(n, f)] = {"tot": {x: v / k for x, v in tot.items()}, "per": np.array(per), "reads": reads / k}
    return out


def evaluate(sim: dict, m: int, a: int, util: float) -> dict:
    cyc = lambda mul, add: np.maximum(mul * s11.MUL_OCC / m, add / a) / util
    env = []
    for f in FS:
        best = None
        for n in NS:
            s = sim[(n, f)]
            cpp = float(cyc(s["tot"]["mul"], s["tot"]["add"]))
            rate = MHZ * 1e6 / cpp
            if best is None or rate > best["proofs_per_s"]:
                lat = cyc(s["per"][:, 0], s["per"][:, 1]) / (MHZ * 1e3)
                mj = s13.energy_mJ(s["tot"]) + s["reads"] * 64 * DRAM_PJ_B * 1e-9
                best = {"f": f, "n": n, "proofs_per_s": rate, "mJ_per_proof": mj,
                        "dram_GBs_at_flood": s["reads"] * 64 * FLOOD / 1e9, "batch_work_ms_p99": float(np.percentile(lat, 99))}
        env.append(best)
    worst = min(env, key=lambda e: e["proofs_per_s"])
    area = s13.CP12_VERIFY_MM2 + (m - 5) * s13.MUL_MM2 + (a - 4) * s13.ADD_MM2
    leak = s13.CP12_VERIFY_LEAK_MW + (m - 5) * s13.MUL_LEAK_MW + (a - 4) * s13.ADD_LEAK_MW
    chip_W = s13.CP12_CHIP_ORDINARY_W + max(min(e["proofs_per_s"], FLOOD) * e["mJ_per_proof"] * 1e-3 for e in env) + leak * 1e-3
    return {"M": m, "A": a, "util": util, "worst": worst, "envelope": env, "verify_mm2": area, "chip_flood_W": chip_W,
            "max_dram_GBs": max(e["dram_GBs_at_flood"] for e in env),
            "ok": worst["proofs_per_s"] >= FLOOD and chip_W <= 2.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    su = s13.Setup()
    res = {"selftest": selftest(su)}
    print(res["selftest"], flush=True)
    assert all(res["selftest"].values()), "an optimization changes the result (rule S3)"

    costs = {}
    for name, v in VARIANTS.items():
        c = Costs(su, v)
        co = {"decode": s13.linear(c.decode), "batch": s13.linear(c.batch), "single": s13.counts(c.single())}
        if v.get("defer"):
            co["subgroup"] = s13.linear(c.subgroup)
        co["reads_single"] = c.reads.get("single", 0)
        co["reads_batch"] = c.reads.get(("batch", 8), 0)
        inv = {x: co["decode"][1][x] + co["single"][x] for x in ("mul", "add")}
        co["invalid_proof"] = inv
        costs[name] = co
        print(name, "single", co["single"], "decode/proof", co["decode"][1], "invalid proof", inv, flush=True)
    res["op_counts"] = costs

    built = {n: s11.build(n) for n in (4, 8)}
    utils = {f"{m}x{a}": s13.lockstep_util(m, a, built) for m, a in MACHINES}
    res["lockstep_util"] = utils
    print("util", {k: round(u, 3) for k, u in utils.items()}, flush=True)
    # one test at a time (sensitivity): the fastest single trace on 6 x 6
    c = Costs(su, VARIANTS["V4 +O4 (w8)"])
    r = s11.schedule(c.single(), 6, 6, 256)
    res["alone_single_V4_M6A6"] = r
    print("alone single V4 6x6", r["cycles"], flush=True)

    rng = np.random.default_rng(14)
    res["machines"] = {}
    for name, v in VARIANTS.items():
        sim = simulate(costs[name], bool(v.get("defer")), rng)
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
