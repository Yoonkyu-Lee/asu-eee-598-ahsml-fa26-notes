"""CP11: replay a Groth16 batch verification, base-field operation by operation, on a cycle model of the
verification unit. Rules are pre-registered in cp/CP11.md. Also the core CP13 and CP14 build on: the proofs and key
(`Setup`), the traced tests (decode, batch test, single test), the list scheduler (`schedule`, `per_proof`).

Run from project/:  .venv/Scripts/python.exe chip/model/verifier_cycles.py   (CP11: was src/sim_cp11.py)
Writes results/chip/cp11_sim.json.

Workload (per batch of n proofs, BN254, bn254_trace.py), written in lock-step over the batch:
  decode each proof: decompress A, C (G1) and B (G2), G2 subgroup check
  weights: r_i A_i (128-bit r_i), sum r_i C_i (shared-doubling multi-scalar), sum r_i L_i (vk IC points),
           (sum r_i) alpha; all to affine with one batch inversion (Montgomery's trick)
  multi-Miller loop over n + 3 pairs (lines for the n fresh B_i computed in the loop, right before use;
  three precomputed vk lines), one final exponentiation
Machine: M multipliers (each busy `MUL_OCC` cycles per product, result after `MUL_LAT`), A adders (1 cycle).
Synthetic proofs (random group elements), so the final equality does not hold; only the work matters.
"""
from __future__ import annotations

import heapq
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bn254_trace as bn  # noqa: E402
from bn254_trace import C, Fp, Fp2, Jac, P, R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "chip"
MHZ = 300.0              # CP10 S=1 multiplier clock
MUL_OCC, MUL_LAT = 4, 4  # CP10 S=1: a product every 4 cycles per unit, result 4 cycles after issue
NJ_MUL = 1.758           # CP10 S=1, synthesized (results/cp10)
PJ_ADD = 1.6             # CP7: 254-bit add + conditional subtract, L7 45 nm
PJ_TRAFFIC = 24 * 5.0    # CP7: two operands in, one out, 8 words each, 8 KB SRAM 5 pJ per 32-bit word
FLOOD = 12_207           # proofs/s that fill 100 Mbps
R2 = 2 * FLOOD           # CP11 rule R2


def straus(points, scalars):
    """sum k_i P_i with shared doublings (affine inputs, Jacobian result)."""
    bits = max(k.bit_length() for k in scalars)
    acc = None
    for b in range(bits - 1, -1, -1):
        if acc is not None:
            acc = acc.dbl()
        for (x, y), k in zip(points, scalars):
            if (k >> b) & 1:
                acc = Jac(x, y, bn.one_like(x)) if acc is None else acc.add_affine(x, y)
    return acc


def batch_affine(pts):
    """Montgomery's trick: one inversion for all points."""
    zs = [p.z for p in pts]
    pref = [zs[0]]
    for z in zs[1:]:
        pref.append(pref[-1] * z)
    inv = pref[-1].inv()
    out = [None] * len(pts)
    for i in range(len(pts) - 1, -1, -1):
        zi = inv * pref[i - 1] if i else inv
        if i:
            inv = inv * zs[i]
        zi2 = zi.sq()
        out[i] = (pts[i].x * zi2, pts[i].y * zi2 * zi)
    return out


# ---- lock-step versions: the same operation applied to every proof of the batch before the next one,
# so independent work sits next to each other in program order (as microcode looping over a batch would)

def ls_pow(xs, e):
    rs = list(xs)
    for bit in bin(e)[3:]:
        rs = [r.sq() for r in rs]
        if bit == "1":
            rs = [r * x for r, x in zip(rs, xs)]
    return rs


def ls_inv(xs):
    """Montgomery's trick over a list (Fp or Fp2): one inversion."""
    pref = [xs[0]]
    for x in xs[1:]:
        pref.append(pref[-1] * x)
    inv = pref[-1].inv()
    out = [None] * len(xs)
    for i in range(len(xs) - 1, -1, -1):
        out[i] = inv * pref[i - 1] if i else inv
        if i:
            inv = inv * xs[i]
    return out


def ls_scalar_mul(pts, ks):
    """Double-and-add for every point at once (affine in, Jacobian out); ks may differ per point."""
    bits = max(k.bit_length() for k in ks)
    acc = [None] * len(pts)
    for b in range(bits - 1, -1, -1):
        acc = [a.dbl() if a is not None else None for a in acc]
        for i, ((x, y), k) in enumerate(zip(pts, ks)):
            if (k >> b) & 1:
                acc[i] = Jac(x, y, bn.one_like(x)) if acc[i] is None else acc[i].add_affine(x, y)
    return acc


def ls_fp2_sqrt(vals):
    """Fp2 square roots for a list (complex method), roots of Fp elements and the final inversion done jointly."""
    half = C((P + 1) // 2)
    alpha = [v.c0.sq() + v.c1.sq() for v in vals]
    gamma = ls_pow(alpha, (P + 1) // 4)
    delta = [(v.c0 + g) * half for v, g in zip(vals, gamma)]
    x0 = ls_pow(delta, (P + 1) // 4)
    redo = [i for i, (x, d) in enumerate(zip(x0, delta)) if not (x.sq() == d)]
    if redo:
        d2 = [(vals[i].c0 - gamma[i]) * half for i in redo]
        for i, r in zip(redo, ls_pow(d2, (P + 1) // 4)):
            x0[i] = r
    inv2x0 = ls_inv([x.dbl() for x in x0])
    return [Fp2(x, v.c1 * t) for x, v, t in zip(x0, vals, inv2x0)]


def ls_recover(proofs):
    """Decompress A, C (G1) and B (G2) of every proof: G1 x's (all A, then all C), their y's, the G2 points."""
    g1x = [fresh(a[0]) for a, _, _ in proofs] + [fresh(c[0]) for _, _, c in proofs]
    bx = [fresh(b[0]) for _, b, _ in proofs]
    g1y = ls_pow([x.sq() * x + bn.B1 for x in g1x], (P + 1) // 4)
    by = ls_fp2_sqrt([x.sq() * x + bn.B2 for x in bx])
    return g1x, g1y, list(zip(bx, by))


def ls_subgroup(bs):
    """G2 subgroup check psi(B) == [6x^2] B for every point."""
    m = ls_scalar_mul(bs, [6 * bn.X * bn.X] * len(bs))
    zi = ls_inv([j.z for j in m])
    for j, z, b in zip(m, zi, bs):
        z2 = z.sq()
        (j.x * z2, j.y * z2 * z), bn._frob_point(b)  # affine [6x^2]B and psi(B), then compared


def ls_decode(proofs):
    """Decompress every proof and check B is in G2: [(A, B, C)] as traced values."""
    g1x, g1y, bs = ls_recover(proofs)
    ls_subgroup(bs)
    n = len(proofs)
    return [((g1x[i], g1y[i]), bs[i], (g1x[n + i], g1y[n + i])) for i in range(n)]


def rand_g1(rng):
    return bn.scalar_mul(bn.G1_GEN, rng.randrange(1, R)).to_affine()


def rand_g2(rng):
    return bn.scalar_mul(bn.G2_GEN, rng.randrange(1, R)).to_affine()


def fresh(v):
    """An input value (no producing operation)."""
    return Fp(v.v) if isinstance(v, Fp) else Fp2(Fp(v.c0.v), Fp(v.c1.v))


def trace_of(fn) -> list:
    bn.TRACE.clear()
    bn.tracing(True)
    fn()
    bn.tracing(False)
    return list(bn.TRACE)


class Setup:
    """n synthetic proofs, verifying key, prepared vk lines and batch weights r_i, made without tracing.
    The tests take the vk multi-scalar products and the final exponentiation as arguments (CP14's variants)."""

    def __init__(self, n: int, seed: int = 598):
        rng = random.Random(seed)
        self.rng = rng
        bn.tracing(False)
        self.proofs = [(rand_g1(rng), rand_g2(rng), rand_g1(rng)) for _ in range(n)]
        self.alpha = rand_g1(rng)
        beta, gamma, delta = (rand_g2(rng) for _ in range(3))
        self.ic = [rand_g1(rng) for _ in range(6)]
        self.publics = [[rng.randrange(R) for _ in range(5)] for _ in range(n)]
        self.lines = {k: bn.prepare(q) for k, q in (("beta", beta), ("gamma", gamma), ("delta", delta))}
        self.r = [rng.getrandbits(128) | 1 for _ in range(n)]

    def pts(self, k):
        """Decoded points of the first k proofs as fresh inputs (read from the 256 B per-proof store)."""
        pt = lambda p: (fresh(p[0]), fresh(p[1]))
        return [(pt(a), pt(b), pt(c)) for a, b, c in self.proofs[:k]]

    def vk_msm(self, scalars):
        """sum_j s_j IC_j (Straus) and s_0 alpha."""
        return straus(self.ic, scalars), bn.scalar_mul(self.alpha, scalars[0])

    def ic_msm(self, x):
        """IC_0 + sum_j x_j IC_(j+1) for one proof's public inputs."""
        return straus(self.ic, [1] + x)

    def weights_tail(self, k, cs, vk_msm=None):
        """sum r_i C_i, public-input weighting, sum r L, (sum r) alpha: three G1 points (Jacobian)."""
        r = self.r[:k]
        sC = straus(cs, r)
        coeff = [sum(ri * x[j] for ri, x in zip(r, self.publics)) % R for j in range(5)]
        for _ in range(k):  # scalar-field weighting of public inputs, counted as base-field work
            for _ in range(5):
                C(self.rng.randrange(P)) * C(self.rng.randrange(P))
        return [sC, *(vk_msm or self.vk_msm)([sum(r) % R] + coeff)]

    def vk_pairs(self, aff):
        """(-sum r C, delta), (-sum r L, gamma), (-(sum r) alpha, beta) from the three affine points."""
        return [((p[0], -p[1]), self.lines[k]) for p, k in zip(aff, ("delta", "gamma", "beta"))]

    def batch_test(self, d, final=bn.final_exp, vk_msm=None):
        """Batch test of the decoded proofs d (== 1 for a valid batch)."""
        k = len(d)
        rA = ls_scalar_mul([a for a, _, _ in d], self.r[:k])
        aff = batch_affine(rA + self.weights_tail(k, [c for _, _, c in d], vk_msm))
        final(bn.miller_loop_otf([(p, b) for p, (_, b, _) in zip(aff[:k], d)], self.vk_pairs(aff[k:])))

    def single_test(self, proof, x, final=bn.final_exp, ic_msm=None):
        """Single verification of a decoded proof (compared with the precomputed e(alpha, beta))."""
        A, B, Cp = proof
        L = batch_affine([(ic_msm or self.ic_msm)(x)])[0]
        pre = [((L[0], -L[1]), self.lines["gamma"]), ((Cp[0], -Cp[1]), self.lines["delta"])]
        final(bn.miller_loop_otf([(A, B)], pre))


def build(n: int, single: bool = False, seed: int = 598) -> dict:
    """The traced batch of n proofs (or one proof verified alone), decoding included."""
    su = Setup(n, seed)
    bn.TRACE.clear()
    bn.tracing(True)
    dec = ls_decode(su.proofs)
    n_decode = len(bn.TRACE)
    if single:
        su.single_test(dec[0], su.publics[0])
    else:
        su.batch_test(dec)
    bn.tracing(False)
    return {"trace": list(bn.TRACE), "n": n, "decode_ops": n_decode}


def level_order(trace):
    """Reorder the trace by dependency depth (stable), the way a compiler would interleave independent work
    (other proofs' decoding, other pairs' lines) before handing a static schedule to the unit. Ids are renumbered."""
    depth = [0] * len(trace)
    for op, out, a, b in trace:
        depth[out] = 1 + max(depth[a] if a >= 0 else 0, depth[b] if b >= 0 else 0)
    order = sorted(range(len(trace)), key=lambda i: (depth[i], i))
    new_id = [0] * len(trace)
    for k, i in enumerate(order):
        new_id[i] = k
    m = lambda x: new_id[x] if x >= 0 else -1
    return [(trace[i][0], new_id[i], m(trace[i][2]), m(trace[i][3])) for i in order]


def schedule(trace, mults: int, adders: int, window: int = 1 << 30) -> dict:
    """Cycle-accurate list scheduling: ready ops issue in program order to free units, but no further than
    `window` ops ahead of the oldest op not yet issued (a static schedule with bounded look-ahead)."""
    nops = len(trace)
    kind = [t[0] == "m" for t in trace]
    deps = [0] * nops
    users = [[] for _ in range(nops)]
    for op, out, a, b in trace:
        for s in {a, b}:
            if s >= 0:
                deps[out] += 1
                users[s].append(out)
    ready_m, ready_a = [], []
    for i in range(nops):
        if deps[i] == 0:
            heapq.heappush(ready_m if kind[i] else ready_a, i)
    done_ev = []  # (time, op)
    mul_free = [0] * mults
    issue_t = [0] * nops
    finish_t = [0] * nops
    t, issued, busy_mul_cycles = 0, 0, 0
    done = bytearray(nops)
    head = 0
    while issued < nops:
        while head < nops and done[head]:
            head += 1
        limit = head + window
        while done_ev and done_ev[0][0] <= t:
            _, op = heapq.heappop(done_ev)
            for u in users[op]:
                deps[u] -= 1
                if deps[u] == 0:
                    heapq.heappush(ready_m if kind[u] else ready_a, u)
        for k in range(mults):
            if mul_free[k] <= t and ready_m and ready_m[0] < limit:
                op = heapq.heappop(ready_m)
                done[op] = 1
                mul_free[k] = t + MUL_OCC
                issue_t[op], finish_t[op] = t, t + MUL_LAT
                heapq.heappush(done_ev, (t + MUL_LAT, op))
                issued += 1
                busy_mul_cycles += MUL_OCC
        for _ in range(adders):
            if not ready_a or ready_a[0] >= limit:
                break
            op = heapq.heappop(ready_a)
            done[op] = 1
            issue_t[op], finish_t[op] = t, t + 1
            heapq.heappush(done_ev, (t + 1, op))
            issued += 1
        # jump to the next moment something can change
        nxt = [done_ev[0][0]] if done_ev else []
        if ready_m and ready_m[0] < limit:
            nxt.append(min(mul_free))
        if ready_a and ready_a[0] < limit:
            nxt.append(t + 1)
        t = max(t + 1, min(nxt)) if nxt else t + 1
    total = max(finish_t) if nops else 0
    # values alive at once: from finish of the producer to issue of its last user
    last_use = [-1] * nops
    for op, out, a, b in trace:
        for s in (a, b):
            if s >= 0:
                last_use[s] = max(last_use[s], issue_t[out])
    ev = []
    for i in range(nops):
        if last_use[i] >= 0:
            ev.append((finish_t[i], 1))
            ev.append((last_use[i] + 1, -1))
    ev.sort()
    live = peak = 0
    for _, d in ev:
        live += d
        peak = max(peak, live)
    n_mul = sum(kind)
    return {"cycles": total, "mul_ops": n_mul, "add_ops": nops - n_mul,
            "mul_util": busy_mul_cycles / (mults * total) if total else 0, "peak_live_values": peak}


def per_proof(traces: dict, mults: int, adders: int, window: int = 1 << 30):
    """Batches of 8 and 4 on one machine: (r8, r4, marginal cycles of one more proof in the batch)."""
    r8 = schedule(traces[8], mults, adders, window)
    r4 = schedule(traces[4], mults, adders, window)
    return r8, r4, (r8["cycles"] - r4["cycles"]) / 4


def energy_mJ(ops: dict) -> float:
    return (ops["mul"] * NJ_MUL * 1e3 + ops["add"] * PJ_ADD + (ops["mul"] + ops["add"]) * PJ_TRAFFIC) * 1e-9


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"machine": {"MHz": MHZ, "mul_occupancy": MUL_OCC, "mul_latency": MUL_LAT, "nJ_mul": NJ_MUL,
                       "pJ_add": PJ_ADD, "pJ_traffic_per_op": PJ_TRAFFIC}}
    t0 = time.time()
    built = {n: build(n) for n in (4, 8)}
    single = build(1, single=True)
    res["trace_s"] = time.time() - t0
    counts = {}
    for key, b in [("batch4", built[4]), ("batch8", built[8]), ("single", single)]:
        m = sum(1 for t in b["trace"] if t[0] == "m")
        counts[key] = {"ops": len(b["trace"]), "mul": m, "add": len(b["trace"]) - m, "decode_ops": b["decode_ops"]}
    marg = {k: (counts["batch8"][k] - counts["batch4"][k]) / 4 for k in ("mul", "add")}
    res["op_counts"] = counts
    res["per_proof_marginal"] = marg
    print("counts", counts, "per proof", marg, flush=True)

    program = {n: built[n]["trace"] for n in (4, 8)}
    mj = energy_mJ(marg)
    sweep = []
    for mults in (1, 2, 4, 5, 6, 8):
        for adders in (1, 2, 3, 4, 6):
            r8, _, cpp = per_proof(program, mults, adders)
            sweep.append({"mults": mults, "adders": adders, "batch8": r8, "cycles_per_proof": cpp,
                          "proofs_per_s": MHZ * 1e6 / cpp, "mJ_per_proof": mj, "R1": mj <= 0.08,
                          "R2": MHZ * 1e6 / cpp >= R2, "buffer_KB": r8["peak_live_values"] * 32 / 1024})
            print(mults, adders, round(cpp), round(MHZ * 1e6 / cpp), f"util {r8['mul_util']:.2f}",
                  f"live {r8['peak_live_values']}", flush=True)
    res["sweep"] = sweep
    win = []
    for order, tr in (("program", program), ("level", {n: level_order(program[n]) for n in (4, 8)})):
        for mults, adders in ((4, 4), (5, 4), (6, 4), (6, 6)):
            for w in (32, 64, 128, 256, 1024, 4096):
                r8, _, cpp = per_proof(tr, mults, adders, w)
                win.append({"order": order, "mults": mults, "adders": adders, "window": w, "cycles_per_proof": cpp,
                            "proofs_per_s": MHZ * 1e6 / cpp, "mul_util": r8["mul_util"],
                            "peak_live_values": r8["peak_live_values"], "buffer_KB": r8["peak_live_values"] * 32 / 1024})
                print("window", order, mults, adders, w, round(MHZ * 1e6 / cpp), f"util {r8['mul_util']:.2f}",
                      f"live {r8['peak_live_values']}", flush=True)
    res["window_sweep"] = win
    s1 = schedule(single["trace"], 4, 2)
    res["single_proof_M4_A2"] = {**s1, "us": s1["cycles"] / MHZ}
    (OUT / "cp11_sim.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
