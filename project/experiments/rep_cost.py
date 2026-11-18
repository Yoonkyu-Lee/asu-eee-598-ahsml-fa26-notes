"""CP37 (1) gauges from the measured REP-N bench (journal/cp/CP37.md, rules written before the run).

  .venv/Scripts/python.exe -W ignore experiments/rep_cost.py   -> results/chip/cp37_rep_gauges.json

Inputs: results/chip/cp37_rep.json (measured on this PC), results/neighborhoods/b_scale*.json (CP39 B1 scenes),
chip/model/chip.py. G1 author time on a Pi 5 (by ratio, estimate), G2 per-box verification share (estimate),
G3 chip time for the proof's G1 MSMs (Pippenger count x the CP10 multiplier, estimate; the Fr NTT stays on the host).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chip.model.chip import SPEC  # noqa: E402

PI5_SLOW = 2.93                  # CP34: Pi 5 core = laptop P-core x 2.93 (Geekbench 6)
K1_S = 5.0                       # new post visible within 5 s (NORTHSTAR 6)
MUL_HZ, MUL_CYCLES = 300e6, 4    # CP10 pipelined multiplier (chip.py source text)
MIXED_ADD_M, FULL_ADD_M = 11, 16  # Fq multiplications per mixed / full Jacobian addition (estimate)
SCALAR_BITS = 254
FQ2_MUL_IN_FQ = 3                # Karatsuba: one Fq2 multiplication = 3 Fq multiplications (estimate)


def pippenger_mults(n: int) -> float:
    """Fq multiplications for one G1 MSM of size n (estimate): windows of c bits, n mixed adds per window,
    2^(c+1) full adds for the bucket sums, c doublings per window (doubling ~ 7M)."""
    best = math.inf
    for c in range(4, 22):
        w = math.ceil(SCALAR_BITS / c)
        best = min(best, w * (n * MIXED_ADD_M + 2 ** (c + 1) * FULL_ADD_M + c * 7))
    return best


def main():
    rep = json.loads((ROOT / "results" / "chip" / "cp37_rep.json").read_text(encoding="utf-8"))
    q = rep["chain_2_16_prove_ms"] / rep["cp6_laptop_chain_2_16_prove_ms"]       # this PC / laptop
    units = SPEC["verifier"]["multipliers"]
    mul_per_s = MUL_HZ / MUL_CYCLES * units
    out = {"what": "CP39/CP37 (1) gauges (journal/cp/CP37.md)", "label": "estimate from measured bench",
           "this_pc_over_laptop": q, "pi5_slowdown_vs_laptop": PI5_SLOW, "k1_s": K1_S,
           "chip_mul_per_s": mul_per_s, "rows": [], "g2": {}}
    for r in rep["rows"]:
        pi5_1 = r["prove_us"] / 1e6 / q * PI5_SLOW
        m = r["witnesses"] + r["public_inputs"] + 1
        dom = 1 << math.ceil(math.log2(r["constraints"] + r["public_inputs"]))
        g1_sizes = [m, m, r["witnesses"], dom]                      # A, B in G1, C (L query), H query
        mults = sum(pippenger_mults(s) for s in g1_sizes)
        sp = r["split_s"]
        host_rest = max(r["prove_us"] / 1e6 - sp["g1_msms_four"], 0.0)      # what stays on the host with the chip
        out["rows"].append({"n": r["n"], "constraints": r["constraints"], "prove_s_this_pc": r["prove_us"] / 1e6,
                            "split_this_pc_s": sp, "g1_msm_share_of_prove": sp["g1_msms_four"] / (r["prove_us"] / 1e6),
                            "pi5_with_chip_1core_s": host_rest / q * PI5_SLOW + mults / mul_per_s,
                            "pi5_with_chip_4core_ideal_s": host_rest / q * PI5_SLOW / 4 + mults / mul_per_s,
                            # added after the results (marked in CP37.md): the G2 MSM is Fq2 arithmetic, which the
                            # verifier's Fq multipliers already do for pairings; only the Fr FFTs stay on the host
                            "after_results_chip_g2_msm_s": FQ2_MUL_IN_FQ * pippenger_mults(m) / mul_per_s,
                            "after_results_pi5_with_chip_g1_g2_4core_ideal_s":
                                sp["fft_seven"] / q * PI5_SLOW / 4 + (mults + FQ2_MUL_IN_FQ * pippenger_mults(m)) / mul_per_s,
                            "after_results_pi5_with_chip_g1_g2_1core_s":
                                sp["fft_seven"] / q * PI5_SLOW + (mults + FQ2_MUL_IN_FQ * pippenger_mults(m)) / mul_per_s,
                            "pi5_1core_s": pi5_1, "pi5_4core_ideal_s": pi5_1 / 4,
                            "x_k1_1core": pi5_1 / K1_S, "x_k1_4core": pi5_1 / 4 / K1_S,
                            "verify_ms_this_pc": r["verify_us"] / 1e3, "verify_ms_pi5": r["verify_us"] / 1e3 / q * PI5_SLOW,
                            "proof_plus_publics_bytes": r["proof_bytes"] + r["publics_bytes"], "pk_MB": r["pk_bytes"] / 1e6,
                            "g1_msm_sizes": g1_sizes, "g1_msm_fq_mults": mults, "chip_g1_msm_s": mults / mul_per_s,
                            "note": "G2 MSM (B) and the Fr NTTs are not on the chip (G2: Fq2; NTT: Fr, CP37 table)"})
    v_pi5 = {r["n"]: r["verify_ms_pi5"] / 1e3 for r in out["rows"]}
    for band, suf in (("base", ""), ("high", "_high")):
        b = json.loads((ROOT / "results" / "neighborhoods" / f"b_scale{suf}.json").read_text(encoding="utf-8"))
        peak = b["params"]["peak_factor"]
        for users, s in b["scales"].items():
            texts = (s["posts_per_s_mean"] + s["comments_per_s_mean"]) * peak
            out["g2"][f"{band}_{users}"] = {"texts_per_s_peak": texts,
                                            "pi5_core_share_by_n": {n: texts * v for n, v in v_pi5.items()},
                                            "chip_share": texts / SPEC["verifier"]["min_proofs_per_s"]}
    spam = json.loads((ROOT / "results" / "neighborhoods" / "spam_all.json").read_text(encoding="utf-8"))
    out["g2_spam"] = [{"users": x["users"], "K": x["K"], "identities": x["identities"], "msgs_per_s": x["verify_per_s"],
                       "pi5_core_share_rep_verify": x["verify_per_s"] * v_pi5[10]} for x in spam["rows"]]
    (ROOT / "results" / "chip" / "cp37_rep_gauges.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
