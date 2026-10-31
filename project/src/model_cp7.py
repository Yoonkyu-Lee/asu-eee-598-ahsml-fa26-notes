"""CP7 cost model: a proof-verification unit (BN254 pairings and G1 arithmetic) for the node box.

Run from project/:  .venv/Scripts/python.exe src/model_cp7.py
Reads results/cp6/*.json and results/cp7/cp7_cpu.json, writes results/cp7/cp7_model.json.
Every hardware number is an estimate (추정) built from
  - operation counts of an optimal ate pairing on a 254-bit BN curve (출처: herumi/ate-pairing README, which
    builds on Aranha et al., EUROCRYPT 2011): Miller loop 6785 mu + 3022 r, final exponentiation 3526 mu + 1932 r,
    where mu = 256x256-bit multiply without reduction and r = Montgomery reduction 512 -> 256 bits;
  - G1 formulas from the Explicit-Formulas Database: Jacobian doubling (a = 0) 2M + 5S, mixed addition 7M + 4S;
  - per-operation energies from lecture L7 (Horowitz, 45 nm), as in CP4: 32-bit multiply 3.1 pJ, 32-bit add 0.1 pJ,
    8 KB SRAM 32-bit read 5 pJ.
Dynamic energy only, like CP4.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# operation counts (출처, see docstring); BN254 (the SNARK curve) has a similar loop length, so +-20% (추정)
ML = {"mu": 6785, "r": 3022}
FE = {"mu": 3526, "r": 1932}
DBL, MADD = 2 + 5, 7 + 4  # Fp multiplications (squares counted as multiplications)
LIMBS = 8                  # 254-bit operands as 8 x 32-bit words

# pJ, lecture L7 (Horowitz, 45 nm)
MUL32, ADD32, SRAM_WORD = 3.1, 0.1, 5.0
E_MU = LIMBS * LIMBS * MUL32 + 2 * LIMBS * LIMBS * ADD32            # schoolbook 8x8 words, carry-save adds
E_R = (LIMBS * LIMBS + LIMBS) * MUL32 + 2 * (LIMBS * LIMBS + LIMBS) * ADD32  # Montgomery reduction (CIOS-like)
ADDS_PER_MUL = 2           # (가정) Fp additions/subtractions per multiplication in pairing code
E_ADD = 2 * LIMBS * ADD32  # 254-bit add with conditional subtract
WORDS_PER_OP = 3 * LIMBS   # two operands in, one result out


def scalar_mul(bits: int, w: int = 4) -> float:
    """Fp multiplications of a G1 scalar multiplication, width-w NAF."""
    return bits * DBL + bits / (w + 1) * MADD


def straus(n: int, bits: int = 254, w: int = 4) -> float:
    """n-point multi-scalar multiplication with shared doublings (small n, as for public inputs)."""
    return bits * DBL + n * bits / (w + 1) * MADD


def tally(ml: float, fe: float, g1_mults: float) -> dict:
    mu = ml * ML["mu"] + fe * FE["mu"] + g1_mults
    r = ml * ML["r"] + fe * FE["r"] + g1_mults
    mults = (mu + r) / 2  # full Fp-multiplication equivalents (one Fp mul = one mu + one r; E_mu ~ E_r)
    compute = (mu * E_MU + r * E_R + ADDS_PER_MUL * mults * E_ADD) * 1e-12
    traffic = (1 + ADDS_PER_MUL) * mults * WORDS_PER_OP * SRAM_WORD * 1e-12
    return {"fp_mul_equiv": mults, "uJ_compute_only": compute * 1e6, "uJ_with_8KB_sram_traffic": (compute + traffic) * 1e6}


def main() -> None:
    cpu = json.loads((RES / "cp7" / "cp7_cpu.json").read_text())
    cp6 = json.loads((RES / "cp6" / "cp6_crypto.json").read_text())
    box = json.loads((RES / "cp6" / "cp6_box.json").read_text(encoding="utf-8"))
    out: dict = {"pJ": {"E_mu": E_MU, "E_r": E_R, "E_add254": E_ADD, "sram_word": SRAM_WORD}}

    # Groth16 verification: e(A,B) e(L,-gamma) e(C,-delta) == e(alpha,beta) (precomputed)
    single = tally(ml=3, fe=1, g1_mults=straus(5))
    # batch of n: per proof one Miller loop + r_i A_i (128-bit weights) + share of MSM(C) + (2 ML + FE)/n
    n = 64
    batch = tally(ml=1 + 2 / n, fe=1 / n, g1_mults=scalar_mul(128) + scalar_mul(128) / 4 + straus(5) / n)
    out["per_verify"] = {"single": single, "batch64_per_proof": batch}

    # Check the counts against the CPU: CPU time / Fq-mul time = Fq-mul equivalents the CPU actually spends.
    p = cpu["prims"]["asm_native/p_core"]
    out["cpu_check"] = {
        "miller_loop_fq_mul_equiv": p["miller_loop_1_us"] * 1e3 / p["fq_mul_ns"],
        "final_exp_fq_mul_equiv": p["final_exp_us"] * 1e3 / p["fq_mul_ns"],
        "model_miller_loop": [(ML["mu"] + ML["r"]) / 2, ML["mu"] + ML["r"]],  # [equivalents, every mu and r as a full mul]
        "model_final_exp": [(FE["mu"] + FE["r"]) / 2, FE["mu"] + FE["r"]],
    }

    # Energy per proof: measured CPU vs the unit.
    e6, e7 = cp6["energy"], cpu["energy"]
    cpu_mj = {
        "P-core single (측정)": (e6["p_core/groth16"]["J_per_op"], e6["p_core/groth16"]["active_J_per_op"]),
        "LP-E single (측정)": (e6["lp_e_core/groth16"]["J_per_op"], e6["lp_e_core/groth16"]["active_J_per_op"]),
        "P-core batch 64 (측정)": (e7["p_core/groth16batch"]["J_per_op"], e7["p_core/groth16batch"]["active_J_per_op"]),
        "LP-E batch 64 (측정)": (e7["lp_e_core/groth16batch"]["J_per_op"], e7["lp_e_core/groth16batch"]["active_J_per_op"]),
    }
    out["cpu_mJ_per_proof"] = {k: {"pkg": a * 1e3, "active": b * 1e3} for k, (a, b) in cpu_mj.items()}
    unit = {"unit single": single["uJ_with_8KB_sram_traffic"] / 1e3, "unit batch 64": batch["uJ_with_8KB_sram_traffic"] / 1e3}
    out["unit_mJ_per_proof"] = unit

    # Flood sizing: Fp multiplications per second, units needed at 500 MHz (one Fp mul per cycle, pipelined).
    flood = {}
    for link, v in box["spam_flood"].items():
        msgs = v["msgs_per_s"]
        rows = {}
        for name, per in (("single", single), ("batch64", batch)):
            rate = msgs * per["fp_mul_equiv"]
            rows[name] = {"fp_mul_per_s": rate, "units_at_500MHz": rate / 500e6,
                          "W": msgs * per["uJ_with_8KB_sram_traffic"] * 1e-6}
        flood[link] = {"msgs_per_s": msgs, "budget_mJ_per_proof_2W": v["mJ_per_proof_for_2W"], "unit": rows}
    out["flood"] = flood
    (RES / "cp7" / "cp7_model.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
