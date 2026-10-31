"""CP8 model: where should the proof-verification unit live? Compare alternatives A-D under the rules
pre-registered in cp/CP8.md (R1 energy, R2 throughput, R3 no interference; then smallest multiplier area).

Run from project/:  .venv/Scripts/python.exe src/model_cp8.py
Reads results/cp7/cp7_model.json and results/cp8/prims_*.json, writes results/cp8/cp8_alternatives.json.
All outputs are estimates (추정) on the same basis as CP4/CP7 (lecture L7, 45 nm, dynamic energy only).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FLOOD = 12_207            # proofs/s at 100 Mbps, 1 KB messages (CP6 assumption)
R1_MJ, R2_PER_S = 0.08, 2 * FLOOD
ADDS_PER_MUL = 2          # as in CP7 (가정)
INSTR_PJ = 70             # lecture L7: energy of one simple-core instruction beyond its arithmetic (45 nm)

# Proof decoding with checks (estimate, Fp multiplications), cross-checked against the CPU below:
#   G1 decompression (A, C): sqrt in Fp = exponentiation by (p+1)/4, ~254 S + ~127 M each
#   G2 decompression (B): sqrt in Fp2, about two Fp exponentiations plus Fp2 overhead
#   G2 subgroup check (B): endomorphism method, a ~64-bit G2 scalar multiplication (Fp2 ops = 3 Fp mults)
G1_SQRT = 254 + 127
DECODE = {"g1_decompress_x2": 2 * G1_SQRT, "g2_decompress": 2 * G1_SQRT + 200,
          "g2_subgroup": 64 * 7 * 3 + 64 / 3 * 11 * 3}


def main() -> None:
    cp7 = json.loads((RES / "cp7" / "cp7_model.json").read_text(encoding="utf-8"))
    prims = {p.stem.replace("prims_", ""): json.loads(p.read_text()) for p in (RES / "cp8").glob("prims_*.json")}
    out: dict = {"rules": {"R1_mJ_per_proof": R1_MJ, "R2_proofs_per_s": R2_PER_S, "flood_per_s": FLOOD}}

    # 1. Per-proof work, batch-64 verification + checked decoding.
    batch = cp7["per_verify"]["batch64_per_proof"]
    decode = sum(DECODE.values())
    fp_mul = batch["fp_mul_equiv"] + decode
    nj_per_mul = batch["uJ_with_8KB_sram_traffic"] * 1e3 / batch["fp_mul_equiv"]  # CP7 energy per Fp mul incl. traffic
    p = prims["target-fast_1"]
    out["work_per_proof"] = {
        "verify_batch64_fp_mul": batch["fp_mul_equiv"], "decode_checks_fp_mul": decode, "decode_parts": DECODE,
        "total_fp_mul": fp_mul, "nJ_per_fp_mul_incl_traffic": nj_per_mul,
        "cpu_check": {  # measured on a P-core (asm build): decoding share of the batch per-proof time
            "decode_checked_us": p["proof_decode_checked_us"], "decode_unchecked_us": p["proof_decode_unchecked_us"],
            "batch_verify_us": p["groth16_batch64_per_proof_us"],
            "decode_share_of_total": p["proof_decode_checked_us"] / (p["proof_decode_checked_us"] + p["groth16_batch64_per_proof_us"]),
            "model_decode_share": decode / fp_mul}}

    alts = {}
    # A_w: dedicated unit, one w x 256-bit row of the product and of the Montgomery reduction per cycle,
    # pipelined over independent multiplications (pairing and batch give plenty), two 254-bit adders.
    for w, ghz in ((64, 0.5), (128, 0.5), (256, 0.5), (128, 1.0)):
        cyc = max(256 / w, ADDS_PER_MUL / 2)
        rate = ghz * 1e9 / (fp_mul * cyc)
        alts[f"A{w}@{ghz}GHz"] = {
            "what": f"dedicated Fp unit, {w}-bit rows", "mult32_equiv": 2 * (w // 32) * 8, "cycles_per_fp_mul": cyc,
            "proofs_per_s": rate, "mJ_per_proof": fp_mul * nj_per_mul * 1e-6, "R3": True,
            "note": "1 GHz on a 254-bit datapath at 45 nm is unproven: CP9 synthesis decides" if ghz >= 1 else ""}
    # B: borrow the encoder's 16x16 INT8 weight-stationary array (CP4). A 256x256-bit product is a 32x63 Toeplitz
    # matmul in 8-bit limbs = 2 x 4 tiles; each tile costs (M + 16) cycles with M = products sharing an operand.
    # Variable x variable products share nothing (M = 1); the two reduction products use constants (amortised).
    tiles = math.ceil(32 / 16) * math.ceil(63 / 16)
    cyc_b = tiles * (1 + 16) + 2 * tiles * 1.25
    e_b = (2 * 1024 * (0.2 + 0.1 + 0.5) + 3 * tiles * 16 * 16 * 0.5) * 1e-3  # nJ: INT8 MAC + INT32 add + PE reg; tile loads
    alts["B int8 array"] = {
        "what": "reuse the encoder's 16x16 INT8 array (8-bit limbs)", "mult32_equiv": 0, "cycles_per_fp_mul": cyc_b,
        "proofs_per_s": 1e9 / (fp_mul * cyc_b), "mJ_per_proof": fp_mul * (e_b + nj_per_mul - 0.45) * 1e-6,
        "R3": False, "note": "needs integer carry handling the FP16 vector unit lacks; occupies the encoder during a flood"}
    # C: host in-order core + a 254-bit multiply instruction backed by a 64-bit-row datapath; adds and loads stay
    # instructions (1 mul + 2 add + 2 memory ops per multiplication), IPC 1 at 1 GHz.
    instr = fp_mul * (1 + ADDS_PER_MUL) * (1 + 2 / 3)
    cyc_c = instr / fp_mul + 256 / 64 - 1
    alts["C cpu + bigint instr"] = {
        "what": "host core + 254-bit multiply instruction (64-bit rows)", "mult32_equiv": 2 * 2 * 8,
        "cycles_per_fp_mul": cyc_c, "proofs_per_s": 1e9 / (fp_mul * cyc_c),
        "mJ_per_proof": (fp_mul * nj_per_mul + instr * INSTR_PJ * 1e-3) * 1e-6,
        "R3": False, "note": "keeps the host core fully busy during a flood"}
    # D: software only, measured (CP7, batch on one low-power core).
    lpe = cp7["cpu_mJ_per_proof"]["LP-E batch 64 (측정)"]
    lp = prims["target-fast_1048576"]
    alts["D software (LP-E core)"] = {
        "what": "batch verification on one low-power E-core (measured)", "mult32_equiv": 0,
        "proofs_per_s": 1e6 / (lp["groth16_batch64_per_proof_us"] + lp["proof_decode_checked_us"]),
        "mJ_per_proof": lpe["active"], "R3": False, "note": "measured active energy, decoding not included"}

    for a in alts.values():
        a["R1"] = a["mJ_per_proof"] <= R1_MJ
        a["R2"] = a["proofs_per_s"] >= R2_PER_S
        a["pass"] = a["R1"] and a["R2"] and a["R3"]
    passing = sorted((k for k, a in alts.items() if a["pass"]), key=lambda k: (alts[k]["mult32_equiv"], alts[k]["mJ_per_proof"]))
    out["alternatives"] = alts
    out["ranking"] = passing
    (RES / "cp8" / "cp8_alternatives.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, a in alts.items():
        print(f"{k:24s} R1={a['R1']!s:5} R2={a['R2']!s:5} R3={a['R3']!s:5} {a['proofs_per_s']:9.0f}/s {a['mJ_per_proof']:.4f} mJ mult32={a['mult32_equiv']}")
    print("ranking", passing)
    print(json.dumps(out["work_per_proof"], indent=1))


if __name__ == "__main__":
    main()
