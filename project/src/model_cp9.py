"""CP9: turn the synthesis results of rtl/mont_mul.sv into the CP8 decision quantities.

Run from project/:  .venv/Scripts/python.exe src/model_cp9.py
Reads results/cp9/cp9_synth.json (numbers transcribed from the Apporto reports: Design Compiler 2023.12,
FreePDK45 OSU gscl45nm + DesignWare, power from VCS switching activity (VCD -> SAIF) of 1,000 reference
multiplications) and results/cp8/cp8_alternatives.json. Writes results/cp9/cp9_model.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def evaluate(syn: dict, cp8: dict) -> dict:
    """R1/R2 quantities for every finished synthesis run in `syn` (schema of cp9_synth.json)."""
    rules = cp8["rules"]
    work = cp8["work_per_proof"]
    fp_mul = work["total_fp_mul"]
    traffic_nj = work["nJ_per_fp_mul_incl_traffic"] - 0.45  # CP7's operand traffic per multiply, kept as is
    need_rate = rules["R2_proofs_per_s"] * fp_mul       # Fp multiplications per second for R2
    out = {"need_fp_mul_per_s": need_rate, "units": {}}
    for name, s in syn["runs"].items():
        if s.get("pending"):
            continue
        f = 1e3 / s["fmax_path_ns"]                       # MHz, from the critical path of the run
        cyc = s["W_cycles_per_mul"]                       # K + 2 (load, K rows, final subtraction)
        rate = f * 1e6 / cyc
        e_nj = s["power_mW"] * s["tb_cycles_per_mul"] * s["sim_period_ns"] / 1e3  # mW x ns = pJ; per multiply
        copies = math.ceil(need_rate / rate)
        mj = fp_mul * (e_nj + traffic_nj) * 1e-6
        out["units"][name] = {
            "W": s["W"], "area_mm2": s["area_um2"] / 1e6, "fmax_MHz": f, "cycles_per_mul": cyc,
            "fp_mul_per_s": rate, "nJ_per_mul": e_nj, "proofs_per_s_one_unit": rate / fp_mul,
            "copies_for_R2": copies, "area_for_R2_mm2": copies * s["area_um2"] / 1e6,
            "mJ_per_proof": mj, "R1": mj <= rules["R1_mJ_per_proof"],
            "W_at_flood": rules["flood_per_s"] * mj * 1e-3,
            "vs_cp7_estimate": e_nj / 0.45}
    return out


def main() -> None:
    syn = json.loads((RES / "cp9" / "cp9_synth.json").read_text(encoding="utf-8"))
    cp8 = json.loads((RES / "cp8" / "cp8_alternatives.json").read_text(encoding="utf-8"))
    out = evaluate(syn, cp8)
    (RES / "cp9" / "cp9_model.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
