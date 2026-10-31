"""CP10: the CP9 decision quantities for the interleaved multiplier (rtl/mont_mul_pipe.sv), S = 1..4.

Run from project/:  .venv/Scripts/python.exe src/model_cp10.py
Reads results/cp10/cp10_synth.json (same schema as CP9), writes results/cp10/cp10_model.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cp9 import RES, evaluate  # noqa: E402


def main() -> None:
    syn = json.loads((RES / "cp10" / "cp10_synth.json").read_text(encoding="utf-8"))
    cp8 = json.loads((RES / "cp8" / "cp8_alternatives.json").read_text(encoding="utf-8"))
    out = evaluate(syn, cp8)
    base = json.loads((RES / "cp9" / "cp9_model.json").read_text(encoding="utf-8"))["units"]["W64"]
    for u in out["units"].values():
        u["area_vs_cp9_W64x6"] = u["area_for_R2_mm2"] / base["area_for_R2_mm2"]
        u["clock_vs_cp9_W64"] = u["fmax_MHz"] / base["fmax_MHz"]
    (RES / "cp10" / "cp10_model.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, u in out["units"].items():
        print(f"{k}: {u['area_mm2']:.3f} mm2, {u['fmax_MHz']:.0f} MHz ({u['clock_vs_cp9_W64']:.2f}x), "
              f"{u['nJ_per_mul']:.2f} nJ, x{u['copies_for_R2']} -> {u['area_for_R2_mm2']:.3f} mm2 "
              f"({u['area_vs_cp9_W64x6']:.2f}x), {u['mJ_per_proof']:.4f} mJ/proof, {u['W_at_flood']:.2f} W")


if __name__ == "__main__":
    main()
