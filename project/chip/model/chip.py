"""The latest chip: the one place its numbers come from (CP31).

The node box (box/box_server.py, the virtual accelerator) asks this module how long the chip would take. Experiments
print the summary from here. Every number is an estimate (추정, 45 nm models) unless marked; each says where it comes
from. The full story is in docs/chip.md.

  .venv/Scripts/python.exe chip/model/chip.py          # prints the summary and checks the table
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "chip_table.json"          # made by build_table.py from the encoder model (results/chip/*)
RES = HERE.parents[1] / "results" / "chip"

# The computed numbers come from the result files (CLAUDE.md 6: JSON -> chip.py -> docs; tests/fast/check.py)
_C34 = json.loads((RES / "cp34_chip.json").read_text(encoding="utf-8"))
_AREA = _C34["area"]["16x16 2MB w16 (FP16 MAC)"]
_POWER = _C34["power"]["w16 16x16 2MB"]
_WORST = next(m["worst"]["proofs_per_s"] for m in
              json.loads((RES / "cp14_fixed.json").read_text(encoding="utf-8"))["machines"]["V4 +O4 (w8)"]
              if (m["M"], m["A"]) == (6, 6))

# ---------------------------------------------------------------------------------------------------- the chip
SPEC = {
    "process": "45 nm (FreePDK45 synthesis for the multiplier; Timeloop/Accelergy 45 nm for the encoder)",
    "verifier": {
        "what": "Groth16/BN254 proof verification unit: 254-bit Montgomery multipliers + adders, batch check",
        "multipliers": 6, "adders": 6,
        "area_mm2": round(_AREA["chip_mm2"] - _AREA["encoder_mm2"], 2), "min_proofs_per_s": int(_WORST),
        "single_proof_s": 89.6e-6,
        "source": "CP14 (verifier_fixed.py: re-sized after CP13's invalid-proof flood); multiplier from CP10 "
                  "synthesis of chip/rtl/mont_mul_pipe.sv at S = 1 (multiplier.py: 0.28 mm2, 300 MHz, 1.76 nJ, "
                  "measured synthesis); single-proof time CP19",
    },
    "encoder": {
        "what": "transformer encoder array (MiniLM text, CLIP image) with a global SRAM",
        "array": "16x16", "global_sram_MB": 2, "text_precision": "FP16 (the Recall@10 >= 0.95 contract, CP3/CP27)",
        "source": "CP34: direct 16-bit mappings (timeloop_fp16.py) of 8x8 and 16x16 with 2 MB, sized by CP28's "
                  "rules (encoder_resize_w16.py). 8x8 passes if the FP16 MAC issues one op per cycle but fails at half "
                  "speed, so by the rule written in advance 16x16 is kept",
        "fp16_text_factor": "per text length, encoder_resize_w16.fac(16, L, 'w16'): time x1.12-1.36, energy "
                            "x4.6-6.3 over the INT8 mapping (CP34; CP28 had assumed x4.0 for both)",
    },
    "area_mm2": round(_AREA["chip_mm2"], 1),
    "encoder_mm2": round(_AREA["encoder_mm2"], 2),
    "area_source": "CP34 chip_w16.py (results/chip/cp34_chip.json): encoder (CP12 roll-up on the 16-bit mapping, 2 MB "
                   "SRAM 6.80, FP16 MACs 0.94) + verification unit. The INT8 16x16 roll-up gives 9.62, matching CP28's "
                   "lower bound",
    "power": {"ordinary_day_mW": round(_POWER["S1_chip_mW"], 1), "flood_W": round(_POWER["S4_chip_W"], 2),
              "source": "CP34 chip_w16.py: CP19's box model (S1 ordinary day, S4 invalid-proof flood) with the 16-bit "
                        "text times and energies; load = CP16's top-1 % box receiving every sub (P-all). The ordinary "
                        "day is an upper bound: since CP33 a box receives only its subs (0.25-0.51x the verifications "
                        "in the sync model); the flood is set by the link and does not change"},
    "image_s": 0.552,
    "image_source": "CP20 FP16 CLIP image tower on the 8x8 array, as the demo box has used it since CP20 (open: "
                    "not re-estimated for 16x16, docs/chip.md)",
}
PROOF_S = SPEC["verifier"]["single_proof_s"]
IMAGE_S = SPEC["image_s"]


class TextTable:
    """Chip time and energy for one text of L tokens (L = 1..256): (seconds, joules)."""

    def __init__(self, rows: list):
        self.rows = rows

    def text1(self, L: int) -> tuple[float, float]:
        s, J = self.rows[min(max(int(L), 1), 256) - 1]
        return s, J


def text_table(name: str = "fp16_16x16") -> TextTable:
    """fp16_16x16: the latest chip at contract precision (CP28 size, CP34 times). int8_8x8: the CP18-CP27 chip (INT8 text; breaks
    the Recall contract, kept for the --text int8 boxes)."""
    t = json.loads(TABLE.read_text(encoding="utf-8"))
    return TextTable(t[{"fp16_16x16": "text_fp16_16x16_2MB", "int8_8x8": "text_int8_8x8_1MB"}[name]])


def summary() -> str:
    v, e = SPEC["verifier"], SPEC["encoder"]
    t = text_table()
    lines = [
        f"verification unit: {v['multipliers']} multipliers + {v['adders']} adders, {v['area_mm2']} mm2, "
        f">= {v['min_proofs_per_s']:,} proofs/s, one proof {v['single_proof_s'] * 1e6:.1f} us   [{v['source']}]",
        f"encoder: {e['array']}, {e['global_sram_MB']} MB, text {e['text_precision']}   [{e['source']}]",
        "  text on the chip (s): " + ", ".join(f"{L} tokens {t.text1(L)[0] * 1e3:.1f} ms" for L in (16, 64, 128, 256)),
        f"  one picture: {IMAGE_S} s   [{SPEC['image_source']}]",
        f"chip area: {SPEC['area_mm2']} mm2   [{SPEC['area_source']}]",
        f"power: ordinary day {SPEC['power']['ordinary_day_mW']} mW, flood {SPEC['power']['flood_W']} W   "
        f"[{SPEC['power']['source']}]",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
