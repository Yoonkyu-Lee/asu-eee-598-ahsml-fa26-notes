"""Rebuild chip_table.json (the chip's time per text length and per picture) from the encoder model.

  .venv/Scripts/python.exe chip/model/build_table.py           # writes chip/model/chip_table.json
  .venv/Scripts/python.exe chip/model/build_table.py --check   # only compares with the stored table

encoder_size.Encoder over the INT8 Timeloop mappings and part costs in results/chip/, per text length L = 1..256 and
one 50-patch picture; FP16 text = that x the CP34 factor per length (direct 16-bit mappings, encoder_resize_w16.py).
Through CP33 the factor was CP28's flat x4.0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import encoder_size as m18  # noqa: E402
import encoder_resize_w16 as rz  # noqa: E402
from chip.model.chip import SPEC, TABLE  # noqa: E402


def enc_table(R: int, glb: int):
    parts = json.loads((m18.RES / "cp12_parts.json").read_text(encoding="utf-8"))
    enc = m18.Encoder({"minilm": m18.load(f"minilm_glb{glb}"), "clipv": m18.load(f"clipv_glb{glb}")}, parts, R, glb)
    text = [enc.run(m18.MINILM, np.array([L])) for L in range(1, 257)]
    img = enc.run(m18.CLIPV, np.array([50]))
    return text, img


def build() -> dict:
    t16, i16 = enc_table(16, 2048)
    t8, i8 = enc_table(8, 1024)
    f = [rz.fac(16, L, "w16") for L in range(1, 257)]
    return {"text_fp16_16x16_2MB": [[s * a, J * b] for (s, J), (a, b) in zip(t16, f)],
            "text_int8_8x8_1MB": [list(r) for r in t8],
            "img_16x16": list(i16), "img_8x8": list(i8),
            "note": "seconds and joules per item on the chip (estimates). Built by build_table.py from results/chip."}


if __name__ == "__main__":
    new = build()
    if "--check" in sys.argv:
        old = json.loads(TABLE.read_text(encoding="utf-8"))
        same = all(np.allclose(np.array(old[k], dtype=float), np.array(new[k], dtype=float), rtol=1e-12, atol=0)
                   for k in ("text_fp16_16x16_2MB", "text_int8_8x8_1MB", "img_16x16", "img_8x8"))
        print("same as the stored table:", same)
        sys.exit(0 if same else 1)
    TABLE.write_text(json.dumps(new), encoding="utf-8")
    print("wrote", TABLE)
