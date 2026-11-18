"""CP34: the whole chip with FP16 text from the direct 16-bit mappings: encoder area, chip area, chip power.

  .venv/Scripts/python.exe -W ignore chip/model/chip_w16.py <cp30 worktree>/project   -> results/chip/cp34_chip.json

Area: rollup.Encoder (CP12's roll-up: Timeloop/Accelergy component areas + FP16 attention MACs + vector lanes) on the
16-bit mapping's components, with the 16-bit integer MAC replaced by an FP16 MAC (area x 1.13 = Aladdin 45 nm, 2 ns,
FP32 (mult + add) / INT32 (mult + add)); plus the verify unit (CP14, 6 multipliers + 6 adders, 2.12 mm2).
Power: CP19's box model at tag cp30 (S1 ordinary day; S4 invalid-proof flood, proof check first, the CP19 picks),
with text time and energy from encoder_resize_w16 (mode w16; flat4 = CP28 for reference). Every number is 추정.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import encoder_resize_w16 as rz  # noqa: E402

RES = rz.RES
VERIFY_MM2 = 2.12                                     # CP14 (results/chip/cp14_fixed.json), estimate
MAC_AREA_FP_OVER_INT = (3.01e3 + 3.01e3) / (5.14e3 + 1.79e2)   # Aladdin 45 nm, 2 ns: FP32 (mult+add) / INT32


def encoder_area(R: int, w16: bool) -> dict:
    from chip.model import rollup
    parts = json.loads((RES / "cp12_parts.json").read_text())
    if w16:
        f = "cp34_timeloop_w16.json" if R == 16 else f"cp34_timeloop_w16_R{R}.json"
        d = json.loads((RES / f).read_text())
        tl = {"components": {str(R): d["components"]}, "runs": d["runs"]}
    else:
        tl = json.loads((RES / "cp12_timeloop.json").read_text())
    enc = rollup.Encoder(tl, parts, R, 0.5)
    a = dict(enc.area_mm2)
    if w16:
        a["mac"] *= MAC_AREA_FP_OVER_INT
    return {"parts_mm2": {k: round(v, 4) for k, v in a.items()}, "encoder_mm2": round(sum(a.values()), 3)}


def main() -> None:
    wt = Path(sys.argv[1]).resolve()
    out = {"area": {}, "power": {}}
    for R in (8, 16):
        for w in (False, True):
            e = encoder_area(R, w)
            key = f"{R}x{R} 2MB {'w16 (FP16 MAC)' if w else 'INT8 (CP12)'}"
            out["area"][key] = {**e, "chip_mm2": round(e["encoder_mm2"] + VERIFY_MM2, 2)}
            print(key, out["area"][key], flush=True)
    sys.path.insert(0, str(wt / "src"))
    import bench_cp1
    bench_cp1.DATA = rz.ROOT / "data" / "reddit" / bench_cp1.DATA.name
    import model_cp4
    model_cp4.MODEL_DIR = rz.ROOT / "models" / model_cp4.MODEL_DIR.name
    import numpy as np
    import sim_cp19 as c19
    from sim_cp24 import fixed
    c19.Box.host = fixed(c19.Box.host)
    t1, bt = c19.EncTable.text1, c19.EncTable.batch
    inp = c19.inputs()
    lens = np.minimum(c19.token_lengths(), 256)
    h_conn = json.loads((wt / "results" / "cp19" / "cp19.json").read_text(encoding="utf-8"))["handshake"]["box_s"]
    for mode, R in (("w16", 16), ("w16", 8), ("flat4", 16)):
        def text1(self, n, _m=mode):
            s, J = t1(self, n)
            f, g = rz.fac(self.R, n, _m)
            return s * f, J * g

        def batch(self, ls, _m=mode):
            s, J = bt(self, ls)
            f, g = rz.fac(self.R, len(ls) * max(ls), _m)
            return s * f, J * g
        c19.EncTable.text1, c19.EncTable.batch = text1, batch
        et = c19.EncTable(R, 2048)
        s1 = c19.s1_day(et, inp, lens, "Q2")
        s4 = c19.s4_flood(et, inp, lens, "O-b", "N-b", "invalid", h_conn)
        key = f"{mode} {R}x{R} 2MB"
        out["power"][key] = {"S1_chip_mW": s1["chip_W"] * 1e3, "S1_chip_J_day": s1.get("chip_J_day"),
                             "S1_text_p99_s": s1["text_p99_s"], "S4_chip_W": s4["chip_W"],
                             "S4_verify_busy": s4["verify_busy"], "S4_text_p99_s": s4["text_p99_s"]}
        print(key, out["power"][key], flush=True)
        (RES / "cp34_chip.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    c19.EncTable.text1, c19.EncTable.batch = t1, bt
    print("wrote results/chip/cp34_chip.json")


if __name__ == "__main__":
    main()
