"""CP34: size the encoder again with FP16 text from direct 16-bit mappings instead of CP28's flat "INT8 x 4.0".

  .venv/Scripts/python.exe -W ignore chip/model/encoder_resize_w16.py <cp30 worktree>/project
      -> results/chip/cp34_resize.json

The rules and the box model are CP28's, unchanged (journal/cp/CP34.md): CP19's peak-hour scene S2 and the 8 h / 24 h
backlog scenes S3 (live work first), rules K1 (new post searchable), K2 (query P99 <= 50 ms), K3 (8 h backlog
<= 60 min), batch 8 or 1. That code lives at git tag cp30 (legacy/), so it is imported from a worktree of that tag.
Only the text-encoding time and energy factor changes:
  flat4     CP28: every text time and energy x 4.0 (CP20's image-tower FP16/INT8 ratio)
  w16       per array R and rows P: 16-bit mapping cycles / INT8 mapping cycles (results/chip/cp34_timeloop_w16*.json
            vs cp12_timeloop.json, same template, 2 MB, 500 MHz), interpolated in log P; energy the same way, with the
            mapping's 16-bit integer MAC replaced by an FP16 MAC estimate (below)
  w16_half  sensitivity: the FP16 MAC at half throughput (cycles x 2, an upper bound)
FP16 MAC (estimate): Accelergy's 16-bit integer MAC energy x (Aladdin 45 nm FP32 mult + add) / (Aladdin INT32 mult +
add), all at a 2 ns pipeline step: x (29.7 + 11.026) / (9.55 + 0.0858) = 4.23. Every number is 추정.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results" / "chip"
E_I16_PJ = 4.27651                                   # Accelergy intmac, 16-bit (ERT of the runs)
FP_OVER_INT = (29.7 + 11.026) / (9.55 + 0.0858)      # Aladdin 45 nm, 2 ns: FP32 (mult+add) / INT32 (mult+add)
E_FP16_PJ = E_I16_PJ * FP_OVER_INT


def factors() -> dict:
    """(R, P) -> (time factor, energy factor) from the mappings, summed over the four MiniLM GEMMs."""
    i8 = json.loads((RES / "cp12_timeloop.json").read_text())["runs"]
    out = {}
    for R, f in ((16, "cp34_timeloop_w16.json"), (8, "cp34_timeloop_w16_R8.json")):
        w = json.loads((RES / f).read_text())["runs"]
        for P in sorted({r["P"] for r in w}):
            a = [r for r in w if r["P"] == P]
            b = [r for r in i8 if r["R"] == R and r["P"] == P]
            assert len(a) == 4 and len(b) == 4, (R, P, len(a), len(b))
            mac = sum(r["K"] * r["N"] * r["P"] for r in a)                  # MACs of the four GEMMs
            e16 = sum(r["energy_uJ"] for r in a) + mac * (E_FP16_PJ - E_I16_PJ) * 1e-6
            out[(R, P)] = (sum(r["cycles"] for r in a) / sum(r["cycles"] for r in b),
                           e16 / sum(r["energy_uJ"] for r in b))
    return out


FAC = factors()


def fac(R: int, P: float, mode: str) -> tuple[float, float]:
    if mode == "flat4":
        return 0.545 / 0.136, 0.545 / 0.136
    pts = sorted(p for r, p in FAC if r == R)
    lp = math.log(min(max(P, pts[0]), pts[-1]))
    for lo, hi in zip(pts[:-1], pts[1:]):
        if math.log(lo) <= lp <= math.log(hi):
            u = (lp - math.log(lo)) / (math.log(hi) - math.log(lo))
            t = FAC[(R, lo)][0] * (1 - u) + FAC[(R, hi)][0] * u
            e = FAC[(R, lo)][1] * (1 - u) + FAC[(R, hi)][1] * u
            return (t * 2 if mode == "w16_half" else t), e
    raise ValueError(P)


def main() -> None:
    wt = Path(sys.argv[1]).resolve()                 # <worktree of tag cp30>/project
    sys.path.insert(0, str(wt / "src"))
    import bench_cp1
    bench_cp1.DATA = ROOT / "data" / "reddit" / bench_cp1.DATA.name      # git-ignored inputs of this checkout
    import model_cp4
    model_cp4.MODEL_DIR = ROOT / "models" / model_cp4.MODEL_DIR.name
    import numpy as np
    import sim_cp19 as c19
    from sim_cp24 import fixed
    c19.Box.host = fixed(c19.Box.host)
    base_t1, base_bt = c19.EncTable.text1, c19.EncTable.batch
    inp = c19.inputs()
    lens = np.minimum(c19.token_lengths(), 256)
    L = c19.LIMITS
    out = {"factors": {f"R{R} P{P}": {"time": round(t, 3), "energy": round(e, 3)} for (R, P), (t, e) in FAC.items()},
           "fp16_mac_pJ": round(E_FP16_PJ, 2), "int16_mac_pJ": E_I16_PJ, "runs": {}}
    for mode in ("w16", "w16_half", "flat4"):
        def text1(self, n, _m=mode):
            s, J = base_t1(self, n)
            f, g = fac(self.R, n, _m)
            return s * f, J * g

        def batch(self, ls, _m=mode):
            s, J = base_bt(self, ls)
            f, g = fac(self.R, len(ls) * max(ls), _m)
            return s * f, J * g
        c19.EncTable.text1, c19.EncTable.batch = text1, batch
        for R in (8, 16):
            et = c19.EncTable(R, 2048)
            s2 = c19.s2_peak(et, inp, lens, "Q2")
            rows = [c19.s3_backlog(et, inp, lens, h, "Q2", B, live_first=True) for h in (8, 24) for B in (8, 1)]
            for B in (8, 1):
                s3 = [r for r in rows if r["B"] == B]
                k1 = all(r["text_p99_s"] <= L["K1_text_box_s"] for r in s3 + [s2])
                k2 = all(r["query_p99_ms"] <= L["K2_query_s"] * 1e3 for r in s3 + [s2])
                k3 = all(r["catch_up_s"] <= L["K3_backlog_8h_s"] for r in s3 if r["hours"] == 8)
                key = f"{mode} {R}x{R} 2MB B={B}"
                out["runs"][key] = {"K1": k1, "K2": k2, "K3": k3, "pass": k1 and k2 and k3,
                                    "query_ms_S2": s2["query_p99_ms"], "query_ms_S3_max": max(r["query_p99_ms"] for r in s3),
                                    "text_s_S2": s2["text_p99_s"],
                                    "catch_up_min_8h": next(r["catch_up_min"] for r in s3 if r["hours"] == 8),
                                    "catch_up_min_24h": next(r["catch_up_min"] for r in s3 if r["hours"] == 24),
                                    "text_164_ms": et.text1(164)[0] * 1e3, "text_164_mJ": et.text1(164)[1] * 1e3,
                                    "query_ms": et.text1(12)[0] * 1e3}
                print(key, out["runs"][key], flush=True)
                (RES / "cp34_resize.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    c19.EncTable.text1, c19.EncTable.batch = base_t1, base_bt
    print("wrote results/chip/cp34_resize.json")


if __name__ == "__main__":
    main()
