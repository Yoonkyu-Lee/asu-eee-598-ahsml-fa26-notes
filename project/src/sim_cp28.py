"""CP28 R1: the chip conclusions of CP19, CP20, CP22 and CP23 again with FP16 text encoding (the CP3 precision
contract, CP27): every MiniLM time and energy on the chip x 4.0 (CP20's image-tower GEMM FP16 / INT8, 추정 M-a).

  .venv/Scripts/python.exe src/sim_cp28.py rerun     # results/cp28/{cp19,cp20,cp22,cp23}/   (CPU, about 40 min)
  .venv/Scripts/python.exe src/sim_cp28.py compare   # -> results/cp28/compare.json

CP19 and CP20 run with CP22's host fix (as CP24 R1; R1 and R1b gave the same decisions), and are compared with
results/cp24/rerun (the same host, INT8 text). CP22 and CP23 already had that host and are compared with their own
committed results.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
RR = RES / "cp28"
FP16 = 0.545 / 0.136                     # CP20: CLIP image tower GEMM, FP16 0.545 s vs INT8 0.136 s


def fp16_text(EncTable) -> None:
    t1, bt = EncTable.text1, EncTable.batch

    def text1(self, L):
        s, J = t1(self, L)
        return s * FP16, J * FP16

    def batch(self, lens):
        s, J = bt(self, lens)
        return s * FP16, J * FP16
    EncTable.text1, EncTable.batch = text1, batch


def rerun() -> None:
    import sim_cp19 as c19
    import sim_cp20 as c20
    import sim_cp22 as c22
    import sim_cp23 as c23
    from sim_cp24 import fixed
    fp16_text(c19.EncTable)
    c19.Box.host = fixed(c19.Box.host)
    c20.Box20.host = fixed(c20.Box20.host)
    for sub in ("cp14", "cp19", "cp20", "cp22", "cp23"):
        (RR / sub).mkdir(parents=True, exist_ok=True)
    copies = {RES / "cp14" / "cp14_fixed.json": RR / "cp14" / "cp14_fixed.json",
              RES / "cp20" / "cp20_bench.json": RR / "cp20" / "cp20_bench.json",
              RES / "cp23" / "rln_n.json": RR / "cp23" / "rln_n.json",
              RES / "cp23" / "cp23.json": RR / "cp23" / "cp23.json"}
    for a, b in copies.items():
        shutil.copy(a, b)
    try:
        c19.OUT = RR / "cp19"
        c19.main()
        c20.RES, c20.OUT = RR, RR / "cp20"
        c20.main()
        c22.OUT = RR / "cp22"
        c22.main()
        c23.OUT = RR / "cp23"
        c23.i5()
    finally:
        for b in list(copies.values())[:2]:
            b.unlink(missing_ok=True)


def compare() -> None:
    from sim_cp24 import flat
    res = {}
    pairs = {"cp19": (RES / "cp24" / "rerun" / "cp19" / "cp19.json", ("rule_Q", "rule_C", "rule_S", "encoder_pick")),
             "cp20": (RES / "cp24" / "rerun" / "cp20" / "cp20.json", ("rule_V2", "rule_F", "rule_A")),
             "cp22": (RES / "cp22" / "cp22.json", None), "cp23": (None, None)}
    for cp, (old_p, picks) in pairs.items():
        new_dir = RR / cp
        files = sorted(new_dir.glob("*.json"))
        res[cp] = {"files": [f.name for f in files]}
        if old_p is None or not old_p.exists():
            continue
        new_p = new_dir / old_p.name
        if not new_p.exists():
            continue
        old, new = json.loads(old_p.read_text(encoding="utf-8")), json.loads(new_p.read_text(encoding="utf-8"))
        keys = picks or [k for k in new if k.startswith("rule")]
        r = {}
        for k in keys:
            fo, fn = flat(old.get(k)), flat(new.get(k))
            r[k] = {"same": fo == fn, "changed": {p: (fo.get(p), fn.get(p)) for p in set(fo) | set(fn)
                                                  if fo.get(p) != fn.get(p)}}
        res[cp]["decisions"] = r
        print(cp, {k: v["same"] for k, v in r.items()})
    (RR / "compare.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def encoders() -> None:
    """R1, second part: CP19's rule Q scenarios (S2 peak, S3 8 h and 24 h backlog, Q2 live first) for each encoder
    size at FP16 text: the smallest that keeps K1-K3 (CP19 rule S's encoder list)."""
    import numpy as np
    import sim_cp19 as c19
    from sim_cp24 import fixed
    fp16_text(c19.EncTable)
    c19.Box.host = fixed(c19.Box.host)
    inp = c19.inputs()
    lens = np.minimum(c19.token_lengths(), 256)
    L = c19.LIMITS
    out = {}
    for R, glb in ((8, 1024), (16, 2048), (32, 2048)):
        et = c19.EncTable(R, glb)
        s2 = c19.s2_peak(et, inp, lens, "Q2")
        rows = [c19.s3_backlog(et, inp, lens, h, "Q2", B, live_first=True) for h in (8, 24) for B in (8, 1)]
        best = None
        for B in (8, 1):
            s3 = [r for r in rows if r["B"] == B]
            k1 = all(r["text_p99_s"] <= L["K1_text_box_s"] for r in s3 + [s2])
            k2 = all(r["query_p99_ms"] <= L["K2_query_s"] * 1e3 for r in s3 + [s2])
            k3 = all(r["catch_up_s"] <= L["K3_backlog_8h_s"] for r in s3 if r["hours"] == 8)
            ok = k1 and k2 and k3
            best = best or (ok and B)
            out[f"{R}x{R} B={B}"] = {"K1": k1, "K2": k2, "K3": k3, "pass": ok,
                                     "query_ms_S2": s2["query_p99_ms"], "text_s_S2": s2["text_p99_s"],
                                     "query_ms_S3_max": max(r["query_p99_ms"] for r in s3),
                                     "catch_up_min_8h": next(r["catch_up_min"] for r in s3 if r["hours"] == 8),
                                     "catch_up_min_24h": next(r["catch_up_min"] for r in s3 if r["hours"] == 24),
                                     "text_164_ms": et.text1(164)[0] * 1e3, "query_ms": et.text1(12)[0] * 1e3}
            print(R, B, out[f"{R}x{R} B={B}"], flush=True)
    out["pick"] = next((k for k, v in out.items() if v["pass"]), None)
    print("pick", out["pick"])
    (RR / "encoders.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def rerun16() -> None:
    """R1, third part: CP22 and CP23 (I5 + F1) with the encoder R1 picked at FP16 text (16x16 2 MB): their code
    builds EncTable(8, 1024), so the class is wrapped to build the 16x16 instead."""
    import sim_cp19 as c19
    fp16_text(c19.EncTable)
    base = c19.EncTable

    class Enc16(base):
        def __init__(self, R=8, glb=1024):
            super().__init__(16, 2048)
    c19.EncTable = Enc16
    import sim_cp22 as c22
    import sim_cp23 as c23
    for sub in ("cp22_16", "cp23_16"):
        (RR / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(RES / "cp23" / "rln_n.json", RR / "cp23_16" / "rln_n.json")
    shutil.copy(RES / "cp23" / "cp23.json", RR / "cp23_16" / "cp23.json")
    c22.OUT = RR / "cp22_16"
    c22.main()
    c23.OUT = RR / "cp23_16"
    c23.i5()


if __name__ == "__main__":
    {"rerun": rerun, "compare": compare, "encoders": encoders, "rerun16": rerun16}[sys.argv[1]]()
