"""CP34: the CP28 encoder (16 x 16 array, 2 MB global buffer, 500 MHz) mapped with 16-bit words: FP16 text.

Runs inside WSL like timeloop_minilm.py:
  . chip/model/timeloop/tl_env.sh && python chip/model/timeloop_fp16.py      -> results/chip/cp34_timeloop_w16.json
Same template and search as CP12 (timeloop/arch.yaml) with WB = 16: every buffer, register and the DRAM hold 16-bit
words, the global buffer keeps its 128-bit port (8 words a cycle instead of 16), partial sums stay 32-bit.
The MAC is Accelergy's 16-bit integer MAC: a proxy (lower bound) for an FP16 MAC, whose alignment and normalisation
cost more. Storage, traffic and cycles are those of 16-bit words. Every number is 추정.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeloop_minilm as base  # noqa: E402

ROWS = (32, 128, 512, 2048)


def main() -> None:
    base.WB = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 16
    runs = os.path.expanduser("~/tl_runs")            # run as root (Accelergy's config is root's), like CP12
    os.makedirs(runs, exist_ok=True)
    if "--work" in sys.argv:                          # resume: finished shapes are kept (base.run skips them)
        work = Path(sys.argv[sys.argv.index("--work") + 1])
    else:
        work = Path(tempfile.mkdtemp(prefix=f"cp34_w{base.WB}_", dir=runs))
    print("work", work, flush=True)
    jobs = [(16, g, P, work) for g in base.GEMMS for P in ROWS]
    out, comps = [], None
    with Pool(5) as pool:                      # forked workers inherit WB
        for r in pool.starmap(base.run, jobs):
            if comps is None:
                try:
                    comps = base.tables(Path(r["_dir"]))
                except Exception as e:
                    comps = {"error": repr(e), "dir": r["_dir"]}
            r.pop("_dir")
            out.append(r)
            print(f"w{base.WB} R=16 {r['gemm']:8s} P={r['P']:4d} cycles={r['cycles']:>9d} util={r['utilization']:.2f} "
                  f"E={r['energy_uJ']:.2f} uJ", flush=True)
    res = {"assumptions": {"GHz": base.GHZ, "glb_KB": base.GLB_KB, "word_bits": base.WB, "R": 16,
                           "mac": f"Accelergy intmac {base.WB}-bit multiplier, 32-bit adder (proxy for FP16)",
                           "glb_port_bits": 128},
           "components": comps, "runs": out}
    name = base.ROOT / "results" / "chip" / f"cp34_timeloop_w{base.WB}.json"
    name.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)


if __name__ == "__main__":
    main()
