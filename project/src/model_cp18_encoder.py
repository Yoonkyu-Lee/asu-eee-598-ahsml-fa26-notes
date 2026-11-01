"""CP18 encoder mappings: CP12's Timeloop/Accelergy setup (src/tl_cp12) on other GEMM shapes and global-buffer sizes.

Runs inside WSL like model_cp12_encoder.py:
  . src/tl_cp12/tl_env.sh && python src/model_cp18_encoder.py --model clipv                  # arrays 8/16/32, 2 MB
  . src/tl_cp12/tl_env.sh && python src/model_cp18_encoder.py --model minilm --glb 512 --arrays 16
Writes results/cp18/cp18_timeloop_<model>_glb<KB>.json in the cp12_timeloop.json format. Every number is 추정.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_cp12_encoder as base  # noqa: E402

OUT = base.ROOT / "results" / "cp18"
MODELS = {  # name: (GEMMs (K, N), rows P = tokens x batch)
    "minilm": (base.GEMMS, base.ROWS),
    "clipv": ({"qkv": (768, 2304), "attn_out": (768, 768), "ffn_in": (768, 3072), "ffn_out": (3072, 768)},
              (50, 100, 200, 400, 800, 1600)),        # 50 tokens per image, B = 1 ... 32
    "clipt": ({"qkv": (512, 1536), "attn_out": (512, 512), "ffn_in": (512, 2048), "ffn_out": (2048, 512)},
              (16, 32, 77, 154, 308, 616)),          # up to 77 tokens, B = 1 ... 8
}


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    model = arg("--model", "clipv")
    base.GLB_KB = int(arg("--glb", "2048"))
    base.GEMMS, rows = MODELS[model]
    arrays = tuple(int(x) for x in arg("--arrays", "8,16,32").split(","))
    if "--rows" in sys.argv:
        rows = tuple(int(x) for x in arg("--rows", "").split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    os.makedirs("/opt/tl/runs", exist_ok=True)
    work = Path(arg("--work", "")) if "--work" in sys.argv else \
        Path(tempfile.mkdtemp(prefix=f"cp18_{model}_{base.GLB_KB}_", dir="/opt/tl/runs"))
    print("work", work, flush=True)
    jobs = [(R, g, P, work) for R in arrays for g in base.GEMMS for P in rows]
    out, comps = [], {}
    with Pool(5) as pool:   # forked workers inherit GEMMS and GLB_KB set above
        for r in pool.starmap(base.run, jobs):
            if r["R"] not in comps:
                try:
                    comps[r["R"]] = base.tables(Path(r["_dir"]))
                except Exception as e:
                    comps[r["R"]] = {"error": repr(e), "dir": r["_dir"]}
            r.pop("_dir")
            out.append(r)
            print(f"{model} glb={base.GLB_KB} R={r['R']:2d} {r['gemm']:8s} P={r['P']:4d} cycles={r['cycles']:>9d} "
                  f"util={r['utilization']:.2f} E={r['energy_uJ']:.2f} uJ", flush=True)
    res = {"assumptions": {"GHz": base.GHZ, "dram_GBs": base.DRAM_GBS, "glb_KB": base.GLB_KB, "model": model,
                           "technology": "45nm", "setup": "src/tl_cp12 (CP12)"},
           "components": {str(k): v for k, v in comps.items()}, "runs": out}
    path = OUT / f"cp18_timeloop_{model}_glb{base.GLB_KB}.json"
    if path.exists():   # merge with earlier arrays of the same model and buffer size
        old = json.loads(path.read_text())
        keep = [r for r in old["runs"] if r["R"] not in arrays]
        res["runs"] = keep + res["runs"]
        res["components"] = {**old["components"], **res["components"]}
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if all("error" not in c for c in comps.values()):
        shutil.rmtree(work, ignore_errors=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
