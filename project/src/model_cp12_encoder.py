"""CP12 encoder model: Timeloop mapper + Accelergy (45 nm) for the 24 weight GEMM shapes of MiniLM-L6.

Runs inside WSL (Ubuntu 24.04) with Timeloop, Accelergy and timeloopfe built under /opt/tl (see cp/CP12.md):
  . src/tl_cp12/tl_env.sh && python src/model_cp12_encoder.py [--quick]
Writes results/cp12/cp12_timeloop.json: per (array R, GEMM, rows P) the mapped cycles, energy by component,
utilization, plus each component's area and leakage from Accelergy's ART/ERT. Every number is an estimate (추정).

Rows P = tokens x posts in the batch. A GEMM (K -> N) is a 1x1 convolution with C = K, M = N.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import yaml
import timeloopfe.v4 as tl

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "src" / "tl_cp12"
OUT = ROOT / "results" / "cp12"
COMPONENTS = "/opt/tl/timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components"
D, FF = 384, 1536
GEMMS = {"qkv": (D, 3 * D), "attn_out": (D, D), "ffn_in": (D, FF), "ffn_out": (FF, D)}
ROWS = (32, 64, 128, 256, 512, 1024, 2048)  # B=1 lengths and B=8 sorted batches
ARRAYS = (8, 16, 32)
GHZ = 0.5
DRAM_GBS = 8.5  # as in CP4 (assumed LPDDR4 share)
GLB_KB = 2048   # CP4 sram_needed(B=1, streaming) = 1.97 MB


def problem_yaml(K: int, N: int, P: int) -> dict:
    dims = ["C", "M", "R", "S", "N", "P", "Q"]
    return {"problem": {
        "version": 0.4,
        "instance": {"C": K, "M": N, "P": P, "Q": 1, "R": 1, "S": 1, "N": 1},
        "shape": {
            "name": "gemm_as_conv",
            "dimensions": dims,
            "data_spaces": [
                {"name": "Weights", "projection": [[["C"]], [["M"]], [["R"]], [["S"]]]},
                {"name": "Inputs", "projection": [[["N"]], [["C"]], [["R"], ["P"]], [["S"], ["Q"]]]},
                {"name": "Outputs", "projection": [[["N"]], [["M"]], [["Q"]], [["P"]]], "read_write": True},
            ]}}}


def parse_stats(text: str) -> dict:
    out = {"cycles": int(re.search(r"^Cycles: (\d+)", text, re.M).group(1)),
           "energy_uJ": float(re.search(r"^Energy: ([\d.e+-]+) uJ", text, re.M).group(1)),
           "utilization": float(re.search(r"^Utilization: ([\d.]+)%", text, re.M).group(1)) / 100,
           "computes": int(re.search(r"^Computes = (\d+)", text, re.M).group(1))}
    block = text.split("fJ/Compute", 1)[1]
    fj = {}
    for line in block.splitlines()[1:]:
        m = re.match(r"\s+(\S+)\s+=\s+([\d.e+-]+)", line)
        if not m:
            break
        fj[m.group(1)] = float(m.group(2))
    out["fJ_per_compute"] = fj
    return out


def tables(run_dir: Path) -> dict:
    """Per-instance area (um^2) and leakage (pJ per cycle) of each component, from Accelergy."""
    art = yaml.safe_load((run_dir / "timeloop-mapper.ART_summary.yaml").read_text())["ART_summary"]["table_summary"]
    ert = yaml.safe_load((run_dir / "timeloop-mapper.ERT_summary.yaml").read_text())["ERT_summary"]["table_summary"]
    comp = {}
    for row in art:
        m = re.search(r"(\w+)\[1\.\.(\d+)\]$", row["name"])
        if not m:
            continue
        base, n = m.groups()
        comp[base] = {"instances": int(n), "area_um2": float(row["area"]),
                      "estimator": row["primitive_estimations"] if isinstance(row["primitive_estimations"], str)
                      else [p["estimator"] for p in row["primitive_estimations"]]}
    for row in ert:
        m = re.search(r"(\w+)\[1\.\.\d+\]$", row["name"])
        if not m:
            continue
        base = m.group(1)
        acts = {a["name"]: float(a["energy"]) for a in row["actions"]}
        comp.setdefault(base, {})["actions_pJ"] = acts
    return comp


# Mapper search per array side. R=32 took 20+ min per mapping with the default search, so it gets a shorter one;
# with the spatial factors fixed (C=R, M=R) the remaining choice is the temporal tiling.
SEARCH = {8: (200, 2000), 16: (200, 2000), 32: (50, 300)}  # (victory_condition, timeout)


def run(R: int, gemm: str, P: int, work: Path) -> dict:
    K, N = GEMMS[gemm]
    run_dir = work / f"R{R}_{gemm}_P{P}"
    victory, timeout = SEARCH[R]
    done = run_dir / "timeloop-mapper.stats.txt"
    if done.exists():  # resumed sweep: keep the finished mapping
        stats = parse_stats(done.read_text())
        stats["mapping"] = (run_dir / "timeloop-mapper.map.txt").read_text()
        return {"R": R, "gemm": gemm, "K": K, "N": N, "P": P, "search": [victory, timeout], **stats, "_dir": str(run_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    prob = run_dir / "problem.yaml"
    prob.write_text(yaml.safe_dump(problem_yaml(K, N, P)))
    cycle_s = 1 / (GHZ * 1e9)
    spec = tl.Specification.from_yaml_files(
        str(SPEC / "top.yaml.jinja2"),
        jinja_parse_data={"problem": str(prob), "R": R, "GLB_KB": GLB_KB, "CYCLE_S": cycle_s,
                          "DRAM_WPC": 4096, "COMPONENTS": COMPONENTS, "VICTORY": victory, "TIMEOUT": timeout})
    tl.call_mapper(spec, output_dir=str(run_dir), log_to=str(run_dir / "mapper.log"))
    stats = parse_stats((run_dir / "timeloop-mapper.stats.txt").read_text())
    stats["mapping"] = (run_dir / "timeloop-mapper.map.txt").read_text()
    return {"R": R, "gemm": gemm, "K": K, "N": N, "P": P, "search": [victory, timeout], **stats, "_dir": str(run_dir)}


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    os.makedirs("/opt/tl/runs", exist_ok=True)
    if "--work" in sys.argv:  # resume an interrupted sweep in its directory
        work = Path(sys.argv[sys.argv.index("--work") + 1])
    else:
        work = Path(tempfile.mkdtemp(prefix="cp12_tl_", dir="/opt/tl/runs"))
    rows, comps = [], {}
    arrays = (16,) if quick else ARRAYS
    rows_p = (128,) if quick else ROWS
    jobs = [(R, gemm, P, work) for R in arrays for gemm in GEMMS for P in rows_p]
    with Pool(5) as pool:  # the mapper itself uses 4 threads (top.yaml.jinja2)
        for r in pool.starmap(run, jobs):
            if r["R"] not in comps:
                try:
                    comps[r["R"]] = tables(Path(r["_dir"]))
                except Exception as e:  # keep the runs even if the table format surprises us
                    comps[r["R"]] = {"error": repr(e), "dir": r["_dir"]}
            r.pop("_dir")
            rows.append(r)
            print(f"R={r['R']:2d} {r['gemm']:8s} P={r['P']:4d} cycles={r['cycles']:>9d} "
                  f"util={r['utilization']:.2f} E={r['energy_uJ']:.2f} uJ", flush=True)
    res = {"assumptions": {"GHz": GHZ, "dram_GBs": DRAM_GBS, "glb_KB": GLB_KB, "technology": "45nm",
                           "tools": "Timeloop (NVlabs master 3237082), Accelergy 0.4 + Aladdin/CACTI/library plug-ins, "
                                    "timeloopfe 0.4; components from timeloop-accelergy-exercises 2d55108"},
           "components": {str(k): v for k, v in comps.items()}, "runs": rows}
    name = "cp12_timeloop_quick.json" if quick else "cp12_timeloop.json"
    (OUT / name).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if all("error" not in c for c in comps.values()):
        shutil.rmtree(work, ignore_errors=True)
    print("wrote", OUT / name)


if __name__ == "__main__":
    main()
