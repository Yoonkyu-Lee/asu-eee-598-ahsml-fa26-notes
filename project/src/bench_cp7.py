"""CP7 benchmark: the CPU side of the proof-verification cost model.

Build first (from project/):
  cargo build --release --manifest-path src/crypto_bench/Cargo.toml
  RUSTFLAGS="-C target-cpu=native" cargo build --release --features fast --target-dir src/crypto_bench/target-fast --manifest-path src/crypto_bench/Cargo.toml
Run:  .venv/Scripts/python.exe src/bench_cp7.py
Writes results/cp7/cp7_cpu.json:
  prims    Fq / Fq2 / Fq12 mul, Miller loop, final exponentiation, G1 scalar mul, Groth16 verify (single, batch),
           for the default arkworks build and the x86 assembly build, on a P-core and a low-power E-core
  energy   package joules per proof for batch verification (64 per call), 10 s loop per core
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp2 import idle_watts  # noqa: E402
from bench_cp6 import on_core  # noqa: E402
import bench_cp6  # noqa: E402
from hw_probe import PackageEnergy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp7"
BUILDS = {"default": ROOT / "src/crypto_bench/target/release/crypto_bench.exe",
          "asm_native": ROOT / "src/crypto_bench/target-fast/release/crypto_bench.exe"}
CORES = {"p_core": 0, "lp_e_core": 20}  # as in CP6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cores": CORES, "prims": {}, "energy": {}}
    for build, exe in BUILDS.items():
        bench_cp6.EXE = exe
        for name, core in CORES.items():
            res["prims"][f"{build}/{name}"] = r = json.loads(on_core(core, ["prims"]))
            print(build, name, {k: round(v, 2) for k, v in r.items()}, flush=True)
    bench_cp6.EXE = BUILDS["default"]
    for name, core in CORES.items():
        idle = idle_watts()
        meter = PackageEnergy(); meter.start()
        r = json.loads(on_core(core, ["loop", "groth16batch", "10"]))
        joules, secs = meter.stop()
        res["energy"][f"{name}/groth16batch"] = e = {
            "ops": r["ops"], "us_per_op": r["secs"] / r["ops"] * 1e6, "pkg_W": joules / secs, "idle_W": idle,
            "J_per_op": joules / r["ops"], "active_J_per_op": max(joules - idle * secs, 0) / r["ops"]}
        print(name, "batch energy", {k: round(v, 6) for k, v in e.items()}, flush=True)
    (OUT / "cp7_cpu.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
