"""CP6 benchmark: what the crypto of a community node box costs on a CPU core, in time and energy.

Build first:  cargo build --release --manifest-path src/crypto_bench/Cargo.toml
Run from project/:  .venv/Scripts/python.exe src/bench_cp6.py
Writes results/cp6/cp6_crypto.json:
  per core class (a P-core, a low-power E-core as the weak-device stand-in): the Rust bench's times
  energy: package joules per Groth16 / Ed25519 verification, from a 10 s verification loop on one core
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp2 import idle_watts  # noqa: E402
from hw_probe import PackageEnergy, core_classes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp6"
EXE = ROOT / "src" / "crypto_bench" / "target" / "release" / "crypto_bench.exe"


def on_core(core: int, args: list[str]) -> str:
    """Run the bench pinned to one logical core (Windows children inherit the parent's affinity)."""
    me = psutil.Process()
    old = me.cpu_affinity()
    me.cpu_affinity([core])
    try:
        return subprocess.run([str(EXE), *args], capture_output=True, text=True, check=True).stdout
    finally:
        me.cpu_affinity(old)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cls = core_classes()
    cores = {"p_core": 0, "lp_e_core": [x for x in cls[min(cls)] if x >= 20][0]}  # CP2: 0 is a P-core, 20-21 LP-E
    res = {"cores": cores}
    for name, core in cores.items():
        res[name] = json.loads(on_core(core, []))
        print(name, "groth16 verify us", res[name]["groth16_bn254_5_public_inputs"]["2^12"]["verify_us"], flush=True)
    res["energy"] = {}
    for name, core in cores.items():
        for what in ("groth16", "ed25519"):
            idle = idle_watts()
            meter = PackageEnergy(); meter.start()
            r = json.loads(on_core(core, ["loop", what, "10"]))
            joules, secs = meter.stop()
            res["energy"][f"{name}/{what}"] = e = {
                "ops": r["ops"], "us_per_op": r["secs"] / r["ops"] * 1e6, "pkg_W": joules / secs, "idle_W": idle,
                "J_per_op": joules / r["ops"], "active_J_per_op": max(joules - idle * secs, 0) / r["ops"]}
            print(name, what, {k: round(v, 6) for k, v in e.items()}, flush=True)
    (OUT / "cp6_crypto.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
