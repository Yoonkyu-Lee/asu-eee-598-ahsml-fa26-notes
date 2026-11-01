"""CP12 part costs at 45 nm from the same estimators Accelergy uses (CACTI for SRAM, Aladdin tables for logic).

Runs inside WSL with the Accelergy plug-ins under /opt/tl/venv (see cp/CP12.md):
  /opt/tl/venv/bin/python src/model_cp12_parts.py
Writes results/cp12/cp12_parts.json. The chip roll-up (src/model_cp12.py) reads it. All numbers are estimates (추정).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PLUGINS = Path("/opt/tl/venv/share/accelergy/estimation_plug_ins")
# --hp: sensitivity with high-performance cells (a copy of the plug-in whose default_cfg.cfg says itrs-hp instead of itrs-lstp)
HP = "--hp" in sys.argv
sys.path.insert(0, "/opt/tl/cacti_hp_plugin" if HP else str(PLUGINS / "accelergy-cacti-plug-in"))
from cacti_wrapper import CactiSRAM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp12"

# name: (width bits, depth, banks). Encoder buffers mirror src/tl_cp12/arch.yaml; verify buffers are CP11's.
SRAMS = {
    "enc_glb_2MB": (128, 131072, 16),
    "enc_glb_1MB": (128, 65536, 16),
    "enc_in_buf_16KB": (128, 1024, 1),
    "enc_acc_buf_16KB": (256, 512, 1),
    "ver_buf_8KB": (256, 256, 1),       # 194 live 256-bit values (CP11)
    "ver_ucode_32KB": (256, 1024, 1),   # microcode store (size is an estimate, see CP12.md)
}
ALADDIN = ("adder", "multiplier", "fp_sp_adder", "fp_sp_multiplier", "reg")


def num(x):
    """Accelergy's action decorator may wrap the value; reduce it to a float."""
    for attr in ("value", "energy"):
        if hasattr(x, attr):
            return float(getattr(x, attr))
    return float(x[0] if isinstance(x, (tuple, list)) else x)


def aladdin_row(name: str, latency_ns: float) -> dict:
    """The table row with the largest latency not above latency_ns (one-stage logic at that clock)."""
    rows = list(csv.DictReader(open(PLUGINS / "accelergy-aladdin-plug-in" / "data" / f"{name}.csv")))
    best = max((r for r in rows if float(r["latency(ns)"]) <= latency_ns), key=lambda r: float(r["latency(ns)"]))
    return {"latency_ns": float(best["latency(ns)"]), "dynamic_pJ": float(best["dynamic energy(pJ)"]),
            "leak_mW": float(best["leakage power(mW)"]), "area_um2": float(best["area(um^2)"]),
            "note": best["comment"], "table_bits": 32}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"source": "CACTI (HewlettPackard/cacti master, via accelergy-cacti-plug-in 7649b2c CactiSRAM, 45 nm); "
                     "Aladdin 45 nm tables (accelergy-aladdin-plug-in 5e2e126 data/*.csv, 32-bit entries)",
           "sram": {}, "aladdin_2ns": {}, "aladdin_3ns": {}}
    for name, (w, d, b) in SRAMS.items():
        s = CactiSRAM(technology="45nm", width=w, depth=d, n_banks=b)
        res["sram"][name] = {"width_bits": w, "depth": d, "banks": b, "bytes": w * d // 8,
                             "read_pJ": num(s.read()), "write_pJ": num(s.write()),
                             "leak_mW": num(s.leak(1.0)) * 1e3, "area_um2": num(s.get_area()) * 1e12,
                             "random_cycle_ns": s.random_cycle_time * 1e9}
        print(name, res["sram"][name], flush=True)
    if HP:
        res["source"] += "; SENSITIVITY: CACTI cells itrs-hp"
        (OUT / "cp12_parts_hp.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return
    for lat, key in ((2.0, "aladdin_2ns"), (3.0, "aladdin_3ns")):
        res[key] = {n: aladdin_row(n, lat) for n in ALADDIN}
    (OUT / "cp12_parts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT / "cp12_parts.json")


if __name__ == "__main__":
    main()
