"""CP34: collect the 16-bit mappings that finished (the sweep was cut at 40 min) into a partial result file.
  (WSL, as the user that ran the sweep)  python chip/model/timeloop_fp16_collect.py <work dir>
  -> results/chip/cp34_timeloop_w16_partial.json, with the INT8 runs of the same array (cp12_timeloop.json) beside them.
Every number is 추정. The MAC is a 16-bit integer MAC (a proxy, lower bound for an FP16 MAC)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeloop_minilm as base  # noqa: E402

work = Path(sys.argv[1])
int8 = {(r["gemm"], r["P"]): r for r in json.loads((base.ROOT / "results/chip/cp12_timeloop.json").read_text())["runs"]
        if r["R"] == 16}
rows = []
for d in sorted(work.iterdir()):
    s = d / "timeloop-mapper.stats.txt"
    if not s.exists():
        continue
    st = base.parse_stats(s.read_text())
    _, gemm, p = d.name.split("_", 2)[0], d.name[4:d.name.rindex("_P")], int(d.name[d.name.rindex("_P") + 2:])
    i8 = int8.get((gemm, p))
    rows.append({"gemm": gemm, "P": p, "cycles": st["cycles"], "utilization": st["utilization"],
                 "energy_uJ": st["energy_uJ"],
                 "int8_cycles": i8 and i8["cycles"], "int8_energy_uJ": i8 and i8["energy_uJ"],
                 "cycles_ratio": i8 and round(st["cycles"] / i8["cycles"], 3),
                 "energy_ratio": i8 and round(st["energy_uJ"] / i8["energy_uJ"], 3)})
out = {"what": "CP34 partial: 16 x 16 array, 2 MB, 16-bit words, 16-bit integer MAC as FP16 proxy; sweep cut at 40 min",
       "label": "estimate (Timeloop/Accelergy, 45 nm)", "finished": len(rows), "of": 16, "runs": rows}
(base.ROOT / "results/chip/cp34_timeloop_w16_partial.json").write_text(json.dumps(out, indent=1))
print(len(rows), "runs ->", "results/chip/cp34_timeloop_w16_partial.json")
