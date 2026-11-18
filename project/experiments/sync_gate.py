"""CP30 cost gate from results/sync/cost.json and idle.json (rules in cp/CP30.md, written before results).

  .venv/Scripts/python.exe experiments/sync_gate.py     -> prints the tables, writes results/sync/gate.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "sync"
DUP_MAX, EXCH_MAX, IDLE_MAX = 0.10, 100.0, 100.0          # (i) (ii) and gate 2
MISS_TOL, P99_TOL = 0.001, 1.0                            # (iii)


def main():
    cost = json.loads((R / "cost.json").read_text())
    det = []                                            # S2 is in both grids: the same numbers (determinism)
    if (R / "cost2.json").exists():
        c2 = json.loads((R / "cost2.json").read_text())
        old = {(r["scene"], r["n"], r["seed"]): r for r in cost if r["variant"] == "S2"}
        for r in c2:
            if r["variant"] == "S2":
                o = old.get((r["scene"], r["n"], r["seed"]))
                det.append(o is not None and o["world_trace"] == r["world_trace"]
                           and o["exch_kb_box_day"] == r["exch_kb_box_day"])
        cost = cost + [r for r in c2 if r["variant"] != "S2"]
    for extra in ("cost3.json", "cost4.json"):
      if (R / extra).exists():                     # F' (after cost2, labelled); its S2 also checks determinism
        c3 = json.loads((R / extra).read_text())
        old = {(r["scene"], r["n"], r["seed"]): r for r in cost if r["variant"] == "S2"}
        for r in c3:
            if r["variant"] == "S2" and (r["scene"], r["n"], r["seed"]) in old:
                o = old[(r["scene"], r["n"], r["seed"])]
                det.append(o["world_trace"] == r["world_trace"] and o["exch_kb_box_day"] == r["exch_kb_box_day"])
        have = {(r["scene"], r["n"], r["seed"], r["variant"]) for r in cost}
        cost = cost + [r for r in c3 if (r["scene"], r["n"], r["seed"], r["variant"]) not in have]
    idle = json.loads((R / "idle.json").read_text())
    by = defaultdict(dict)
    for r in cost:
        by[(r["scene"], r["n"], r["seed"])][r["variant"]] = r
    out = {"cells": [], "idle": idle}
    variants = ("S2", "S2+K", "S2+F", "S2+P", "S2c", "S2+P2", "S2+KF", "S2+KF+P2", "S2+KF'", "S2+KF'+P2", "S2+KE", "S2+KF'E")
    fails = defaultdict(list)
    traces_equal = True
    for key in sorted(by, key=lambda k: (k[0], k[1], k[2])):
        cell = by[key]
        base = cell.get("S2")
        tr = {r.get("world_trace_times", r["world_trace"]) for v, r in cell.items() if v != "S0"}
        traces_equal &= len(tr) == 1 or not all("world_trace_times" in r for r in cell.values())
        for v in variants:
            r = cell.get(v)
            if r is None:
                continue
            g1 = r["dup_ratio"] <= DUP_MAX
            g2 = r["exch_kb_box_day"] <= EXCH_MAX
            g3 = True
            if base is not None and v != "S2":
                g3 = (r["miss"] <= base["miss"] + MISS_TOL * max(r["miss_pairs"], 1)
                      and (r["live_p99_s"] or 0) <= (base["live_p99_s"] or 0) + P99_TOL)
            ok_v = not r["violations"]
            row = {"scene": key[0], "n": key[1], "seed": key[2], "variant": v, "dup": round(r["dup_ratio"], 3),
                   "exch_kb": round(r["exch_kb_box_day"], 1), "miss": r["miss"], "miss_pairs": r["miss_pairs"],
                   "p99": r["live_p99_s"], "violations": r["violations"], "gate_i": g1, "gate_ii": g2,
                   "gate_iii": g3, "over_over_needed": r["over_over_needed"],
                   "x2_over": cell["S0"]["over_over_needed"] if "S0" in cell else None,
                   "push_ctl_kb": round(r["push_ctl_kb"], 1), "recon_kb": round(r["recon_kb"], 1),
                   "probe_kb": round(r["probe_kb"], 1)}
            out["cells"].append(row)
            for name, ok in (("i", g1), ("ii", g2), ("iii", g3), ("viol", ok_v)):
                if not ok:
                    fails[v].append((name, key))
    out["world_traces_equal"] = traces_equal
    out["s2_rerun_identical"] = det
    print("S2 rerun identical in cost2:", all(det) if det else None, len(det))
    out["fails"] = {v: [(a, list(k)) for a, k in f] for v, f in fails.items()}
    idle_ok = {v: all(r["idle_kb_box_day"] <= IDLE_MAX for r in idle if r["variant"] == v) for v in ("S2", "S2c", "S2+KF", "S2+KF+P2", "S2+K", "S2+KF'", "S2+KF'+P2")}
    out["idle_ok"] = idle_ok
    (R / "gate.json").write_text(json.dumps(out, indent=1, default=str))
    print("world traces equal across variants in every cell:", traces_equal)
    print("idle <= 100 KB:", idle_ok)
    for v in variants:
        f = fails.get(v, [])
        print(f"{v:5s} gate fails: {len(f)}", sorted({(a, k[0], k[1]) for a, k in f})[:12])
    print()
    print("seed 1, exchange KB/box-day | dup ratio | over/needed (X2)")
    for sc in ("C-a", "C-b", "C-c"):
        for n in (10, 30, 100, 300, 1000):
            cell = by.get((sc, n, 1))
            if not cell:
                continue
            s = " ".join(f"{v}={cell[v]['exch_kb_box_day']:5.1f}/{cell[v]['dup_ratio']:.2f}/{cell[v]['live_p99_s']:.1f}"
                         for v in ("S2", "S2+K", "S2c", "S2+KF'", "S2+KF'+P2") if v in cell)
            x2 = cell.get("S0", {}).get("over_over_needed")
            print(f"{sc} n={n:4d} {s} | over S2 {cell['S2']['over_over_needed']:.2f} S2c "
                  f"{cell['S2c']['over_over_needed']:.2f} X2 {x2:.2f}")


if __name__ == "__main__":
    main()
