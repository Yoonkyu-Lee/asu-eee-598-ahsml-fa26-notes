"""CP26: exchange protocols compared on the churn model (sim.exchange), then the pick by rules E1-E3.

  .venv/Scripts/python.exe src/sim_cp26.py            # -> results/cp26/cp26.json   (CPU, about 30 min)

Grid: scenes C-a (all always-on), C-b (one always-on, the rest laptops), C-c (half and half) x n = 10 ... 1000 x
X0, X2, X3; X1 and X1m as event runs at n <= 30 and X1 by formula above (checked against the runs). Sensitivity at
n = 100: push loss 0 / 1e-2, laptops off 3 days, home-box outages 1 h.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sim.exchange import Exchange, Scene, x1_formula  # noqa: E402

OUT = ROOT / "results" / "cp26"
SCENES = {"C-a": {"always": 1.0}, "C-b": {"always_n": 1, "always": 0.0}, "C-c": {"always": 0.5}}
NS = (10, 30, 100, 300, 1000)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def slim(r: dict) -> dict:
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


def slope(xs, ys) -> float:
    x, y = np.log(np.asarray(xs, float)), np.log(np.maximum(np.asarray(ys, float), 1e-12))
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"runs": [], "x1_formula": [], "sensitivity": []}
    for name, kw in SCENES.items():
        for n in NS:
            for proto in ("X0", "X2", "X3") + (("X1", "X1m") if n <= 30 else ()):
                r = slim(Exchange(Scene(n=n, **kw), proto).run())
                r["scene"] = name
                res["runs"].append(r)
                log(name, n, proto, {k: r[k] for k in ("missing_frac", "live_p99_s", "catch_up_p99_s",
                                                       "over_over_needed", "dup_over_needed")})
            m = next(x for x in res["runs"] if x["scene"] == name and x["n"] == n and x["proto"] == "X2")["messages"]
            f = x1_formula(n, Scene(n=n, **kw), m)
            res["x1_formula"].append({"scene": name, "n": n, **f})
    for label, kw in {"loss 0": {"p_loss": 0.0}, "loss 1e-2": {"p_loss": 1e-2},
                      "laptops off 3 d": {"always_n": 1, "always": 0.0, "lap_off_s": 3 * 86400.0},
                      "home outages 1 h": {"on_out_s": 3600.0}}.items():
        for proto in ("X2", "X3"):
            r = slim(Exchange(Scene(n=100, **kw), proto).run())
            r["case"] = label
            res["sensitivity"].append(r)
            log("sens", label, proto, r["missing_frac"], r["over_over_needed"], r["catch_up_p99_s"])
    # rules E1-E3 per protocol
    pick = {}
    for proto in ("X0", "X2", "X3"):
        rs = [r for r in res["runs"] if r["proto"] == proto]
        e1 = all(r["missing_frac"] <= 1e-4 for r in rs)
        e2 = all(r["live_p99_s"] <= 2.0 for r in rs) and all(
            np.isnan(r["catch_up_all_p99_s"]) or r["catch_up_all_p99_s"] <= 60.0 for r in rs) and proto != "X0"
        sl = {s: slope([r["n"] for r in rs if r["scene"] == s], [r["over_over_needed"] for r in rs if r["scene"] == s])
              for s in SCENES}
        e3 = all(r["over_over_needed"] <= 0.10 for r in rs) and all(v <= 0.1 for v in sl.values())
        # added after the rerun's results (labelled): catch-ups that had to wait because no box was on are apart;
        # the pre-registered E2 above counts them
        waited = sum(r.get("catch_up_waited_n", 0) for r in rs)
        e2_on = all(r["live_p99_s"] <= 2.0 for r in rs) and all(
            np.isnan(r["catch_up_p99_s"]) or r["catch_up_p99_s"] <= 60.0 for r in rs) and proto != "X0"
        pick[proto] = {"E1": e1, "E2": e2, "E3": e3, "ratio_slope": sl, "waited_catch_ups": waited,
                       "E2_when_a_box_is_on": e2_on,
                       "worst_missing": max(r["missing_frac"] for r in rs),
                       "worst_over_ratio": max(r["over_over_needed"] for r in rs)}
    res["rules"] = pick
    res["pick"] = next((p for p in ("X0", "X2", "X3") if all(pick[p][e] for e in ("E1", "E2", "E3"))), None)
    res["pick_when_a_box_is_on"] = next((p for p in ("X0", "X2", "X3")
                                         if pick[p]["E1"] and pick[p]["E2_when_a_box_is_on"] and pick[p]["E3"]), None)
    log("rules", json.dumps(pick), "pick", res["pick"])
    (OUT / "cp26.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("wrote")


if __name__ == "__main__":
    main()
