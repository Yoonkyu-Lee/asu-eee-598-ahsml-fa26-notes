"""CP38 (journal/cp/CP38.md): is CP33's delivery tail a parameter problem or a premise problem? Rules written before.

  .venv/Scripts/python.exe -W ignore experiments/sync_tail.py diag     # the 16 failing cells, sub, member-path diagnosis
  .venv/Scripts/python.exe -W ignore experiments/sync_tail.py sweep    # 16 variants x the 16 failing cells
  .venv/Scripts/python.exe -W ignore experiments/sync_tail.py confirm <variant>   # one variant on all 110 sub cells
  .venv/Scripts/python.exe -W ignore experiments/sync_tail.py judge    # the rules on the files above
Leak runs (the reference) are CP33's, same seeds: results/sync/subs/random.json. Output: results/sync/tail/.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from sim.sync import DAY, Sync  # noqa: E402
import sync as c30  # noqa: E402
import sync_scenes as c29  # noqa: E402
import sync_subs as c33  # noqa: E402

OUT = ROOT / "results" / "sync" / "tail"
T0 = time.time()
NO_PATH = 0.10            # R-premise: a late delivery had a member path for less than 10 % of its wait
PREMISE_SHARE = 0.5       # R-premise: at least half of the late deliveries
VARIANTS = {f"d{d}_r{int(r)}_e{int(e)}_{'K' if k else 'noK'}": dict(d_sub=d, sub_refill_s=r, explore_s=e, known_first=k)
            for d, r, e, k in itertools.product((3, 5), (600.0, 60.0), (3600.0, 600.0), (False, True))}
BASE = "d3_r600_e3600_noK"     # CP33 as it is


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def cp33():
    rnd = json.loads((ROOT / "results" / "sync" / "subs" / "random.json").read_text())
    by = {(r["scene"], r["n"], r["seed"], r["mode"]): r for r in rnd}
    gate = json.loads((ROOT / "results" / "sync" / "subs" / "gate.json").read_text())
    failing = [tuple(f["cell"]) for f in gate["layer1"]["delivery_fail"]]
    return by, failing


def _one(job):
    s, n, seed, var, diag = job
    t = time.time()
    sc = c33.scene(n, "sub", seed=seed, days=3.0, miss_at=(DAY, 2 * DAY, 3 * DAY - 400.0), topics=c33.TOPICS,
                   tail_diag=diag, **VARIANTS[var], **c29.SCENES[s], **c29.FAULTS)
    r = Sync(sc).run()
    keep = {**c30.metrics(r, n, 3.0), "ev": r["ev"], "verify_per_box_day": r["verify_per_box_day"],
            "live_path_p99_s": r.get("live_path_p99_s")}
    keep.update(scene=s, n=n, seed=seed, variant=var, secs=round(time.time() - t, 1))
    if diag:
        keep["tail_diag"] = r["tail_diag"]
    return keep


def passes(r, L) -> bool:
    """CP33's delivery rule against the leak run of the same cell."""
    e, el = r["ev"], L["ev"]
    return (e.get("miss", 0) <= el.get("miss", 0) + c33.MISS_TOL * max(e.get("miss_pairs", 0), 1)
            and (r["live_p99_s"] or 0) <= (L["live_p99_s"] or 0) + c33.P99_TOL)


def run_diag():
    by, failing = cp33()
    out = c30._pool([(*c, BASE, True) for c in failing], _one, "diag")
    rows = []
    for r in out:
        k = (r["scene"], r["n"], r["seed"])
        L, old = by[(*k, "leak")], by[(*k, "sub")]
        thr = (L["live_p99_s"] or 0) + c33.P99_TOL
        late = [x for x in r["tail_diag"] if x[0] > thr and x[1] > 0]
        nopath = [x for x in late if x[2] / x[1] < NO_PATH]
        rows.append({"cell": k, "same_as_cp33": r["live_p99_s"] == old["live_p99_s"] and r["ev"] == old["ev"],
                     "p99": r["live_p99_s"], "leak_p99": L["live_p99_s"], "late": len(late),
                     "late_no_path": len(nopath), "path_share_median": sorted(x[2] / x[1] for x in late)[len(late) // 2]
                     if late else None, "miss": r["ev"].get("miss", 0), "leak_miss": L["ev"].get("miss", 0)})
        log(rows[-1])
    late = sum(x["late"] for x in rows)
    nop = sum(x["late_no_path"] for x in rows)
    save("diag.json", {"rows": rows, "late": late, "late_no_path": nop,
                       "premise": late > 0 and nop / late >= PREMISE_SHARE, "raw": out})
    log("late", late, "no member path", nop, "premise", late > 0 and nop / late >= PREMISE_SHARE)


def run_sweep():
    by, failing = cp33()
    jobs = [(*c, v, False) for v in VARIANTS for c in failing]
    out = c30._pool(jobs, _one, "sweep")
    table = {}
    for v in VARIANTS:
        rs = [r for r in out if r["variant"] == v]
        ok = [passes(r, by[(r["scene"], r["n"], r["seed"], "leak")]) for r in rs]
        table[v] = {"pass": sum(ok), "of": len(rs), "violations": [r["violations"] for r in rs if r["violations"]],
                    "exch_kb_max": max(r["exch_kb_box_day"] for r in rs),
                    "p99_C-a_100_300": [r["live_p99_s"] for r in rs if r["scene"] == "C-a" and r["n"] in (100, 300)]}
        log(v, table[v]["pass"], "/", len(rs), "exch max", round(table[v]["exch_kb_max"], 1))
    save("sweep.json", {"table": table, "raw": out})


def run_confirm(var):
    by, _ = cp33()
    jobs = [(s, n, seed, var, False) for n in sorted(c29.NS, reverse=True) for s in c29.SCENES for seed in c29.SEEDS
            if n < 1000 or seed <= 2]
    out = c30._pool(jobs, _one, f"confirm {var}")
    fails = [(r["scene"], r["n"], r["seed"]) for r in out if (r["scene"], r["n"], r["seed"], "leak") in by
             and not passes(r, by[(r["scene"], r["n"], r["seed"], "leak")])]
    save(f"confirm_{var}.json", {"variant": var, "runs": len(out), "violations": [(r["scene"], r["n"], r["seed"],
         r["violations"]) for r in out if r["violations"]], "delivery_fail": fails,
         "exch_kb_max": max(r["exch_kb_box_day"] for r in out), "dup_max": max(r["dup_ratio"] for r in out),
         "raw": out})
    log(var, "runs", len(out), "delivery fails", len(fails))


def run_judge():
    d = json.loads((OUT / "diag.json").read_text())
    sw = json.loads((OUT / "sweep.json").read_text())["table"]
    full = [v for v, t in sw.items() if t["pass"] == t["of"] and not t["violations"]]
    conf = {}
    for v in full:
        f = OUT / f"confirm_{v}.json"
        if f.exists():
            c = json.loads(f.read_text())
            conf[v] = {"ok": not c["violations"] and len(c["delivery_fail"]) <= 1 and c["exch_kb_max"] <= c33.EXCH_MAX
                       and c["dup_max"] <= c33.DUP_MAX,
                       "delivery_fail": c["delivery_fail"], "exch_kb_max": c["exch_kb_max"]}
    j = {"premise": d["premise"], "late": d["late"], "late_no_path": d["late_no_path"],
         "variants_passing_16": full, "confirmed": conf}
    save("judge.json", j)
    log(j)


# ---------------------------------------------------------------- after the owner's check (2026-11-18): two verdicts
# The delivery rule re-registered after the results (corrected, CP38.md): P99 of the wait counted from the first member
# path <= leak P99 + 1 s, misses as in CP33. The availability cost is reported on the original metric by sub size.
SIZES = ((1, 4), (5, 9), (10, 29), (30, 99), (100, 10 ** 9))        # member boxes of a sub
EXPLORE10 = "d3_r600_e600_noK"                                       # the 10-minute explore, nothing else changed
WORKERS = int(__import__("os").environ.get("WORKERS", "14"))   # fewer while real-box tests run elsewhere
OVERHEAD_MAX = 52.5                                                  # KB per box-day, CP30's S2+KE maximum (owner)
VARIANTS.update({f"sr{k}": dict(VARIANTS[BASE], small_relay=k) for k in (4, 9, 29)})


def pct(xs, q=99.0):
    xs = sorted(xs)
    return xs[int(q / 100 * (len(xs) - 1))] if xs else None


def avail(r, sizes) -> dict:
    """Live member deliveries by sub size: count, share with no member path, P99 of all and of the no-path ones."""
    out = {}
    for lo, hi in sizes:
        rows = [x for x in r["diag_rows"] if lo <= r["sub_size"][x[0]] <= hi]
        nop = [x[1] for x in rows if x[2]]
        out[f"{lo}-{hi if hi < 10 ** 9 else ''}"] = {"n": len(rows), "no_path": len(nop),
                                                      "p99_s": pct([x[1] for x in rows]), "no_path_p99_s": pct(nop)}
    return out


def _cell(job):
    """kind 'rand': CP33 layer 1 (faults, 3 days); kind 'cost': CP33 cost scenes (CP26 faults, 2 days). Diagnosis on."""
    kind, s, n, seed, var, mode = job
    t = time.time()
    if kind == "rand":
        sc = c33.scene(n, mode, seed=seed, days=3.0, miss_at=(DAY, 2 * DAY, 3 * DAY - 400.0), topics=c33.TOPICS,
                       tail_diag=True, **VARIANTS[var], **c29.SCENES[s], **c29.FAULTS)
        days = 3.0
    else:
        kw = dict(c29.SCENES[s])
        kw.update(c29.CP26_FAULTS)
        sc = c33.scene(n, mode, seed=seed, days=2.0, miss_at=c30.MISS_AT, topics=c33.TOPICS, tail_diag=True,
                       **VARIANTS[var], **kw)
        days = 2.0
    r = Sync(sc).run()
    w = r["work"]
    keep = {**c30.metrics(r, n, days), "ev": r["ev"], "verify_per_box_day": r["verify_per_box_day"],
            "verify_foreign_per_box_day": r["verify_per_box_day"] * w["verify_foreign"] / max(w["verify"], 1),
            "live_path_p99_s": r.get("live_path_p99_s"), "avail": avail(r, SIZES),
            "small_subs": sorted(t for t, h in r["sub_size"].items() if h <= sc.small_relay) if sc.small_relay else []}
    keep.update(kind=kind, scene=s, n=n, seed=seed, variant=var, mode=mode, secs=round(time.time() - t, 1))
    return keep


def new_rule(r, L) -> bool:
    e, el = r["ev"], L["ev"]
    return (e.get("miss", 0) <= el.get("miss", 0) + c33.MISS_TOL * max(e.get("miss_pairs", 0), 1)
            and (r["live_path_p99_s"] or 0) <= (L["live_p99_s"] or 0) + c33.P99_TOL)


def rand_cells(failing_only: bool):
    by, failing = cp33()
    if failing_only:
        return by, failing
    return by, [(s, n, seed) for n in sorted(c29.NS, reverse=True) for s in c29.SCENES for seed in c29.SEEDS
                if n < 1000 or seed <= 2]


def run_protocol(var, failing_only=True):
    """New rule on CP33's layer 1 (the 16 failing cells, or all 110), against CP33's leak runs of the same seeds."""
    by, cells = rand_cells(failing_only)
    out = c30._pool([("rand", *c, var, "sub") for c in cells], _cell, f"protocol {var}", workers=WORKERS)
    rows = []
    for r in out:
        k = (r["scene"], r["n"], r["seed"])
        L = by[(*k, "leak")]
        rows.append({"cell": k, "new_rule": new_rule(r, L), "path_p99": r["live_path_p99_s"], "p99": r["live_p99_s"],
                     "leak_p99": L["live_p99_s"], "miss": r["ev"].get("miss", 0), "leak_miss": L["ev"].get("miss", 0),
                     "violations": r["violations"]})
    name = f"protocol_{var}_{'16' if failing_only else '110'}.json"
    save(name, {"variant": var, "cells": len(rows), "pass": sum(x["new_rule"] for x in rows),
                "violations": [x for x in rows if x["violations"]], "rows": rows,
                "raw": [{k: v for k, v in r.items() if k != "avail"} for r in out]})
    log(name, sum(x["new_rule"] for x in rows), "/", len(rows), "violations", sum(1 for x in rows if x["violations"]))


def run_cost(var, mode):
    """CP33's cost grid (sync_subs.run_cost) with the diagnosis: overhead and availability by sub size."""
    jobs = [("cost", s, n, 1, var, mode) for n in (1000, 300, 100, 30, 10) for s in ("C-a", "C-b", "C-c")]
    jobs += [("cost", s, n, seed, var, mode) for n in (100, 30, 10) for s in ("C-a", "C-b", "C-c") for seed in (2, 3)]
    out = c30._pool(jobs, _cell, f"cost {var} {mode}", workers=WORKERS)
    save(f"cost_{var}_{mode}.json", {"variant": var, "mode": mode, "exch_max": max(r["exch_kb_box_day"] for r in out),
                                     "dup_max": max(r["dup_ratio"] for r in out), "rows": out})
    log(var, mode, "exch max", round(max(r["exch_kb_box_day"] for r in out), 1),
        "violations", [(r["scene"], r["n"], r["violations"]) for r in out if r["violations"]])


def run_layer1(var, mode):
    """Availability by sub size on CP33's layer 1 (all 110 cells; faults: liars, link drops, partitions, outages)."""
    _, cells = rand_cells(False)
    out = c30._pool([("rand", *c, var, mode) for c in cells], _cell, f"layer1 {var} {mode}", workers=WORKERS)
    save(f"layer1_{var}_{mode}.json", {"variant": var, "mode": mode, "rows": out})
    log(var, mode, "runs", len(out), "violations", sum(1 for r in out if set(r["violations"]) - {"S5"}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if cmd == "layer1":
        run_layer1(sys.argv[2], sys.argv[3])
    elif cmd == "protocol":
        run_protocol(sys.argv[2], failing_only=sys.argv[3] == "16")
    elif cmd == "cost":
        run_cost(sys.argv[2], sys.argv[3])
    else:
        {"diag": run_diag, "sweep": run_sweep, "judge": run_judge}.get(cmd, lambda: run_confirm(sys.argv[2]))()
