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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diag"
    {"diag": run_diag, "sweep": run_sweep, "judge": run_judge}.get(cmd, lambda: run_confirm(sys.argv[2]))()
