"""CP33: reconcile only the subs both boxes hold (H12). Rules and predictions: journal/cp/CP33.md (written before results).

  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py regress   # CP29+CP30 scenes, topics = 0, fixed model
  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py holes     # layer 2: sub scenes D1-D5, leak vs sub
  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py mc        # layer 3a: sub model + mutants
  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py cost      # CP26 scenes with 20 subs, leak vs sub (~15 min)
  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py idle      # no posts
  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py random    # layer 1, CP29 grid + faults, 20 subs (~30 min)
  .venv/Scripts/python.exe -W ignore experiments/sync_subs.py gate      # the pre-registered rules on the files above
Everything goes to results/sync/subs/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from sim.sync import DAY, HDR, Sync  # noqa: E402
from sim.sync_mc import check_subs  # noqa: E402
import sync as c30  # noqa: E402
import sync_scenes as c29  # noqa: E402

OUT = ROOT / "results" / "sync" / "subs"
V = "S2+KE"                                   # the CP30 default
MODES = ("leak", "sub")
TOPICS = 20                                   # subs in the random and cost scenes (추정, journal/cp/CP33.md)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def scene(n, mode, **kw):
    return c30.scene(n, V, sub_mode=mode, **kw)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str))


# ---------------------------------------------------------------------------------------------- regression
def run_regress():
    """The CP29 and CP30 directed scenes with topics = 0 on the fixed model (the floor restarts on re-mesh)."""
    c30.OUT = OUT / "regress"
    c30.run_holes()
    c30.run_mc()


# ---------------------------------------------------------------------------------------------- layer 2
def sub_scenes():
    from sim.sync import SyncScene
    T = SyncScene().t_bound

    def at(t, tag, box, key):
        def f(s):
            s.__dict__.setdefault("seen", {})[key] = s.holds(box, tag)
        return (t, "call", f)

    def subs(rows_subs):
        return [(0, "subs", b, list(ts)) for b, ts in rows_subs.items()]

    S = []
    d1 = {0: [0, 1], 1: [0], 2: [1]}
    S.append(dict(name="D1 a foreign sub never arrives", n=3, topics=2,
                  rows=subs(d1) + [(0, "on", b) for b in range(3)] +
                  [(100, "pub", 0, "post", "PB", 0.0, None, True, False, 1), (100 + 2 * T, "until")],
                  checks=[("box2 holds PB", lambda s: s.holds(2, "PB")),
                          ("box1 never holds PB", lambda s: not s.holds(1, "PB"))],
                  pred={"sub": "pass", "leak": "fail"}))
    d2 = {b: [0] for b in range(8)}
    d2[0] = d2[7] = [0, 1]
    S.append(dict(name="D2 the sub's two holders are not neighbours", n=8, topics=2, over={"D": 2, "D_HI": 4},
                  rows=subs(d2) + [(0, "on", b) for b in range(8)] +
                  [(600, "pub", 0, "post", "X", 0.0, None, True, False, 1), at(600 + T, "X", 7, "x"),
                   (600 + T + 60, "until")],
                  checks=[("box7 holds X within t_bound", lambda s: s.seen.get("x", False))],
                  pred={"sub": "pass", "leak": None}))
    S.append(dict(name="D3 a delete in sub B", n=3, topics=2,
                  rows=subs(d1) + [(0, "on", b) for b in range(3)] +
                  [(100, "pub", 0, "post", "PB", 0.0, None, True, False, 1), (200, "pub", 0, "delete", "DB", 0.0, "PB"),
                   (200 + 2 * T, "until")],
                  checks=[("box2 forgot PB, holds DB", lambda s: s.forgot(2, "PB") and s.holds(2, "DB")),
                          ("box1 holds neither", lambda s: not s.holds(1, "PB") and not s.holds(1, "DB")
                           and not s.forgot(1, "PB"))],
                  pred={"sub": "pass", "leak": "fail"}))
    d4 = {0: [0], 1: [0], 2: [0], 3: [2]}
    S.append(dict(name="D4 a sub with one holder", n=4, topics=3,
                  rows=subs(d4) + [(0, "on", b) for b in range(4)] +
                  [(t, "pub", 3, "post", f"C{t}", 0.0, None, True, False, 2) for t in (100, 3600, 7200)] +
                  [(DAY, "until")],
                  checks=[("box3: no sessions", lambda s: s.boxes[3].c["recon_sessions"] == 0),
                          ("box3: <= 100 hellos a day", lambda s: s.boxes[3].c["tx_probe"] <= 100 * HDR)],
                  report=lambda s: {"box3_sessions": s.boxes[3].c["recon_sessions"],
                                    "box3_hellos": s.boxes[3].c["tx_probe"] // HDR},
                  pred={"sub": "pass", "leak": None}))
    d5 = {b: [0] for b in range(8)}
    for b in (0, 1, 6, 7):
        d5[b] = [0, 1]
    S.append(dict(name="D5 a 2 h partition inside a sub, heal, post crosses", n=8, topics=2, over={"D": 3, "D_HI": 6},
                  rows=subs(d5) + [(0, "on", b) for b in range(8)] +
                  [(100, "part", [0, 1, 2, 3]), (7300, "heal"), (7400, "pub", 0, "post", "X", 0.0, None, True, False, 1),
                   (7400 + 7200, "until")],
                  checks=[("boxes 6, 7 hold X", lambda s: s.holds(6, "X") and s.holds(7, "X"))],
                  pred={"sub": "pass", "leak": None}))
    return S


def run_holes():
    out = []
    for d in sub_scenes():
        for mode in MODES:
            kw = dict(c29.QUIET)
            kw.update(d.get("over", {}))
            sim = Sync(scene(d["n"], mode, days=10.0, topics=d["topics"], **kw), script=list(d["rows"]))
            r = sim.run()
            checks = {lab: bool(fn(sim)) for lab, fn in d["checks"]}
            passed = all(checks.values()) and not r["violations"]
            pred = d["pred"][mode]
            ok = None if pred is None else (passed == (pred == "pass"))
            out.append({"scene": d["name"], "mode": mode, "pass": passed, "predicted": pred, "as_predicted": ok,
                        "checks": checks, "violations": r["violations"],
                        "report": d["report"](sim) if "report" in d else None})
            log(f"{d['name'][:52]:52s} {mode:4s} {'PASS' if passed else 'FAIL'} "
                f"{'(no prediction)' if ok is None else ('' if ok else '<-- NOT AS PREDICTED')} {checks} {r['violations']}")
    save("holes.json", out)
    log("holes:", len(out), "runs,", sum(1 for x in out if x["as_predicted"] is False), "not as predicted")


# ---------------------------------------------------------------------------------------------- layer 3a
def run_mc():
    out = []
    for m in (None, "accept_any", "recon_all+accept_any", "push_any+accept_any", "recon_all", "push_any"):
        r = check_subs(mutate=m)
        log(f"mc subs mutate={m}: {r['states']} states, violations {r['violations']}")
        out.append(r)
    save("mc.json", out)


# ---------------------------------------------------------------------------------------------- cost
def metrics(r, n, days):
    m = c30.metrics(r, n, days)
    m.update(verify_per_box_day=r["verify_per_box_day"], encode_per_box_day=r["encode_per_box_day"],
             stranded=r["ev"].get("stranded", 0), stranded_pairs=r["ev"].get("stranded_pairs", 0),
             rej_sub=r["work"]["rej_sub"], verify_foreign=r["work"]["verify_foreign"],
             subs_per_box=r["subs_per_box"])
    return m


def _cost_one(job):
    s, n, mode, seed = job
    t = time.time()
    kw = dict(c29.SCENES[s])
    kw.update(c29.CP26_FAULTS)
    sc = scene(n, mode, seed=seed, days=2.0, miss_at=c30.MISS_AT, topics=TOPICS, **kw)
    r = Sync(sc).run()
    return {"scene": s, "n": n, "variant": mode, "seed": seed, **metrics(r, n, 2.0), "secs": round(time.time() - t, 1)}


def run_cost():
    jobs = [(s, n, m, 1) for n in (1000, 300, 100, 30, 10) for s in ("C-a", "C-b", "C-c") for m in MODES]
    jobs += [(s, n, m, seed) for n in (100, 30, 10) for s in ("C-a", "C-b", "C-c") for m in MODES for seed in (2, 3)]
    out = c30._pool(jobs, _cost_one, "cost")
    save("cost.json", out)


def _idle_one(job):
    n, mode = job
    sc = scene(n, mode, seed=1, days=1.0, always=1.0, msgs_per_day=0.0, p_fail=0.0, p_silent=0.0,
               link_drops_per_day=0.0, partitions_per_day=0.0, invalid_frac=0.0, on_out_per_day=0.0, topics=TOPICS)
    r = Sync(sc).run()
    tx = r["tx"]
    return {"n": n, "variant": mode, "idle_kb_box_day": (tx.get("recon", 0) + tx.get("probe", 0)) / n / 1000,
            "recon_kb": tx.get("recon", 0) / n / 1000, "probe_kb": tx.get("probe", 0) / n / 1000,
            "violations": r["violations"]}


def run_idle():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, m) for n in (10, 30, 100, 300) for m in MODES]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(_idle_one, jobs))
    for r in out:
        log(f"idle n={r['n']:3d} {r['variant']:4s} {r['idle_kb_box_day']:.1f} KB/box-day viol={r['violations']}")
    save("idle.json", out)


# ---------------------------------------------------------------------------------------------- layer 1
def _rand_one(job):
    s, n, seed, mode = job
    t = time.time()
    sc = scene(n, mode, seed=seed, days=3.0, miss_at=(DAY, 2 * DAY, 3 * DAY - 400.0), topics=TOPICS,
               **c29.SCENES[s], **c29.FAULTS)
    r = Sync(sc).run()
    keep = {k: r[k] for k in ("n", "violations", "examples", "l1_checks", "l1_items", "live_p99_s", "ev", "work",
                              "verify_per_box_day", "encode_per_box_day", "subs_per_box")}
    keep.update(scene=s, seed=seed, mode=mode, secs=round(time.time() - t, 1))
    return keep


def run_random(workers=14):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(s, n, seed, m) for n in sorted(c29.NS, reverse=True) for s in c29.SCENES for seed in c29.SEEDS
            for m in MODES if n < 1000 or seed <= 2]
    out = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_rand_one, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            v = {x: y for x, y in r["violations"].items() if not (r["mode"] == "leak" and x == "S5")}
            log(f"{k:3d}/{len(jobs)} {r['scene']} n={r['n']:4d} s={r['seed']} {r['mode']:4s} {r['secs']:6.1f}s "
                f"viol={v}{' +S5' if 'S5' in r['violations'] else ''} miss={r['ev'].get('miss', 0)}/"
                f"{r['ev'].get('miss_pairs', 0)} stranded={r['ev'].get('stranded', 0)}")
            if k % 10 == 0 or k == len(jobs):
                save("random.json", out)
    log("random done", len(out))


# ---------------------------------------------------------------------------------------------- the rules
DUP_MAX, EXCH_MAX, IDLE_MAX = 0.10, 100.0, 100.0
MISS_TOL, P99_TOL = 0.001, 1.0


def run_gate():
    g = {}
    rnd = json.loads((OUT / "random.json").read_text())
    by = {(r["scene"], r["n"], r["seed"], r["mode"]): r for r in rnd}
    sub_viol = [(k, r["violations"]) for k, r in by.items() if k[3] == "sub" and r["violations"]]
    leak_s5 = sum(1 for k, r in by.items() if k[3] == "leak" and r["violations"].get("S5"))
    leak_other = [(k, {x: y for x, y in r["violations"].items() if x != "S5"}) for k, r in by.items()
                  if k[3] == "leak" and set(r["violations"]) - {"S5"}]
    deliv, strand = [], []
    for k, r in by.items():
        if k[3] != "sub" or (k[0], k[1], k[2], "leak") not in by:
            continue
        L = by[(k[0], k[1], k[2], "leak")]
        e, el = r["ev"], L["ev"]
        ok = (e.get("miss", 0) <= el.get("miss", 0) + MISS_TOL * max(e.get("miss_pairs", 0), 1)
              and (r["live_p99_s"] or 0) <= (L["live_p99_s"] or 0) + P99_TOL)
        if not ok:
            deliv.append({"cell": k[:3], "miss": (e.get("miss"), el.get("miss")), "pairs": e.get("miss_pairs"),
                          "p99": (r["live_p99_s"], L["live_p99_s"])})
        sok = e.get("stranded", 0) <= el.get("stranded", 0) + MISS_TOL * max(e.get("stranded_pairs", 0), 1)
        if not sok:
            strand.append({"cell": k[:3], "stranded": (e.get("stranded"), el.get("stranded")),
                           "pairs": e.get("stranded_pairs"), "judged": k[0] in ("C-a", "C-c")})
    g["layer1"] = {"runs": len(rnd), "sub_violations": sub_viol, "leak_S5_runs": leak_s5,
                   "leak_other_violations": leak_other, "delivery_fail": deliv,
                   "stranded_fail_judged": [x for x in strand if x["judged"]],
                   "stranded_report_C_b_etc": [x for x in strand if not x["judged"]]}
    cost = json.loads((OUT / "cost.json").read_text())
    idle = json.loads((OUT / "idle.json").read_text())
    sub = [r for r in cost if r["variant"] == "sub"]
    g["cost"] = {"dup_fail": [(r["scene"], r["n"], r["seed"], r["dup_ratio"]) for r in sub if r["dup_ratio"] > DUP_MAX],
                 "exch_fail": [(r["scene"], r["n"], r["seed"], r["exch_kb_box_day"]) for r in sub
                               if r["exch_kb_box_day"] > EXCH_MAX],
                 "exch_max_sub": max(r["exch_kb_box_day"] for r in sub),
                 "idle_max_sub": max(r["idle_kb_box_day"] for r in idle if r["variant"] == "sub"),
                 "idle_fail": [r for r in idle if r["variant"] == "sub" and r["idle_kb_box_day"] > IDLE_MAX]}
    table = []
    for s in ("C-a", "C-b", "C-c"):
        for n in (10, 30, 100, 300, 1000):
            c = {r["variant"]: r for r in cost if r["scene"] == s and r["n"] == n and r["seed"] == 1}
            if len(c) == 2:
                table.append({"scene": s, "n": n,
                              **{f"{m}_{x}": round(c[m][y], 2) if isinstance(c[m][y], float) else c[m][y]
                                 for m in MODES for x, y in (("exch_kb", "exch_kb_box_day"), ("verify", "verify_per_box_day"),
                                                             ("p99_s", "live_p99_s"), ("miss", "miss"))},
                              "verify_sub_over_leak": round(c["sub"]["verify_per_box_day"] / c["leak"]["verify_per_box_day"], 3)
                              if c["leak"]["verify_per_box_day"] else None})
    g["table_seed1"] = table
    holes = json.loads((OUT / "holes.json").read_text())
    g["layer2"] = {"runs": len(holes), "not_as_predicted": [(h["scene"], h["mode"]) for h in holes if h["as_predicted"] is False]}
    mc = json.loads((OUT / "mc.json").read_text())
    g["layer3a"] = [{"mutate": m["mutate"], "states": m["states"], "violations": m["violations"]} for m in mc]
    g["adopt"] = (not sub_viol and not deliv and not g["layer1"]["stranded_fail_judged"] and not g["cost"]["dup_fail"]
                  and not g["cost"]["exch_fail"] and not g["cost"]["idle_fail"] and not g["layer2"]["not_as_predicted"]
                  and not mc[0]["violations"])
    save("gate.json", g)
    print(json.dumps({k: v for k, v in g.items() if k != "table_seed1"}, indent=1, default=str)[:4000])
    for row in table:
        print(row)


if __name__ == "__main__":
    {"regress": run_regress, "holes": run_holes, "mc": run_mc, "cost": run_cost, "idle": run_idle,
     "random": run_random, "gate": run_gate}[sys.argv[1] if len(sys.argv) > 1 else "holes"]()
