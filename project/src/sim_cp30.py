"""CP30: squeeze the sync cost (K fetch dedupe + retry, F floor only for changed pairs, P peer discovery) and verify
again in layers (cp/CP30.md has the rules, written before results).

  .venv/Scripts/python.exe -W ignore src/sim_cp30.py holes    # layer 2: CP29 scenes + new ones -> results/cp30/holes.json
  .venv/Scripts/python.exe -W ignore src/sim_cp30.py mc       # layer 3a: clean flags + mutants  -> results/cp30/mc.json
  .venv/Scripts/python.exe -W ignore src/sim_cp30.py cost     # CP26 scenes, gate (i)-(iii)       -> results/cp30/cost.json
  .venv/Scripts/python.exe -W ignore src/sim_cp30.py idle     # gate 2: no posts, no faults       -> results/cp30/idle.json
  .venv/Scripts/python.exe -W ignore src/sim_cp30.py traffic  # info: 10 posts a box a day        -> results/cp30/traffic.json
  .venv/Scripts/python.exe -W ignore src/sim_cp30.py random   # layer 1: S2+KE and S2+KF'E         -> results/cp30/random.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sim.sync import DAY, ENV, HDR, Sync, SyncScene  # noqa: E402
import sim_cp29 as c29  # noqa: E402

OUT = ROOT / "results" / "cp30"
T0 = time.time()
WORLD = {"split_rng": True}                       # every CP30 run, the S2 baseline too (Codex 1st review)
VAR = {"S2": {}, "S2+K": {"dedupe": True}, "S2+F": {"floor_change": True}, "S2+P": {"px": True},
       "S2c": {"dedupe": True, "floor_change": True, "px": True},
       # after the cost grid: P1 made live delivery worse in C-b n = 1000, so by the rule (cp/CP30.md) P1 -> P2
       "S2+P2": {"cand_probe": (60.0, 3600.0)},
       "S2+KF": {"dedupe": True, "floor_change": True},
       "S2+KF+P2": {"dedupe": True, "floor_change": True, "cand_probe": (60.0, 3600.0)},
       # after cost2 (labelled): F left boxes with forgotten one-sided peers cut off -> F' ("heard")
       "S2+KF'": {"dedupe": True, "floor_change": "heard"},
       "S2+KF'+P2": {"dedupe": True, "floor_change": "heard", "cand_probe": (60.0, 3600.0)},
       # after layer 2 (labelled): S2+K kept H11 (a long partition never heals) -> E, exploration on its own
       "S2+KE": {"dedupe": True, "explore": True},
       "S2+KF'E": {"dedupe": True, "floor_change": "heard", "explore": True}}
MISS_AT = (DAY, 1.5 * DAY, 2.0 * DAY - 400.0)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def scene(n, v, **kw):
    if v == "S0":
        return SyncScene(n=n, variant="S0", floor=False, **WORLD, **kw)
    return SyncScene(n=n, variant="S2", floor=True, **WORLD, **VAR[v], **kw)


# ---------------------------------------------------------------------------------------------- layer 2: directed
def new_scenes():
    """name, n, rows, checks, predicted failures (variants), overrides. Predictions: cp/CP30.md (before results)."""
    S = []
    T = SyncScene().t_bound

    # K: the owner leg of X fails (links to box 0 cut while the answers are on their way) while the session that
    # skipped X is still running; box 0 must get X again without waiting for the floor
    def k_hook(s):
        orig = s._serve_at
        done = {}

        def serve(y, x, keys, state, silent=False):
            X = s.tag.get("X")
            if not done and x == 0 and X is not None and any(k[1] == X.mid for k in keys):
                done["t"] = s.eng.t
                for q in (1, 2):
                    s.link_down.add(s.pair(0, q))
                s.eng.after(1.0, lambda: [s.link_down.discard(s.pair(0, q)) for q in (1, 2)])
            return orig(y, x, keys, state, silent=silent)
        s._serve_at = serve
        s.k_cut = done

    def at(t_check, tag, box, key):
        def f(s):
            s.__dict__.setdefault("seen", {})[key] = s.holds(box, tag)
        return (t_check, "call", f)

    S.append(dict(name="K owner leg fails, skipped session still running", n=3,
                  rows=[(0, "on", 0), (5, "pub", 0, "post", "Y"), (6, "off", 0), (10, "on", 1), (10, "on", 2),
                        (20, "pub", 1, "post", "X"), (99, "call", k_hook), (100, "on", 0),
                        at(130, "X", 0, "x30"), (100 + T + 60, "until")],
                  checks=[("box0 holds X within 30 s", lambda s: s.seen.get("x30", False)),
                          ("box0 holds X within t_bound", lambda s: s.holds(0, "X"))],
                  fail={"S2", "S2+F", "S2+P"}, fail_check={"box0 holds X within 30 s"}))

    # F: box 0 takes W from box 2 while it is clean towards box 1; the push 0->1 (and 2->1) is lost silently
    S.append(dict(name="F third-party change, push to Y silently lost", n=3,
                  rows=[(0, "on", 0), (0, "on", 1), (0, "on", 2), (3000, "force", 2, 1, "pass"),
                        (3000, "force", 2, 1, "silent"), (3000, "force", 0, 1, "pass"), (3000, "force", 0, 1, "silent"),
                        (3001, "pub", 2, "post", "W"), (3001 + T + 60, "until")],
                  checks=[("box1 holds W", lambda s: s.holds(1, "W"))], fail=set()))

    # F: box 1 restarts within a second (nobody notices), then a silent loss towards it
    S.append(dict(name="F peer restarts unnoticed, then a silent loss", n=3,
                  rows=[(0, "on", 0), (0, "on", 1), (0, "on", 2), (4000, "off", 1), (4001, "on", 1),
                        (4100, "force", 2, 1, "pass"), (4100, "force", 2, 1, "silent"),
                        (4100, "force", 0, 1, "pass"), (4100, "force", 0, 1, "silent"),
                        (4200, "pub", 2, "post", "W"), (4200 + T + 60, "until")],
                  checks=[("box1 holds W", lambda s: s.holds(1, "W"))], fail=set()))

    # F: clean, then a quiet day: floor sessions on day 2
    def mark(s):
        s.day1 = sum(B.c["recon_f"] for B in s.boxes)
    S.append(dict(name="F clean, then a quiet day (floor sessions day 2 <= 3)", n=3,
                  rows=[(0, "on", 0), (0, "on", 1), (0, "on", 2), (100, "pub", 0, "post", "X"),
                        (DAY, "call", mark), (2 * DAY, "until")],
                  checks=[("floor sessions on day 2 <= 3",
                           lambda s: sum(B.c["recon_f"] for B in s.boxes) - s.day1 <= 3)],
                  report=lambda s: {"floor_sessions_day2": sum(B.c["recon_f"] for B in s.boxes) - s.day1,
                                    "floor_skips": sum(B.c["floor_skip"] for B in s.boxes)},
                  fail={"S2", "S2+K", "S2+P", "S2+KF'", "S2+KF'+P2", "S2+KE", "S2+KF'E"}))

    # P: only the always-on box is on when a laptop comes on (twice)
    def hellos(s):
        s.h1 = s.boxes[5].c["tx_probe"]
    S.append(dict(name="P laptop comes on, only the always-on box is on", n=10,
                  rows=[(0, "on", 0), (50, "pub", 0, "post", "X"), (100, "on", 5), at(160, "X", 5, "x"),
                        (400, "off", 5), (450, "pub", 0, "post", "Z"), (999, "call", hellos), (1000, "on", 5),
                        (1060, "call", lambda s: setattr(s, "h2", s.boxes[5].c["tx_probe"] - s.h1)),
                        at(1060, "Z", 5, "z"), (1100, "until")],
                  checks=[("box5 holds X within 60 s", lambda s: s.seen.get("x", False)),
                          ("box5 holds Z within 60 s of coming on again", lambda s: s.seen.get("z", False))],
                  report=lambda s: {"probe_bytes_second_come_on": s.h2}, fail=set()))

    # P: a 2 h partition with D = 3 (each side can fill its mesh), then healing; a post on one side must cross
    S.append(dict(name="P 2 h partition (D=3), heal, post crosses within 2 h", n=8, over={"D": 3, "D_HI": 6},
                  rows=[(0, "on", b) for b in range(8)] + [(100, "part", [0, 1, 2, 3]), (7300, "heal"),
                                                            (7400, "pub", 0, "post", "X"), (7400 + 7200, "until")],
                  checks=[("boxes 4-7 hold X", lambda s: all(s.holds(b, "X") for b in (4, 5, 6, 7)))],
                  fail={"S2", "S2+K", "S2+F", "S2+KF'", "S2+KF'+P2"}))
    return S


def run_holes():
    out = []
    names = list(VAR)
    for sc_def in c29.scene_list():                    # CP29 scenes: S2 and S2c must pass as predicted there
        for v in ("S2", "S2c"):
            sc = scene(sc_def["n"], v, days=10.0, **c29.QUIET)
            sim = Sync(sc, script=list(sc_def["rows"]))
            r = sim.run()
            checks = {lab: bool(fn(sim)) for lab, fn in sc_def["checks"]}
            passed = all(checks.values()) and not r["violations"]
            pred_fail = "S2" in sc_def["fail"] or any(f.startswith("S2:") for f in sc_def["fail"])
            if sc_def.get("report_only"):
                pred_fail = not passed
            out.append({"scene": "CP29: " + sc_def["name"], "variant": v, "pass": passed, "predicted_fail": pred_fail,
                        "as_predicted": passed != pred_fail, "checks": checks, "violations": r["violations"],
                        "report": sc_def["report"](sim) if "report" in sc_def else None})
            log(f"{sc_def['name'][:50]:50s} {v:5s} {'PASS' if passed else 'FAIL'}"
                f"{'' if passed != pred_fail else '  <-- NOT AS PREDICTED'} {checks}")
    for inject in (None, 3 * c29.SIG):                 # tombstone invariant
        for v in ("S2", "S2c"):
            rows, lo, hi = c29.tomb_scene(inject)
            sim = Sync(scene(2, v, days=10.0, **c29.QUIET), script=rows)
            found: list = []
            sim.eng.at(lo, c29.tomb_monitor(sim, 1.0, lo, hi, found), sim)
            r = sim.run()
            passed = not found and not r["violations"]
            pred_fail = bool(inject)
            label = "CP29: tombstone invariant" + (" (injected 3 sigma)" if inject else " (+-sigma)")
            out.append({"scene": label, "variant": v, "pass": passed, "predicted_fail": pred_fail,
                        "as_predicted": passed != pred_fail, "breaks": len(found), "violations": r["violations"]})
            log(f"{label:50s} {v:5s} {'PASS' if passed else 'FAIL'}{'' if passed != pred_fail else '  <-- NOT AS PREDICTED'}")
    for sc_def in new_scenes():
        for v in names:
            kw = dict(c29.QUIET)
            kw.update(sc_def.get("over", {}))
            sim = Sync(scene(sc_def["n"], v, days=10.0, **kw), script=list(sc_def["rows"]))
            r = sim.run()
            checks = {lab: bool(fn(sim)) for lab, fn in sc_def["checks"]}
            fc = sc_def.get("fail_check")
            passed = all(checks.values()) and not r["violations"]
            pred_fail = v in sc_def["fail"]
            if v in ("S2+P2", "S2+KF", "S2+KF+P2"):         # added after the cost grid; no prediction written
                as_pred_note = "no prediction"
            else:
                as_pred_note = ""
            if fc:                                     # the predicted failure is on these checks only
                ok_other = all(val for lab, val in checks.items() if lab not in fc) and not r["violations"]
                as_pred = ok_other and (all(checks[l] for l in fc) != pred_fail)
            else:
                as_pred = passed != pred_fail
            rep = sc_def["report"](sim) if "report" in sc_def else None
            if as_pred_note:
                as_pred = None
            out.append({"scene": sc_def["name"], "variant": v, "pass": passed, "predicted_fail": pred_fail,
                        "as_predicted": as_pred, "checks": checks, "violations": r["violations"], "report": rep,
                        "triggers": r["triggers"]})
            log(f"{sc_def['name'][:50]:50s} {v:5s} {'PASS' if passed else 'FAIL'}"
                f"{'' if as_pred or as_pred is None else '  <-- NOT AS PREDICTED'} {checks} {rep or ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "holes.json").write_text(json.dumps(out, indent=1, default=str))
    bad = [o for o in out if o["as_predicted"] is False]
    log(f"holes: {len(out)} runs, {len(bad)} not as predicted")


# ---------------------------------------------------------------------------------------------- layer 3a
def run_mc():
    from sim.sync_mc import check, check_clean
    out = []
    for m in (None, "keep_on_publish", "keep_on_adversary", "set_on_difference"):
        r = check_clean(mutate=m)
        out.append(r)
        log(f"mc clean flags mutate={m}: {r['states']} states, {r['transitions']} transitions, "
            f"truncated={r['truncated']}, violations={r['violations']}")
    for mode in ("S12", "S0"):                         # CP29 regression (keyset knows P3 now)
        r = check(mode)
        out.append(r)
        log(f"mc {mode}: {r['states']} states, violations={r['violations']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mc.json").write_text(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------- cost (gate)
def metrics(r, n, days):
    tx, w = r["tx"], r["work"]
    bd = n * days
    need_env = w["needed_env"]
    dup_b = w["dup_env"] * (ENV + HDR)
    exch = tx.get("recon", 0) + tx.get("fetch", 0) + tx.get("probe", 0) + dup_b
    sent = r["sent_bytes"]
    return {"needed_env": need_env, "dup_env": w["dup_env"], "dup_ratio": w["dup_env"] / need_env if need_env else 0.0,
            "exch_kb_box_day": exch / bd / 1000, "recon_kb": tx.get("recon", 0) / bd / 1000,
            "probe_kb": tx.get("probe", 0) / bd / 1000, "fetch_kb": tx.get("fetch", 0) / bd / 1000,
            "push_ctl_kb": (tx.get("ihave", 0) + tx.get("iwant", 0)) / bd / 1000,
            "over_over_needed": r["over_over_needed"], "sent_kb_box_day": sent / bd / 1000,
            "miss": r["ev"].get("miss", 0), "miss_pairs": r["ev"].get("miss_pairs", 0), "live_p99_s": r["live_p99_s"],
            "violations": r["violations"], "world_trace": r["world_trace"],
            "world_trace_times": r["world_trace_times"], "floor_skip": w["floor_skip"],
            "triggers": r["triggers"]}


def _cost_one(job):
    scene_name, n, v, seed, extra = job
    t = time.time()
    kw = dict(c29.SCENES[scene_name])
    kw.update(c29.CP26_FAULTS)
    kw.update(extra)
    sc = scene(n, v, seed=seed, days=2.0, miss_at=MISS_AT, **kw)
    r = Sync(sc).run()
    out = {"scene": scene_name, "n": n, "variant": v, "seed": seed, "extra": extra, **metrics(r, n, 2.0),
           "secs": round(time.time() - t, 1)}
    return out


def _pool(jobs, fn, name, workers=14):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(fn, j) for j in jobs]):
            r = f.result()
            out.append(r)
            log(f"{name} {r['scene']} n={r['n']:4d} s={r['seed']} {r['variant']:5s} dup={r['dup_ratio']:.3f} "
                f"exch={r['exch_kb_box_day']:6.1f} KB miss={r['miss']}/{r['miss_pairs']} p99={r['live_p99_s']} "
                f"viol={r['violations']} {r['secs']}s")
    return out


def run_cost():
    jobs = [(s, n, v, 1, {}) for n in (1000, 300, 100, 30, 10) for s in ("C-a", "C-b", "C-c")
            for v in ("S0",) + tuple(VAR)]
    jobs += [(s, n, v, seed, {}) for n in (100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in ("S2", "S2c")
             for seed in (2, 3)]
    out = _pool(jobs, _cost_one, "cost")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost.json").write_text(json.dumps(out, indent=1, default=str))


def run_cost2():
    """After the cost grid (labelled in cp/CP30.md): S2+P2, S2+KF, S2+KF+P2 and S2 again (for the trace check)."""
    vs = ("S2", "S2+P2", "S2+KF", "S2+KF+P2")
    jobs = [(s, n, v, 1, {}) for n in (1000, 300, 100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in vs]
    jobs += [(s, n, v, seed, {}) for n in (100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in ("S2", "S2+KF+P2")
             for seed in (2, 3)]
    out = _pool(jobs, _cost_one, "cost2")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost2.json").write_text(json.dumps(out, indent=1, default=str))


def run_cost3():
    """After cost2 (labelled in cp/CP30.md): F' variants; S2 again for the trace and miss baseline."""
    vs = ("S2", "S2+KF'", "S2+KF'+P2")
    jobs = [(s, n, v, 1, {}) for n in (1000, 300, 100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in vs]
    jobs += [(s, n, v, seed, {}) for n in (100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in vs
             for seed in (2, 3, 4, 5)]
    out = _pool(jobs, _cost_one, "cost3")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost3.json").write_text(json.dumps(out, indent=1, default=str))


def run_cost4():
    """After layer 2 (labelled): E (exploration on its own)."""
    vs = ("S2+KE", "S2+KF'E")
    jobs = [(s, n, v, 1, {}) for n in (1000, 300, 100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in vs]
    jobs += [(s, n, v, seed, {}) for n in (100, 30, 10) for s in ("C-a", "C-b", "C-c") for v in vs
             for seed in (2, 3, 4, 5)]
    out = _pool(jobs, _cost_one, "cost4")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost4.json").write_text(json.dumps(out, indent=1, default=str))


def run_traffic():
    jobs = [(s, 100, v, 1, {"msgs_per_day": mpd}) for s in ("C-a", "C-b") for v in ("S0", "S2", "S2c")
            for mpd in (1.0, 10.0)]
    out = _pool(jobs, _cost_one, "traffic")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traffic.json").write_text(json.dumps(out, indent=1, default=str))


def _idle_one(job):
    n, v = job
    sc = scene(n, v, seed=1, days=1.0, always=1.0, msgs_per_day=0.0, p_fail=0.0, p_silent=0.0, link_drops_per_day=0.0,
               partitions_per_day=0.0, invalid_frac=0.0, on_out_per_day=0.0)
    r = Sync(sc).run()
    tx = r["tx"]
    per = [sum(B.c.get(k, 0) for k in ("tx_recon", "tx_probe")) / 1000 for B in []]
    return {"n": n, "variant": v, "idle_kb_box_day": (tx.get("recon", 0) + tx.get("probe", 0)) / n / 1000,
            "recon_kb": tx.get("recon", 0) / n / 1000, "probe_kb": tx.get("probe", 0) / n / 1000,
            "violations": r["violations"]}


def run_idle():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, v) for n in (10, 30, 100, 300) for v in ("S2", "S2c", "S2+KF", "S2+KF+P2", "S2+K", "S2+KF'", "S2+KF'+P2", "S2+KE", "S2+KF'E")]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(_idle_one, jobs))
    for r in out:
        log(f"idle n={r['n']:3d} {r['variant']:4s} {r['idle_kb_box_day']:.1f} KB/box-day "
            f"(recon {r['recon_kb']:.1f}, probe {r['probe_kb']:.1f}) viol={r['violations']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "idle.json").write_text(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------- layer 1
def _rand_one(job):
    scene_name, n, seed, v = job
    t = time.time()
    sc = scene(n, v, seed=seed, days=3.0, miss_at=(DAY, 2 * DAY, 3 * DAY - 400.0),
               **c29.SCENES[scene_name], **c29.FAULTS)
    r = Sync(sc).run()
    r.update(scene=scene_name, name=v, secs=round(time.time() - t, 1))
    return r


def run_random(ns=c29.NS, seeds=c29.SEEDS, workers=14, variants=("S2+KE", "S2+KF'E")):
    """Layer 1 for the default (S2+KE) and the option (S2+KF'E); CP29's grid and faults."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(s, n, seed, v) for n in sorted(ns, reverse=True) for s in c29.SCENES for seed in seeds for v in variants
            if n < 1000 or seed <= 2]
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_rand_one, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            log(f"{k:3d}/{len(jobs)} {r['scene']} n={r['n']:4d} s={r['seed']} {r['name']:8s} {r['secs']:6.1f}s "
                f"viol={r['violations']} miss={r['ev'].get('miss', 0)}/{r['ev'].get('miss_pairs', 0)}")
            if k % 10 == 0 or k == len(jobs):
                (OUT / "random.json").write_text(json.dumps(out, default=str))
    log("random done", len(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "holes"
    if cmd == "random":
        ns = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else c29.NS
        run_random(ns)
    else:
        {"holes": run_holes, "mc": run_mc, "cost": run_cost, "cost2": run_cost2, "cost3": run_cost3, "cost4": run_cost4, "idle": run_idle, "traffic": run_traffic}[cmd]()
