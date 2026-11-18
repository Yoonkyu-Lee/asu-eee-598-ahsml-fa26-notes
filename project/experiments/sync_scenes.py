"""CP29: sync by set reconciliation, verified in layers (cp/CP29.md has the rules, written before results).
The scene data here (scene_list, QUIET, SCENES, NS, SEEDS, FAULTS, CP26_FAULTS, tomb_scene) is shared by the CP30,
CP33 and CP38 runners.

  .venv/Scripts/python.exe experiments/sync_scenes.py holes     # layer 2, S2 and S2e (no floor)  -> results/sync/holes.json
  .venv/Scripts/python.exe experiments/sync_scenes.py random    # layer 1, random + coverage    -> results/sync/random.json
  .venv/Scripts/python.exe experiments/sync_scenes.py props     # layer 3b, reconciler tests    -> results/sync/props.json

Retired in recast R1b: the variants S0 (the CP26 X2 box), S1 and S1e (fixed 1 h buckets), the X2-vs-S2 cost grid
(cost) and this runner's layer 3a (mc: its S2 search runs in experiments/sync.py mc, the S0 contrast search is gone).
Their results stay in results/sync/; the commands run at tag pre-r1, the CP29 originals at tag cp30.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.sync import DAY, Sync, SyncScene  # noqa: E402

OUT = ROOT / "results" / "sync"
T0 = time.time()
VARIANTS = (("S2", True), ("S2", False))


def vname(v, fl):
    return v + ("" if fl else "e")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- layer 2: directed
QUIET = dict(p_fail=0.0, p_silent=0.0, link_drops_per_day=0.0, partitions_per_day=0.0, invalid_frac=0.0, D=6)
SIG = 120.0


def tb(sc):
    return sc.t_bound


def scene_list():
    """Each scene: name, n, script builder (sc -> rows), checks [(label, fn(sim) -> bool)], predicted failures."""
    S = []
    base = SyncScene(**QUIET)
    T = base.t_bound

    # H1: one failed push, two boxes, nobody comes on afterwards
    S.append(dict(name="H1 failed push, nobody comes on", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (99, "force", 0, 1, "fail"), (100, "pub", 0, "post", "X"),
                        (100 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail=set()))
    # H2: an unfilled gap is never retried
    S.append(dict(name="H2 gap never retried", n=3,
                  rows=[(0, "on", 0), (100, "pub", 0, "post", "X"), (200, "off", 0), (300, "on", 0), (400, "off", 0),
                        (500, "on", 2), (600, "on", 1), (700, "on", 0), (700 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X")), ("box2 holds X", lambda s: s.holds(2, "X"))],
                  fail=set()))
    # H3: author clock skew beyond the 60 s margin (both within sigma)
    S.append(dict(name="H3 skew beyond the catch-up margin", n=2,
                  rows=[(0, "offset", 0, -110.0), (0, "offset", 1, 110.0), (0, "on", 0), (0, "on", 1),
                        (200, "off", 1), (210, "pub", 0, "post", "X"), (300, "on", 1), (300 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail=set()))
    # H4: a liar claims coverage
    S.append(dict(name="H4 liar claims coverage", n=3,
                  rows=[(0, "liar", 2, "withhold"), (0, "on", 2), (50, "on", 0), (60, "pub", 0, "post", "X"),
                        (100, "off", 0), (110, "on", 0), (120, "off", 0), (200, "on", 1), (300, "on", 0),
                        (300 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail=set()))
    # H5: the offer goes to one box, which leaves
    S.append(dict(name="H5 offer to one box that leaves", n=3,
                  rows=[(0, "on", 0), (10, "pub", 0, "post", "X"), (20, "off", 0), (100, "on", 2), (110, "on", 1),
                        (190, "link", 0, 1, "down"), (200, "on", 0), (200.27, "off", 2), (205, "link", 0, 1, "up"),
                        (205 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail=set()))
    # H6: deletes - tombstones forever (F1) vs resurrection after expiry
    S.append(dict(name="H6 tombstone expiry, no resurrection", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (100, "pub", 0, "post", "X"), (150, "off", 1),
                        (200, "pub", 0, "delete", "D", 0.0, "X"),
                        (DAY + 4 * SIG + 1200, "on", 1), (DAY + 4 * SIG + 1200 + T + 60, "until")],
                  checks=[("box0 never holds X again", lambda s: not s.holds(0, "X")),
                          ("box1 dropped X (expired)", lambda s: not s.holds(1, "X"))],
                  fail=set()))
    # H8: store before validate / squatting by a liar
    S.append(dict(name="H8 squatting liar, invalid first", n=3,
                  rows=[(0, "liar", 2, "squat"), (0, "on", 0), (0, "on", 1), (0, "on", 2), (99, "force", 0, 1, "fail"),
                        (100, "pub", 0, "post", "X"), (100 + T + 60, "until")],
                  checks=[("box1 holds the valid X", lambda s: s.holds(1, "X") and
                           s.boxes[1].held[s.tag["X"].mid] is s.tag["X"])], fail=set()))
    # idle silent loss: an envelope is acked and lost, then nothing happens
    S.append(dict(name="silent loss, then idle", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (99, "force", 0, 1, "pass"), (99, "force", 0, 1, "silent"),
                        (100, "pub", 0, "post", "X"),
                        (100 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail={"S2e"}))
    # a link comes back without any traffic
    S.append(dict(name="silent link return", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (50, "link", 0, 1, "down"), (100, "pub", 0, "post", "X"),
                        (400, "link", 0, 1, "up"), (400 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail=set()))
    # partition, then heal
    S.append(dict(name="partition heals", n=4,
                  rows=[(0, "on", 0), (0, "on", 1), (0, "on", 2), (0, "on", 3), (100, "part", [0, 1]),
                        (200, "pub", 0, "post", "X"), (1000, "heal"), (1000 + T + 60, "until")],
                  checks=[("box2 holds X", lambda s: s.holds(2, "X")), ("box3 holds X", lambda s: s.holds(3, "X"))],
                  fail=set()))
    # delete before post
    S.append(dict(name="delete arrives before its post", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (99, "force", 0, 1, "fail"), (100, "pub", 0, "post", "X"),
                        (101, "pub", 0, "delete", "D", 0.0, "X"), (101 + T + 60, "until")],
                  checks=[("box1 never holds X", lambda s: not s.holds(1, "X")),
                          ("box1 forgot X", lambda s: s.forgot(1, "X"))], fail=set()))
    # comment before its post (orphan) and a comment whose target is outside the band
    S.append(dict(name="comment before its post", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (99, "force", 0, 1, "fail"), (100, "pub", 0, "post", "X"),
                        (101, "pub", 0, "comment", "C", 0.0, "X"), (101 + T + 60, "until")],
                  checks=[("box1 holds X and C", lambda s: s.holds(1, "X") and s.holds(1, "C")),
                          ("C not orphan at box1", lambda s: s.tag["C"].mid not in s.boxes[1].orphan)],
                  fail=set()))
    # window edge: stamped near now + 2 sigma, receiver sigma behind
    S.append(dict(name="future edge: +2 sigma received by a box sigma behind", n=2,
                  rows=[(0, "offset", 0, SIG), (0, "offset", 1, -SIG), (0, "on", 0), (0, "on", 1),
                        (100, "pub", 0, "post", "X"), (100 + T + 60, "until")],
                  checks=[("box1 holds X", lambda s: s.holds(1, "X"))], fail=set()))
    # window edge: an item leaves the band while the link is down (no claim, no violation)
    S.append(dict(name="lower edge: item leaves the band", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (50, "link", 0, 1, "down"),
                        (100, "pub", 0, "post", "X", -(DAY - 2 * SIG - 30)), (200, "link", 0, 1, "up"),
                        (200 + T + 60, "until")],
                  checks=[("no claim on X (left B), no violation", lambda s: True)], fail=set()))
    # dishonest stamps: backdated 6 h and far future (+3 sigma)
    S.append(dict(name="dishonest stamps: -6 h and +3 sigma", n=2,
                  rows=[(0, "on", 0), (0, "on", 1), (100, "pub", 0, "post", "B", -6 * 3600.0),
                        (100, "pub", 0, "post", "F", 3 * SIG), (100 + 3 * SIG + T + 60, "until")],
                  checks=[("backdated held", lambda s: s.holds(1, "B"))], report_only=True,
                  report=lambda s: {"backdated: held by box1": s.holds(1, "B"),
                                    "far future: held by its own box": s.holds(0, "F"),
                                    "far future: held by box1": s.holds(1, "F")},
                  fail=set()))
    # foreign delete by a liar
    S.append(dict(name="foreign delete", n=3,
                  rows=[(0, "liar", 2, "squat"), (0, "on", 0), (0, "on", 1), (0, "on", 2),
                        (100, "pub", 0, "post", "X"), (200, "pub", 2, "delete", "FD", 0.0, "X", True, True),
                        (200 + T + 60, "until")],
                  checks=[("X still held by 0 and 1", lambda s: s.holds(0, "X") and s.holds(1, "X"))],
                  fail=set()))
    return S


def tomb_monitor(sim, every=10.0, start=0.0, stop=0.0, log_=None):
    """Tombstone invariant: no honest box has dropped the tombstone while another honest box still accepts the target."""
    def tick(s):
        X, D = s.tag.get("X"), s.tag.get("D")
        if X is None or D is None:
            return
        R, sg = s.sc.R, s.sc.sigma
        for x in s.boxes:
            if X.mid not in x.applied_del:
                continue
            rule_dropped = X.t0 < x.now() - R - 2 * sg
            impl_gone = D.mid not in x.held and X.mid not in x.forgotten
            if impl_gone and not rule_dropped:
                log_.append({"t": round(s.eng.t, 1), "early_drop_at": x.i})
            if not rule_dropped:
                continue
            for y in s.boxes:
                if y is x or not y.honest:
                    continue
                if y.now() - R <= X.t0 <= y.now() + 2 * sg:
                    log_.append({"t": round(s.eng.t, 1), "dropped_at": x.i, "accepts": y.i})
        if s.eng.t + every <= stop:
            s.eng.after(every, tick, s)
    return tick


def tomb_scene(inject: float | None):
    """Both boxes hold the tombstone; watch the cutoff at exact boundaries with offsets +-sigma (or an injected one)."""
    o1 = -inject if inject else -SIG
    t_x = 100.0
    lo = t_x + DAY - 4 * SIG
    hi = t_x + DAY + 6 * SIG + 1300
    rows = [(0, "offset", 0, SIG), (0, "offset", 1, o1), (0, "on", 0), (0, "on", 1),
            (t_x, "pub", 0, "post", "X"), (200, "pub", 0, "delete", "D", 0.0, "X"), (hi, "until")]
    return rows, lo, hi


def run_holes():
    out = []
    for sc_def in scene_list():
        for v, fl in VARIANTS:
            name = vname(v, fl)
            sc = SyncScene(n=sc_def["n"], floor=fl, days=10.0, **QUIET)
            sim = Sync(sc, script=list(sc_def["rows"]))
            r = sim.run()
            checks = {lab: bool(fn(sim)) for lab, fn in sc_def["checks"]}
            viol = r["violations"]
            passed = all(checks.values()) and not viol
            pred_fail = name in sc_def["fail"] or any(f.startswith(name + ":") for f in sc_def["fail"])
            rep = sc_def["report"](sim) if "report" in sc_def else None
            if sc_def.get("report_only"):
                pred_fail = not passed                     # effect reported, not judged
            out.append({"scene": sc_def["name"], "variant": name, "pass": passed, "predicted_fail": pred_fail,
                        "as_predicted": passed != pred_fail, "checks": checks, "violations": viol,
                        "report": rep, "examples": {k: v[:2] for k, v in r["examples"].items()}})
            log(f"{sc_def['name'][:46]:46s} {name:4s} {'PASS' if passed else 'FAIL'}"
                f"{'' if passed != pred_fail else '  <-- NOT AS PREDICTED'}  {checks} {viol}")
    # tombstone invariant: within the clock bound, and with an injected violation
    for inject in (None, 3 * SIG):
        for v, fl in VARIANTS:
            rows, lo, hi = tomb_scene(inject)
            sc = SyncScene(n=2, floor=fl, days=10.0, **QUIET)
            sim = Sync(sc, script=rows)
            found: list = []
            sim.eng.at(lo, tomb_monitor(sim, 1.0, lo, hi, found), sim)
            r = sim.run()
            label = "tombstone invariant" + (" (injected offset 3 sigma)" if inject else " (offsets +-sigma)")
            passed = not found and not r["violations"]
            pred_fail = bool(inject)
            out.append({"scene": label, "variant": vname(v, fl), "pass": passed, "predicted_fail": pred_fail,
                        "as_predicted": passed != pred_fail, "breaks": found[:3], "violations": r["violations"]})
            log(f"{label:46s} {vname(v, fl):4s} {'PASS' if passed else 'FAIL'}"
                f"{'' if passed != pred_fail else '  <-- NOT AS PREDICTED'}  breaks={len(found)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "holes.json").write_text(json.dumps(out, indent=1, default=str))
    bad = [o for o in out if not o["as_predicted"]]
    log(f"holes: {len(out)} runs, {len(bad)} not as predicted")


# ---------------------------------------------------------------------------------------------- layer 1: random
SCENES = {"C-a": {"always": 1.0}, "C-b": {"always_n": 1, "always": 0.0}, "C-c": {"always": 0.5},
          "C-0": {"always": 0.0}, "C-x": {"always": 1.0, "corr_at": 1.5 * DAY}}
NS = (10, 30, 100, 300, 1000)
SEEDS = (1, 2, 3, 4, 5)
FAULTS = dict(liars=0.1, invalid_frac=0.02, foreign_del_frac=0.2, backdate_frac=0.02)


def _one(job):
    scene, n, seed, v, fl = job
    t = time.time()
    sc = SyncScene(n=n, floor=fl, seed=seed, days=3.0, **SCENES[scene], **FAULTS)
    r = Sync(sc).run()
    r.update(scene=scene, name=vname(v, fl), secs=round(time.time() - t, 1))
    return r


def run_random(ns=NS, seeds=SEEDS, workers=14):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # n = 1000 (20-40 min a run): only S2, seeds 1-2 (time; the floor rule uses n <= 300)
    jobs = [(s, n, seed, v, fl) for n in sorted(ns, reverse=True) for s in SCENES for seed in seeds
            for v, fl in VARIANTS
            if n < 1000 or ((v, fl) == ("S2", True) and seed <= 2)]
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            log(f"{k:3d}/{len(jobs)} {r['scene']} n={r['n']:4d} s={r['seed']} {r['name']:4s} {r['secs']:6.1f}s "
                f"viol={r['violations']} over={r['over_over_needed']}")
            if k % 25 == 0 or k == len(jobs):
                (OUT / "random.json").write_text(json.dumps(out, default=str))
    log("random done", len(out))


# ---------------------------------------------------------------------------------------------- cost (pick rule, C1)
CP26_FAULTS = dict(p_fail=0.0, p_silent=1e-3, link_drops_per_day=0.0, partitions_per_day=0.0, invalid_frac=0.0)


# ---------------------------------------------------------------------------------------------- layer 3b: reconciler
B_FIXED, C_LOG = 256, 128          # C1 gate, cp/CP29.md (written before results)
ENV_FETCH = 1000 + 40 + 40 + 12    # one fetched envelope: request header + key + answer header + envelope


def run_props(branches=(16, 8)):
    import math
    import random
    from sim.recon import KeySet, Session, reconcile
    rng = random.Random(7)
    band = ((0, -1), (86400, -1))

    def mk(n, lo=0, hi=86400):
        return [(rng.randrange(lo, hi), rng.getrandbits(63) + 1) for _ in range(n)]

    def cases(N):
        base = mk(N)
        yield "identical", base, list(base)
        if N:
            yield "one-off", base, base[:-1]
            k = max(1, N // 100)
            yield "1% each side", base[k:] + mk(k), base[:-k] + mk(k)
            yield "many small ranges", [x for j, x in enumerate(base) if j % 50], [x for j, x in enumerate(base) if j % 50 != 7]
        yield "disjoint", base, mk(N)
        dense = [(4242, rng.getrandbits(63) + 1) for _ in range(N)]       # everything in one second
        yield "dense second", dense, dense[: max(0, N - 3)] + mk(3, 4242, 4243)
        edge = [(0, rng.getrandbits(63) + 1) for _ in range(5)] + [(86399, rng.getrandbits(63) + 1) for _ in range(5)]
        yield "band edges", base + edge, base

    rows, bad = [], []
    for N in (0, 1, 10, 100, 1000, 10000, 100000):
        for label, A, B in cases(N):
            a, b = KeySet(A), KeySet(B)
            ta, tb = set(a.keys), set(b.keys)
            ka, kb = list(a.keys), list(b.keys)
            for v in ("S2",):
                for br in branches:
                    r = reconcile(a, b, *band, branch=br, leaf=16)
                    d = len(tb - ta) + len(ta - tb)
                    exact = r["need_a"] == tb - ta and r["need_b"] == ta - tb and r["done"]
                    nn = max(len(ta), len(tb), 2)
                    total = r["bytes"] + d * ENV_FETCH
                    gate = B_FIXED + d * (1000 + C_LOG * math.log2(nn))
                    row = {"N": N, "case": label, "variant": v, "branch": br, "d": d, "exact": exact,
                           "recon_bytes": r["bytes"], "total_bytes": total, "gate": round(gate),
                           "c1": total <= gate, "rounds": r["rounds"]}
                    # interrupted at every round: partial results are true differences, sets untouched
                    for k in range(1, r["rounds"] + 1):
                        p = reconcile(a, b, *band, branch=br, leaf=16, abort_after=k)
                        if not (p["need_a"] <= tb - ta and p["need_b"] <= ta - tb):
                            bad.append(("abort wrong", N, label, v, br, k))
                    if a.keys != ka or b.keys != kb:
                        bad.append(("sets changed", N, label, v, br))
                    if not exact:
                        bad.append(("not exact", N, label, v, br))
                    rows.append(row)
    # concurrent inserts and expiries between rounds
    conc = 0
    for trial in range(300):
        N = rng.choice((10, 100, 1000, 5000))
        base = mk(N)
        a, b = KeySet(base + mk(rng.randrange(0, 20))), KeySet(base + mk(rng.randrange(0, 20)))
        seen_a, seen_b = set(a.keys), set(b.keys)
        sa, sb = Session(a), Session(b)
        msg, side = sa.initiate(*band), sb
        while msg:
            for _ in range(rng.randrange(0, 4)):                     # someone writes, something expires
                x = (rng.randrange(0, 86400), rng.getrandbits(63) + 1)
                (a if rng.random() < 0.5 else b).add(x)
                seen_a |= set(a.keys)
                seen_b |= set(b.keys)
            if rng.random() < 0.2:
                cut = rng.randrange(0, 2000)
                a.drop_before(cut)
                b.drop_before(cut)
            msg = side.process(msg)
            side = sa if side is sb else sb
        if not (sa.need <= seen_b and sb.need <= seen_a):
            bad.append(("concurrent: need not held by peer", trial))
        a.keys = sorted(set(a.keys) | sa.need & seen_b)
        a._pre = None
        r2 = reconcile(a, b, *band)                                  # a second, quiet session closes the rest
        if r2["need_a"] != set(b.keys) - set(a.keys) or r2["need_b"] != set(a.keys) - set(b.keys):
            bad.append(("concurrent: second session not exact", trial))
        conc += 1
    res = {"rows": rows, "concurrent_trials": conc, "failures": bad[:50], "n_failures": len(bad)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "props.json").write_text(json.dumps(res, indent=1, default=str))
    for v, br in tuple(("S2", b) for b in branches):
        rs = [x for x in rows if x["variant"] == v and x["branch"] == br]
        fails = [x for x in rs if not x["c1"]]
        log(f"{v} branch {br}: {len(rs)} cases, exact {sum(x['exact'] for x in rs)}, C1 pass {len(rs) - len(fails)};"
            f" C1 fails: {[(x['N'], x['case'], x['total_bytes'], x['gate']) for x in fails][:8]}")
    log(f"props: {len(bad)} failures (abort / unchanged / exact / concurrent), {conc} concurrent trials")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "holes"
    if cmd == "random":
        ns = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else NS
        run_random(ns)
    else:
        {"holes": run_holes, "props": run_props}[cmd]()
