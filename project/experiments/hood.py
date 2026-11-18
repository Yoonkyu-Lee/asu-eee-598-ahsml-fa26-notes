"""CP39 R2-R5 (journal/cp/CP39.md, rules written before the runs): the semantic-neighbourhood model on Reddit posts.

  .venv/Scripts/python.exe -W ignore experiments/hood.py all     -> results/neighborhoods/r2_r5.json
Needs data/neighborhoods/posts.npz (experiments/hood_data.py). Every number is an estimate (model).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.hood import DAY, Hood, HoodScene, unit  # noqa: E402

DATA = ROOT / "data" / "neighborhoods" / "posts.npz"
OUT = ROOT / "results" / "neighborhoods" / "r2_r5.json"
SEEDS = (1, 2, 3)
T0 = time.time()


def world(variant="fp32"):
    z = np.load(DATA)
    names = [str(x) for x in z["names"]]
    small = {names.index(str(s)) for s in z["small"]}
    return {"emb": unit(z[variant].astype(np.float64)), "label": z["label"], "names": names, "small_ids": sorted(small)}


W = {}


def _w():
    if not W:
        W.update(world())
    return W


def _run(sc: HoodScene):
    t = time.time()
    r = Hood(sc, _w()).run()
    r["secs"] = round(time.time() - t, 1)
    r["scene"] = {k: v for k, v in sc.__dict__.items()}
    return r


def _travel(sc: HoodScene):
    """R3: one box per seed picks up a new interest at day 1 after its owner reads 5 posts of it (a friend's links)."""
    w = _w()
    h = Hood(sc, w)
    rng = np.random.default_rng(sc.seed + 100)
    out = []

    def start(b, s):
        h.true[b].add(s)
        for ix in rng.choice(h.by_sub[s], 5):
            h.u[b] = unit(h.u[b] + h.sc.alpha * w["emb"][ix])
        out.append((b, s, h.eng.t))
    for b in rng.choice(sc.n, 20, replace=False):
        s = int(rng.choice([x for x in range(h.n_sub) if x not in h.true[b]]))
        h.eng.at(DAY, start, int(b), s)
    for b in range(sc.n):
        h.eng.after(h.rng.exponential(DAY / sc.posts_per_box_day), h.write, b)
        h.eng.after(h.rng.uniform(0, sc.refresh_s), h.refresh, b)
    h.eng.run(sc.days * DAY)
    res = []
    for b, s, t0 in out:
        posts = [(pid, p["t"]) for pid, p in enumerate(h.posts) if p["label"] == s and p["t"] >= t0]
        got = [(pt, h.has[b].get(pid)) for pid, pt in posts]
        hours = []
        for k in range(1, int((sc.days * DAY - t0) // 3600)):
            win = [(pt, g) for pt, g in got if t0 + (k - 1) * 3600 <= pt < t0 + k * 3600]
            if win:
                hours.append((k, sum(g is not None for _, g in win) / len(win)))
        first = next((k for k, x in hours if x >= 0.5), None)
        res.append({"box": b, "sub": s, "hours_to_half": first, "posts_after": len(posts)})
    return res


def _rumour(sc: HoodScene):
    """Share of all boxes a post reaches, by its vote count at publication (votes given up front)."""
    w = _w()
    out = {}
    for v in (0, 1, 3, 7, 15, 31, 63):
        h = Hood(replace(sc, posts_per_box_day=0.0), w)
        for b in range(sc.n):
            h.eng.after(h.rng.uniform(0, sc.refresh_s), h.refresh, b)
        reach = []
        rng = np.random.default_rng(sc.seed + v)

        def pub(b, ix, s, vv=v):
            h._publish(b, int(ix), label=int(s), votes=vv)
        for k in range(30):
            b = int(rng.integers(sc.n))
            s = int(rng.choice(sorted(h.true[b])))
            h.eng.at(3600.0 + 60 * k, pub, b, rng.choice(h.by_sub[s]), s)
        h.eng.run(3600.0 + 60 * 30 + 1800.0)
        for pid, p in enumerate(h.posts):
            outside = [b for b in range(sc.n) if p["label"] not in h.true[b]]
            reach.append(sum(pid in h.has[b] for b in outside) / max(len(outside), 1))
        out[v] = float(np.mean(reach))
    return out


def run_all():
    base = HoodScene()
    jobs = {"r2_always": [replace(base, seed=s) for s in SEEDS],
            "r2_laptops": [replace(base, always=0.3, seed=s) for s in SEEDS],
            "r4_sigma_0.5": [replace(base, advertise_sigma=0.05, seed=s) for s in SEEDS],
            "r4_sigma_1": [replace(base, advertise_sigma=0.1, seed=s) for s in SEEDS]}
    sub_target = 0
    for tol in (0.0, 12.0):
        for votes in (0, 31):
            jobs[f"r5_tol{tol:g}_votes{votes}"] = [replace(base, spam_boxes=10, spam_per_day=20.0, spam_target=sub_target,
                                                           tol_deg=tol, bought_votes=votes, seed=s) for s in SEEDS]
    res = {"what": "CP39 R2-R5 (journal/cp/CP39.md)", "label": "estimate (model) on measured embeddings"}
    with ProcessPoolExecutor(14) as ex:
        futs = {k: [ex.submit(_run, sc) for sc in v] for k, v in jobs.items()}
        trav = [ex.submit(_travel, replace(base, days=2.0, seed=s)) for s in SEEDS]
        rum = [ex.submit(_rumour, replace(base, seed=s)) for s in SEEDS]
        for k, fs in futs.items():
            rs = [f.result() for f in fs]
            res[k] = {m: float(np.mean([r[m] for r in rs if r.get(m) is not None])) for m in
                      ("recall_60s", "recall_1h", "recall_small_60s", "recall_small_1h", "waste_share",
                       "receipts_per_box_day", "members_receipts_per_box_day", "flood_receipts_per_box_day",
                       "exposure_top1", "spam_reach_target", "spam_reach_all") if any(r.get(m) is not None for r in rs)}
            res[k]["secs"] = [r["secs"] for r in rs]
            print(f"[{time.time() - T0:6.0f}s]", k, {m: round(x, 3) for m, x in res[k].items() if m != 'secs'}, flush=True)
        res["r3_travel"] = [x for f in trav for x in f.result()]
        rr = [f.result() for f in rum]
        res["r3_rumour_reach_by_votes"] = {v: float(np.mean([r[v] for r in rr])) for v in rr[0]}
    h = [x["hours_to_half"] for x in res["r3_travel"]]
    res["r3_travel_summary"] = {"boxes": len(h), "never": sum(x is None for x in h),
                                "median_hours": float(np.median([x for x in h if x is not None])) if any(x is not None for x in h) else None}
    print("travel", res["r3_travel_summary"], "rumour", res["r3_rumour_reach_by_votes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    run_all()
