"""CP27: the algorithm audit. A: push duplicates on the exchange model. B: author embeddings instead of every box
encoding every post (B1 tolerance, B2 recall from data/cp3_cache; B3 encoder work on the CP19 chip costs).

  .venv/Scripts/python.exe -W ignore src/sim_cp27.py push     # A -> results/cp27/push.json   (CPU, about 10 min)
  .venv/Scripts/python.exe src/sim_cp27.py embed               # B1, B2 -> results/cp27/embed.json
  .venv/Scripts/python.exe src/sim_cp27.py enc                 # B3 -> results/cp27/enc.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "cp27"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- A: push duplicates
PUSH = {"P0": {"push": "eager", "D": 6}, "P1": {"push": "eager", "D": 3, "D_LO": 2},
        "P2": {"push": "lazy", "D": 6}, "P3": {"push": "tree", "D": 6}, "P4": {"push": "idontwant", "D": 6}}
SCENES = {"C-a": {"always": 1.0}, "C-b": {"always_n": 1, "always": 0.0}, "C-c": {"always": 0.5}}
SIMPLE = ("P0", "P1", "P4", "P2", "P3")          # pre-registered order, simplest first


def push() -> None:
    from sim.exchange import Exchange, Scene
    res = {"runs": []}
    for sc_name, skw in SCENES.items():
        for n in (100, 1000):
            for loss in (1e-3, 1e-2):
                for name, pkw in PUSH.items():
                    r = Exchange(Scene(n=n, p_loss=loss, **skw, **pkw), "X2").run()
                    total = 1 + r["dup_over_needed"] + r["over_over_needed"]
                    row = {"scene": sc_name, "n": n, "loss": loss, "push": name, "total_over_needed": total,
                           **{k: float(r[k]) if isinstance(r[k], (float, np.floating)) else r[k] for k in
                              ("missing_frac", "live_p50_s", "live_p99_s", "dup_over_needed", "over_over_needed",
                               "per_box_needed_B", "catch_up_p99_s", "messages")}}
                    res["runs"].append(row)
                    log(sc_name, n, loss, name, {k: round(row[k], 4) for k in
                                                 ("missing_frac", "live_p99_s", "total_over_needed")})
    ok = {}
    for name in PUSH:
        rs = [r for r in res["runs"] if r["push"] == name]
        ok[name] = {"E1": all(r["missing_frac"] <= 1e-4 for r in rs), "E2": all(r["live_p99_s"] <= 2.0 for r in rs),
                    "worst_total": max(r["total_over_needed"] for r in rs),
                    "mean_total": float(np.mean([r["total_over_needed"] for r in rs])),
                    "worst_live_p99_s": max(r["live_p99_s"] for r in rs)}
    passing = [p for p in PUSH if ok[p]["E1"] and ok[p]["E2"]]
    best = min(passing, key=lambda p: ok[p]["mean_total"]) if passing else None
    # pre-registered: within 10% of the least total bytes, take the simplest
    pick = next((p for p in SIMPLE if p in passing and ok[p]["mean_total"] <= 1.10 * ok[best]["mean_total"]), None)
    res.update({"rules": ok, "least_bytes": best, "pick": pick})
    log("rules", json.dumps(ok), "least", best, "pick", pick)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "push.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- B1, B2: author embeddings (data/cp3_cache)
PREC = ("fp32", "fp16", "int8_dyn_hf", "int8_pc")


def embed() -> None:
    """The four precisions of the same 20,000 posts (CP3) stand in for "the author's device" and "the box": the
    same model run by different kernels gives slightly different vectors."""
    C = {p: np.load(ROOT / "data" / "cp3_cache" / f"{p}.npz") for p in PREC}
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    c = {p: unit(C[p]["c"].astype(np.float64)) for p in PREC}
    q = {p: unit(C[p]["q"].astype(np.float64)) for p in PREC}
    res = {"pairs": {}, "n_posts": int(len(c["fp32"])), "n_queries": int(len(q["fp32"]))}
    lows = []
    for a in PREC:                                   # author's precision
        for b in PREC:                               # the box's precision
            if a == b:
                continue
            cs = np.sum(c[a] * c[b], axis=1)
            res["pairs"][f"{a}->{b}"] = {"cos_min": float(cs.min()), "cos_q1e-4": float(np.quantile(cs, 1e-4)),
                                         "cos_q1e-3": float(np.quantile(cs, 1e-3)), "cos_p50": float(np.median(cs))}
            lows.append(float(np.quantile(cs, 1e-4)))
    tau = float(min(lows))                           # every pair passes 99.99% of honest posts
    res["tau"] = tau
    log("tau", tau, {k: round(v["cos_q1e-4"], 5) for k, v in res["pairs"].items()})
    # B1: an attacker turns its post's true vector e towards a target query by at most arccos(tau): its best score
    # for query q is cos(max(angle(q, e) - theta, 0)). How often does a random unrelated post then reach the top 10?
    theta = float(np.arccos(tau))
    truth_c, truth_q = c["fp32"], q["fp32"]
    S = truth_q @ truth_c.T                          # 1,000 x 20,000
    s10 = np.sort(S, axis=1)[:, -10]
    rng = np.random.default_rng(27)
    qi = rng.integers(0, S.shape[0], 200_000)
    ci = rng.integers(0, S.shape[1], 200_000)
    sc = S[qi, ci]
    honest_in = sc >= s10[qi]
    phi = np.arccos(np.clip(sc, -1, 1))
    att = np.cos(np.maximum(phi - theta, 0.0))
    res["B1"] = {"theta_deg": float(np.degrees(theta)), "pairs_sampled": int(len(qi)),
                 "in_top10_honest": float(honest_in.mean()), "in_top10_with_turn": float((att >= s10[qi]).mean()),
                 "gain_x": float((att >= s10[qi]).mean() / max(honest_in.mean(), 1e-12)),
                 "score_gain_p50": float(np.median(att - sc)), "score_gain_max": float((att - sc).max()),
                 "s10_p50": float(np.median(s10))}
    log("B1", res["B1"])
    # B1 curve (added after B1): tighter tolerance needs a more exact verifier; how much leverage is left at each one
    curve = []
    for t in (0.965, 0.98, 0.99, 0.995, 0.999, 0.9995):
        th = float(np.arccos(t))
        at = np.cos(np.maximum(phi - th, 0.0))
        row = {"tau": t, "theta_deg": float(np.degrees(th)), "in_top10_with_turn": float((at >= s10[qi]).mean())}
        for a_, b_ in (("fp32", "fp16"), ("fp32", "int8_pc"), ("fp32", "int8_dyn_hf"), ("int8_pc", "int8_dyn_hf")):
            row[f"honest_pass {a_}->{b_}"] = float((np.sum(c[a_] * c[b_], axis=1) >= t).mean())
        curve.append(row)
    res["B1_curve"] = curve
    for r_ in curve:
        log("B1 curve", {k: round(v, 5) for k, v in r_.items()})
    # B2: rank with author vectors (precision a) against the box's query (precision b); truth = FP32 exact top 10
    top = lambda Q, Cc: np.argpartition(-(Q @ Cc.T), 10, axis=1)[:, :10]
    gt = top(truth_q, truth_c)
    res["B2"] = {}
    for a in PREC:
        for b in PREC:
            got = top(q[b], c[a])
            rec = np.mean([len(set(g) & set(h)) / 10 for g, h in zip(gt, got)])
            res["B2"][f"author {a}, query {b}"] = float(rec)
    log("B2", res["B2"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "embed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- B3: encoder work (chip model, CP19 / CP20)
def enc() -> None:
    """First-order chip time per box (추정): CP19's chip table (8x8 INT8 at 500 MHz: 164-token post 70.9 ms, query
    10.9 ms) and day volume (P-all: 25,889 texts, 1,440 queries), FP16 = 4.0x INT8 (CP20 image tower GEMM 0.545 s vs
    0.136 s). Recall and attacker leverage come from embed.json (B1, B2)."""
    cp19 = json.loads((ROOT / "results" / "cp24" / "rerun_gap" / "cp19" / "cp19.json").read_text(encoding="utf-8"))
    emb = json.loads((OUT / "embed.json").read_text(encoding="utf-8"))
    t8, q8 = cp19["encoder_pick"]["text_164_ms"] / 1e3, cp19["encoder_pick"]["query_ms"] / 1e3
    f16 = 0.545 / 0.136
    texts, queries = cp19["inputs"]["day"]["text"], cp19["S1"]["n_query"]
    search_s = cp19["inputs"]["host_box_s"]["search_fixed"] + cp19["inputs"]["host_box_s"]["search_per_vec"] * texts
    K2 = cp19["inputs"]["limits"]["K2_query_s"]
    curve = {round(r["tau"], 4): r for r in emb["B1_curve"]}
    honest10 = emb["B1"]["in_top10_honest"]
    B2 = emb["B2"]
    opts = {
        "E0a INT8 ingest (the chip as modelled since CP12)": dict(ingest=t8, query=q8, verify_q=0.0,
            recall=B2["author int8_pc, query int8_pc"], leverage=1.0, extra_B=0, searchable="after INT8 encode"),
        "E0a + CP3 re-rank of top 20 (FP16) at query": dict(ingest=t8, query=q8 + 20 * f16 * t8, verify_q=0.0,
            recall=0.993, leverage=1.0, extra_B=0, searchable="after INT8 encode"),
        "E0b FP16 ingest": dict(ingest=f16 * t8, query=f16 * q8, verify_q=0.0,
            recall=B2["author fp16, query fp16"], leverage=1.0, extra_B=0, searchable="after FP16 encode"),
        "E2 author FP32 vector, verify top 10 at query (FP16)": dict(ingest=0.0, query=f16 * q8, verify_q=10 * f16 * t8,
            recall=B2["author fp32, query fp16"], leverage=curve[0.9995]["in_top10_with_turn"] / honest10, extra_B=768,
            searchable="at once (author vector)"),
        "E3-int8 author FP32 vector, INT8 check when idle (tau 0.965)": dict(ingest=t8, query=f16 * q8, verify_q=0.0,
            recall=B2["author fp32, query fp16"], leverage=curve[0.965]["in_top10_with_turn"] / honest10, extra_B=768,
            searchable="at once, checked later"),
        "E3-fp16 author FP32 vector, FP16 check when idle (tau 0.9995)": dict(ingest=f16 * t8, query=f16 * q8,
            verify_q=0.0, recall=B2["author fp32, query fp16"], leverage=curve[0.9995]["in_top10_with_turn"] / honest10,
            extra_B=768, searchable="at once, checked later"),
    }
    res = {"inputs": {"t_int8_s": t8, "q_int8_s": q8, "fp16_over_int8": f16, "texts_per_day": texts,
                      "queries_per_day": queries, "search_s": search_s, "K2_s": K2}, "options": {}}
    for name, o in opts.items():
        day = texts * o["ingest"] + queries * (o["query"] + o["verify_q"])
        qlat = o["query"] + o["verify_q"] + search_s
        backlog24 = texts * o["ingest"]
        res["options"][name] = {**o, "chip_s_per_day": day, "busy_frac": day / 86400, "query_s": qlat,
                                "K2_pass": qlat <= K2, "recall_pass": o["recall"] >= 0.95,
                                "backlog_24h_chip_s": backlog24, "spam_texts_per_s_to_saturate": 1 / o["ingest"]
                                if o["ingest"] else None}
        log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["options"][name].items()
                   if k in ("chip_s_per_day", "query_s", "K2_pass", "recall", "recall_pass", "leverage",
                            "backlog_24h_chip_s", "spam_texts_per_s_to_saturate")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "enc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"push": push, "embed": embed, "enc": enc}[sys.argv[1]]()
