"""CP3 benchmark: which quality contract is meaningful, and the lowest precision that keeps it.

Run from project/:  .venv/Scripts/python.exe src/bench_cp3.py [--speed]
Without --speed only the quality part runs (no timing, so no power preflight needed).
Writes results/cp3/cp3_results.json. Mixed-precision models go to models/variants/ and
embeddings are cached in data/cp3_cache/ (both git-ignored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp1 import load_posts, text_of  # noqa: E402
from bench_cp2 import FP32, HF_INT8, VAR, embed_all, run  # noqa: E402
from hw_probe import core_classes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp3"
CACHE = ROOT / "data" / "cp3_cache"
PRE = VAR / "model_pre.onnx"
RECALL_TARGET = 0.95
LAYERS = range(6)
# Weight MatMuls of one MiniLM layer. Dynamic INT8 quantizes exactly these 36; QK^T and AV stay FP32.
KINDS = {"qkv": ["attention/self/query", "attention/self/key", "attention/self/value"],
         "attn_out": ["attention/output/dense"], "ffn_in": ["intermediate/dense"], "ffn_out": ["output/dense"]}
KIND_MACS = {"qkv": 3 * 384 * 384, "attn_out": 384 * 384, "ffn_in": 384 * 1536, "ffn_out": 1536 * 384}
LAYER_MACS = sum(KIND_MACS.values())


def node_names(layers, kinds) -> list[str]:
    return [f"/encoder/layer.{l}/{k}/MatMul" for l in layers for kind in kinds for k in KINDS[kind]]


def fp32_mac_share(layers, kinds) -> float:
    """Share of weight-MatMul MACs left in FP32 (per token, independent of length)."""
    return len(layers) * sum(KIND_MACS[k] for k in kinds) / (6 * LAYER_MACS)


def mixed(layers, kinds) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    tag = "L" + "".join(map(str, sorted(layers))) + "_" + "-".join(sorted(kinds))
    p = VAR / f"int8_pc_keep_{tag}.onnx"
    if not p.exists():
        quantize_dynamic(str(PRE), str(p), per_channel=True, weight_type=QuantType.QInt8,
                         nodes_to_exclude=node_names(layers, kinds))
    return p


def embeddings(name: str, path: Path, corpus: list[str], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return z["c"], z["q"]
    c, q = embed_all(path, corpus), embed_all(path, queries)
    np.savez(f, c=c, q=q)
    return c, q


def topk(s: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(s, idx, 1), axis=1), 1)


def self_retrieval(s: np.ndarray, mask: np.ndarray) -> dict:
    """Query i is the title of corpus post i. Rank of the post itself."""
    own = s[np.arange(len(s)), np.arange(len(s))]
    rank = (s > own[:, None]).sum(1)[mask]
    return {"hit1": float((rank == 0).mean()), "hit10": float((rank < 10).mean()), "mrr": float((1 / (rank + 1)).mean())}


def metrics(s0: np.ndarray, s: np.ndarray, s_asym: np.ndarray, body: np.ndarray) -> dict:
    """s0: FP32 scores (queries x corpus), s: variant scores, s_asym: FP32 query vs variant corpus."""
    gt = topk(s0, 11)
    gt10 = gt[:, :10]
    tenth = s0[np.arange(len(s0)), gt[:, 9]]
    got100 = topk(s, 100)
    got10 = got100[:, :10]
    rec = np.array([len(set(a) & set(b)) / 10 for a, b in zip(gt10, got10)])
    r = {"recall_at_10": float(rec.mean()), "recall_p10": float(np.percentile(rec, 10)),
         "top1_agree": float((gt[:, 0] == got10[:, 0]).mean())}
    # Tie-tolerant recall: a returned post counts if FP32 scores it within delta of FP32's own 10th result.
    got_fp32_score = np.take_along_axis(s0, got10, 1)
    for d in (0.005, 0.01, 0.02):
        r[f"tolerant_recall_d{d}"] = float((got_fp32_score >= tenth[:, None] - d).mean())
    # Rerank: FP32 re-encodes the variant's top-K, so the result equals FP32's top-10 within those K.
    for k in (20, 50, 100):
        r[f"recall_10_at_{k}"] = float(np.mean([len(set(a) & set(b[:k])) / 10 for a, b in zip(gt10, got100)]))
    asym = topk(s_asym, 10)
    r["recall_asym_fp32_query"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(gt10, asym)]))
    r["self_all"] = self_retrieval(s, np.ones(len(s), bool))
    r["self_with_body"] = self_retrieval(s, body)
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "cp3_results.json"
    res = json.loads(out_file.read_text()) if out_file.exists() else {}
    res.setdefault("quality", {})

    def save():
        out_file.write_text(json.dumps(res, indent=2, default=float))

    # Same draws as CP2 (module-level rng in bench_cp1): timing set first, then the 20k corpus.
    timing_posts = load_posts(1000)
    corpus_posts = load_posts(20_000, skip={p["_i"] for p in timing_posts})
    timing = [text_of(p) for p in timing_posts]
    corpus = [text_of(p) for p in corpus_posts]
    queries = [p["title"] for p in corpus_posts[:1000]]
    body = np.array([len(p["body"].strip()) > 0 for p in corpus_posts[:1000]])
    res["queries_with_body"] = int(body.sum())

    c0, q0 = embeddings("fp32", FP32, corpus, queries)
    s0 = q0 @ c0.T
    gt = topk(s0, 11)
    rows = np.arange(len(s0))
    res["fp32_score_gaps"] = {
        "median_rank10_minus_rank11": float(np.median(s0[rows, gt[:, 9]] - s0[rows, gt[:, 10]])),
        "median_rank1_minus_rank10": float(np.median(s0[rows, gt[:, 0]] - s0[rows, gt[:, 9]])),
        "median_rank10_score": float(np.median(s0[rows, gt[:, 9]]))}
    res["fp32_self"] = {"all": self_retrieval(s0, np.ones(len(s0), bool)), "with_body": self_retrieval(s0, body)}

    made: dict[str, Path] = {}

    def evaluate(name: str, path: Path, share: float) -> dict:
        made[name] = path
        if name not in res["quality"]:
            c, q = embeddings(name, path, corpus, queries)
            res["quality"][name] = {"fp32_mac_share": share, **metrics(s0, q @ c.T, q0 @ c.T, body)}
            save()
        m = res["quality"][name]
        print(f"{name:40s} share={share:.2f} R@10={m['recall_at_10']:.3f} tol.01={m['tolerant_recall_d0.01']:.3f} "
              f"self@10={m['self_all']['hit10']:.3f} asym={m['recall_asym_fp32_query']:.3f}", flush=True)
        return m

    print("contracts ...", flush=True)
    evaluate("fp16", VAR / "fp16.onnx", 1.0)
    evaluate("int8_dyn_hf", HF_INT8, 0.0)
    base = evaluate("int8_pc", VAR / "int8_dyn_perchannel.onnx", 0.0)["recall_at_10"]

    print("sensitivity: keep one layer / one kind in FP32 ...", flush=True)
    all_kinds = list(KINDS)
    gain_layer = {l: evaluate(f"keep_L{l}", mixed([l], all_kinds), fp32_mac_share([l], all_kinds))["recall_at_10"] - base
                  for l in LAYERS}
    gain_kind = {k: evaluate(f"keep_{k}", mixed(LAYERS, [k]), fp32_mac_share(LAYERS, [k]))["recall_at_10"] - base
                 for k in all_kinds}
    res["gain_layer"], res["gain_kind"] = gain_layer, gain_kind

    print("cumulative: add FP32 groups in order of gain until the target ...", flush=True)
    res["cumulative"] = {"layer": [], "kind": []}
    order = sorted(LAYERS, key=lambda l: -gain_layer[l])
    for n in range(2, 6):
        ls = order[:n]
        name = "keep_L" + "".join(map(str, sorted(ls)))
        res["cumulative"]["layer"].append(name)
        if evaluate(name, mixed(ls, all_kinds), fp32_mac_share(ls, all_kinds))["recall_at_10"] >= RECALL_TARGET:
            break
    order = sorted(all_kinds, key=lambda k: -gain_kind[k])
    for n in range(2, 4):
        ks = order[:n]
        name = "keep_" + "+".join(sorted(ks))
        res["cumulative"]["kind"].append(name)
        if evaluate(name, mixed(LAYERS, ks), fp32_mac_share(LAYERS, ks))["recall_at_10"] >= RECALL_TARGET:
            break
    save()

    # Speed and energy: same conditions as the CP2 baseline (P-core 1 thread; all cores, bucketed batch 8).
    passing = [n for n, q in res["quality"].items() if n.startswith("keep") and q["recall_at_10"] >= RECALL_TARGET]
    cheapest = min(passing, key=lambda n: res["quality"][n]["fp32_mac_share"]) if passing else None
    res["cheapest_passing_mixed"] = cheapest
    cls = core_classes()
    p_cores = cls[max(cls)]
    paths = {"fp32": FP32, "int8_pc": VAR / "int8_dyn_perchannel.onnx"}
    if cheapest:
        paths[cheapest] = made[cheapest]
    save()
    if "--speed" not in sys.argv:
        return
    print("speed and energy ...", flush=True)
    res["speed"] = {}
    for name, path in paths.items():
        res["speed"][name] = {"p_core_1thread": run(path, timing, p_cores, 1),
                              "all_cores_bucket8": run(path, timing, None, 0, 8, True)}
        print(name, {k: round(v["ms_per_post"], 2) for k, v in res["speed"][name].items()}, flush=True)
        save()
    # FP32 rerank at query time: re-encode one unsorted batch of 20 candidates per query.
    res["speed"]["fp32"]["all_cores_batch20"] = run(FP32, timing, None, 0, 20, False)
    save()


if __name__ == "__main__":
    main()
