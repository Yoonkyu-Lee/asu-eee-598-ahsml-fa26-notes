"""CP1 benchmark: per-stage cost of one ingest and one query on this laptop's CPU.

Run from project/:  .venv/Scripts/python.exe src/bench_cp1.py
Writes results/cp1/cp1_results.json. Numbers are preliminary unless the power preflight in cp/CP1.md was confirmed.
"""
from __future__ import annotations

import gzip
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parent))
from node import DIM, MAX_TOKENS, Encoder, ExactIndex, decrypt, seal, verify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "reddit" / "reddit_title_text_2010.jsonl.gz"
OUT = ROOT / "results" / "cp1"
rng = random.Random(0)


def load_posts(n: int, skip: set[int] | None = None) -> list[dict]:
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    idx = [i for i in range(len(rows)) if not skip or i not in skip]
    pick = rng.sample(idx, n)
    return [dict(rows[i], _i=i) for i in pick]


def text_of(p: dict) -> str:
    return p["title"] + "\n\n" + p["body"]


def stats(xs_ns: list[int]) -> dict:
    a = np.array(xs_ns, dtype=np.float64) / 1e6  # ms
    return {"n": len(a), "mean_ms": a.mean(), "p50_ms": np.percentile(a, 50), "p99_ms": np.percentile(a, 99)}


def now() -> int:
    return time.perf_counter_ns()


def bench_ingest(posts: list[dict], threads: int, warmup: int = 30) -> dict:
    enc = Encoder("int8", threads=threads)
    room_key = os.urandom(32)
    authors = [Ed25519PrivateKey.generate() for _ in range(50)]
    envs = [seal(rng.choice(authors), room_key, text_of(p)) for p in posts]  # done by authors, not timed
    index = ExactIndex(len(envs))
    t = {k: [] for k in ["verify", "decrypt", "tokenize", "encode", "insert", "total"]}
    for i, env in enumerate(envs):
        t0 = now(); verify(env)
        t1 = now(); body = decrypt(env, room_key)
        t2 = now(); feed = enc.tokenize([body])
        t3 = now(); v = enc.encode_tokens(feed)[0]
        t4 = now(); index.insert(v)
        t5 = now()
        if i < warmup:
            continue
        for k, a, b in [("verify", t0, t1), ("decrypt", t1, t2), ("tokenize", t2, t3), ("encode", t3, t4), ("insert", t4, t5), ("total", t0, t5)]:
            t[k].append(b - a)
    return {k: stats(v) for k, v in t.items()}


def bench_batched_encode(texts: list[str], threads: int, batch: int) -> dict:
    enc = Encoder("int8", threads=threads)
    enc.encode(texts[:batch])  # warmup
    t0 = now()
    for i in range(0, len(texts), batch):
        enc.encode(texts[i : i + batch])
    dt = (now() - t0) / 1e6
    return {"threads": threads, "batch": batch, "posts": len(texts), "ms_per_post": dt / len(texts)}


def bench_query(titles: list[str], sizes: list[int], reps: int = 200) -> dict:
    enc = Encoder("int8", threads=1)
    qt = []
    for i, ttl in enumerate(titles):
        t0 = now(); q = enc.encode([ttl])[0]; dt = now() - t0
        if i >= 20:
            qt.append(dt)
    out = {"encode_query_1thread": stats(qt), "search": {}}
    gen = np.random.default_rng(0)
    for n in sizes:  # random unit vectors: scan time does not depend on content
        idx = ExactIndex(n)
        idx.vecs[:] = gen.standard_normal((n, DIM), dtype=np.float32)
        idx.vecs /= np.linalg.norm(idx.vecs, axis=1, keepdims=True)
        idx.n = n
        qs = gen.standard_normal((reps + 10, DIM), dtype=np.float32)
        ts = []
        for j, q in enumerate(qs):
            t0 = now(); idx.search(q / np.linalg.norm(q)); dt = now() - t0
            if j >= 10:
                ts.append(dt)
        out["search"][str(n)] = dict(stats(ts), index_MB=idx.vecs.nbytes / 1e6)
        del idx
    return out


def bench_recall(corpus: list[dict], n_queries: int = 300, k: int = 10) -> dict:
    texts = [text_of(p) for p in corpus]
    queries = [p["title"] for p in corpus[:n_queries]]
    res = {}
    emb = {}
    for variant in ["fp32", "int8"]:
        enc = Encoder(variant, threads=0)
        c = np.concatenate([enc.encode(texts[i : i + 32]) for i in range(0, len(texts), 32)])
        q = enc.encode(queries)
        emb[variant] = (c, q)
    gt = np.argsort(-(emb["fp32"][0] @ emb["fp32"][1].T), axis=0)[:k].T
    got = np.argsort(-(emb["int8"][0] @ emb["int8"][1].T), axis=0)[:k].T
    rec = [len(set(a) & set(b)) / k for a, b in zip(gt, got)]
    res = {"corpus": len(texts), "queries": n_queries, "recall_at_10_mean": float(np.mean(rec)),
           "recall_at_10_min": float(np.min(rec)), "frac_queries_below_0.9": float(np.mean(np.array(rec) < 0.9))}
    return res


def token_lengths(posts: list[dict]) -> dict:
    enc = Encoder("int8", threads=1)
    enc.tok.no_truncation()
    lens = np.array([len(enc.tok.encode(text_of(p)).ids) for p in posts])
    return {"n": len(lens), "p50": float(np.percentile(lens, 50)), "p90": float(np.percentile(lens, 90)),
            "p99": float(np.percentile(lens, 99)), "mean": float(lens.mean()),
            "frac_truncated_at_256": float(np.mean(lens > MAX_TOKENS))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    posts = load_posts(1500)
    corpus = load_posts(5000, skip={p["_i"] for p in posts})
    r = {"env": {"cpu": platform.processor(), "python": platform.python_version(), "onnxruntime": ort.__version__,
                 "numpy": np.__version__, "logical_cpus": os.cpu_count()}}
    print("token lengths ..."); r["token_lengths"] = token_lengths(posts)
    print("ingest, 1 thread ..."); r["ingest_1thread"] = bench_ingest(posts[:1030], threads=1)
    print("ingest, all threads ..."); r["ingest_allthreads"] = bench_ingest(posts[:1030], threads=0)
    texts = [text_of(p) for p in posts[:1024]]
    print("batched encode ...")
    r["batched_encode"] = [bench_batched_encode(texts, th, b) for th in (1, 0) for b in (8, 32)]
    print("query ..."); r["query"] = bench_query([p["title"] for p in posts[:320]], [10_000, 100_000, 1_000_000])
    print("recall ..."); r["recall"] = bench_recall(corpus)
    (OUT / "cp1_results.json").write_text(json.dumps(r, indent=2, default=float))
    print(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
