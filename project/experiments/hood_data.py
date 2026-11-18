"""CP39: the content world. Reddit 2010 posts with their subreddit as a hidden label, embedded by MiniLM (FP32 and the
HF dynamic INT8 variant), for the semantic-neighbourhood model (sim/hood.py).

  .venv/Scripts/python.exe -W ignore experiments/hood_data.py         -> data/neighborhoods/posts.npz (git-ignored)
  .venv/Scripts/python.exe -W ignore experiments/hood_data.py mpnet   -> adds "mpnet" (all-mpnet-base-v2, 768-d, CP39 A)

Large crowds: the 30 largest subreddits after dropping catch-alls, at most 800 posts each. Small crowds: 30 subreddits
with 50-200 posts, drawn with a fixed seed. Text = title + body, cut by the tokenizer at 256 tokens (MAX_TOKENS).
"""
from __future__ import annotations

import collections
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.node import MAX_TOKENS, Encoder  # noqa: E402

SRC = ROOT / "data" / "reddit" / "reddit_title_text_2010.jsonl.gz"
OUT = ROOT / "data" / "neighborhoods" / "posts.npz"
CATCH_ALL = {"AskReddit", "reddit.com", "self", "IAmA", "DoesAnybodyElse", "funny", "WTF", "circlejerk", "pics",
             "answers", "Favors", "tipofmytongue", "shittyadvice", "needadvice", "todayilearned", "secretsanta"}
LARGE, LARGE_CAP, SMALL, SMALL_RANGE, SEED = 30, 800, 30, (50, 200), 39
MPNET = ROOT / "models" / "all-mpnet-base-v2"       # HF sentence-transformers/all-mpnet-base-v2, onnx/model.onnx


def select():
    rows = collections.defaultdict(list)
    with gzip.open(SRC, "rt", encoding="utf-8") as g:
        for line in g:
            r = json.loads(line)
            rows[r["subreddit"]].append((r["title"] or "") + "\n" + (r["body"] or ""))
    rng = random.Random(SEED)
    by_size = sorted(rows, key=lambda s: -len(rows[s]))
    large = [s for s in by_size if s not in CATCH_ALL][:LARGE]
    small_pool = sorted(s for s in rows if SMALL_RANGE[0] <= len(rows[s]) <= SMALL_RANGE[1] and s not in CATCH_ALL)
    small = rng.sample(small_pool, SMALL)
    texts, labels = [], []
    for s in large + small:
        posts = rows[s] if len(rows[s]) <= LARGE_CAP else rng.sample(rows[s], LARGE_CAP)
        texts += posts
        labels += [s] * len(posts)
    return texts, labels, large, small


def embed(enc, texts, dim):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))     # length buckets: less padding
    emb = np.zeros((len(texts), dim), np.float32)
    t0 = time.time()
    for k in range(0, len(order), 64):
        idx = order[k:k + 64]
        emb[idx] = enc.encode([texts[i] for i in idx])
    print(len(texts), "posts", dim, "d", round(time.time() - t0, 1), "s", flush=True)
    return emb


def add_mpnet():
    """Same posts, same order, same token cut (MAX_TOKENS); the larger encoder for CP39 A (c)."""
    from tokenizers import Tokenizer
    texts, labels, _, _ = select()
    z = dict(np.load(OUT))
    names = [str(x) for x in z["names"]]
    assert (np.array([names.index(x) for x in labels]) == z["label"]).all()
    enc = Encoder(str(MPNET / "onnx" / "model.onnx"), threads=0)
    enc.tok = Tokenizer.from_file(str(MPNET / "tokenizer.json"))
    enc.tok.enable_truncation(MAX_TOKENS)
    z["mpnet"] = embed(enc, texts, 768)
    np.savez_compressed(OUT, **z)
    print("wrote mpnet into", OUT)


def main():
    texts, labels, large, small = select()
    out = {v: embed(Encoder(v, threads=0), texts, 384) for v in ("fp32", "int8")}
    names = sorted(set(labels))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, fp32=out["fp32"], int8=out["int8"], label=np.array([names.index(x) for x in labels]),
                        names=np.array(names), large=np.array(large), small=np.array(small))
    print("wrote", OUT, {"subs": len(names), "large": len(large), "small": len(small)})


if __name__ == "__main__":
    add_mpnet() if sys.argv[1:] == ["mpnet"] else main()
