"""CP20 measurements on the laptop CPU (no GPU, no NPU):
  1. Photo quality of the CLIP image tower (COCO val2017; first 300 images calibrate, the other 4,700 evaluate):
     FP32, dynamic INT8 (CP17), static INT8 with synthetic calibration (CP18), static INT8 with photo calibration,
     and, if rule P needs it, static INT8 with the first and last blocks left in float (mixed).
  2. MiniLM cut at 128 tokens (the Hailo-8L input) on CP3's self-retrieval.
  3. One-thread CPU encoder times (MiniLM INT8 by length, CLIP image), x 2.93 later for a Pi 5 core (추정).

  ../.venv/Scripts/python.exe -I src/prep_coco.py    (once, Pillow)
  .venv/Scripts/python.exe src/bench_cp20.py         -> results/cp20/cp20_bench.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp1 import load_posts, text_of  # noqa: E402
from bench_cp18 import ClipText, self_retrieval  # noqa: E402
from node import Encoder  # noqa: E402
from sim import ml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp20"
ONNX = ROOT / "models" / "clip-vit-base-patch32" / "onnx"
COCO = ROOT / "data" / "coco"
N_CAL = 300
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def pixels(imgs: np.ndarray) -> np.ndarray:
    return ((imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - ml.MEAN) / ml.STD).astype(np.float32)


def encode(path: Path, imgs: np.ndarray, batch: int = 32, threads: int = 0) -> np.ndarray:
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    out = [s.run(None, {"pixel_values": pixels(imgs[i:i + batch])})[0] for i in range(0, len(imgs), batch)]
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


ATTN = ("/self_attn/MatMul", "/self_attn/MatMul_1")     # QK^T and AV: FP16 on the chip, not INT8


def gemm_nodes(blocks: tuple = ()) -> tuple[list[str], list[str]]:
    """(attention matmuls, linear nodes of the given blocks) by name."""
    import onnx
    m = onnx.load(str(ONNX / "vision_model_op13.onnx"), load_external_data=False)
    names = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Conv")]
    attn = [n for n in names if n.endswith(ATTN)]
    edge = [n for n in names if n not in attn and any(f"/{b}/" in n for b in blocks)]
    return attn, edge


def static_photo(cal: np.ndarray, name: str, *, chip_scope: bool, float_blocks: tuple = ()) -> Path:
    """Static QDQ INT8 calibrated on photos. chip_scope: only the linear layers (patch Conv, q/k/v/out, fc1/fc2,
    projection) as on the chip's INT8 array; attention matmuls, adds, norms stay float (the chip's FP16 units)."""
    out = ONNX / f"{name}.onnx"
    if out.exists():
        return out
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"pixel_values": pixels(cal[i:i + 1])} for i in range(len(cal))])

        def get_next(self):
            return next(self.it, None)
    kw = {}
    if chip_scope:
        attn, edge = gemm_nodes(float_blocks)
        kw = {"op_types_to_quantize": ["MatMul", "Conv"], "nodes_to_exclude": attn + edge}
    quantize_static(str(ONNX / "vision_model_op13.onnx"), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, **kw)
    return out


def fp16_weights() -> Path:
    """Weights rounded to FP16 (activations stay FP32): an indication for an FP16 datapath, not the full effect."""
    out = ONNX / "vision_model_fp16w.onnx"
    if out.exists():
        return out
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(ONNX / "vision_model.onnx"))
    for t in m.graph.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype == np.float32:
            t.CopyFrom(numpy_helper.from_array(a.astype(np.float16).astype(np.float32), t.name))
    onnx.save(m, str(out))
    return out


def top10(q: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-10 image ids per query (unordered) and the rank of each query's own image."""
    tops, ranks = [], []
    for i in range(0, len(q), 2048):
        s = q[i:i + 2048] @ e.T
        tops.append(np.argpartition(-s, 10, axis=1)[:, :10])
        ranks.append(s)
    return np.concatenate(tops), ranks


def quality(name: str, e: np.ndarray, ref: np.ndarray, q: np.ndarray, owner: np.ndarray, top_ref: np.ndarray) -> dict:
    top, chunks = top10(q, e)
    rank = np.concatenate([(s > s[np.arange(len(s)), owner[i * 2048:i * 2048 + len(s)]][:, None]).sum(1)
                           for i, s in enumerate(chunks)])
    agree = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top.tolist(), top_ref.tolist())]))
    same = (e * ref).sum(1)
    return {"variant": name, "recall@1": float((rank < 1).mean()), "recall@5": float((rank < 5).mean()),
            "recall@10": float((rank < 10).mean()), "neighbor_recall@10_vs_fp32": agree,
            "same_image_cos_min": float(same.min()), "same_image_cos_p001": float(np.percentile(same, 0.1)),
            "same_image_cos_p50": float(np.median(same))}


def photos() -> dict:
    imgs = np.load(COCO / "val2017_224.npy", mmap_mode="r")
    idx = json.loads((COCO / "val2017_index.json").read_text(encoding="utf-8"))
    cal, ev = np.asarray(imgs[:N_CAL]), imgs[N_CAL:]
    caps = idx["captions"][N_CAL:]
    q_text = [c for cs in caps for c in cs]
    owner = np.array([i for i, cs in enumerate(caps) for _ in cs])
    ct = ClipText()
    q = ct.encode(q_text, 256)
    log("captions", len(q_text))
    variants = {"fp32": ONNX / "vision_model.onnx", "int8_dynamic": ml.quantize_clip(),
                "int8_static_synthetic": ONNX / "vision_model_int8_static.onnx",
                "int8_static_photo_all_ops": static_photo(cal, "vision_model_int8_static_photo", chip_scope=False),
                "int8_static_photo": static_photo(cal, "vision_model_int8_static_photo_gemm", chip_scope=True),
                "fp16_weights": fp16_weights()}
    emb = {}
    for k, p in variants.items():
        emb[k] = encode(p, ev)
        log("encoded", k)
    ref = emb["fp32"]
    top_ref = top10(q, ref)[0]
    res = {"n_eval_images": len(ev), "n_captions": len(q_text), "n_calibration": N_CAL, "rows": []}
    for k in variants:
        res["rows"].append(quality(k, emb[k], ref, q, owner, top_ref))
        log(res["rows"][-1])
    cross = ref @ ref.T
    np.fill_diagonal(cross, -1)
    res["fp32_different_images_cos_max"] = float(cross.max())
    res["fp32_different_images_cos_p99"] = float(np.percentile(cross[cross > -1], 99))

    def passes(r):
        f = res["rows"][0]
        return r["neighbor_recall@10_vs_fp32"] >= 0.95 and r["recall@10"] >= f["recall@10"] - 0.01
    order = ["int8_static_photo"]
    pick = "int8_static_photo" if passes(next(r for r in res["rows"] if r["variant"] == "int8_static_photo")) else None
    if pick is None:
        pm = static_photo(cal, "vision_model_int8_static_photo_gemm_mixed", chip_scope=True,
                          float_blocks=("embeddings", "layers.0", "layers.11"))
        e = encode(pm, ev)
        r = quality("int8_static_photo_mixed", e, ref, q, owner, top_ref)
        r["float_nodes"] = len(gemm_nodes(("embeddings", "layers.0", "layers.11"))[1])
        res["rows"].append(r)
        log(r)
        order.append("int8_static_photo_mixed")
        pick = "int8_static_photo_mixed" if passes(r) else "fp16"
    res["rule_P"] = {"order": order + ["fp16"], "pass": {r["variant"]: passes(r) for r in res["rows"]}, "pick": pick}
    return res


def minilm_128() -> dict:
    timing = load_posts(1000)
    corpus_posts = load_posts(20_000, skip={p["_i"] for p in timing})
    corpus = [text_of(p) for p in corpus_posts]
    queries = [p["title"] for p in corpus_posts[:1000]]
    z = np.load(ROOT / "data" / "cp3_cache" / "fp32.npz")
    full = self_retrieval(z["q"] @ z["c"].T, np.ones(1000, bool))
    enc = Encoder("fp32", threads=0)
    enc.tok.enable_truncation(128)
    c = np.concatenate([enc.encode(corpus[i:i + 64]) for i in range(0, len(corpus), 64)])
    qq = np.concatenate([enc.encode(queries[i:i + 64]) for i in range(0, len(queries), 64)])
    cut = self_retrieval(qq @ c.T, np.ones(1000, bool))
    lens = np.array([len(e.ids) for e in Encoder("fp32").tok.encode_batch(corpus)])
    return {"tokens_256": full, "tokens_128": cut, "hit10_diff": cut["hit10"] - full["hit10"],
            "share_over_128": float((lens > 128).mean()), "same_quality": cut["hit10"] >= full["hit10"] - 0.005}


def cpu_times() -> dict:
    """예비 측정 (power preflight not confirmed): one thread, median of repeats."""
    def med(fn, n):
        fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts))
    enc = Encoder("int8", threads=1)
    rng = np.random.default_rng(20)
    text = {}
    for L in (12, 32, 64, 128, 192, 256):
        ids = rng.integers(1000, 30000, (1, L)).astype(np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if "token_type_ids" in enc.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        text[L] = med(lambda: enc.session.run(None, feed), 30)
    img = {}
    x = pixels(ml.synth_images([1]))
    for k, p in (("fp32", ONNX / "vision_model.onnx"), ("int8_static_photo", ONNX / "vision_model_int8_static_photo_gemm.onnx"),
                 ("int8_static_photo_mixed", ONNX / "vision_model_int8_static_photo_gemm_mixed.onnx")):
        if not p.exists():
            continue
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        s = ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])
        img[k] = med(lambda: s.run(None, {"pixel_values": x}), 10)
    return {"minilm_int8_s_by_tokens": text, "clip_image_s": img, "threads": 1, "label": "예비 측정 (laptop 185H)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cp20_bench.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k, fn in (("photos", photos), ("minilm_128", minilm_128), ("cpu_times", cpu_times)):
        if k not in res:
            res[k] = fn()
            log(k, json.dumps(res[k], default=float, ensure_ascii=True)[:2000])
            path.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    log("rule P", res["photos"]["rule_P"])


if __name__ == "__main__":
    main()
