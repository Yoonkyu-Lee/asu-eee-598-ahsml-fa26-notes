"""CP18 measurements on the laptop CPU (no GPU, no NPU):
  1. X5': self-retrieval on CP3's corpus (20k posts, 1,000 title queries): CLIP text tower FP32 vs MiniLM FP32.
  2. Determinism of the CLIP image tower: FP32, dynamic INT8 (CP17), static INT8, across batch size, thread count
     and implementation (onnxruntime CPU vs OpenVINO CPU).

  .venv/Scripts/python.exe src/bench_cp18.py      -> results/cp18/cp18_bench.json
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
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp1 import load_posts, text_of  # noqa: E402
from sim import ml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp18"
CLIP = ROOT / "models" / "clip-vit-base-patch32"
CACHE = ROOT / "data" / "cp3_cache"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ 1. X5'

class ClipText:
    def __init__(self, threads: int = 0):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(CLIP / "onnx" / "text_model.onnx"), o, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(CLIP / "tokenizer.json"))
        self.tok.enable_truncation(77)
        self.pad = self.tok.token_to_id("<|endoftext|>")

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            ids = self.ids(texts[i:i + batch])
            L = max(len(x) for x in ids)
            a = np.array([x + [self.pad] * (L - len(x)) for x in ids], dtype=np.int64)
            out.append(self.s.run(None, {"input_ids": a})[0])
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def self_retrieval(s: np.ndarray, mask: np.ndarray) -> dict:
    """Same as bench_cp3: query i is the title of corpus post i; rank of the post itself."""
    d = s[np.arange(len(s)), np.arange(len(s))]
    rank = (s > d[:, None]).sum(1)[mask]
    return {"hit1": float((rank == 0).mean()), "hit10": float((rank < 10).mean()), "mrr": float((1 / (rank + 1)).mean()),
            "n": int(mask.sum())}


def x5() -> dict:
    timing_posts = load_posts(1000)                       # same draws as CP2/CP3
    corpus_posts = load_posts(20_000, skip={p["_i"] for p in timing_posts})
    corpus = [text_of(p) for p in corpus_posts]
    queries = [p["title"] for p in corpus_posts[:1000]]
    body = np.array([len(p["body"].strip()) > 0 for p in corpus_posts[:1000]])
    z = np.load(CACHE / "fp32.npz")                        # MiniLM FP32 embeddings cached by CP3
    s0 = z["q"] @ z["c"].T
    ct = ClipText()
    # padding check: a padded batch must give the same vectors as one text at a time
    few = queries[:8]
    pad_diff = float(np.abs(ct.encode(few, 8) - np.concatenate([ct.encode([t], 1) for t in few])).max())
    log("clip text padding max abs diff", pad_diff)
    lens = np.array([len(x) for x in ct.ids(corpus)])
    trunc_ok = all(x[-1] == ct.pad for x in ct.ids(corpus[:200]))
    q = ct.encode(queries)
    c = ct.encode(corpus)
    s1 = q @ c.T
    long_body = lens[:1000] >= 77
    res = {"minilm_fp32": {"all": self_retrieval(s0, np.ones(1000, bool)), "with_body": self_retrieval(s0, body),
                           "long_77": self_retrieval(s0, long_body)},
           "clip_text_fp32": {"all": self_retrieval(s1, np.ones(1000, bool)), "with_body": self_retrieval(s1, body),
                              "long_77": self_retrieval(s1, long_body)},
           "clip_tokens": {"truncated_share": float((lens >= 77).mean()), "p50": float(np.median(lens)),
                           "eos_kept_after_truncation": bool(trunc_ok)},
           "padding_max_abs_diff": pad_diff}
    d = res["clip_text_fp32"]["all"]["hit10"] - res["minilm_fp32"]["all"]["hit10"]
    res["rule"] = {"hit10_diff": d, "pick": "M2" if d >= -0.005 else "M1"}
    return res


# ------------------------------------------------------------------ 2. determinism

def static_int8() -> Path:
    out = CLIP / "onnx" / "vision_model_int8_static.onnx"
    if out.exists():
        return out
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            x = ml.synth_images(range(20_000, 20_100)).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            self.it = iter([{"pixel_values": ((xi - ml.MEAN) / ml.STD)[None]} for xi in x])

        def get_next(self):
            return next(self.it, None)

    import onnx
    from onnx import version_converter
    up = CLIP / "onnx" / "vision_model_op13.onnx"     # per-channel DequantizeLinear needs opset >= 13 (file is 11)
    if not up.exists():
        onnx.save(version_converter.convert_version(onnx.load(str(CLIP / "onnx" / "vision_model.onnx")), 13), str(up))
    quantize_static(str(up), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    return out


def ort_run(path: Path, x: np.ndarray, batch: int, threads: int) -> np.ndarray:
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"pixel_values": x[i:i + batch]})[0] for i in range(0, len(x), batch)])


def ov_run(path: Path, x: np.ndarray, batch: int) -> np.ndarray:
    import openvino as ov
    core = ov.Core()
    m = core.read_model(str(path))
    m.reshape({"pixel_values": [-1, 3, 224, 224]})
    cm = core.compile_model(m, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    return np.concatenate([cm(x[i:i + batch])[0] for i in range(0, len(x), batch)])


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    na = a / np.linalg.norm(a, axis=1, keepdims=True)
    nb = b / np.linalg.norm(b, axis=1, keepdims=True)
    return {"bit_exact": bool(np.array_equal(a, b)), "max_abs": float(np.abs(a - b).max()),
            "cos_min": float((na * nb).sum(1).min())}


def determinism(n: int = 48) -> dict:
    imgs = ml.synth_images(range(30_000, 30_000 + n))
    x = ((imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - ml.MEAN) / ml.STD).astype(np.float32)
    paths = {"fp32": CLIP / "onnx" / "vision_model.onnx", "int8_dynamic": ml.quantize_clip(), "int8_static": static_int8()}
    ref32 = ort_run(paths["fp32"], x, 16, 0)
    res = {"n_images": n}
    for name, p in paths.items():
        base = ort_run(p, x, 16, 0)
        r = {"batch_1_vs_16": compare(base, ort_run(p, x, 1, 0)),
             "threads_1_vs_all": compare(base, ort_run(p, x, 16, 1)),
             "rerun": compare(base, ort_run(p, x, 16, 0)),
             "cos_to_fp32": compare(ref32, base)["cos_min"]}
        try:
            ov = ov_run(p, x, 16)
            r["ort_vs_openvino"] = compare(base, ov)
            r["openvino_batch_1_vs_16"] = compare(ov, ov_run(p, x, 1))
        except Exception as e:
            r["ort_vs_openvino"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        res[name] = r
        log(name, r)
    s = res["int8_static"]
    axes = [s["batch_1_vs_16"]["bit_exact"], s["threads_1_vs_all"]["bit_exact"],
            s.get("ort_vs_openvino", {}).get("bit_exact", False)]
    res["rule"] = {"axes_bit_exact": axes, "I3c_target": all(axes),
                   "static_cos_to_fp32_min": s["cos_to_fp32"]}
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    res["determinism"] = determinism()
    log("D", res["determinism"]["rule"])
    res["x5"] = x5()
    log("X5", res["x5"]["rule"], res["x5"]["minilm_fp32"]["all"], res["x5"]["clip_text_fp32"]["all"])
    (OUT / "cp18_bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("wrote")


if __name__ == "__main__":
    main()
