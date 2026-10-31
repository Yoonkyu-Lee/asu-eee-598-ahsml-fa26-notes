"""CP2 benchmark: the cheapest encoder setting that keeps Recall@10 >= 0.95, with its ms and J per post.

Run from project/:  .venv/Scripts/python.exe src/bench_cp2.py
Writes results/cp2/cp2_results.json. Model variants are generated into models/variants/ (git-ignored).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp1 import load_posts, now, stats, text_of  # noqa: E402
from hw_probe import PackageEnergy, core_classes  # noqa: E402
from node import MODEL_DIR, Encoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp2"
VAR = ROOT / "models" / "variants"
FP32 = MODEL_DIR / "onnx" / "model.onnx"
HF_INT8 = MODEL_DIR / "onnx" / "model_quint8_avx2.onnx"
RECALL_TARGET = 0.95
BACKLOG_POSTS = 1_000_000 / 3  # 8 h of a community posting 10^6 per day (assumed rate, see CP1)


# ---------------------------------------------------------------- variants

def make_variants(calib_texts: list[str]) -> dict[str, Path]:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationMethod
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.transformers.float16 import convert_float_to_float16

    VAR.mkdir(parents=True, exist_ok=True)
    pre = VAR / "model_pre.onnx"
    if not pre.exists():
        quant_pre_process(str(FP32), str(pre), skip_symbolic_shape=True)  # symbolic inference fails on this export
    out = {"fp32": FP32, "int8_dyn_hf": HF_INT8}

    p = VAR / "int8_dyn_perchannel.onnx"
    if not p.exists():
        quantize_dynamic(str(pre), str(p), per_channel=True, weight_type=QuantType.QInt8)
    out["int8_dyn_perchannel"] = p

    p = VAR / "int8_static_qdq.onnx"
    if not p.exists():
        enc = Encoder("fp32", threads=0)
        # Percentile calibration stacks activations across samples, so every sample needs the same length.
        feed = enc.tokenize(calib_texts)
        if feed["input_ids"].shape[1] < 256:
            pad = 256 - feed["input_ids"].shape[1]
            feed = {k: np.pad(v, ((0, 0), (0, pad))) for k, v in feed.items()}
        samples = [{k: v[i: i + 1] for k, v in feed.items()} for i in range(len(calib_texts))]

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(samples)

            def get_next(self):
                return next(self.it, None)

        quantize_static(str(pre), str(p), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.99})
    out["int8_static_qdq"] = p

    p = VAR / "fp16.onnx"
    if not p.exists():
        onnx.save(convert_float_to_float16(onnx.load(str(FP32)), keep_io_types=True), str(p))
    out["fp16"] = p
    return out


# ---------------------------------------------------------------- quality

def embed_all(variant: Path, texts: list[str], batch: int = 32) -> np.ndarray:
    enc = Encoder(str(variant), threads=0)
    order = np.argsort([len(t) for t in texts])  # group similar lengths to avoid padding waste
    out = np.empty((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), batch):
        ids = order[i: i + batch]
        out[ids] = enc.encode([texts[j] for j in ids])
    return out


def recall(gt_c, gt_q, c, q, k: int = 10) -> dict:
    gt = np.argsort(-(gt_c @ gt_q.T), axis=0)[:k].T
    got = np.argsort(-(c @ q.T), axis=0)[:k].T
    r = np.array([len(set(a) & set(b)) / k for a, b in zip(gt, got)])
    return {"recall_at_10_mean": float(r.mean()), "recall_at_10_p10": float(np.percentile(r, 10)),
            "frac_below_0.9": float((r < 0.9).mean())}


# ---------------------------------------------------------------- speed and energy

def idle_watts(seconds: float = 2.0) -> float:
    m = PackageEnergy(); m.start(); time.sleep(seconds); j, s = m.stop()
    return j / s


def run(variant: Path, texts: list[str], cpus: list[int] | None, threads: int, batch: int = 1, bucket: bool = False) -> dict:
    proc = psutil.Process()
    all_cpus = list(range(psutil.cpu_count()))
    proc.cpu_affinity(cpus or all_cpus)
    try:
        enc = Encoder(str(variant), threads=threads)
        for t in texts[:20]:
            enc.encode([t])
        idle = idle_watts()
        if bucket:
            order = np.argsort([len(t) for t in texts])
            groups = [[texts[j] for j in order[i: i + batch]] for i in range(0, len(texts), batch)]
        else:
            groups = [texts[i: i + batch] for i in range(0, len(texts), batch)]
        lat = []
        meter = PackageEnergy(); meter.start()
        for g in groups:
            t0 = now(); enc.encode(g); lat.append(now() - t0)
        joules, secs = meter.stop()
    finally:
        proc.cpu_affinity(all_cpus)
    n = len(texts)
    r = {"cpus": "all" if cpus is None else cpus, "threads": threads, "batch": batch, "bucket": bucket,
         "ms_per_post": secs * 1000 / n, "pkg_W": joules / secs, "idle_W": idle,
         "J_per_post": joules / n, "active_J_per_post": max(joules - idle * secs, 0) / n,
         "backlog_min": BACKLOG_POSTS * secs / n / 60}
    if batch == 1:
        r["latency"] = stats(lat)
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cls = core_classes()
    p_cores, e_cores = cls[max(cls)], [c for c in cls[min(cls)] if c < 20]  # 20, 21 are the low-power E-cores
    lp_cores = [c for c in cls[min(cls)] if c >= 20]
    timing_posts = load_posts(1000)
    corpus_posts = load_posts(20_000, skip={p["_i"] for p in timing_posts})
    calib_posts = load_posts(200, skip={p["_i"] for p in timing_posts + corpus_posts})  # disjoint from evaluation
    timing = [text_of(p) for p in timing_posts]
    corpus = [text_of(p) for p in corpus_posts]
    queries = [p["title"] for p in corpus_posts[:1000]]
    variants = make_variants([text_of(p) for p in calib_posts])
    res = {"cores": {"p": p_cores, "e": e_cores, "lp_e": lp_cores}, "quality": {}, "speed": {}, "extra": {}}

    print("quality ...")
    gt_c, gt_q = embed_all(FP32, corpus), embed_all(FP32, queries)
    for name, path in variants.items():
        if name == "fp32":
            res["quality"][name] = {"recall_at_10_mean": 1.0}
            continue
        try:
            res["quality"][name] = recall(gt_c, gt_q, embed_all(path, corpus), embed_all(path, queries))
        except Exception as e:  # e.g. a CPU kernel missing for FP16
            res["quality"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(name, res["quality"][name])

    print("speed and energy ...")
    for name, path in variants.items():
        if "error" in res["quality"][name]:
            continue
        res["speed"][name] = {"p_core_1thread": run(path, timing, p_cores, 1), "all_cores": run(path, timing, None, 0)}
        print(name, {k: round(v["ms_per_post"], 2) for k, v in res["speed"][name].items()})

    passing = [n for n, q in res["quality"].items() if q.get("recall_at_10_mean", 0) >= RECALL_TARGET]
    cheapest = min(passing, key=lambda n: res["speed"][n]["p_core_1thread"]["ms_per_post"])
    res["cheapest_passing"] = cheapest
    print("extra runs for", cheapest, "and int8_dyn_hf ...")
    for name in dict.fromkeys([cheapest, "int8_dyn_hf"]):
        path = variants[name]
        res["extra"][name] = {
            "batch32_unsorted": run(path, timing, None, 0, 32, False),
            "batch8_bucketed": run(path, timing, None, 0, 8, True),
            "batch32_bucketed": run(path, timing, None, 0, 32, True),
            "e_core_1thread": run(path, timing, e_cores[:1], 1),
            "e_cores_8threads": run(path, timing, e_cores, len(e_cores)),
            "lp_e_core_1thread": run(path, timing, lp_cores[:1], 1),
        }
    (OUT / "cp2_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
