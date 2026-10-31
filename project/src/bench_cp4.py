"""CP4 benchmark: is existing hardware already enough? The laptop's Intel NPU vs the CPU, same encoder.

Run from project/:  .venv/Scripts/python.exe src/bench_cp4.py [--batched]
Writes results/cp4/cp4_npu.json. The NPU needs static shapes, so each post is padded up to a length
bucket (32, 64, 128, 256) and one compiled model per bucket is used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp1 import load_posts, now, stats, text_of  # noqa: E402
from bench_cp2 import BACKLOG_POSTS, FP32, VAR, idle_watts, run  # noqa: E402
from bench_cp3 import CACHE, metrics  # noqa: E402
from hw_probe import PackageEnergy, core_classes  # noqa: E402
from node import Encoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp4"
BUCKETS = (32, 64, 128, 256)


MASKED = VAR / "fp32_mask_bias.onnx"


def mask_bias_model() -> Path:
    """The export builds the additive attention mask from an int64 mask inside the graph, which this NPU
    compiler cannot lower. Feed the FP additive mask (B, 1, 1, L) directly instead; the math is unchanged."""
    if MASKED.exists():
        return MASKED
    import onnx
    from onnx.utils import Extractor
    m = onnx.load(str(FP32))
    m.graph.input.append(onnx.helper.make_tensor_value_info("mask_bias", onnx.TensorProto.FLOAT, ["B", 1, 1, "L"]))
    for n in m.graph.node:
        if n.op_type == "Add" and "/attention/self/Add" in n.name:
            n.input[1] = "mask_bias"
    inputs = [i.name for i in m.graph.input if i.name != "attention_mask"]
    onnx.save(Extractor(m).extract_model(inputs, [o.name for o in m.graph.output]), str(MASKED))
    return MASKED


class NpuEncoder:
    def __init__(self, device: str = "NPU", batch: int = 1, config: dict | None = None):
        self.batch = batch
        self.tok = Encoder("fp32", threads=1)  # only for its tokenizer
        core = ov.Core()
        self.models = {}
        for L in BUCKETS:
            m = core.read_model(str(mask_bias_model()))
            m.reshape({i.get_any_name(): [batch, 1, 1, L] if i.get_any_name() == "mask_bias" else [batch, L]
                       for i in m.inputs})
            ppp = ov.preprocess.PrePostProcessor(m)
            for i in m.inputs:  # this NPU driver's compiler rejects I64 inputs
                if i.get_any_name() != "mask_bias":
                    ppp.input(i.get_any_name()).tensor().set_element_type(ov.Type.i32)
            self.models[L] = core.compile_model(ppp.build(), device, config or {})  # NPU runs FP16 by default
        self.names = [i.get_any_name() for i in self.models[BUCKETS[0]].inputs]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def prepare(self, texts: list[str]) -> tuple[int, dict, np.ndarray]:
        """Pad rows and tokens to a compiled static shape. Returns (bucket, inputs, mask)."""
        feed = self.tok.tokenize(texts)
        b, n = feed["input_ids"].shape
        L = next(x for x in BUCKETS if x >= n)
        feed = {k: np.pad(v, ((0, self.batch - b), (0, L - n))) for k, v in feed.items()}
        mask = feed["attention_mask"][..., None].astype(np.float32)
        bias = ((1 - mask[..., 0]) * -1e4)[:, None, None, :].astype(np.float32)  # -1e4: safe in FP16
        return L, {k: bias if k == "mask_bias" else feed[k].astype(np.int32) for k in self.names}, mask

    def encode(self, texts: list[str]) -> np.ndarray:
        """Up to `batch` texts (this NPU compiler only accepts batch 1 for the attention MatMul)."""
        L, inputs, mask = self.prepare(texts)
        b = len(texts)
        hidden = self.models[L](inputs)[0]
        pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        return (pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None))[:b]


def npu_async(texts: list[str], jobs: int) -> dict:
    """Throughput mode for backlog. Batched shapes do not compile on this NPU driver, so keep `jobs`
    batch-1 requests in flight instead (OpenVINO THROUGHPUT hint), in length order."""
    enc = NpuEncoder(config={"PERFORMANCE_HINT": "THROUGHPUT"})
    order = np.argsort([len(t) for t in texts])
    prepared = [enc.prepare([texts[j]]) for j in order]  # tokenization is host work, kept out of the timing
    queues = {L: ov.AsyncInferQueue(m, jobs) for L, m in enc.models.items()}
    for L, inputs, _ in prepared[:jobs]:
        queues[L].start_async(inputs)
    for q in queues.values():
        q.wait_all()
    idle = idle_watts()
    meter = PackageEnergy(); meter.start()
    for L, inputs, _ in prepared:
        queues[L].start_async(inputs)
    for q in queues.values():
        q.wait_all()
    joules, secs = meter.stop()
    n = len(texts)
    return {"jobs": jobs, "ms_per_post": secs * 1000 / n, "pkg_W": joules / secs, "idle_W": idle,
            "J_per_post": joules / n, "active_J_per_post": max(joules - idle * secs, 0) / n,
            "backlog_min": BACKLOG_POSTS * secs / n / 60}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    timing_posts = load_posts(1000)  # same draws as CP2/CP3
    corpus_posts = load_posts(20_000, skip={p["_i"] for p in timing_posts})
    timing = [text_of(p) for p in timing_posts]
    corpus = [text_of(p) for p in corpus_posts]
    queries = [p["title"] for p in corpus_posts[:1000]]
    out = OUT / "cp4_npu.json"
    if "--batched" in sys.argv:  # added after the batch-1 run: does batching close the gap?
        res = json.loads(out.read_text())
        for jobs in (1, 4, 8):
            res[f"npu_async{jobs}_sorted"] = r = npu_async(timing, jobs)
            print(f"npu async{jobs}", {k: round(v, 3) for k, v in r.items()}, flush=True)
        out.write_text(json.dumps(res, indent=2, default=float))
        return
    res = {"device": ov.Core().get_property("NPU", "FULL_DEVICE_NAME"), "openvino": ov.__version__}

    t0 = time.time()
    enc = NpuEncoder()
    res["compile_s"] = time.time() - t0

    # Speed and energy, batch 1, the same 1,000 posts as CP2/CP3.
    for t in timing[:20]:
        enc.encode_one(t)
    idle = idle_watts()
    lat = []
    meter = PackageEnergy(); meter.start()
    for t in timing:
        a = now(); enc.encode_one(t); lat.append(now() - a)
    joules, secs = meter.stop()
    n = len(timing)
    res["npu_speed"] = {"ms_per_post": secs * 1000 / n, "latency": stats(lat), "pkg_W": joules / secs, "idle_W": idle,
                        "J_per_post": joules / n, "active_J_per_post": max(joules - idle * secs, 0) / n,
                        "backlog_min": BACKLOG_POSTS * secs / n / 60}
    print("npu", {k: v for k, v in res["npu_speed"].items() if k != "latency"}, flush=True)

    # Quality against the cached FP32 embeddings from CP3.
    z = np.load(CACHE / "fp32.npz")
    c0, q0 = z["c"], z["q"]
    c = np.stack([enc.encode_one(t) for t in corpus])
    q = np.stack([enc.encode_one(t) for t in queries])
    s0 = q0 @ c0.T
    body = np.ones(len(q), bool)
    res["npu_quality"] = metrics(s0, q @ c.T, q0 @ c.T, body)
    print("npu quality", res["npu_quality"]["recall_at_10"], flush=True)

    # The weak-device CPU baseline under the CP3 contract: INT8 on one low-power E-core.
    cls = core_classes()
    lp = [x for x in cls[min(cls)] if x >= 20]
    res["cpu_lp_e_int8"] = run(VAR / "int8_dyn_perchannel.onnx", timing, lp[:1], 1)
    print("lp-e int8", res["cpu_lp_e_int8"]["ms_per_post"], flush=True)
    out.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
