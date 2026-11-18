"""CP5 benchmark: why does the NPU idle? Separate the hardware's limits from the software's.

Run from project/:  .venv/Scripts/python.exe src/bench_cp5.py [profile] [micro] [chain] [attn] [mask] [ablate] [int8] [fixed] [igpu]
(no argument = all parts). Each part merges its results into results/cp5/cp5.json.
  profile  per-layer task times the NPU plugin reports (PERF_COUNT), per length bucket
  micro    synthetic graphs: empty call, one GEMM per shape
  chain    synthetic graphs: the 36 weight GEMMs alone (FP16, INT8), softmax alone
  attn     synthetic graphs: attention core cut after QK^T / softmax / mask add / AV
  mask     synthetic graphs: QK^T + mask + softmax with the mask in four equivalent forms
  ablate   the full encoder with softmax removed, or rewritten as max/exp/sum/div
  int8     NNCF post-training INT8 of the full encoder on the NPU: speed, energy, quality
  fixed    the full encoder with the fix from 'mask' (per-head mask): speed, energy, quality
  igpu     the same FP16 encoder on the Arc iGPU, batch 1 and length-bucketed batch 8
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import openvino.opset13 as ops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cp1  # noqa: E402
from bench_cp1 import load_posts, now, stats, text_of  # noqa: E402
from bench_cp2 import BACKLOG_POSTS, VAR, idle_watts  # noqa: E402
from bench_cp3 import CACHE, metrics  # noqa: E402
from bench_cp4 import BUCKETS, NpuEncoder, mask_bias_model  # noqa: E402
from hw_probe import PackageEnergy  # noqa: E402
from model_cp4 import GEMMS, LAYERS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp5"
RES = OUT / "cp5.json"
# Chips and Cheese, "Intel Meteor Lake's NPU" (2024): 2 NCE tiles, 4096 INT8 MAC/cycle, FP16 at half rate, 1.16 GHz measured
NPU_GHZ, NPU_MAC_INT8, NPU_MAC_FP16 = 1.16, 4096, 2048


def save(key: str, val) -> None:
    res = json.loads(RES.read_text()) if RES.exists() else {}
    res[key] = val
    RES.write_text(json.dumps(res, indent=2, default=float))
    print(key, "saved", flush=True)


def call_ms(compiled, inputs, n: int = 100) -> float:
    """Median wall time of one synchronous call with fixed inputs."""
    req = compiled.create_infer_request()
    for _ in range(10):
        req.infer(inputs)
    t = []
    for _ in range(n):
        a = now(); req.infer(inputs); t.append(now() - a)
    return float(np.median(t)) / 1e6


def bucket_inputs(enc: NpuEncoder, L: int) -> dict:
    """A post that fills bucket L exactly (no padding), so every call does the full L-token work."""
    return enc.prepare(["word " * (L - 2)])[1]


def gemm_macs(L: int) -> int:
    return LAYERS * L * sum(k * n for _, k, n in GEMMS)


# ---------------------------------------------------------------- profile

def profile_part() -> None:
    enc = NpuEncoder(config={"PERF_COUNT": True})
    out = {}
    for L in BUCKETS:
        inputs = bucket_inputs(enc, L)
        wall = call_ms(enc.models[L], inputs)
        req = enc.models[L].create_infer_request()
        for _ in range(5):
            req.infer(inputs)
        task, span = collections.defaultdict(float), collections.defaultdict(float)
        for p in req.get_profiling_info():
            k = f"{p.node_type}/{p.exec_type}"
            task[k] += p.cpu_time.total_seconds() * 1e3   # the plugin's per-task execution time
            span[k] += p.real_time.total_seconds() * 1e3  # includes waiting on barriers; overlaps across tasks
        out[str(L)] = {"wall_ms": wall, "task_ms": dict(task), "span_ms": dict(span)}
        print(L, round(wall, 2), {k: round(v, 2) for k, v in sorted(task.items(), key=lambda x: -x[1])[:5]}, flush=True)
    save("profile_fp16", out)


# ---------------------------------------------------------------- micro

def compile_npu(model: ov.Model):
    return ov.Core().compile_model(model, "NPU")


def gemm_chain(M: int, rng) -> ov.Model:
    """The 36 weight GEMMs of MiniLM-L6 alone, no attention or vector ops, M = tokens."""
    x = ops.parameter([M, 384], np.float32, name="x")
    h = x
    for _ in range(LAYERS):
        for _, K, N in GEMMS:
            if h.get_output_partial_shape(0)[1].get_length() != K:
                h = ops.slice(h, np.array([0, 0]), np.array([M, K]), np.array([1, 1]))  # qkv -> one of q, k, v
            w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32)  # [N, K]: per-channel INT8 fails to compile as [K, N]
            h = ops.matmul(h, ops.constant(w), False, True)
    return ov.Model([h.output(0)], [x])


def softmax_chain(L: int) -> ov.Model:
    """Six softmaxes of the attention-score shape (1, 12, L, L), as in one post's encoder."""
    x = ops.parameter([1, 12, L, L], np.float32, name="x")
    h = x
    for _ in range(LAYERS):
        h = ops.softmax(h, -1)
    return ov.Model([h.output(0)], [x])


def micro_part() -> None:
    rng = np.random.default_rng(0)
    out = {}
    x = ops.parameter([1, 16], np.float32, name="x")
    tiny = ov.Model([ops.add(x, ops.constant(np.ones((1, 16), np.float32))).output(0)], [x])
    out["empty_call_ms"] = call_ms(compile_npu(tiny), {"x": np.zeros((1, 16), np.float32)}, 300)
    print("empty", out["empty_call_ms"], flush=True)

    single = []
    for name, K, N in GEMMS:
        for M in (32, 64, 128, 256, 512, 1024, 2048):
            x = ops.parameter([M, K], np.float32, name="x")
            w = ops.constant((rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32))
            ms = call_ms(compile_npu(ov.Model([ops.matmul(x, w, False, False).output(0)], [x])),
                         {"x": rng.standard_normal((M, K)).astype(np.float32)})
            single.append({"gemm": name, "M": M, "K": K, "N": N, "ms": ms, "GOPS": 2 * M * K * N / ms / 1e6})
            print(single[-1], flush=True)
    out["single_gemm_fp16"] = single
    save("micro", out)


def chain_part() -> None:
    import nncf
    rng = np.random.default_rng(0)
    out = {}
    chain = {}
    for L in BUCKETS + (512, 1024):
        feed = {"x": rng.standard_normal((L, 384)).astype(np.float32)}
        m = gemm_chain(L, rng)
        fp16 = call_ms(compile_npu(m), feed)
        calib = nncf.Dataset([{"x": rng.standard_normal((L, 384)).astype(np.float32)} for _ in range(20)])
        q = nncf.quantize(gemm_chain(L, np.random.default_rng(0)), calib, subset_size=20)
        int8 = call_ms(compile_npu(q), feed)
        macs = gemm_macs(L)
        chain[str(L)] = {"fp16_ms": fp16, "int8_ms": int8, "fp16_GOPS": 2 * macs / fp16 / 1e6, "int8_GOPS": 2 * macs / int8 / 1e6,
                         "ideal_fp16_ms": macs / (NPU_MAC_FP16 * NPU_GHZ * 1e9) * 1e3,
                         "ideal_int8_ms": macs / (NPU_MAC_INT8 * NPU_GHZ * 1e9) * 1e3}
        print("chain", L, {k: round(v, 3) for k, v in chain[str(L)].items()}, flush=True)
    out["gemm_chain"] = chain

    soft = {}
    for L in BUCKETS:
        ms = call_ms(compile_npu(softmax_chain(L)), {"x": rng.standard_normal((1, 12, L, L)).astype(np.float32)})
        soft[str(L)] = {"ms": ms, "M_elems_per_s": LAYERS * 12 * L * L / ms / 1e3}
        print("softmax", L, soft[str(L)], flush=True)
    out["softmax_chain"] = soft
    save("micro_chain", out)


def attn_chain(L: int, stage: str) -> ov.Model:
    """Six attention cores (1, 12 heads, L, 32), cut at a stage, to see where softmax becomes slow:
    'qk' = QK^T only, 'qk_sm' = + softmax, 'qk_mask_sm' = + mask add, 'full' = + AV (the encoder's attention core)."""
    q = ops.parameter([1, 12, L, 32], np.float32, name="q")
    k = ops.parameter([1, 12, L, 32], np.float32, name="k")
    v = ops.parameter([1, 12, L, 32], np.float32, name="v")
    mask = ops.parameter([1, 1, 1, L], np.float32, name="mask")
    h, outs = q, []
    for _ in range(LAYERS):
        s = ops.matmul(h, k, False, True)
        if stage == "qk_mask_sm" or stage == "full":
            s = ops.add(s, mask)
        if stage != "qk":
            s = ops.softmax(s, -1)
        if stage == "full":
            h = ops.matmul(s, v, False, False)
        else:
            outs.append(s.output(0))  # keep every layer's scores alive; the next layer reuses q
    return ov.Model([h.output(0)] if stage == "full" else outs, [q, k, v, mask])


def attn_part() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for L in BUCKETS:
        feed = {n: rng.standard_normal((1, 12, L, 32)).astype(np.float32) for n in ("q", "k", "v")}
        feed["mask"] = np.zeros((1, 1, 1, L), np.float32)
        out[str(L)] = {st: call_ms(compile_npu(attn_chain(L, st)), feed) for st in ("qk", "qk_sm", "qk_mask_sm", "full")}
        print("attn", L, {k: round(v, 2) for k, v in out[str(L)].items()}, flush=True)
    save("micro_attn", out)


def mask_chain(L: int, form: str) -> ov.Model:
    """Six QK^T + mask + softmax, with the mask in different forms (same math):
    'row' = bias (1, 1, 1, L) broadcast over rows and heads (the encoder's form), 'full' = (1, 1, L, L) given whole,
    'heads' = (1, 12, L, L), 'folded' = no Add: [q, 1] x [k, bias]^T puts the bias inside the matmul."""
    q = ops.parameter([1, 12, L, 32], np.float32, name="q")
    k = ops.parameter([1, 12, L, 32], np.float32, name="k")
    shape = {"row": [1, 1, 1, L], "full": [1, 1, L, L], "heads": [1, 12, L, L], "folded": [1, 12, L, 1]}[form]
    mask = ops.parameter(shape, np.float32, name="mask")
    outs = []
    for _ in range(LAYERS):
        if form == "folded":
            ones = ops.constant(np.ones((1, 12, L, 1), np.float32))
            s = ops.matmul(ops.concat([q, ones], -1), ops.concat([k, mask], -1), False, True)
        else:
            s = ops.add(ops.matmul(q, k, False, True), mask)
        outs.append(ops.softmax(s, -1).output(0))
    return ov.Model(outs, [q, k, mask])


def mask_part() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for L in BUCKETS:
        row = {}
        for form in ("row", "full", "heads", "folded"):
            m = mask_chain(L, form)
            feed = {"q": rng.standard_normal((1, 12, L, 32)).astype(np.float32),
                    "k": rng.standard_normal((1, 12, L, 32)).astype(np.float32),
                    "mask": np.zeros(m.input("mask").get_shape(), np.float32)}
            row[form] = call_ms(compile_npu(m), feed)
        out[str(L)] = row
        print("mask", L, {k: round(v, 2) for k, v in row.items()}, flush=True)
    save("micro_mask", out)


# ---------------------------------------------------------------- ablate

def rewrite_softmax(mode: str) -> Path:
    """mode 'identity': drop softmax (wrong output, timing only). The rest keep the math:
    'decomposed': max, sub, exp, sum, div. 'flat2d': softmax over (12 L, L) rows, then back to (1, 12, L, L).
    'transposed': swap the last two axes, softmax over axis 2, swap back."""
    path = VAR / f"fp32_mask_bias_softmax_{mode}.onnx"
    if path.exists():
        return path
    m = onnx.load(str(mask_bias_model()))
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    nodes = []
    axes = onnx.helper.make_tensor("sm_axes", onnx.TensorProto.INT64, [1], [-1])
    m.graph.initializer.append(axes)
    for n in m.graph.node:
        if n.op_type != "Softmax":
            nodes.append(n)
            continue
        x, y, p = n.input[0], n.output[0], n.name
        if mode == "identity":
            nodes.append(onnx.helper.make_node("Identity", [x], [y], name=p))
            continue
        if mode == "flat2d":
            nodes += [onnx.helper.make_node("Shape", [x], [f"{p}_shape"], name=f"{p}_Shape"),
                      onnx.helper.make_node("Flatten", [x], [f"{p}_2d"], axis=3, name=f"{p}_Flatten"),
                      onnx.helper.make_node("Softmax", [f"{p}_2d"], [f"{p}_sm"], axis=-1, name=p),
                      onnx.helper.make_node("Reshape", [f"{p}_sm", f"{p}_shape"], [y], name=f"{p}_Reshape")]
            continue
        if mode == "transposed":
            nodes += [onnx.helper.make_node("Transpose", [x], [f"{p}_t"], perm=[0, 1, 3, 2], name=f"{p}_T0"),
                      onnx.helper.make_node("Softmax", [f"{p}_t"], [f"{p}_sm"], axis=2, name=p),
                      onnx.helper.make_node("Transpose", [f"{p}_sm"], [y], perm=[0, 1, 3, 2], name=f"{p}_T1")]
            continue
        red = (lambda op, a, b: onnx.helper.make_node(op, [a, "sm_axes"], [b], keepdims=1, name=f"{p}_{op}")) if opset >= 18 \
            else (lambda op, a, b: onnx.helper.make_node(op, [a], [b], axes=[-1], keepdims=1, name=f"{p}_{op}"))
        nodes += [red("ReduceMax", x, f"{p}_max"),
                  onnx.helper.make_node("Sub", [x, f"{p}_max"], [f"{p}_sub"], name=f"{p}_Sub"),
                  onnx.helper.make_node("Exp", [f"{p}_sub"], [f"{p}_exp"], name=f"{p}_Exp")]
        if opset >= 13:
            nodes.append(onnx.helper.make_node("ReduceSum", [f"{p}_exp", "sm_axes"], [f"{p}_sum"], keepdims=1, name=f"{p}_ReduceSum"))
        else:
            nodes.append(onnx.helper.make_node("ReduceSum", [f"{p}_exp"], [f"{p}_sum"], axes=[-1], keepdims=1, name=f"{p}_ReduceSum"))
        nodes.append(onnx.helper.make_node("Div", [f"{p}_exp", f"{p}_sum"], [y], name=f"{p}_Div"))
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, str(path))
    return path


def ablate_part() -> None:
    out = json.loads(RES.read_text()).get("ablate_softmax", {}) if RES.exists() else {}
    for mode in ("base", "identity", "decomposed", "flat2d", "transposed"):
        if mode in out:
            continue
        enc = NpuEncoder(model=None if mode == "base" else rewrite_softmax(mode), config={"PERF_COUNT": True})
        row = {}
        for L in BUCKETS:
            inputs = bucket_inputs(enc, L)
            row[str(L)] = {"wall_ms": call_ms(enc.models[L], inputs)}
            req = enc.models[L].create_infer_request()
            for _ in range(5):
                req.infer(inputs)
            task = collections.defaultdict(float)
            for p in req.get_profiling_info():
                task[f"{p.node_type}/{p.exec_type}"] += p.cpu_time.total_seconds() * 1e3
            row[str(L)]["task_ms"] = dict(task)
        if mode not in ("base", "identity"):  # same math as softmax: check the output still matches
            base = NpuEncoder(device="CPU", model=mask_bias_model())
            texts = [text_of(p) for p in load_posts(50)]
            row["cos_vs_cpu_fp32"] = float(np.mean([enc.encode_one(t) @ base.encode_one(t) for t in texts]))
        out[mode] = row
        print(mode, {L: round(v["wall_ms"], 2) for L, v in row.items() if L.isdigit()}, flush=True)
        save("ablate_softmax", out)


# ---------------------------------------------------------------- shared: speed, energy, quality

def measure(enc: NpuEncoder, timing: list[str]) -> dict:
    """1,000 posts, in arrival order for batch 1, length-sorted groups for larger batches."""
    if enc.batch == 1:
        units = [[t] for t in timing]
    else:
        order = np.argsort([len(t) for t in timing])
        units = [[timing[j] for j in order[i: i + enc.batch]] for i in range(0, len(timing), enc.batch)]
    for u in units[:20]:
        enc.encode(u)
    idle = idle_watts()
    lat = []
    meter = PackageEnergy(); meter.start()
    for u in units:
        a = now(); enc.encode(u); lat.append(now() - a)
    joules, secs = meter.stop()
    n = len(timing)
    return {"batch": enc.batch, "ms_per_post": secs * 1000 / n, "latency_per_call": stats(lat), "pkg_W": joules / secs,
            "idle_W": idle, "J_per_post": joules / n, "active_J_per_post": max(joules - idle * secs, 0) / n,
            "backlog_min": BACKLOG_POSTS * secs / n / 60}


def encode_all(enc: NpuEncoder, texts: list[str]) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 384), np.float32)
    for i in range(0, len(texts), enc.batch):
        idx = order[i: i + enc.batch]
        out[idx] = enc.encode([texts[j] for j in idx])
    return out


def quality(enc: NpuEncoder, corpus: list[str], queries: list[str]) -> dict:
    z = np.load(CACHE / "fp32.npz")
    c0, q0 = z["c"], z["q"]
    c, q = encode_all(enc, corpus), encode_all(enc, queries)
    return metrics(q0 @ c0.T, q @ c.T, q0 @ c.T, np.ones(len(q), bool))


def bucket_times(enc: NpuEncoder) -> dict:
    return {str(L): call_ms(enc.models[L], enc.prepare(["word " * (L - 2)] * enc.batch)[1]) for L in BUCKETS}


def posts():
    bench_cp1.rng.seed(0)  # load_posts draws from a shared RNG: reset it so these are the CP2-CP4 draws
    timing_posts = load_posts(1000)
    corpus_posts = load_posts(20_000, skip={p["_i"] for p in timing_posts})
    return ([text_of(p) for p in timing_posts], [text_of(p) for p in corpus_posts],
            [p["title"] for p in corpus_posts[:1000]])


# ---------------------------------------------------------------- int8

def weights_nk(model: ov.Model) -> ov.Model:
    """Store each weight as [N, K] with transpose_b. With the ONNX [K, N] layout this NPU compiler rejects the
    per-channel INT8 weight quantizers NNCF inserts (FakeQuantize shape check). Same math."""
    for op in model.get_ordered_ops():
        if op.get_type_name() != "MatMul" or op.get_attributes()["transpose_b"]:
            continue
        w = op.input_value(1).get_node()
        if w.get_type_name() != "Constant":
            continue
        new = ops.matmul(op.input_value(0), ops.constant(np.ascontiguousarray(w.get_data().T)), op.get_attributes()["transpose_a"], True)
        new.set_friendly_name(op.get_friendly_name())
        op.output(0).replace(new.output(0))
    model.validate_nodes_and_infer_types()
    return model


def int8_part() -> None:
    import nncf
    timing, corpus, queries = posts()
    path = VAR / "npu_int8.xml"
    if not path.exists():
        cpu = NpuEncoder(device="CPU")  # only for prepare(); calibration runs on the CPU plugin
        model = weights_nk(ov.Core().read_model(str(mask_bias_model())))
        names = [i.get_any_name() for i in model.inputs]
        feeds = []
        for t in corpus[-300:]:  # calibration posts: corpus tail, not among the 1,000 queries
            f = cpu.tok.tokenize([t])
            bias = ((1 - f["attention_mask"].astype(np.float32)) * -1e4)[:, None, None, :]
            feeds.append({k: bias if k == "mask_bias" else f[k].astype(np.int64) for k in names})
        q = nncf.quantize(model, nncf.Dataset(feeds), model_type=nncf.ModelType.TRANSFORMER, subset_size=300)
        ov.save_model(q, str(path))
    enc = NpuEncoder(model=path)
    res = {"bucket_ms": bucket_times(enc)}
    print("int8 buckets", res["bucket_ms"], flush=True)
    res["speed"] = measure(enc, timing)
    print("int8 speed", {k: v for k, v in res["speed"].items() if k != "latency_per_call"}, flush=True)
    res["quality"] = quality(enc, corpus, queries)
    print("int8 recall", res["quality"]["recall_at_10"], flush=True)
    save("npu_int8", res)


# ---------------------------------------------------------------- fixed

def fixed_part() -> None:
    """The software fix found by 'mask': same FP16 encoder, mask given per head (no broadcast in the graph)."""
    timing, corpus, queries = posts()
    m = onnx.load(str(mask_bias_model()))
    path = VAR / "fp32_mask_bias_heads.onnx"
    if not path.exists():
        dims = next(i for i in m.graph.input if i.name == "mask_bias").type.tensor_type.shape.dim
        dims[1].dim_value, dims[2].dim_param = 12, "L"
        onnx.save(m, str(path))
    enc = NpuEncoder(model=path, heads_mask=True)
    res = {"bucket_ms": bucket_times(enc)}
    print("fixed buckets", res["bucket_ms"], flush=True)
    res["speed"] = measure(enc, timing)
    print("fixed speed", {k: v for k, v in res["speed"].items() if k != "latency_per_call"}, flush=True)
    res["quality"] = quality(enc, corpus, queries)
    print("fixed recall", res["quality"]["recall_at_10"], flush=True)
    save("npu_fp16_heads_mask", res)


# ---------------------------------------------------------------- igpu

def igpu_part() -> None:
    timing, corpus, queries = posts()
    res = {}
    for batch in (1, 8):
        enc = NpuEncoder(device="GPU.0", batch=batch)  # GPU plugin default: FP16 inference precision
        r = {"bucket_ms": bucket_times(enc), "speed": measure(enc, timing)}
        print("igpu", batch, r["bucket_ms"], {k: v for k, v in r["speed"].items() if k != "latency_per_call"}, flush=True)
        if batch == 8:
            r["quality"] = quality(enc, corpus, queries)
            print("igpu recall", r["quality"]["recall_at_10"], flush=True)
        res[f"batch{batch}"] = r
    save("igpu_fp16", res)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1:] or ["profile", "micro", "chain", "attn", "mask", "ablate", "int8", "fixed", "igpu"]
    for part in parts:
        globals()[f"{part}_part"]()
