"""CP4 analytical cost model: an INT8 ingest accelerator for MiniLM-L6 on a weak device.

Run from project/:  .venv/Scripts/python.exe src/model_cp4.py
Writes results/cp4/cp4_model.json. Every number here is an estimate (추정) built from
  - measured token lengths of real posts (the CP2/CP3 corpus), and
  - per-operation energies from the Horowitz table in lecture L7 (45 nm, 0.9 V).

Hardware template (one design point):
  INT8 weight-stationary array R x C   -> the 36 weight GEMMs (CP3 contract: INT8 is fine for these)
  FP16 unit, F MACs/cycle              -> QK^T and AV (FP32 in the CP3 contract; FP16 kept 0.999 recall in CP2)
  vector unit, V elements/cycle        -> LayerNorm, GELU, softmax, residual, dynamic quantization
  one global SRAM buffer               -> activations, plus either all weights or one GEMM's weights double-buffered
  optional 32 KB local buffers         -> input reuse across column tiles and INT32 accumulation
  DRAM                                 -> weights (if not resident) and the embedding table
Leakage is not modelled: energies are dynamic only.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cp1 import load_posts, text_of  # noqa: E402
from node import MAX_TOKENS, MODEL_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp4"
D, FF, LAYERS, HEADS = 384, 1536, 6, 12
GEMMS = [("qkv", D, 3 * D), ("attn_out", D, D), ("ffn_in", D, FF), ("ffn_out", FF, D)]  # (K, N) per layer
W_BYTES = LAYERS * sum(k * n for _, k, n in GEMMS)  # INT8 weights of the 36 GEMMs
BACKLOG_POSTS = 1_000_000 / 3  # 8 h of a community posting 10^6 per day (assumed, as in CP1-CP3)

# pJ, lecture L7 Horowitz table (45 nm, 0.9 V)
INT8_MUL, INT32_ADD, FP16_MUL, FP16_ADD = 0.2, 0.1, 1.1, 0.4
SRAM_32B = {8 << 10: 5.0, 32 << 10: 10.0, 1 << 20: 100.0}  # per 32-bit read, by array size
DRAM_BYTE = 640 / 4
E_MAC8 = INT8_MUL + INT32_ADD   # INT8 multiply, INT32 accumulate
E_MAC16 = FP16_MUL + FP16_ADD
E_VEC_OP = (FP16_MUL + FP16_ADD) / 2  # one FP16 vector op, mean of add and multiply
VEC_OPS_PER_ELEM = 5                  # (추정) e.g. LayerNorm ~ sub, mul, add, plus reductions
PE_REG = 0.5  # (추정) pipeline registers per MAC in a systolic PE; not in the lecture table
LOCAL_BYTES = 32 << 10  # input buffer and accumulator next to the array, as in a TPU-style design


def sram_byte_pj(size: int) -> float:
    """Per-byte SRAM access energy, log-log interpolated over the table (extrapolated past 1 MB)."""
    xs = sorted(SRAM_32B)
    lx, ly = np.log([float(x) for x in xs]), np.log([SRAM_32B[x] for x in xs])
    s = math.log(max(size, xs[0]))
    if s >= lx[-1]:
        y = ly[-1] + (s - lx[-1]) * (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    else:
        y = np.interp(s, lx, ly)
    return math.exp(y) / 4


def token_lengths(n: int = 20_000) -> np.ndarray:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(MAX_TOKENS)
    timing = load_posts(1000)  # same draws as CP2/CP3
    corpus = load_posts(n, skip={p["_i"] for p in timing})
    return np.array([len(e.ids) for e in tok.encode_batch([text_of(p) for p in corpus])])


def anatomy(L: int) -> dict:
    """Work for one post of L tokens."""
    gemm = L * W_BYTES
    attn = LAYERS * 2 * L * L * D
    vec = LAYERS * (4 * L * D + L * FF + HEADS * L * L + 4 * L * D + L * FF)  # LN x2, GELU, softmax, residual x2, quant
    return {"L": L, "gemm_mac": gemm, "attn_mac": attn, "vec_elems": vec, "attn_share": attn / (gemm + attn)}


def batch_cost(lens: np.ndarray, hw: dict) -> tuple[float, float, dict]:
    """Seconds and joules for one bucketed batch: every post padded to the longest in the batch."""
    R, C, f = hw["R"], hw["C"], hw["GHz"] * 1e9
    B, L = len(lens), int(lens.max())
    M = B * L
    e = {"mac": 0.0, "pe": 0.0, "sram_global": 0.0, "sram_local": 0.0, "dram": 0.0, "attn": 0.0, "vec": 0.0}
    sb = sram_byte_pj(hw["sram_bytes"])
    sl = sram_byte_pj(LOCAL_BYTES) if hw["local_buf"] else sb  # without local buffers every access goes global
    t = 0.0
    for _ in range(LAYERS):
        for _, K, N in GEMMS:
            kt, nt = math.ceil(K / R), math.ceil(N / C)
            cycles = kt * nt * (M + R)               # weight tile load overlaps poorly: + R per tile
            t_dram = 0.0
            if not hw["weights_resident"]:
                t_dram = K * N / (hw["dram_GBs"] * 1e9)
                e["dram"] += K * N * DRAM_BYTE
                e["sram_global"] += K * N * sb       # write into the buffer
            t += max(cycles / f, t_dram)             # double-buffered: next GEMM's weights stream meanwhile
            macs = M * K * N
            e["mac"] += macs * E_MAC8
            e["pe"] += macs * PE_REG
            # global: weights once, inputs once, FP16 outputs once.
            # local (input buffer, accumulators): inputs re-read per column tile, INT32 partials across row tiles.
            e["sram_global"] += (K * N + M * K + M * N * 2) * sb
            e["sram_local"] += (M * K * (nt - 1) + M * N * 4 * 2 * (kt - 1)) * sl
        attn_macs = B * 2 * L * L * D
        t += attn_macs / (hw["F"] * f)
        e["attn"] += attn_macs * E_MAC16 + B * (3 * L * D + 2 * HEADS * L * L + L * D) * 2 * sb
        n_vec = B * anatomy(L)["vec_elems"] // LAYERS
        t += n_vec / (hw["V"] * f)
        e["vec"] += n_vec * (VEC_OPS_PER_ELEM * E_VEC_OP + 4 * sb)
    emb = int(lens.sum()) * D * 2 * 2           # word + position rows, FP16, from DRAM
    e["dram"] += emb * DRAM_BYTE
    t += emb / (hw["dram_GBs"] * 1e9)
    return t, sum(e.values()) * 1e-12, e


def evaluate(lens: np.ndarray, hw: dict) -> dict:
    order = np.sort(lens)
    B = hw["B"]
    t = j = 0.0
    parts: dict[str, float] = {}
    for i in range(0, len(order), B):
        dt, dj, e = batch_cost(order[i: i + B], hw)
        t, j = t + dt, j + dj
        for k, v in e.items():
            parts[k] = parts.get(k, 0.0) + v * 1e-12
    n = len(lens)
    live_t, live_j, _ = batch_cost(np.array([int(np.percentile(lens, 99))]), {**hw, "B": 1})
    return {**hw, "ms_per_post": t / n * 1e3, "mJ_per_post": j / n * 1e3, "avg_W": j / t,
            "backlog_min": BACKLOG_POSTS * t / n / 60, "live_p99_len_ms": live_t * 1e3,
            "energy_share": {k: v / j for k, v in parts.items()}}


def sram_needed(B: int, resident: bool) -> int:
    act = B * MAX_TOKENS * (2 * D * 2 + FF)          # residual stream FP16 x2 + FFN intermediate INT8
    weights = W_BYTES if resident else 2 * D * FF    # all weights, or the largest GEMM double-buffered
    return act + weights


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lens = token_lengths()
    res = {"tokens": {"n": len(lens), "mean": float(lens.mean()), "p50": float(np.median(lens)),
                      "p90": float(np.percentile(lens, 90)), "p99": float(np.percentile(lens, 99)),
                      "truncated_share": float((lens >= MAX_TOKENS).mean())},
           "weights_MB": W_BYTES / 1e6}
    per = [anatomy(int(L)) for L in lens]
    res["per_post_mean"] = {k: float(np.mean([p[k] for p in per])) for k in ("gemm_mac", "attn_mac", "vec_elems", "attn_share")}
    res["anatomy_at"] = {str(L): anatomy(L) for L in (64, 128, 256)}
    for B in (8, 32):
        padded = sum(int(np.sort(lens)[i: i + B].max()) * len(np.sort(lens)[i: i + B]) for i in range(0, len(lens), B))
        res[f"bucket_pad_overhead_B{B}"] = padded / lens.sum() - 1

    sample = np.random.default_rng(0).choice(lens, 2000, replace=False)  # keeps the sweep fast
    rows = []
    grid = itertools.product((8, 16, 32, 64), (0.5, 1.0), (1, 8, 32), (False, True), (8.5, 17.0), (False, True))
    for R, GHz, B, resident, bw, local in grid:
        hw = {"R": R, "C": R, "GHz": GHz, "B": B, "weights_resident": resident, "dram_GBs": bw, "local_buf": local,
              "F": max(16, R * R // 4), "V": max(16, R), "sram_bytes": sram_needed(B, resident)}
        rows.append(evaluate(sample, hw))
    res["sweep"] = rows
    ok = [r for r in rows if r["backlog_min"] <= 60 and r["live_p99_len_ms"] <= 5000]
    res["smallest_meeting_target"] = min(ok, key=lambda r: (r["R"], r["sram_bytes"], r["mJ_per_post"])) if ok else None
    res["lowest_energy"] = min(rows, key=lambda r: r["mJ_per_post"])
    (OUT / "cp4_model.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=2, default=float))


if __name__ == "__main__":
    main()
