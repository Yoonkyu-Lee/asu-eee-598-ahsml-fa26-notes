"""CP18 encoder re-sizing: CP12's encoder roll-up (model_cp12.Encoder) for any transformer shape, judged by rule E
(cp/CP18.md) on the CP16 behaviour-model load instead of "10^6 posts per box per day".

  .venv/Scripts/python.exe src/model_cp18.py        -> results/cp18/cp18_encoder.json

Reads Timeloop mappings: results/cp12/cp12_timeloop.json (MiniLM, 2 MB) and results/cp18/cp18_timeloop_*.json
(CLIP towers, other buffer sizes). Every output is an estimate (추정).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cp12 import DRAM_GBS, VEC_OPS_PER_ELEM, Encoder as Enc12, interp_run  # noqa: E402
from model_cp4 import token_lengths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "cp18"
DAY_TEXT, DAY_IMG = 25_889, 1_143          # CP16 analytic, 10^6 users, P-all: texts and image posts per box per day
BACKLOG_H = 8
CP14_CHIP_MM2, CP12_ENC_MM2 = 11.3, 9.2161520145
QUERY_TOKENS = 12
LIMITS = {"backlog_min": 60.0, "live_text_p99_s": 5.0, "query_s": 0.050, "image_s": 1.0}   # rule E


@dataclass(frozen=True)
class Shape:
    name: str
    D: int
    FF: int
    layers: int
    heads: int
    max_tokens: int
    image: bool = False

    @property
    def gemms(self) -> dict:
        D, FF = self.D, self.FF
        return {"qkv": (D, 3 * D), "attn_out": (D, D), "ffn_in": (D, FF), "ffn_out": (FF, D)}

    def act_bytes(self, L: int) -> int:
        """Residual stream FP16 x2 + FFN intermediate INT8, per sequence (CP4 sram_needed without the weights)."""
        return L * (4 * self.D + self.FF)


MINILM = Shape("minilm", 384, 1536, 6, 12, 256)
CLIPV = Shape("clipv", 768, 3072, 12, 12, 50, image=True)
CLIPT = Shape("clipt", 512, 2048, 12, 8, 77)


class Encoder(Enc12):
    """CP12's encoder (same area, leakage and unit prices) running a given shape from its own mappings."""

    def __init__(self, tl_by_shape: dict, parts: dict, R: int, glb_kb: int):
        first = next(iter(tl_by_shape.values()))
        super().__init__(first, parts, R, 0.5)
        self.tl = tl_by_shape
        glb = self.comp["glb"]["actions_pJ"]
        self.glb_pJ_B = glb["read"] / 16            # 128-bit word (arch.yaml)
        self.glb_bytes = glb_kb * 1024

    def b_max(self, s: Shape, L: int) -> int:
        panel = 2 * max(k for k, _ in s.gemms.values()) * self.R   # weight column panel, double-buffered
        return max(0, (self.glb_bytes - panel) // s.act_bytes(L))

    def run(self, s: Shape, lens: np.ndarray) -> tuple[float, float]:
        """Seconds and joules for one batch of sequences padded to the longest."""
        runs = self.tl[s.name]["runs"]
        B, L = len(lens), int(lens.max())
        P = B * L
        t = e = 0.0
        for _ in range(s.layers):
            for g, (K, N) in s.gemms.items():
                cyc, on_pj, _ = interp_run(runs, self.R, g, P)
                t += max(cyc / self.f, K * N / (DRAM_GBS * 1e9))     # weights stream once per GEMM
                e += on_pj + K * N * self.dram_pJ_B
            macs = B * 2 * L * L * s.D
            t += macs / (self.F * self.f)
            e += macs * self.fp16_mac_pJ + B * (3 * L * s.D + 2 * s.heads * L * L + L * s.D) * 2 * self.glb_pJ_B
            vec = B * (4 * L * s.D + L * s.FF + s.heads * L * L + 4 * L * s.D + L * s.FF)
            t += vec / (self.V * self.f)
            e += vec * (VEC_OPS_PER_ELEM * self.vec_op_pJ + 4 * self.glb_pJ_B)
        if s.image:   # pixels in, patch embedding = one 3072 -> 768 GEMM at 49 rows per image
            px = B * 224 * 224 * 3
            cyc, on_pj, _ = interp_run(runs, self.R, "ffn_out", B * 49)
            t += px / (DRAM_GBS * 1e9) + max(cyc / self.f, 3072 * 768 / (DRAM_GBS * 1e9))
            e += px * self.dram_pJ_B + on_pj + 3072 * 768 * self.dram_pJ_B
        else:         # token embeddings, FP16, read twice (as CP12)
            emb = int(lens.sum()) * s.D * 2 * 2
            t += emb / (DRAM_GBS * 1e9)
            e += emb * self.dram_pJ_B
        e += sum(self.leak_mW.values()) * 1e-3 * t * 1e12
        return t, e * 1e-12

    def many(self, s: Shape, lens: np.ndarray, B: int) -> tuple[float, float, int]:
        """Length-sorted batches of up to B that fit the global buffer. Returns (s, J, smallest batch used)."""
        order = np.sort(lens)
        t = j = 0.0
        used = B
        i = 0
        while i < len(order):
            L = int(order[min(i + B, len(order)) - 1])
            b = min(B, self.b_max(s, L))
            if b < 1:
                return float("inf"), float("inf"), 0
            used = min(used, b)
            dt, dj = self.run(s, order[i:i + b])
            t, j, i = t + dt, j + dj, i + b
        return t, j, used


def evaluate(enc: Encoder, text: Shape, lens: np.ndarray, rng) -> dict:
    n_t, n_i = round(DAY_TEXT * BACKLOG_H / 24), round(DAY_IMG * BACKLOG_H / 24)
    lt = np.minimum(rng.choice(lens, n_t), text.max_tokens)
    bt, bj, bt_b = enc.many(text, lt, 8)
    bi, bij, bi_b = enc.many(CLIPV, np.full(n_i, 50), 8)
    p99 = int(min(np.percentile(lens, 99), text.max_tokens))
    live_t = enc.many(text, np.array([p99]), 1)[0]
    query = enc.many(text, np.array([QUERY_TOKENS]), 1)[0]
    img = enc.many(CLIPV, np.array([50]), 1)
    day_lens = np.minimum(rng.choice(lens, DAY_TEXT), text.max_tokens)
    day_t = enc.many(text, day_lens, 1)
    r = {"R": enc.R, "glb_KB": enc.glb_bytes // 1024, "text_model": text.name,
         "backlog_min": (bt + bi) / 60, "backlog_text_min": bt / 60, "backlog_image_min": bi / 60,
         "backlog_batch_min_used": {"text": bt_b, "image": bi_b},
         "live_text_p99_s": live_t, "query_s": query, "image_s": img[0], "image_mJ": img[1] * 1e3,
         "text_mJ_mean": day_t[1] / DAY_TEXT * 1e3,
         "day_active_J": day_t[1] + img[1] * DAY_IMG, "day_busy_s": day_t[0] + img[0] * DAY_IMG,
         "area_mm2": sum(enc.area_mm2.values()), "area_parts_mm2": enc.area_mm2, "leak_mW": sum(enc.leak_mW.values())}
    r["day_idle_leak_J"] = r["leak_mW"] * 1e-3 * 86_400
    r["max_users_backlog"] = 1e6 * LIMITS["backlog_min"] / r["backlog_min"] if r["backlog_min"] > 0 else None
    r["pass"] = {k: r[k] <= v for k, v in LIMITS.items()}
    r["ok"] = all(r["pass"].values())
    r["chip_mm2"] = CP14_CHIP_MM2 - CP12_ENC_MM2 + r["area_mm2"]
    return r


def load(name: str) -> dict | None:
    p = (RES / "cp12" / "cp12_timeloop.json") if name == "minilm_glb2048" else (OUT / f"cp18_timeloop_{name}.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = json.loads((OUT / "cp18_bench.json").read_text(encoding="utf-8")) if (OUT / "cp18_bench.json").exists() else {}
    pick = bench.get("x5", {}).get("rule", {}).get("pick", "M1")
    text = CLIPT if pick == "M2" else MINILM
    parts = json.loads((RES / "cp12" / "cp12_parts.json").read_text(encoding="utf-8"))
    lens = token_lengths()
    rows = []
    for glb in (2048, 1024, 512):
        tl_t, tl_v = load(f"{text.name}_glb{glb}"), load(f"clipv_glb{glb}")
        if not tl_t or not tl_v:
            continue
        for R in (8, 16, 32):
            if not any(r["R"] == R for r in tl_t["runs"]) or not any(r["R"] == R for r in tl_v["runs"]):
                continue
            enc = Encoder({text.name: tl_t, "clipv": tl_v}, parts, R, glb)
            r = evaluate(enc, text, lens, np.random.default_rng(18))
            rows.append(r)
            print(f"R={R:2d} glb={glb:4d} KB backlog {r['backlog_min']:6.1f} min (text {r['backlog_text_min']:.1f}, "
                  f"img {r['backlog_image_min']:.1f}) live {r['live_text_p99_s'] * 1e3:.1f} ms query "
                  f"{r['query_s'] * 1e3:.2f} ms image {r['image_s'] * 1e3:.1f} ms {r['image_mJ']:.1f} mJ "
                  f"area {r['area_mm2']:.2f} leak {r['leak_mW']:.1f} mW ok={r['ok']} users<= {r['max_users_backlog']:.3g}",
                  flush=True)
    ok = [r for r in rows if r["ok"]]
    best = min(ok, key=lambda r: (r["area_mm2"], r["day_active_J"])) if ok else None
    res = {"text_model": text.name, "x5_pick": pick, "limits": LIMITS,
           "load": {"day_text": DAY_TEXT, "day_image": DAY_IMG, "backlog_hours": BACKLOG_H},
           "rows": rows, "pick": best}
    if best:
        # flip rule: the picked array must hold a 10^6-user community; otherwise one step up
        res["flip_users_ok"] = best["max_users_backlog"] >= 1e6
    (OUT / "cp18_encoder.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("pick", best and {k: best[k] for k in ("R", "glb_KB", "area_mm2", "chip_mm2", "leak_mW", "backlog_min")})


if __name__ == "__main__":
    main()
