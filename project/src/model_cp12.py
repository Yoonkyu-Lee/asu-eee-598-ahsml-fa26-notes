"""CP12 box chip roll-up: verify unit (CP10 synthesis + CP11 cycle model) + encoder (Timeloop/Accelergy) on one 45 nm chip,
then the box (Pi 5-class host + chip) next to existing combinations.

Run from project/:  .venv/Scripts/python.exe src/model_cp12.py
Reads results/cp12/cp12_timeloop.json and cp12_parts.json (made in WSL by model_cp12_encoder.py and model_cp12_parts.py),
results/cp10, cp11, cp6, cp5. Writes results/cp12/cp12_chip.json. Every output is an estimate (추정) unless it is copied
from a measurement or a source, which the field name or note says.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_cp4 import BACKLOG_POSTS, HEADS, LAYERS, anatomy, token_lengths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
D = 384
GEMMS = {"qkv": (D, 3 * D), "attn_out": (D, D), "ffn_in": (D, 1536), "ffn_out": (1536, D)}
POSTS_PER_S = 1_000_000 / 86_400          # ordinary day, as in CP1-CP6
FLOOD_PROOFS_PER_S = 100e6 / 8 / 1024     # 100 Mbps of 1 KB messages (CP6)
DRAM_GBS = 8.5
VEC_OPS_PER_ELEM = 5                      # as in CP4
GATE_LEAK_MW, GATE_IDLE_SHARE = 1.0, 0.5  # rule D2

# Pi 5-class host and the other boxes: sources are in cp/cp6_survey.md unless noted
PI5_IDLE_W = (2.4, 3.3)                   # Jeff Geerling 2024-08-29 (2GB / 4GB)
PI5_STRESS_W = (8.9, 9.8)                 # same source, stress-ng on 4 cores
GB6_PI5, GB6_185H = 764, 2240             # Geekbench 6 single-core: raspberrypi.com (Pi 5 avg), Geekbench aggregate via HotHardware
HAILO8L_PER_S, HAILO8L_W = 121, (1.4, 1.9)  # model zoo v2.20.0 batch 1 (x86 host); datasheet power on vision models
JETSON_IDLE_W = (2.5, 4.7)                # NVIDIA forum report, Orin Nano Super dev kit
JETSON_CORE_VS_PI5 = (1.7 / 2.4, 1.7 / 2.4 * 1.28)  # (추정) clock ratio, times A78-vs-A76 IPC 1.0-1.28


def interp_run(runs: list[dict], R: int, gemm: str, P: float) -> tuple[float, float, float]:
    """(cycles, on-chip pJ, Timeloop DRAM pJ) at P rows, linear in P between mapped points."""
    pts = sorted((r for r in runs if r["R"] == R and r["gemm"] == gemm), key=lambda r: r["P"])
    xs = [r["P"] for r in pts]
    cyc = [r["cycles"] for r in pts]
    tot = [r["energy_uJ"] * 1e6 for r in pts]
    dram = [r["fJ_per_compute"].get("DRAM", 0.0) * r["computes"] / 1e3 for r in pts]
    on = [t - d for t, d in zip(tot, dram)]
    if P > xs[-1]:  # beyond the largest mapped size: scale the last point linearly
        s = P / xs[-1]
        return cyc[-1] * s, on[-1] * s, dram[-1] * s
    return float(np.interp(P, xs, cyc)), float(np.interp(P, xs, on)), float(np.interp(P, xs, dram))


class Encoder:
    def __init__(self, tl: dict, parts: dict, R: int, ghz: float, hp: dict | None = None):
        self.R, self.f = R, ghz * 1e9
        self.runs = tl["runs"]
        comp = tl["components"][str(R)]
        self.comp = comp
        self.dram_pJ_B = comp["DRAM"]["actions_pJ"]["read"] / 8         # one 64-bit LPDDR4 access
        glb = parts["sram"]["enc_glb_2MB"]
        self.glb_pJ_B = glb["read_pJ"] * 1e12 / (glb["width_bits"] / 8)
        a = parts["aladdin_2ns"]
        fp_mul = a["fp_sp_multiplier"]["dynamic_pJ"] * (15 / 32) ** 2  # Aladdin plug-in scaling for 5+10 bits
        fp_add = a["fp_sp_adder"]["dynamic_pJ"] * (15 / 32)
        self.fp16_mac_pJ = fp_mul + fp_add
        self.vec_op_pJ = (fp_mul + fp_add) / 2
        self.F, self.V = max(16, R * R // 4), max(16, R)                # FP16 attention MACs and vector lanes (CP4)
        fp_area = a["fp_sp_multiplier"]["area_um2"] * (15 / 32) ** 2 + a["fp_sp_adder"]["area_um2"] * (15 / 32)
        fp_leak = a["fp_sp_multiplier"]["leak_mW"] * (15 / 32) ** 2 + a["fp_sp_adder"]["leak_mW"] * (15 / 32)
        cyc_s = 1 / 0.5e9  # Timeloop/Accelergy tables were made at 500 MHz
        area, leak = {}, {}
        for name, c in comp.items():
            if name == "DRAM" or "actions_pJ" not in c:
                continue
            area[name] = c["area_um2"] * c["instances"] / 1e6
            leak[name] = c["actions_pJ"].get("leak", 0.0) * c["instances"] * 1e-12 / cyc_s * 1e3
        if hp:  # sensitivity: SRAM leakage with high-performance cells instead of CACTI's low-standby-power default
            for name, key in (("glb", "enc_glb_2MB"), ("in_buf", "enc_in_buf_16KB"), ("acc_buf", "enc_acc_buf_16KB")):
                leak[name] = hp["sram"][key]["leak_mW"]
        area["fp16_attention"] = self.F * fp_area / 1e6
        area["vector"] = self.V * fp_area / 1e6
        leak["fp16_attention"] = self.F * fp_leak
        leak["vector"] = self.V * fp_leak
        self.area_mm2, self.leak_mW = area, leak

    def batch(self, lens: np.ndarray) -> tuple[float, float, dict]:
        """Seconds and joules for one length-sorted batch, padded to its longest post (as in CP4)."""
        B, L = len(lens), int(lens.max())
        P = B * L
        t, e = 0.0, {"gemm_onchip": 0.0, "dram_weights": 0.0, "attention": 0.0, "vector": 0.0, "dram_embed": 0.0}
        for _ in range(LAYERS):
            for g, (K, N) in GEMMS.items():
                cyc, on_pj, _ = interp_run(self.runs, self.R, g, P)
                t_w = K * N / (DRAM_GBS * 1e9)           # weights stream in once per GEMM, double-buffered
                t += max(cyc / self.f, t_w)
                e["gemm_onchip"] += on_pj
                e["dram_weights"] += K * N * self.dram_pJ_B
            macs = B * 2 * L * L * D
            t += macs / (self.F * self.f)
            e["attention"] += macs * self.fp16_mac_pJ + B * (3 * L * D + 2 * HEADS * L * L + L * D) * 2 * self.glb_pJ_B
            n_vec = B * anatomy(L)["vec_elems"] // LAYERS
            t += n_vec / (self.V * self.f)
            e["vector"] += n_vec * (VEC_OPS_PER_ELEM * self.vec_op_pJ + 4 * self.glb_pJ_B)
        emb = int(lens.sum()) * D * 2 * 2
        e["dram_embed"] += emb * self.dram_pJ_B
        t += emb / (DRAM_GBS * 1e9)
        leak_j = sum(self.leak_mW.values()) * 1e-3 * t
        e["leak_while_busy"] = leak_j * 1e12
        return t, sum(e.values()) * 1e-12, e

    def evaluate(self, lens: np.ndarray, B: int) -> dict:
        order = np.sort(lens)
        t = j = 0.0
        parts: dict[str, float] = {}
        for i in range(0, len(order), B):
            dt, dj, e = self.batch(order[i: i + B])
            t, j = t + dt, j + dj
            for k, v in e.items():
                parts[k] = parts.get(k, 0.0) + v * 1e-12
        n = len(lens)
        p99 = int(np.percentile(lens, 99))
        live_t, _, _ = self.batch(np.array([p99]))
        return {"R": self.R, "GHz": self.f / 1e9, "B": B, "ms_per_post": t / n * 1e3, "mJ_per_post": j / n * 1e3,
                "busy_W": j / t, "backlog_min": BACKLOG_POSTS * t / n / 60, "live_p99_ms": live_t * 1e3,
                "energy_share": {k: v / j for k, v in parts.items()}}


def verify_unit(parts: dict, hp: dict | None = None) -> dict:
    syn = json.loads((RES / "cp10" / "cp10_synth.json").read_text(encoding="utf-8"))["runs"]["pipe_W64_S1"]
    sim = json.loads((RES / "cp11" / "cp11_sim.json").read_text(encoding="utf-8"))
    a = parts["aladdin_2ns"]
    M, A, live = 5, 4, 194                      # CP11 decision
    add_bits = 256
    mod_add_pJ = 2 * a["adder"]["dynamic_pJ"] * add_bits / 32          # add + conditional subtract
    mod_add_um2 = 2 * a["adder"]["area_um2"] * add_bits / 32
    mod_add_leak = 2 * a["adder"]["leak_mW"] * add_bits / 32
    rf_bits = live * 256                        # the 8 KB buffer as flip-flops: it needs ~16 ports per cycle
    uc = parts["sram"]["ver_ucode_32KB"]
    per = sim["per_proof_marginal"]
    ops = per["mul"] + per["add"]
    e = {"mul": per["mul"] * syn["power_mW"] * 1e-3 * syn["sim_period_ns"] * 1e-9 * syn["tb_cycles_per_mul"] * 1e12,
         "add": per["add"] * mod_add_pJ,
         "operand_traffic": ops * sim["machine"]["pJ_traffic_per_op"],   # CP11 assumption, kept (conservative)
         "ucode_fetch": ops / 4 * uc["read_pJ"] * 1e12}                   # 4 micro-ops per 256-bit word (추정)
    area = {"multipliers": M * syn["area_um2"] / 1e6, "mod_adders": A * mod_add_um2 / 1e6,
            "operand_regfile": rf_bits * a["reg"]["area_um2"] / 1e6, "ucode_32KB": uc["area_um2"] / 1e6}
    leak = {"multipliers": M * syn["leakage_mW"], "mod_adders": A * mod_add_leak,
            "operand_regfile": rf_bits * a["reg"]["leak_mW"],
            "ucode_32KB": (hp["sram"]["ver_ucode_32KB"] if hp else uc)["leak_mW"]}
    window = next(r for r in sim["window_sweep"] if r["mults"] == M and r["adders"] == A and r["window"] == 256)
    single = sim["op_counts"]["single"]
    single_pJ = (single["mul"] * e["mul"] / per["mul"] + single["add"] * mod_add_pJ
                 + (single["mul"] + single["add"]) * (sim["machine"]["pJ_traffic_per_op"] + uc["read_pJ"] * 1e12 / 4))
    return {"mJ_per_proof": sum(e.values()) * 1e-9, "energy_share": {k: v / sum(e.values()) for k, v in e.items()},
            "mJ_single_proof": single_pJ * 1e-9, "max_proofs_per_s": window["proofs_per_s"],
            "area_mm2": area, "leak_mW": leak}


def box_rows(enc_day: dict, enc_backlog: dict, enc: Encoder, ver: dict) -> dict:
    """Chip power in each regime with rule D2 gating, plus the box (Pi 5-class host idle + chip)."""
    enc_leak, ver_leak = sum(enc.leak_mW.values()), sum(ver["leak_mW"].values())
    enc_busy_day = POSTS_PER_S * enc_day["ms_per_post"] / 1e3
    ver_busy_day = POSTS_PER_S / ver["max_proofs_per_s"]
    ver_busy_flood = (FLOOD_PROOFS_PER_S + POSTS_PER_S) / ver["max_proofs_per_s"]
    gate_enc = enc_leak > GATE_LEAK_MW and enc_busy_day < GATE_IDLE_SHARE
    gate_ver = ver_leak > GATE_LEAK_MW and ver_busy_day < GATE_IDLE_SHARE

    def leak_w(enc_busy: float, ver_busy: float) -> float:
        le = enc_leak * (enc_busy if gate_enc else 1.0)
        lv = ver_leak * (ver_busy if gate_ver else 1.0)
        return (le + lv) * 1e-3

    # encoder dynamic energy already includes its leakage while busy; remove it to avoid counting twice
    enc_dyn_mJ = enc_day["mJ_per_post"] * (1 - enc_day["energy_share"].get("leak_while_busy", 0.0))
    day_dyn = POSTS_PER_S * (enc_dyn_mJ + ver["mJ_per_proof"]) * 1e-3
    flood_dyn = POSTS_PER_S * enc_dyn_mJ * 1e-3 + (FLOOD_PROOFS_PER_S + POSTS_PER_S) * ver["mJ_per_proof"] * 1e-3
    chip = {
        "gating": {"encoder": gate_enc, "verify": gate_ver, "encoder_leak_mW": enc_leak, "verify_leak_mW": ver_leak,
                   "encoder_busy_day": enc_busy_day, "verify_busy_day": ver_busy_day, "verify_busy_flood": ver_busy_flood},
        "ordinary_W": day_dyn + leak_w(enc_busy_day, ver_busy_day),
        "flood_100Mbps_W": flood_dyn + leak_w(enc_busy_day, min(1.0, ver_busy_flood)),
        "flood_ok": ver_busy_flood <= 1.0,
        "backlog_W": enc_backlog["busy_W"] + (0.0 if gate_ver else ver_leak * 1e-3),  # busy_W includes encoder leakage
        "backlog_min": enc_backlog["backlog_min"],
        "idle_W": leak_w(0.0, 0.0),
        "idle_W_no_gating": (enc_leak + ver_leak) * 1e-3,
    }
    chip["box_ordinary_W"] = [h + chip["ordinary_W"] for h in PI5_IDLE_W]
    chip["box_flood_W"] = [h + chip["flood_100Mbps_W"] for h in PI5_IDLE_W]
    chip["box_idle_W"] = [h + chip["idle_W"] for h in PI5_IDLE_W]
    return chip


def others() -> dict:
    """Existing combinations from measurements (CP5, CP6) and sources; ARM numbers are estimates."""
    crypto = json.loads((RES / "cp6" / "cp6_crypto.json").read_text(encoding="utf-8"))
    cp5 = json.loads((RES / "cp5" / "cp5.json").read_text(encoding="utf-8"))["npu_fp16_heads_mask"]["speed"]
    e = crypto["energy"]
    x86_batch_ms = crypto["p_core"]["groth16_bn254_batch64_verify_per_proof_us"] / 1e3   # portable build, no asm (like arm64)
    x86_single_ms = crypto["p_core"]["groth16_bn254_5_public_inputs"]["2^12"]["verify_us"] / 1e3
    slow = GB6_185H / GB6_PI5
    a76_batch_ms, a76_single_ms = x86_batch_ms * slow, x86_single_ms * slow
    a76_core_W = [(s - i) / 4 for s, i in zip(PI5_STRESS_W, PI5_IDLE_W)]      # active watts per loaded core
    pi_cap = 4 / (a76_batch_ms / 1e3)
    jet_cap = [6 * r / (a76_batch_ms / 1e3) for r in JETSON_CORE_VS_PI5]
    hailo_mJ = [w / HAILO8L_PER_S * 1e3 for w in HAILO8L_W]
    pi_verify_day_W = [POSTS_PER_S * a76_batch_ms / 1e3 * w for w in a76_core_W]
    return {
        "arm_estimate": {"a76_batch_ms_per_proof": a76_batch_ms, "a76_single_ms": a76_single_ms, "slowdown_vs_185H": slow,
                         "a76_active_W_per_core": a76_core_W,
                         "basis": "x86 P-core portable build (CP6, no asm, like arkworks on arm64) x Geekbench 6 single-core ratio"},
        "Pi 5 + Hailo-8L": {
            "encoder": "Hailo-8L, 128 tokens fixed (17% of posts truncated at 256 already; 128 cut unmeasured)",
            "encoder_mJ_per_post": hailo_mJ,
            "ordinary_W": [PI5_IDLE_W[0] + POSTS_PER_S * hailo_mJ[0] / 1e3 + pi_verify_day_W[0],
                           PI5_IDLE_W[1] + POSTS_PER_S * hailo_mJ[1] / 1e3 + pi_verify_day_W[1]],
            "ordinary_note": "Hailo-8L idle power unknown (not included)",
            "flood_max_proofs_per_s": pi_cap, "flood_ok": pi_cap >= FLOOD_PROOFS_PER_S,
            "flood_W_at_max": [i + 4 * w for i, w in zip(PI5_IDLE_W, a76_core_W)],
            "idle_W": list(PI5_IDLE_W),
        },
        "Jetson Orin Nano": {
            "encoder": "GPU TensorRT, dynamic shapes; MiniLM speed/power on Orin Nano unknown",
            "flood_max_proofs_per_s": jet_cap, "flood_ok": min(jet_cap) >= FLOOD_PROOFS_PER_S,
            "flood_note": "CPU only; GPU ZK numbers on Jetson not found (cp6_survey.md)",
            "idle_W": list(JETSON_IDLE_W),
        },
        "Laptop Core Ultra 9 185H (measured)": {
            "encoder": "NPU FP16, per-head mask (CP5)",
            "encoder_active_mJ_per_post": cp5["active_J_per_post"] * 1e3,
            "idle_W": cp5["idle_W"],
            "ordinary_W": cp5["idle_W"] + POSTS_PER_S * (cp5["active_J_per_post"]
                                                         + e["lp_e_core/groth16"]["active_J_per_op"]),
            "flood_W": json.loads((RES / "cp6" / "cp6_box.json").read_text(encoding="utf-8"))["spam_flood"]["100Mbps"]["verifiers"]
            ["P-core, batch of 64"]["active_W"],
            "flood_note": "4.3 P-cores batched (CP6); meets the rate, not the 2 W budget",
        },
    }


def main() -> None:
    out = RES / "cp12"
    tl = json.loads((out / "cp12_timeloop.json").read_text(encoding="utf-8"))
    parts = json.loads((out / "cp12_parts.json").read_text(encoding="utf-8"))
    lens = token_lengths()
    sample = np.random.default_rng(0).choice(lens, 2000, replace=False)  # same sample as CP4
    res: dict = {"assumptions": {"posts_per_s": POSTS_PER_S, "flood_proofs_per_s": FLOOD_PROOFS_PER_S,
                                 "dram_GBs": DRAM_GBS, "gating_rule": [GATE_LEAK_MW, GATE_IDLE_SHARE]}}
    sweep = []
    for R in (8, 16, 32):
        for ghz in (0.5, 1.0):
            enc = Encoder(tl, parts, R, ghz)
            for B in (1, 8):
                r = enc.evaluate(sample, B)
                r["area_mm2"] = sum(enc.area_mm2.values())
                r["leak_mW"] = sum(enc.leak_mW.values())
                sweep.append(r)
                print(f"R={R:2d} {ghz} GHz B={B}: {r['ms_per_post']:.2f} ms {r['mJ_per_post']:.2f} mJ "
                      f"backlog {r['backlog_min']:.0f} min live {r['live_p99_ms']:.1f} ms area {r['area_mm2']:.2f}")
    res["encoder_sweep"] = sweep
    ok = [r for r in sweep if r["GHz"] == 0.5 and r["B"] == 8 and r["backlog_min"] <= 60]
    ok = [r for r in ok if next(s for s in sweep if s["R"] == r["R"] and s["GHz"] == 0.5 and s["B"] == 1)["live_p99_ms"] <= 5000]
    pick = min(ok, key=lambda r: (r["R"], r["mJ_per_post"])) if ok else None
    res["encoder_pick_D1"] = pick
    R = pick["R"] if pick else 16
    enc = Encoder(tl, parts, R, 0.5)
    day = next(s for s in sweep if s["R"] == R and s["GHz"] == 0.5 and s["B"] == 1)
    backlog = next(s for s in sweep if s["R"] == R and s["GHz"] == 0.5 and s["B"] == 8)
    res["encoder"] = {"R": R, "area_mm2": enc.area_mm2, "leak_mW": enc.leak_mW, "ordinary_B1": day, "backlog_B8": backlog,
                      "fp16_mac_pJ": enc.fp16_mac_pJ, "dram_pJ_per_B": enc.dram_pJ_B, "glb_pJ_per_B": enc.glb_pJ_B,
                      "cp4_mJ_per_post": 8.55, "ratio_vs_cp4": day["mJ_per_post"] / 8.55}
    ver = verify_unit(parts)
    res["verify"] = ver
    chip = box_rows(day, backlog, enc, ver)
    chip["area_mm2"] = {"encoder": sum(enc.area_mm2.values()), "verify": sum(ver["area_mm2"].values())}
    chip["area_mm2"]["total"] = sum(chip["area_mm2"].values())
    res["chip_A"] = chip
    hp = json.loads((out / "cp12_parts_hp.json").read_text(encoding="utf-8"))
    enc_hp, ver_hp = Encoder(tl, parts, R, 0.5, hp), verify_unit(parts, hp)
    day_hp = enc_hp.evaluate(sample, 1)
    backlog_hp = enc_hp.evaluate(sample, 8)
    chip_hp = box_rows(day_hp, backlog_hp, enc_hp, ver_hp)
    res["chip_A_sram_hp_cells"] = {"encoder_leak_mW": enc_hp.leak_mW, "verify_leak_mW": ver_hp["leak_mW"], **chip_hp}
    res["others"] = others()
    (out / "cp12_chip.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("encoder_sweep",)}, indent=1, ensure_ascii=False,
                     default=float))


if __name__ == "__main__":
    main()
