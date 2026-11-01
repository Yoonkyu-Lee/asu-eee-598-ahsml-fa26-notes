"""CP20: the CP19 scenarios and contracts on four boxes with the same box software (proof before signature, live work
first, queries first), the verify unit re-picked after proof-first, and the CP19 knobs switched off one at a time.

  X0 Pi 5 only            verify on the CPU (4 cores), encoders on one CPU core (laptop one-thread x 2.93)
  X1 Pi 5 + Hailo-8L      verify on the CPU, encoders on the Hailo-8L (vendor rates; MiniLM fixed at 128 tokens)
  X2 Pi 5 + verify chip + Hailo-8L
  X3 Pi 5 + our chip (CP18 / CP19)

  .venv/Scripts/python.exe src/bench_cp20.py      (first: CPU encoder times, photo quality)
  .venv/Scripts/python.exe src/sim_cp20.py        -> results/cp20/cp20.json
Every number here is a model (추정) on top of sources and 예비 측정.
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_cp18 as m18  # noqa: E402
import sim_cp19 as c19  # noqa: E402
from model_cp12 import DRAM_GBS, JETSON_IDLE_W, PI5_IDLE_W, interp_run  # noqa: E402
from sim_cp17 import PI5  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "cp20"
T0 = time.time()

CPU_SINGLE, CPU_BATCH = 4.068e-3 * c19.W3, 1.038e-3 * c19.W3   # A76 Groth16 verify (CP12 estimate) + W3 input
HAILO_W = (1.4, 1.9)                                            # active; idle unknown (0 here = best case for Hailo)
JETSON_VERIFY = (4_095, 5_241)                                  # CPU proofs/s, batched (CP12 estimate)
MUL_MM2, ADD_MM2, FIXED_MM2 = 1.416467 / 5, 0.011456 / 4, 0.29699072 + 0.103395   # CP12 verify parts
ENC_LEAK_MW = 1.36297467                                        # CP18 pick
L_B = 6 / 600                                                   # CP19 text limit per identity per second


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ verify machines (CP14 V4 w8)

def machines() -> dict:
    r = json.loads((RES / "cp14" / "cp14_fixed.json").read_text(encoding="utf-8"))["machines"]["V4 +O4 (w8)"]
    out = {}
    for m in r:
        env = {e["f"]: e for e in m["envelope"]}
        bad, good = env[1.0], env[0.0]
        k = f"{m['M']}x{m['A']}"
        scale = 14_633 / bad["proofs_per_s"]
        out[k] = {"M": m["M"], "A": m["A"], "area_mm2": m["M"] * MUL_MM2 + m["A"] * ADD_MM2 + FIXED_MM2,
                  "f1_per_s": bad["proofs_per_s"], "worst_mix_per_s": m["worst"]["proofs_per_s"],
                  "V_bad": 1 / bad["proofs_per_s"] * c19.W3, "E_bad": bad["mJ_per_proof"] * 1e-3 * c19.W3,
                  "V_single": c19.V_SINGLE * scale, "E_single": c19.E_SINGLE * bad["mJ_per_proof"] / 0.0685,
                  "V_128": 128 / good["proofs_per_s"] * c19.W3, "E_128": good["mJ_per_proof"] * 1e-3 * 128 * c19.W3,
                  "leak_mW": 7.257 / 5 * m["M"] + 0.1536 / 4 * m["A"] + 3.47648 + 0.000929}
    return out


# ------------------------------------------------------------------ encoder tables other than the chip

class HailoTable:
    """Hailo-8L, model zoo v2.20.0: MiniLM 121/s (B1), 492/s (B8), fixed 128 tokens; CLIP ViT-B/32 image 30/s, 130/s.
    Energy = active watts x time (HAILO_W[0]; scaled to HAILO_W[1] in the report)."""
    kind = "hailo"

    def __init__(self, W: float = HAILO_W[0]):
        self.W, self.enc = W, self
        self.img = (1 / 30, W / 30)
        self.bmax_img = 8

    def b_max(self, shape, L) -> int:
        return 8

    def text1(self, L: int) -> tuple[float, float]:
        return 1 / 121, self.W / 121

    def batch(self, lens: tuple) -> tuple[float, float]:
        s = 1 / 121 + (len(lens) - 1) * (8 / 492 - 1 / 121) / 7
        return s, s * self.W

    def imgs(self, b: int) -> tuple[float, float]:
        s = 1 / 30 + (b - 1) * (8 / 130 - 1 / 30) / 7
        return s, s * self.W


class CpuTable:
    """One Pi 5 core: laptop one-thread times (예비 측정) x 2.93 (Geekbench 6). Energy is counted as host core time."""
    kind = "cpu"

    def __init__(self, bench: dict, image_variant: str):
        t = {int(k): v for k, v in bench["minilm_int8_s_by_tokens"].items()}
        self.Ls, self.ts = np.array(sorted(t)), np.array([t[k] for k in sorted(t)]) * PI5
        self.img = (bench["clip_image_s"][image_variant] * PI5, 0.0)
        self.enc, self.bmax_img = self, 1

    def b_max(self, shape, L) -> int:
        return 1

    def text1(self, L: int) -> tuple[float, float]:
        return float(np.interp(L, self.Ls, self.ts)), 0.0

    def batch(self, lens: tuple) -> tuple[float, float]:
        return sum(self.text1(L)[0] for L in lens), 0.0

    def imgs(self, b: int) -> tuple[float, float]:
        return b * self.img[0], 0.0


class Fp16Image:
    """The chip's encoder with the CLIP image tower's GEMMs in FP16 (rule P): they run on the FP16 unit
    (R^2/4 MACs per cycle, CP4/CP12) instead of the INT8 array, and the weights come from DRAM in 2 bytes.
    Everything else as model_cp18 (추정)."""

    def __init__(self, et, img_chunks: int = 12):
        self.base, self.enc, self.kind, self.img_chunks = et, et.enc, "chip", img_chunks
        e, sh = et.enc, m18.CLIPV
        runs = e.tl["clipv"]["runs"]
        gem = [(g, K, N, 50) for _ in range(sh.layers) for g, (K, N) in sh.gemms.items()] + [("ffn_out", 3072, 768, 49)]
        t8 = j8 = t16 = j16 = 0.0
        for g, K, N, P in gem:
            cyc, on_pj, _ = interp_run(runs, e.R, g, P)
            t8 += max(cyc / e.f, K * N / (DRAM_GBS * 1e9))
            j8 += on_pj + K * N * e.dram_pJ_B
            t16 += max(P * K * N / (e.F * e.f), 2 * K * N / (DRAM_GBS * 1e9))
            j16 += P * K * N * e.fp16_mac_pJ + 2 * K * N * e.dram_pJ_B
        s, J = et.img
        leak = sum(e.leak_mW.values()) * 1e-3
        self.img = (s - t8 + t16, J + (j16 - j8) * 1e-12 + leak * (t16 - t8))
        self.gemm_s = {"int8": t8, "fp16": t16}
        self.bmax_img = 1

    def text1(self, L):
        return self.base.text1(L)

    def batch(self, lens):
        return self.base.batch(lens)

    def imgs(self, b):
        return b * self.img[0], b * self.img[1]


class Box20(c19.Box):
    """sim_cp19.Box with the verify unit on the chip (any CP14 machine) or on the host CPU, the encoder on the chip,
    a Hailo-8L or one host core, and a host of `cores` cores."""

    def __init__(self, et, cost, *, vkind="chip", vm=None, cores=4, **kw):
        self.vkind, self.vm, self.nc = vkind, vm, cores
        self.ekind = getattr(et, "kind", "chip")
        self._hs = 0.0
        if self.ekind != "chip":
            kw["sched"] = "Q1"               # no layer-boundary yielding on a Hailo or a CPU session
        super().__init__(et, cost, **kw)

    @property
    def host_s(self) -> float:
        return self._hs + (self.e_s if self.ekind == "cpu" else 0.0)

    @host_s.setter
    def host_s(self, v: float) -> None:
        self._hs = v - (getattr(self, "e_s", 0.0) if getattr(self, "ekind", "") == "cpu" else 0.0)

    def host(self, t, dur, name="", back=False):
        n = self.nc
        cores = range(n // 2, n) if (back and self.live_first) else range(n)
        i = min(cores, key=self.host_free.__getitem__)
        s = max(t, self.host_free[i])
        self.host_free[i] = s + dur
        self.host_s += dur
        h = int(s // 3600)
        if 0 <= h < len(self.host_hour):
            self.host_hour[h] += dur
        return s + dur

    def _to_verify(self, m):
        if self.vkind == "chip":
            return super()._to_verify(m)
        dur = CPU_BATCH if self.vmode == "batch" else CPU_SINGLE
        t1 = self.host(self.eng.t, dur, "verify", self.live_first and m["cls"] == 2)
        self.v_s += dur
        self.v_n += 1
        self.eng.at(t1, self._v_done, [m])

    def _v_start(self):
        if self.vkind != "chip":
            self.v_busy = False
            return
        vm = self.vm
        if not self.vq and not self.vq_back:
            self.v_busy = False
            return
        self.v_busy = True
        if self.vmode == "batch":
            batch = [self.vq.popleft() for _ in range(min(len(self.vq), 128))]
            while self.vq_back and len(batch) < 128:
                batch.append(self.vq_back.popleft())
            k = len(batch)
            vc, ec = (vm["V_128"] - vm["V_single"]) / 127, (vm["E_128"] - vm["E_single"]) / 127
            dur, J = vm["V_single"] - vc + vc * k, vm["E_single"] - ec + ec * k
        else:
            m = self.vq.popleft() if self.vq else self.vq_back.popleft()
            batch = [m]
            dur, J = (vm["V_single"], vm["E_single"]) if m["kind"] != "flood" else (vm["V_bad"], vm["E_bad"])
        self.v_s += dur
        self.v_J += J
        self.v_n += len(batch)
        self._chip(self.eng.t, J)
        self.eng.after(dur, self._v_done, batch)


# ------------------------------------------------------------------ runs

def scenarios(name, et, make, inp, lens, h_conn, sched) -> dict:
    r = {"S1": c19.s1_day(et, inp, lens, sched, make=make)}
    r["S2"] = c19.s2_peak(et, inp, lens, sched, make=make)
    r["S3"] = c19.s3_backlog(et, inp, lens, 8, sched, 8, make=make)
    r["S4"] = c19.s4_flood(et, inp, lens, "O-b", "N-b", "invalid", h_conn, make=make)
    r["S5"] = c19.k_crit(et, inp, lens, L_B, make=make)
    lim = c19.LIMITS
    s4 = r["S4"]
    r["contracts"] = {
        "K1": all(x is not None and x <= lim["K1_text_box_s"] for x in
                  (r["S2"]["text_p99_s"], r["S3"]["text_p99_s"], s4["text_p99_s"])),
        "K2": all(x <= lim["K2_query_s"] * 1e3 for x in (r["S2"]["query_p99_ms"], r["S3"]["query_p99_ms"],
                                                          s4["query_p99_ms"])),
        "K3": r["S3"]["catch_up_s"] <= lim["K3_backlog_8h_s"],
        "K4": s4["host_offered_cores"] <= lim["K4_host_cores"],
        "K5": s4["chip_W"] <= lim["K5_chip_W"]}
    log(name, r["contracts"], "S1 box", r["S1"]["box_W"], "S4 cores", round(s4["host_offered_cores"], 2),
        "K_crit", r["S5"]["K_crit"])
    return r


def slim(r: dict) -> dict:
    keep = ("text_p99_s", "image_p99_s", "query_p99_ms", "host_cores", "host_offered_cores", "verify_busy",
            "encoder_busy", "chip_W", "box_W", "catch_up_min", "unfinished_text", "chip_J_day", "verify_J_day",
            "encoder_J_day", "n_text")
    return {k: v for k, v in r.items() if k in keep}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = json.loads((OUT / "cp20_bench.json").read_text(encoding="utf-8"))
    img_variant = bench["photos"]["rule_P"]["pick"]
    img_variant = img_variant if img_variant in bench["cpu_times"]["clip_image_s"] else "fp32"
    inp = c19.inputs()
    lens = np.minimum(c19.token_lengths(), 256)
    h_conn = json.loads((RES / "cp19" / "cp19.json").read_text(encoding="utf-8"))["handshake"]["box_s"]
    M = machines()
    res = {"inputs": {"cpu_verify_ms": [CPU_SINGLE * 1e3, CPU_BATCH * 1e3], "hailo_W": HAILO_W,
                      "cpu_image_variant": img_variant, "machines": M}}

    # decision 4 first (the verify machine every chip config uses)
    chip = c19.EncTable(8, 1024)
    v2 = {}
    for k, vm in sorted(M.items(), key=lambda kv: kv[1]["area_mm2"]):
        if vm["f1_per_s"] <= c19.FLOOD_PER_S:
            v2[k] = {"area_mm2": vm["area_mm2"], "f1_per_s": vm["f1_per_s"], "pass": False, "why": "rate < flood"}
            continue
        s4 = c19.s4_flood(chip, inp, lens, "O-b", "N-b", "invalid", h_conn,
                          make=partial(Box20, vkind="chip", vm=vm))
        ok = (s4["text_p99_s"] <= c19.LIMITS["K1_text_box_s"] and s4["query_p99_ms"] <= c19.LIMITS["K2_query_s"] * 1e3
              and s4["host_offered_cores"] <= c19.LIMITS["K4_host_cores"] and s4["chip_W"] <= c19.LIMITS["K5_chip_W"]
              and s4["verify_busy"] <= 0.95)
        v2[k] = {"area_mm2": vm["area_mm2"], "f1_per_s": vm["f1_per_s"], "pass": ok, **slim(s4)}
        log("V2", k, round(vm["area_mm2"], 3), round(vm["f1_per_s"]), ok, round(s4["verify_busy"], 3),
            s4["text_p99_s"])
    pick = min((k for k in v2 if v2[k]["pass"]), key=lambda k: v2[k]["area_mm2"], default="6x6")
    res["rule_V2"] = {"rows": v2, "pick": pick, "cp14": "6x6", "area_saved_mm2": M["6x6"]["area_mm2"] - M[pick]["area_mm2"]}
    log("rule V2", pick)
    vm = M[pick]

    # decision 3: the four boxes
    cpu = CpuTable(bench["cpu_times"], img_variant)
    hailo = HailoTable()
    chip16 = Fp16Image(chip)
    res["image_fp16_chip"] = {"image_s": chip16.img[0], "image_mJ": chip16.img[1] * 1e3, "gemm_s": chip16.gemm_s,
                              "image_int8_s": chip.img[0], "image_int8_mJ": chip.img[1] * 1e3}
    log("fp16 image", res["image_fp16_chip"])
    cfg = {"X0 Pi 5 only": (cpu, partial(Box20, vkind="cpu", cores=3), "Q1"),
           "X1 Pi 5 + Hailo-8L": (hailo, partial(Box20, vkind="cpu"), "Q1"),
           "X2 Pi 5 + verify chip + Hailo-8L": (hailo, partial(Box20, vkind="chip", vm=vm), "Q1"),
           "X3 Pi 5 + our chip": (chip16, partial(Box20, vkind="chip", vm=vm), "Q2"),
           "X3g our chip, FP16 image, GEMM-boundary yield": (Fp16Image(chip, 12 * 4 + 1), partial(Box20, vkind="chip", vm=vm),
                                                             "Q2"),
           "X3i our chip, INT8 image (CP18 assumption)": (chip, partial(Box20, vkind="chip", vm=vm), "Q2")}
    res["boxes"] = {}
    for name, (et, make, sched) in cfg.items():
        r = scenarios(name, et, make, inp, lens, h_conn, sched)
        for k in ("S1", "S2", "S3", "S4"):
            r[k] = slim(r[k])
        if et is hailo:      # Hailo energy was computed at 1.4 W; the range goes to 1.9 W
            for k in ("S1", "S2", "S3", "S4"):
                r[k]["chip_W_range"] = [r[k]["chip_W"], r[k]["chip_W"] + (r[k]["encoder_busy"] * (HAILO_W[1] - HAILO_W[0]))]
        res["boxes"][name] = r
    res["boxes"]["Jetson Orin Nano (CPU verify only)"] = {
        "S4": {"verify_capacity_per_s": JETSON_VERIFY, "host_offered_cores": [6 * c19.FLOOD_PER_S / x for x in JETSON_VERIFY]},
        "contracts": {"K1": False, "K4": False}, "idle_W": JETSON_IDLE_W}
    B = res["boxes"]
    f_ok = (not any(B[k]["contracts"]["K1"] and B[k]["contracts"]["K4"] for k in
                    ("X0 Pi 5 only", "X1 Pi 5 + Hailo-8L", "Jetson Orin Nano (CPU verify only)"))
            and B["X3 Pi 5 + our chip"]["contracts"]["K1"] and B["X3 Pi 5 + our chip"]["contracts"]["K4"])
    x2, x3 = B["X2 Pi 5 + verify chip + Hailo-8L"], B["X3 Pi 5 + our chip"]
    a_ok = (all(x3["contracts"][k] or not x2["contracts"][k] for k in x2["contracts"])
            and x3["S1"]["chip_W"] <= x2["S1"]["chip_W"])
    x3g = B["X3g our chip, FP16 image, GEMM-boundary yield"]
    a_ok_g = (all(x3g["contracts"][k] or not x2["contracts"][k] for k in x2["contracts"])
              and x3g["S1"]["chip_W"] <= x2["S1"]["chip_W"])
    res["rule_F"] = {"C2_holds": f_ok}
    res["rule_A"] = {"one_chip_kept": a_ok, "X3_S1_chip_W": x3["S1"]["chip_W"], "X2_S1_accel_W": x2["S1"]["chip_W"],
                     "X2_S1_accel_W_at_1.9W": x2["S1"].get("chip_W_range"),
                     "with_GEMM_yield (CP19 rule Q fallback)": {"one_chip_kept": a_ok_g,
                                                                "X3g_S1_chip_W": x3g["S1"]["chip_W"]}}
    log("rule F", f_ok, "rule A", a_ok)

    # decision 5: ablations on X3 (CP19 runs where the knob was already varied; new runs otherwise)
    c = json.loads((RES / "cp19" / "cp19.json").read_text(encoding="utf-8"))
    s3 = lambda q, B_, lf: next(x for x in c["S3"] if x["hours"] == 8 and x["sched"] == q and x["B"] == B_
                                and x["live_first"] == lf)
    s4 = lambda o, p, m: next(x for x in c["S4"] if x["order"] == o and x["policy"] == p and x["mix"] == m)
    base_s1 = x3["S1"]
    abl = {"X3 (all on)": {"S2_query_ms": x3["S2"]["query_p99_ms"], "S3_text_s": x3["S3"]["text_p99_s"],
                           "S3_query_ms": x3["S3"]["query_p99_ms"], "S4_host_cores": x3["S4"]["host_offered_cores"],
                           "S4_text_s": x3["S4"]["text_p99_s"], "K_crit": x3["S5"]["K_crit"],
                           "chip_mm2": 11.3 - 9.2161520145 + 3.5254714345 - (M["6x6"]["area_mm2"] - vm["area_mm2"]),
                           "S1_chip_mW": base_s1["chip_W"] * 1e3}}
    abl["proof first -> signature first"] = {"S4_host_cores": s4("O-a", "N-b", "invalid")["host_offered_cores"],
                                             "S4_text_s": s4("O-a", "N-b", "invalid")["text_p99_s"],
                                             "S4_replay_host_cores": s4("O-a", "N-a", "replay")["host_offered_cores"]}
    abl["layer yield (Q2) -> job priority (Q1)"] = {"S2_query_ms": c["S2"]["Q1"]["query_p99_ms"],
                                                    "S3_query_ms": s3("Q1", 8, True)["query_p99_ms"]}
    abl["live first -> FIFO host and verify"] = {"S3_text_s": s3("Q2", 8, False)["text_p99_s"]}
    abl["limit L-b -> L-a"] = {"K_crit": c["S5"]["8x8 1MB L-a"]["K_crit"]}
    abl["verify " + pick + " -> 6x6 (CP14)"] = {"chip_mm2_delta": M["6x6"]["area_mm2"] - vm["area_mm2"],
                                               "S4_verify_busy": v2["6x6"].get("verify_busy")}
    e16 = c19.EncTable(16, 2048)
    s1_16 = c19.s1_day(e16, inp, lens, "Q2", make=partial(Box20, vkind="chip", vm=vm))
    abl["encoder 8x8 1MB -> 16x16 2MB"] = {"K_crit": c["S5"]["16x16 2MB L-b"]["K_crit"],
                                          "chip_mm2_delta": 7.5007 - 3.5255, "S1_chip_mW": s1_16["chip_W"] * 1e3}
    xi = B["X3i our chip, INT8 image (CP18 assumption)"]
    abl["image tower FP16 (rule P) -> INT8 (CP18)"] = {"K_crit": xi["S5"]["K_crit"], "S1_chip_mW": xi["S1"]["chip_W"] * 1e3,
                                                      "S3_catch_up_min": xi["S3"]["catch_up_min"],
                                                      "S2_query_ms": xi["S2"]["query_p99_ms"]}
    abl["power gating -> always on"] = {"S1_chip_mW": base_s1["chip_W"] * 1e3 + ENC_LEAK_MW + vm["leak_mW"],
                                        "idle_chip_mW": ENC_LEAK_MW + vm["leak_mW"]}
    res["ablation"] = abl
    log("ablation", json.dumps(abl, default=float)[:1500])
    (OUT / "cp20.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("wrote")


if __name__ == "__main__":
    main()
