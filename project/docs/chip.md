# The chip: what it is, why, and where every number comes from

State after CP34 (2026-11-17). Labels: **measured**, **sourced** (someone else's paper or datasheet), **estimate** (our model). The research log behind each line is in `journal/` (CP number given).

## Why a chip at all (the workload)

A server-blind community has no operator, so each member's always-on **node box** does the operator's work: checking that a message is real (signature, anonymous membership proof, rate limit), decrypting it, embedding it for search, and indexing it.

| Finding | Number | Label | CP |
|---|---|---|---|
| Ingesting one post on a laptop CPU is almost all text encoding | encoder ≈ 98 %, crypto (verify + decrypt) ≈ 0.5 % | measured | 1 |
| A laptop NPU already meets the backlog-time goal, once its compiler is worked around | 8 h of posts in 45 min (from 174 min) | measured | 4, 5 |
| So speed on an ordinary day is not the reason for a chip. The box is defined by a **spam flood**: 100 Mbps of 1 KB messages, each with a proof to verify | about 12,000 proofs/s; 24–181 W on CPUs | estimate | 6 |
| Edge AI accelerators (Hailo, Coral, Jetson NPU, RK3588) cannot do this math | 8/16-bit MAC arrays, no 254-bit modular arithmetic | sourced (`journal/cp/cp6_survey.md`) | 6 |
| A real community sends a box far less than "a million posts a day" | top-1 % box at peak: 1/26 of the earlier assumption | estimate (community simulator) | 16 |
| Each valid message is verified exactly once per box, also with the CP30 sync | counted over 21 M receptions | estimate (model, counted) | 29, 30 |
| **The flood numbers above assume no limit on new connections.** With a connection policy (known neighbours first, a cap on unknown connections, cut on the first invalid proof), a Pi 5 without the verification unit survives unless an attacker hits almost all of a newcomer's candidate boxes with about 3×10⁴ connection attempts/s each; at 10⁵/s the handshakes alone overload the host with or without the chip. The unit's remaining job: an admitted unknown connection costs the host 0.57 ms instead of 4.9 ms, so the cap can be about 8× higher | band 1–10 × 10⁴ attempts/s, sensitive to the unmeasured cost of refusing a connection (2–50 µs) | estimate (model) | 35 |

## The chip (45 nm)

```
            ┌───────────────────────────── node-box chip ─────────────────────────────┐
 messages ─▶│ verification unit                         encoder                       │─▶ index
 (proofs)   │  6 × 254-bit Montgomery multipliers        16 × 16 MAC array (FP16 text)  │   (host)
            │  6 × adders, batch check, invalid-proof    2 MB global SRAM               │
            │  isolation (CP13–CP14)                     MiniLM text, CLIP image        │
            └──────────────────────────────────────────────────────────────────────────┘
 host: a Pi 5-class CPU (signatures, decryption, index, networking)
```

| Part | Number | Label | Where it comes from |
|---|---|---|---|
| Multiplier (one) | 0.28 mm², 300 MHz, 1.76 nJ per multiply, 4 cycles | **measured** (Design Compiler, FreePDK45, power from simulated switching) | `chip/rtl/mont_mul_pipe.sv` at S = 1 (CP10), `chip/model/multiplier.py` |
| Verification unit | 6 multipliers + 6 adders, 2.12 mm², ≥ 13,429 proofs/s under the worst invalid-proof flood | estimate (cycle model replaying BN254 field operations) | `chip/model/verifier_cycles.py`, `verifier_flood.py`, `verifier_fixed.py` (CP11, CP13, CP14) |
| One proof | 89.6 µs | estimate | CP19 |
| Encoder | 16 × 16, 2 MB, text at FP16 (128-token post 17.9 ms on the chip) | estimate (direct 16-bit Timeloop/Accelergy mappings + CP28's latency/backlog rules) | `timeloop_fp16.py`, `encoder_resize_w16.py` (CP34); `timeloop_minilm.py`, `encoder_size.py` (CP12, CP18) |
| Chip area | **10.3 mm²** (encoder 8.18, of which 2 MB SRAM 6.80; verifier 2.12) | estimate | `chip_w16.py` (CP34): CP12's roll-up on the 16-bit mapping with FP16 MACs |
| Power | ordinary day 16.6 mW average, flood 0.99 W (budget 2 W) | estimate | `chip_w16.py` (CP34): CP19's box model, load = top-1 % box receiving every sub-community (ordinary day is an upper bound: since CP33 a box receives only its subs, 0.25–0.51× the verifications) |

The box asks `chip/model/chip.py` for these numbers (the virtual accelerator: real work in software, chip time from the model). `python chip/model/chip.py` prints them.

## How the design got here (short)

1. CP7–CP8: a dedicated verification unit beats borrowing the encoder array (40× short) or a CPU instruction extension.
2. CP9–CP10: the multiplier in RTL. A clock lower than assumed and energy 4.2× the estimate. Interleaving did not raise the clock enough (1.2×), so S = 1 was kept.
3. CP11–CP14: a cycle model of whole Groth16 batch verification. One invalid proof in a batch makes the batch fail, so the worst case is a flood of invalid proofs. The unit grew to 8 + 8, then optimizations brought it back to 6 + 6.
4. CP12: one chip (verifier + encoder) against Pi 5 + Hailo-8L. Ordinary-day power is a wash (0.06–0.11 W apart). Only the chip survives the flood.
5. CP16–CP18: under real community load the encoder shrank to 8 × 8, 1 MB (chip 11.3 → 5.61 mm²).
6. CP27–CP28: the chip model had counted INT8 text since CP12, which breaks the Recall@10 ≥ 0.95 contract (CP3). Assuming FP16 text costs 4× INT8, 8 × 8 misses the latency and backlog rules, so 16 × 16 was picked.
7. CP34: direct 16-bit mappings show FP16 text takes only 1.0–1.4× the INT8 time (but 3.9–6.3× the energy, from the FP16 MAC). 8 × 8 then passes if the FP16 MAC issues one operation per cycle, and fails at half speed; by the rule written in advance, 16 × 16 stays. The chip is 10.3 mm² (computed, no longer a lower bound).

## Open (known gaps, not hidden)

- **The FP16 MAC is an estimate** scaled from Aladdin's 45 nm FP32 tables (CP34). Whether it issues one operation per cycle decides between 8 × 8 and 16 × 16.
- Power assumes every box receives every sub-community (the CP16 load used since CP19); per-sub-community sync (CP33) would lower it.
- **Picture time in the box (0.552 s) is CP20's FP16 image tower on the 8 × 8 array.** The text time uses 16 × 16. CP31 kept both as they were (a restructure must not change numbers).
- Only the multiplier has RTL. The rest of the unit and the encoder are models.
- Embeddings differ across devices at INT8 (CP17, CP18). A deterministic datapath is a hardware requirement candidate.

## Re-running the chip model

```
.venv/Scripts/python.exe chip/model/chip.py              # the summary
.venv/Scripts/python.exe chip/model/multiplier.py        # synthesis -> multiplier numbers      -> results/chip/cp9_model.json, cp10_model.json
.venv/Scripts/python.exe chip/model/verifier_cycles.py   # Groth16 batch on the unit (~3 min)    -> results/chip/cp11_sim.json
.venv/Scripts/python.exe chip/model/verifier_flood.py    # invalid-proof flood                   -> results/chip/cp13_flood.json
.venv/Scripts/python.exe chip/model/verifier_fixed.py    # 6 + 6 sizing                          -> results/chip/cp14_fixed.json
.venv/Scripts/python.exe chip/model/encoder_analytic.py  # token lengths, analytic encoder        -> results/chip/cp4_model.json
.venv/Scripts/python.exe chip/model/rollup.py            # one chip vs existing boxes (CP12)     -> results/chip/cp12_chip.json
.venv/Scripts/python.exe chip/model/encoder_size.py      # encoder sizing on community load      -> results/chip/cp18_encoder.json
.venv/Scripts/python.exe chip/model/build_table.py --check   # the box's time table == the model
# Timeloop/Accelergy (WSL, see chip/model/timeloop/build_wsl.sh): timeloop_minilm.py, timeloop_shapes.py, parts.py
# RTL on the Apporto CAD lab (VCS, Design Compiler, FreePDK45): chip/rtl/run_syn.csh <W> <S> <period_ns>
```
Every script re-creates its committed result file (checked in CP31; only the wall-clock field `trace_s` differs).
