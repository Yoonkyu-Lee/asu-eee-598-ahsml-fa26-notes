# legacy/ — code and results the latest conclusions no longer need

Everything here was moved in CP31 (2026-11-17) with `git mv`, so its history is intact. It is frozen: it was not
updated to the new layout and may not run from here. **To run an old CP's commands as written in its
`journal/cp/CPn.md`, check out the tag `cp30`** (the last commit before the move):

```
git worktree add ../asu-cp30 cp30      # a second checkout at cp30, next to this one
```

What is here and why it is not current:

| Here | Was | Why it is legacy |
|---|---|---|
| `src/bench_cp2.py` … `bench_cp7.py` | `src/` | CP2–CP7 measurements (encoder precision, NPU, crypto, first verifier estimate). Their conclusions are in `docs/` with sources; the numbers the chip model still reads are in `results/chip/` |
| `src/bench_cp18.py`, `bench_cp20.py` | `src/` | CP18/CP20 measurements. `ClipText` moved to `sim/ml.py` |
| `src/model_cp6.py`, `model_cp7.py`, `model_cp8.py` | `src/` | box load model, first verifier estimate, placement alternatives. Superseded by the cycle model (`chip/model/verifier_*`) and the roll-up; their outputs the chip model reads are frozen in `results/chip/` |
| `src/sim_cp16.py` … `sim_cp28.py` | `src/` | per-CP experiment runners (community load, scenarios, spam limits, consistency, exchange, audit, precision). Conclusions in `journal/` and `docs/` |
| `src/sim/exchange.py`, `src/demo/x2.py`, `src/demo/exchange_check.py`, `src/demo/REMOTE.md` | `src/sim`, `src/demo` | the CP26 exchange (X2) model and its box/model cross-check, the CP25 remote-run notes. Replaced by set reconciliation (CP29–CP30) |
| `src/prep_coco.py` | `src/` | one-off data preparation (CP20) |
| `rtl/mont_mul.sv`, `tb_mont_mul.sv`, `syn.tcl`, `run_cp9.csh` | `rtl/` | CP9's first multiplier (6 cycles, 295 MHz). CP10's design at S = 1 (`chip/rtl/mont_mul_pipe.sv`) replaced it |
| `results/cpN/` | `results/` | every CP's outputs except those in `results/chip`, `results/sync`, `results/bench`, `results/tests` |

## Old name → new name (files that stayed current)

| Old | New |
|---|---|
| `rtl/mont_mul_pipe.sv`, `tb_mont_mul_pipe.sv`, `syn_pipe.tcl`, `run_cp10.csh` | `chip/rtl/` (`run_cp10.csh` → `run_syn.csh`) |
| `src/tl_cp12/` | `chip/model/timeloop/` |
| `src/model_cp12_encoder.py`, `model_cp18_encoder.py`, `model_cp12_parts.py` | `chip/model/timeloop_minilm.py`, `timeloop_shapes.py`, `parts.py` |
| `src/model_cp4.py`, `model_cp12.py`, `model_cp18.py` | `chip/model/encoder_analytic.py`, `rollup.py`, `encoder_size.py` |
| `src/model_cp9.py` + `model_cp10.py` | `chip/model/multiplier.py` |
| `src/bn254_trace.py`, `sim_cp11.py`, `sim_cp13.py`, `sim_cp14.py` | `chip/model/bn254_trace.py`, `verifier_cycles.py`, `verifier_flood.py`, `verifier_fixed.py` |
| `sim_cp19.EncTable` + CP28 FP16 factor (in the box) | `chip/model/chip.py` + `chip_table.json` (`build_table.py`) |
| `src/sim/*`, `src/node.py` | `sim/`, `sim/node.py` |
| `src/demo/` (box files) | `box/` |
| `src/demo/sync_check.py`, `lan_check.py`, `persist_check.py` | `tests/` |
| `src/bench_cp1.py`, `hw_probe.py`, `crypto_bench/` | `bench/ingest_stages.py`, `hw_probe.py`, `crypto_bench/` |
| `src/sim_cp29.py`, `sim_cp30.py`, `cp30_gate.py` | `experiments/sync_scenes.py`, `sync.py`, `sync_gate.py` |
| `results/cp4, cp5, cp6, cp8–cp14, cp18` (files the chip model reads or writes) | `results/chip/` (file names kept) |
| `results/cp30`, `cp29/props.json`, `cp29/cpu_prelim.json` | `results/sync/` |
| `results/cp1`, `cp1_official` | `results/bench/` |
| `CHECKPOINT.md`, `CP_GUIDE.md`, `cp/` | `journal/` |
