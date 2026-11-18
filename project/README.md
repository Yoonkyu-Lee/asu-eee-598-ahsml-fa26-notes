# A chip for the node box of an operator-free community

EEE 598 (ML hardware and systems) final project, Fall 2026.

**The idea**
- A community with no operator: no server can read the posts, and no one has privileged access.
- Search, feeds, spam filtering and identity checks still work, because each member's always-on home **node box** does the operator's work.
- Our part is the hardware that makes that work cheap: a 45 nm chip with a proof-verification unit and a transformer encoder.

Why this direction and what may not change: [NORTHSTAR.md](NORTHSTAR.md).

## Conclusions so far

| | Result | Label | Details |
|---|---|---|---|
| Workload | On an ordinary day the box is easy: text encoding is ≈ 98 % of ingest, and a laptop NPU already meets the backlog goal. The box is defined by a **spam flood**: about 12,000 anonymous proofs per second to verify (24–181 W on CPUs), which edge AI accelerators cannot compute at all | measured / estimate / sourced | [docs/chip.md](docs/chip.md) |
| Chip | Verification unit with 6 × 254-bit Montgomery multipliers + 6 adders (2.12 mm², ≥ 13,429 proofs/s in the worst flood) and a 16 × 16 FP16 encoder with 2 MB of SRAM. Chip **≥ 9.6 mm²** at 45 nm | estimate | [docs/chip.md](docs/chip.md) |
| RTL | The multiplier, synthesized with FreePDK45: 0.28 mm², 300 MHz, 1.76 nJ per multiply | measured (synthesis) | [chip/rtl/](chip/rtl/) |
| Protocol | Boxes sync by set reconciliation and fetch each message once. 0 violations across random runs, directed scenes, exhaustive small-scope search and real-box fault tests. ≤ 52.5 KB per box per day of sync overhead | estimate (model) + measured (real boxes) | [docs/protocol.md](docs/protocol.md) |
| Chip load | Each valid message is verified and encoded exactly once per box, so the chip's work is what the box receives | counted in the model | [docs/protocol.md](docs/protocol.md) |

Known gaps:
- **H12 (found in CP32):** sync reconciles every message regardless of sub, so a box also receives, verifies and stores the subs it never joined (it cannot read them without their key, so they never show). The design replicates only joined subs (CP16). Next: CP33.
- The 16 × 16 FP16 encoder was not mapped, so the area is a lower bound.
- Power was not re-estimated for it.
- Only the multiplier has RTL.

See the "Open" section of [docs/chip.md](docs/chip.md).

## Folders

| Folder | What | Look here when you want to |
|---|---|---|
| [docs/](docs/) | chip design, protocol, numbers with sources | understand the design |
| [chip/rtl/](chip/rtl/) | the multiplier (SystemVerilog), testbench, synthesis scripts | see the real circuit |
| [chip/model/](chip/model/) | `chip.py` (the latest chip, one source), encoder (Timeloop/Accelergy), verification-unit cycle model, roll-up | check or change a chip number |
| [box/](box/) | the node box: real crypto, embeddings, search, sync, web UI; chip time from `chip.py` (virtual accelerator); `stage.py`: five users on one screen | run the community |
| [sim/](sim/) | protocol core shared with the box; community and sync simulators; oracle; model checker | reason about thousands of boxes |
| [bench/](bench/) | workload measurements (ingest stages, CPU energy, crypto in Rust) | see why the chip is shaped this way |
| [experiments/](experiments/) | scripts that reproduce the conclusions | re-run a result |
| [tests/](tests/) | real box processes with faults injected | check the box still behaves |
| [results/](results/) | `chip/`, `sync/`, `bench/`, `tests/`: the numbers behind the conclusions | read the raw numbers |
| [journal/](journal/) | research log: one entry per checkpoint (CP0–CP32), plus a plain-Korean walkthrough | follow how we got here (Korean) |
| [legacy/](legacy/) | code and results the conclusions no longer need, frozen at tag `cp30` | reproduce an old checkpoint |

## Try it (CPU only, a few minutes)

```
python -m venv .venv
.venv/Scripts/pip install -r box/requirements.txt      # numpy, onnxruntime, tokenizers, cryptography
.venv/Scripts/python.exe box/fetch_models.py           # MiniLM + CLIP ONNX from Hugging Face, SHA-256 checked
.venv/Scripts/python.exe chip/model/chip.py            # the chip and where each number comes from
.venv/Scripts/python.exe box/stage.py                  # five users' clients on one page: open http://127.0.0.1:8620/, press Play
.venv/Scripts/python.exe -W ignore experiments/sync.py holes   # the protocol's directed scenes (< 1 min)
```
(On macOS/Linux use `.venv/bin/python`.) Several computers: [box/LAN.md](box/LAN.md).

Run everything from this folder (`project/`). Not in git:
- `data/` (posts and pictures)
- `models/` (from `fetch_models.py`)
- `refs/` (papers)
- `logs/`, `state/`
