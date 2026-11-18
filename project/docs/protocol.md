# The protocol the node boxes run

State after CP30 (2026-11-17). Code: `sim/` (shared by the box and the simulators), `box/box_server.py` (the real box). Labels as in `docs/chip.md`. The research log is in `journal/` (CP numbers given).

## Invariants (from `NORTHSTAR.md`)

- No server or storage node sees plaintext.
- There is no privileged operator.
- Usability features are replaced, not dropped.
- Spam must not exhaust ingest.

## A message

| Field | What | CP |
|---|---|---|
| kind | post · comment · vote · delete · room message · slash notice | 17, 24 |
| body | encrypted per topic (ChaCha20-Poly1305). Only members holding the topic or room key can read it. Other boxes relay it | 17 |
| one-time key + signature | Ed25519 over the header. The header includes the author's time stamp t0 (H9) and, for a delete, the target's header (`ref`) | 17, 29 |
| membership proof | "a member, not saying which". In the demo the proof is shown, not verified; its time is taken from the chip model | 17, 24 |
| rate limit (RLN) | a number per time window. Two messages on the same number reveal the identity; every box drops that identity's later messages | 22–24 |
| id | `hash(header, one-time key, nonce, hash of the body)`. Nobody can claim someone else's id | 29 |

Deleting (forgetting): only the post's one-time key can sign its delete. A box that applies a delete keeps the target's id as "forgotten" and refuses the post afterwards (no resurrection). A copy that someone already took cannot be recalled, as in real life (CP25, CP28).

## Receiving: check everything before keeping anything

```
envelope ─▶ already held / forgotten? ─▶ id == hash(header…)? ─▶ delete: is ref bound to this key?
         ─▶ time key inside the acceptance window A = [now − R, now + 2σ]?
         ─▶ proof and signature (the cache is keyed by id + signature, H10) ─▶ only now: store, index, serve, announce
         failure: counted, never stored or served, the id is not taken (H8)
```

## Sending: ids first

A new message's id goes to the mesh peers (IHAVE). A peer asks for the envelope only if it lacks it (IWANT), from the first announcer. If that fails, it asks the next one, and finally reconciles (CP27, CP28). This brought the push from 6.8× to 1.2× the needed bytes.

## Staying in sync: set reconciliation (default S2+KE)

When two boxes meet, they compare their sets instead of trusting each other's claims.

- **What is compared**: (time key, id) for every message held, every forgotten id and every delete, inside the band B = [now − R + 2σ, now + 2σ]. The time key of a delete is its target's t0.
- **How**: range-based reconciliation (Negentropy-like).
  - Each range has a fingerprint (sum of ids mod 2⁶⁴ + count).
  - A range that differs is split 8 ways; small ranges send their ids.
  - The bytes are proportional to the difference (`sim/recon.py`, CP29).
- **When** (events, plus one slow floor):

| Trigger | When |
|---|---|
| (a) | I came on (or woke up) |
| (b) | a peer reappeared |
| (c) | an observed failure. With an up peer, retry with backoff (CP30) |
| (d) | a message points at something I do not know |
| (e) | a peer went down: reconcile with the others |
| (f) | **floor**: each pair every 10–30 min, skipped if the pair just reconciled. It repairs a loss nobody saw, and it is also the heartbeat ("do you still count me as a peer?", CP30) |

- **Fetch each id once (K, CP30)**:
  - An id that another session or an IWANT is already bringing is not fetched again.
  - If that fetch fails, the sessions that skipped the id run again.
  - In the box, the side that lacks ids is offered them by id first (`/offer`).
- **Explore (E, CP30)**: one hello an hour to a random box. Without it, two sides of a partition longer than the 1 h probe give-up never meet again (H11).
- **Clock bound** σ = 120 s. Forgotten ids and deletes are kept until now − R − 2σ, so no honest box drops a delete while another can still accept its target. R is a test placeholder (1 day in the model, 7 days in the box).
- **Option F′ (`--floor-skip`)**: skip the floor for a pair that has not changed since a clean session and has spoken lately. A generation number guards a session that ran while the set changed. It passes every test and halves the idle cost. It is not the default because it was fixed after its first results (CP30).

### What is checked, and how

| Property | Meaning |
|---|---|
| S1–S4 | no invalid message kept · no resurrection · no delete without the target's key · an invalid copy never blocks the valid one |
| L1, L2 | two honest boxes that are on and linked for long enough end up with the same messages and the same deletes |
| F1 | nothing older than the windows is kept |
| delivery (CP30) | a box on for a while lacks nothing that another on box has. L1 only judges pairs in contact, so it cannot see an overlay that splits |

| Layer | Default S2+KE (CP30) | Label |
|---|---|---|
| 1 random (5 scenes × n 10–1000, liars, forged and invalid messages, clock offsets, no final all-on phase) | 0 violations in 110 runs; 0 delivery misses | estimate (model) |
| 2 directed scenes (holes H1–H11 and the new rules) | all as predicted | estimate (model) |
| 3 exhaustive small scope (3 boxes, adversary) | 2,496 states with 0 violations; clean-flag rule 17,408 states with 0 violations; 3 deliberate bugs caught | estimate (model) |
| 4 real boxes (4 processes: link cuts, clock offsets, kills, forged copies, a held session) | 12 scripts pass; regressions L1–L7, P1–P4 pass | measured |
| demo (CP32, CP33): five users on one screen, 15 steps | 14 pass in CP32 (step 3: H12); 15 pass after CP33 | measured |

Cost (model, CP26 scenes, n = 10–1000):

| | S2+KE |
|---|---|
| duplicate envelopes | 0 (CP29 S2: up to 3.7× the needed ones) |
| exchange overhead | ≤ 52.5 KB per box per day (gate 100) |
| idle | 20–31 KB per box per day |

Re-run:

```
.venv/Scripts/python.exe -W ignore experiments/sync.py holes     # layer 2           -> results/sync/holes.json
.venv/Scripts/python.exe -W ignore experiments/sync.py mc        # layer 3a          -> results/sync/mc.json
.venv/Scripts/python.exe -W ignore experiments/sync.py cost4     # cost grid, S2+KE and S2+KF'E      -> results/sync/cost4.json
.venv/Scripts/python.exe -W ignore experiments/sync.py random    # layer 1 (~25 min, 14 processes)   -> results/sync/random.json
.venv/Scripts/python.exe -W ignore experiments/sync_scenes.py props   # layer 3b: the reconciler alone -> results/sync/props.json
.venv/Scripts/python.exe -W ignore tests/sync_check.py           # layer 4 (~12 min)                 -> results/tests/sync_check.json
```

The CP30 comparison against the retired variants (S0 = X2, F, P, P2; the `cost`, `cost2`, `cost3`, `traffic` grids and
the gate table `results/sync/gate.json`) runs at tag `pre-r1` (recast R1b).

## Sub-communities (H12 found in CP32, fixed in CP33)

A box receives only the sub-communities (subs) it joined.
- A session reconciles only the subs both boxes hold. A fingerprint is a sum of ids, so the subs' fingerprints add, and one session with the same wire format covers them (`sim/recon.py` `TopicKeySet`).
- A box refuses a message of a sub it does not hold before verifying it (`rej_sub`). The exhaustive check shows this receive check alone keeps S5 ("no box holds a sub it did not join"); filtering the session keeps the cost down.
- Each box keeps at least 3 online neighbours in each of its subs, and explores among boxes that share a sub. Assumption: a sub's member boxes can be found (the config here; a rendezvous in a real deployment).
- Before CP33 (CP29-CP32), every box received, verified and stored every sub (H12). The sync model had no subs, so layers 1-3 could not see it; the one-screen demo did.

| CP33 (model, 20 subs, about 3 per box) | box as built (CP30) | per-sub (CP33) |
|---|---|---|
| sync overhead, max | 352 KB per box per day | 51.6 |
| proof verifications per box | 1 | 0.25-0.51 |
| live P99 | 392-1,181 s | <= 1.5 s in 13 of 15 cells; 60 and 188 s in two always-on cells (open) |

Layers: random runs 0 violations in 110, directed scenes as predicted, exhaustive 2,304 states 0 violations, real boxes (14 scripts, regressions, the 15-step story) pass. **But the pre-registered delivery rule fails** (P99 no worse than the box as built + 1 s): in 16 of 110 fault-injected runs the per-sub box delivers its slowest posts later, up to 8,994 s vs 795 s. Status: provisional. Labels: estimate (model) and measured (real boxes). Record: `journal/cp/CP33.md`.

Open: a small sub loses the store-and-forward help that non-members gave (more messages wait for a member to come on in the laptop-heavy scene); the P99 tail above.

## Not covered

- metadata privacy (who talks to whom)
- choosing neighbours against an eclipse attack (exploration and peer hints can be abused)
- the real retention R
- real Groth16 verification in the box
- the run on real home networks (deferred to the end of the project; CP32 shows the same thing with emulated boxes on one screen)
