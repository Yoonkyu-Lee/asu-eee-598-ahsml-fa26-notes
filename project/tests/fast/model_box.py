"""Model vs real box: the same directed script must get the same verdict (CLAUDE.md 7-15; recast R3).

  .venv/Scripts/python.exe tests/fast/model_box.py      reads results only (seconds); exit 1 on a mismatch

Each real-box scene in tests/sync_check.py has a model twin in the layer-2 scripts. The box runs the CP30 default
plus F' (--floor-skip), so its twin is the model variant S2+KF'E (CP29 scripts: S2; CP33 scripts: sub). A pair marked
"other" checks a different property under the same story and is listed, not compared; a box scene with no entry
fails, so a new scene must name its twin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOX = ROOT / "results" / "tests" / "sync_check.json"
CP29 = ROOT / "results" / "sync" / "subs" / "regress" / "holes.json"
SUBS = ROOT / "results" / "sync" / "subs" / "holes.json"

# box scene -> (model file, model scene, model variant, "same" | "other: why")
TWINS = {
    "silent link return (H1 class)": (CP29, "CP29: silent link return", "S2", "same"),
    "silent loss, then idle": (CP29, "CP29: silent loss, then idle", "S2", "same"),
    "skewed author, receiver was off (H3 class)": (CP29, "CP29: H3 skew beyond the catch-up margin", "S2", "same"),
    "forged and bad-signature copies first (H8)": (CP29, "CP29: H8 squatting liar, invalid first", "S2", "same"),
    "delete before its post": (CP29, "CP29: delete arrives before its post", "S2", "same"),
    "partition heals": (CP29, "CP29: partition heals", "S2", "same"),
    "author crashes after posting, restarts": (CP29, "CP29: H5 offer to one box that leaves", "S2", "same"),
    "crash and restart": (CP29, "CP29: H2 gap never retried", "S2", "same"),
    "come on with 3 peers: each missing post comes once (K)":
        (CP29, "K owner leg fails, skipped session still running", "S2+KF'E",
         "other: the box counts duplicate fetches on come-on; the model fails the owner's fetch leg"),
    "held session, change meanwhile: fence keeps clean(1) off":
        (CP29, "F third-party change, push to Y silently lost", "S2+KF'E", "same"),
    "same with the fence off (mutation): clean(1) set wrongly":
        (None, None, None, "other: a mutant of the box; its model twin is a layer-3a mutant (sync_mc), not a script"),
    "quiet: some floor checks skipped (F')":
        (CP29, "F clean, then a quiet day (floor sessions day 2 <= 3)", "S2+KF'E",
         "other: the box checks that skips happen (> 0); the model checks day-2 floor sessions <= 3, which F' "
         "fails by design (CP30)"),
    "sub B post stays in sub B": (SUBS, "D1 a foreign sub never arrives", "sub", "same"),
    "sub B members cut apart, non-members between":
        (SUBS, "D5 a 2 h partition inside a sub, heal, post crosses", "sub", "same"),
}


def main() -> int:
    box = {s["scene"]: s["pass"] for s in json.loads(BOX.read_text(encoding="utf-8"))["scenes"]}
    models = {}
    for f in {CP29, SUBS}:
        for x in json.loads(f.read_text(encoding="utf-8")):
            models[(f, x["scene"], x.get("variant") or x.get("mode"))] = x["pass"]
    bad = 0
    for scene, ok in box.items():
        if scene not in TWINS:
            print(f"FAIL  no twin   {scene!r}: name its model script in tests/fast/model_box.py")
            bad += 1
            continue
        f, ms, var, rel = TWINS[scene]
        if rel != "same":
            print(f"skip  other     {scene!r} ({rel.split(':', 1)[1].strip()})")
            continue
        m = models.get((f, ms, var))
        if m is None:
            print(f"FAIL  missing   {scene!r}: model {ms!r} / {var} not in {f.name}")
            bad += 1
        elif m != ok:
            print(f"FAIL  mismatch  {scene!r}: box {ok}, model {ms!r} / {var} {m}")
            bad += 1
        else:
            print(f"PASS  same      {scene!r} = {ms!r} / {var}: {ok}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
