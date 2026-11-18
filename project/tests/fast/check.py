"""Fast checks (CLAUDE.md 7, gate option G-a): seconds, no boxes, run before every commit.

  .venv/Scripts/python.exe tests/fast/check.py        exit 1 on any FAIL

  numbers   chip/model/chip.py reads its chip numbers from results/chip/*.json, and README.md and docs/chip.md quote
            the same numbers (CLAUDE.md 6: numbers flow JSON -> chip.py -> docs)
  excuse    no added comment justifies a workaround or defers a fix (CLAUDE.md 7-6; pattern from the recast gate.py)
  legacy    current code does not import from legacy/ (CP31: legacy is frozen and may not run)
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CURRENT = ("box", "sim", "chip", "bench", "experiments", "tests")
EXCUSE = re.compile(r"(?i:\b(workaround|hacky?|for now|in case|temporar\w*|quick[- ]?fix|band-?aid|shortcut:))"
                    r"|\b(TODO|FIXME|XXX|HACK)\b")
fails: list[str] = []


def fail(check: str, msg: str) -> None:
    fails.append(f"FAIL  {check:8} {msg}")


def sources(suffix: str = ".py"):
    for d in CURRENT:
        yield from (p for p in (ROOT / d).rglob(f"*{suffix}") if "__pycache__" not in p.parts and
                    "target" not in p.parts)


def check_numbers() -> None:
    from chip.model import chip
    s = chip.SPEC
    res = ROOT / "results" / "chip"
    c34 = json.loads((res / "cp34_chip.json").read_text(encoding="utf-8"))
    area = c34["area"]["16x16 2MB w16 (FP16 MAC)"]
    power = c34["power"]["w16 16x16 2MB"]
    worst = [m for m in json.loads((res / "cp14_fixed.json").read_text(encoding="utf-8"))["machines"]["V4 +O4 (w8)"]
             if (m["M"], m["A"]) == (s["verifier"]["multipliers"], s["verifier"]["adders"])][0]["worst"]
    want = {"area_mm2": round(area["chip_mm2"], 1), "encoder_mm2": round(area["encoder_mm2"], 2),
            "verifier_mm2": round(area["chip_mm2"] - area["encoder_mm2"], 2),
            "min_proofs_per_s": int(worst["proofs_per_s"]),
            "ordinary_day_mW": round(power["S1_chip_mW"], 1), "flood_W": round(power["S4_chip_W"], 2)}
    have = {"area_mm2": s["area_mm2"], "encoder_mm2": s["encoder_mm2"], "verifier_mm2": s["verifier"]["area_mm2"],
            "min_proofs_per_s": s["verifier"]["min_proofs_per_s"],
            "ordinary_day_mW": s["power"]["ordinary_day_mW"], "flood_W": s["power"]["flood_W"]}
    for k, v in want.items():
        if have[k] != v:
            fail("numbers", f"chip.py {k} = {have[k]} but results/chip says {v}")
    quotes = [f"{s['area_mm2']} mm²", f"{s['verifier']['area_mm2']} mm²", f"{s['verifier']['min_proofs_per_s']:,}",
              f"{s['power']['ordinary_day_mW']} mW", f"{s['power']['flood_W']} W"]
    for doc in ("README.md", "docs/chip.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for q in quotes:
            if q not in text:
                fail("numbers", f"{doc} does not quote chip.py's '{q}'")


def check_excuse() -> None:
    for p in sources():
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "#" in line:
                comment = line[line.index("#"):]
                if EXCUSE.search(comment):
                    fail("excuse", f"{p.relative_to(ROOT)}:{i}: {comment.strip()[:80]}")


def check_legacy() -> None:
    for p in sources():
        if p.resolve() == Path(__file__).resolve():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                if n.split(".")[0] == "legacy":
                    fail("legacy", f"{p.relative_to(ROOT)}:{node.lineno} imports {n}")


def main() -> int:
    for check in (check_numbers, check_excuse, check_legacy):
        n = len(fails)
        check()
        print(f"{'PASS' if len(fails) == n else 'FAIL'}  {check.__name__[6:]}")
    print("\n".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
