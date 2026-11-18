"""The oracle in two tiers (CLAUDE.md 7-11, 7-12; recast R3). Run from project/.

  .venv/Scripts/python.exe tests/oracle.py smoke     about 2 min: fast checks, model-box twins, layer 2 and 3a,
                                                      L1-L7, the one-screen story (headless)
  .venv/Scripts/python.exe tests/oracle.py full      about 25 min: smoke + sync_check (14 real-box scripts),
                                                      P1-P4; a recast piece merges only on a full pass

Every step runs as its own process and is judged from its exit code or its last lines. Real-box tests need
project/models and project/data (git-ignored; link them into a worktree) and a non-loopback address for L4 (the owner
API must refuse calls from the network), so lan_check and persist_check get this computer's first LAN address.
Logs: logs/oracle_<tier>_<step>.log.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FAIL = re.compile(r"^FAIL|^SOME FAIL|Traceback|[1-9]\d* not as predicted", re.M)


def lan_address() -> str:
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "box")]
    import lan_check
    return lan_check.addresses()[0]


def steps(tier: str) -> list[tuple[str, list[str]]]:
    a = lan_address()
    s = [("fast", [PY, "tests/fast/check.py"]),
         ("model_box", [PY, "tests/fast/model_box.py"]),
         ("vote_order", [PY, "tests/fast/vote_order.py"]),
         ("regress", [PY, "-W", "ignore", "experiments/sync_subs.py", "regress"]),
         ("holes", [PY, "-W", "ignore", "experiments/sync_subs.py", "holes"]),
         ("mc", [PY, "-W", "ignore", "experiments/sync_subs.py", "mc"]),
         ("lan_recon", [PY, "-W", "ignore", "tests/lan_check.py", a, "results/tests/lan_check_recon.json"]),
         ("stage", [PY, "-W", "ignore", "box/stage.py", "--headless"])]
    if tier == "full":
        s += [("sync_check", [PY, "-W", "ignore", "tests/sync_check.py"]),
              ("persist", [PY, "-W", "ignore", "tests/persist_check.py", a, "results/tests/persist_check_recon.json"])]
    return s


def main() -> int:
    tier = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    (ROOT / "logs").mkdir(exist_ok=True)
    bad = []
    for name, cmd in steps(tier):
        t0 = time.time()
        log = ROOT / "logs" / f"oracle_{tier}_{name}.log"
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        tail = "\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-3:])
        ok = rc == 0 and not FAIL.search(tail)
        print(f"{'PASS' if ok else 'FAIL'}  {name:10} {time.time() - t0:6.0f} s", flush=True)
        if not ok:
            bad.append(name)
    print(f"{tier}: {'ALL PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
