"""A checkout of an old tag for models that only run as they were (CLAUDE.md 6: old variants live on tags).

    from chip.model.at_tag import tag_tree
    wt = tag_tree("cp30")          # <repo>/.worktrees/cp30/project, made with `git worktree add` on first use

The CP19/CP20 box models (sim_cp19, sim_cp20, sim_cp24) and their inputs (results/cp14, cp16, cp17, cp19) run only at
tag cp30; chip_w16.py (CP34) and connection_duel.py (CP35) import them from here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def tag_tree(tag: str = "cp30") -> Path:
    wt = REPO / ".worktrees" / tag
    if not (wt / "project").is_dir():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(wt), tag], check=True,
                       capture_output=True)
    return wt / "project"
