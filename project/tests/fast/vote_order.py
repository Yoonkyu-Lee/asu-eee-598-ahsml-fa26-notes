"""A vote that reaches a box before its post still counts (set reconciliation and offers deliver in any order).

  .venv/Scripts/python.exe tests/fast/vote_order.py      exit 1 on failure

Found as the flaky step 8 of the one-screen story (about 2 runs in 7): the laptop box came back, got the photo's vote
before the photo, and its score stayed 0 (journal/cp/CP38.md, recast R3).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.box import Box, Shared, World, sub_key  # noqa: E402


def score_after(order: str) -> int:
    w, S = World(2), Shared()
    box = Box(0, [0], {}, S, {"text": 4})
    post = w.make(0, "post", 0, 1.0, {"title": "a photo"}, 1000.0, sub_key(0))
    vote = w.make(1, "vote", 0, 2.0, {"target": post.mid, "dir": 1}, 440.0, sub_key(0), target=post.mid)
    for m in ((vote, post) if order == "vote first" else (post, vote)):
        box.apply(m, 3.0, box.validate(m))
    return box.items[post.mid]["score"]


def main() -> int:
    bad = 0
    for order in ("post first", "vote first"):
        s = score_after(order)
        ok = s == 1
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {order}: score {s} (want 1)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
