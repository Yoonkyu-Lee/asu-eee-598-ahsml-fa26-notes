"""Traces in the Chrome/Perfetto JSON format (open in ui.perfetto.dev or chrome://tracing)."""
from __future__ import annotations

import json
from pathlib import Path


class Trace:
    def __init__(self):
        self.ev: list[dict] = []

    def counter(self, pid: str, name: str, t_s: float, values: dict) -> None:
        self.ev.append({"ph": "C", "pid": pid, "name": name, "ts": t_s * 1e6, "args": values})

    def span(self, pid: str, tid: str, name: str, t0_s: float, dur_s: float, args: dict | None = None) -> None:
        self.ev.append({"ph": "X", "pid": pid, "tid": tid, "name": name, "ts": t0_s * 1e6, "dur": dur_s * 1e6,
                        "args": args or {}})

    def write(self, path: Path) -> None:
        path.write_text(json.dumps({"traceEvents": self.ev, "displayTimeUnit": "ms"}), encoding="utf-8")
