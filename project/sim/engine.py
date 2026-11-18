"""Discrete-event engine: a heap of (time, seq, fn, args). Virtual seconds."""
from __future__ import annotations

import heapq


class Engine:
    def __init__(self):
        self.t = 0.0
        self._q: list = []
        self._seq = 0
        self.events = 0

    def at(self, t: float, fn, *args) -> None:
        self._seq += 1
        heapq.heappush(self._q, (t, self._seq, fn, args))

    def after(self, dt: float, fn, *args) -> None:
        self.at(self.t + dt, fn, *args)

    def run(self, until: float = float("inf")) -> None:
        q = self._q
        while q and q[0][0] <= until:
            t, _, fn, args = heapq.heappop(q)
            self.t = t
            self.events += 1
            fn(*args)
        self.t = max(self.t, until if until != float("inf") else self.t)
