"""Gossip over topic meshes on the event engine (gossipsub v1.1 eager push): a box validates a message the first
time it sees it, then forwards it to its mesh peers in that topic; later copies are duplicates (counted, not
validated). Each box's uplink sends one copy after another, so big messages queue behind each other."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .community import Community, bytes_per_msg
from .engine import Engine

VERIFY_S = 1 / 13_429   # CP14: one proof on the 6 x 6 verify unit, worst-mix rate (추정)


def meshes(members: dict, D: int, rng) -> dict:
    """members[topic] = array of online holders. Each picks D random peers; links are made symmetric."""
    mesh = {}
    for t, m in members.items():
        peers = defaultdict(set)
        if len(m) <= D + 1:
            for a in m:
                peers[a] = set(m) - {a}
        else:
            for a in m:
                for b in rng.choice(m, size=D + 1, replace=False):
                    if b != a and len(peers[a]) < D:
                        peers[a].add(b)
                        peers[b].add(a)
        mesh[t] = {a: np.array(sorted(peers[a])) for a in m}
    return mesh


class Gossip:
    def __init__(self, c: Community, hb, ht, D: int = 6, seed: int = 16, watch: tuple = ()):
        self.c, self.p = c, c.p
        rng = np.random.default_rng(seed)
        self.rng = rng
        M = c.p.users
        self.on = rng.random(M) < c.p.online
        members = defaultdict(list)
        for b, t in zip(hb.tolist(), ht.tolist()):
            if self.on[b]:
                members[t].append(b)
        self.members = {t: np.array(v) for t, v in members.items()}
        self.mesh = meshes(self.members, D, rng)
        self.eng = Engine()
        self.uplink_free = np.zeros(M)
        self.seen = [set() for _ in range(M)]
        self.count = defaultdict(lambda: np.zeros(M))       # count[kind][box] unique messages
        self.dup_bytes = np.zeros(M)
        self.rx_bytes = np.zeros(M)
        self.msgs: list[dict] = []
        self.watch = set(watch)
        self.watch_log: list = []                            # (box, t, kind) for the trace

    def publish(self, author: int, topic: int, kind: str, size: float, extra: dict) -> None:
        mid = len(self.msgs)
        self.msgs.append({"t0": self.eng.t, "topic": topic, "kind": kind, "size": size, "n": 0, "tmax": 0.0, **extra})
        self.deliver(author, mid, -1)

    def deliver(self, box: int, mid: int, frm: int) -> None:
        m = self.msgs[mid]
        if mid in self.seen[box]:
            self.dup_bytes[box] += m["size"]
            return
        self.seen[box].add(mid)
        if frm >= 0:
            self.rx_bytes[box] += m["size"]
        m["n"] += 1
        m["tmax"] = self.eng.t
        self.count[m["kind"]][box] += 1
        if box in self.watch:
            self.watch_log.append((box, self.eng.t, m["kind"]))
        t = self.eng.t + VERIFY_S
        tx = m["size"] * 8 / self.p.uplink_bps
        for peer in self.mesh[m["topic"]].get(box, ()):
            if peer == frm:
                continue
            start = max(t, self.uplink_free[box])
            self.uplink_free[box] = start + tx
            self.eng.at(start + tx + self.p.hop_s, self.deliver, int(peer), mid, box)

    def schedule_hour(self, hour_factor: float, H: float, vote_mode: str) -> None:
        """Poisson arrivals over [0, H) for every (user, topic) pair whose user box is online."""
        c, p, rng = self.c, self.p, self.rng
        ok = self.on[c.su]
        su, st = c.su[ok], c.st[ok]
        for kind in ("post", "comment"):
            lam = c.pair_rate(kind)[ok] * hour_factor * H
            n = rng.poisson(lam)
            for u, t, k in zip(su[n > 0], st[n > 0], n[n > 0]):
                for t0 in rng.random(k) * H:
                    img = kind == "post" and rng.random() < p.image_frac
                    size = p.overhead_bytes + (p.post_bytes + (p.image_bytes if img else 0) if kind == "post"
                                               else p.comment_bytes)
                    self.eng.at(t0, self.publish, int(u), int(t), kind, size, {"image": bool(img)})
        lam = c.pair_rate("vote")[ok] * hour_factor * H
        n = rng.poisson(lam)
        T = p.vote_bundle_s
        for u, t, k in zip(su[n > 0], st[n > 0], n[n > 0]):
            times = rng.random(k) * H
            if vote_mode == "V1":
                for t0 in times:
                    self.eng.at(t0, self.publish, int(u), int(t), "vote", bytes_per_msg(p, "vote", "V1"), {"votes": 1})
            else:  # one message per bundle window that holds at least one vote, sent at the window's end
                w, cnt = np.unique(np.floor(times / T), return_counts=True)
                for wi, ci in zip(w, cnt):
                    size = p.overhead_bytes + p.vote_bytes * ci
                    self.eng.at(min((wi + 1) * T, H), self.publish, int(u), int(t), "vote", size,
                                {"votes": int(ci), "wait": float((wi + 1) * T - times[np.floor(times / T) == wi].min())})

    def spread(self) -> dict:
        """Per message: time until every online holder of its topic has it; delivery completeness."""
        out = defaultdict(list)
        missing = total = 0
        for m in self.msgs:
            n_hold = len(self.members.get(m["topic"], ()))
            total += n_hold
            missing += n_hold - m["n"]
            key = "image post" if m.get("image") else m["kind"]
            out[key].append(m["tmax"] - m["t0"] if m["n"] else np.nan)
        return {"delay_s": {k: np.array(v) for k, v in out.items()}, "completeness": 1 - missing / max(total, 1)}
