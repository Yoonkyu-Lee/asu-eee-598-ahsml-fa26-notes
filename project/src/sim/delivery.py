"""Image post delivery inside one topic mesh (CP17 decision 1).

I1  push the whole post (image inside) to every mesh peer (CP16).
I2  push a small envelope (header + thumbnail); each box then pulls the image once from the peer that gave it the
    envelope, which serves as soon as it has the image itself.
I3  push an envelope that also carries the author's image embedding: a box indexes the post on arrival. A box checks
    with probability q = min(1, c / n): pulls the image the same way, re-encodes, compares. A box that finds a
    mismatch publishes a report; a box receiving a report hides the post, then pulls and re-encodes to confirm.
Pulls go to the gossip parent (never straight to the author), so the author is no easier to find than with I1.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .engine import Engine
from .network import VERIFY_S, meshes


class ImageDelivery:
    def __init__(self, members: np.ndarray, M: int, mode: str, *, D: int = 6, uplink_bps: float, hop_s: float,
                 enc_s: float, full_bytes: float, env_bytes: float, report_bytes: float = 600.0, c: float = 14.0,
                 seed: int = 17):
        self.mode, self.M = mode, M
        self.rng = np.random.default_rng(seed)
        self.mesh = meshes({0: members}, D, self.rng)[0]
        self.n = len(members)
        self.q = min(1.0, c / self.n)
        self.up, self.hop, self.enc_s = uplink_bps, hop_s, enc_s
        self.full_b, self.env_b, self.rep_b = full_bytes, env_bytes, report_bytes
        self.eng = Engine()
        self.uplink_free = np.zeros(M)
        self.enc_free = np.zeros(M)
        self.encodes = np.zeros(M)
        self.tx_bytes = np.zeros(M)
        self.dup_bytes = np.zeros(M)
        self.posts: list[dict] = []

    # ---- sending
    def _send(self, frm: int, to: int, nbytes: float, t_ready: float, fn, *args) -> None:
        start = max(t_ready, self.uplink_free[frm])
        self.uplink_free[frm] = start + nbytes * 8 / self.up
        self.tx_bytes[frm] += nbytes
        self.eng.at(self.uplink_free[frm] + self.hop, fn, *args)

    def _encode(self, box: int, t_ready: float) -> float:
        start = max(t_ready, self.enc_free[box])
        self.enc_free[box] = start + self.enc_s
        self.encodes[box] += 1
        return self.enc_free[box]

    # ---- publish
    def publish(self, author: int, false_emb: bool = False) -> None:
        M, t = self.M, self.eng.t
        m = {"t0": t, "author": author, "false": false_emb, "env": np.full(M, np.nan), "full": np.full(M, np.nan),
             "search": np.full(M, np.nan), "hidden": np.full(M, np.nan), "report_seen": np.zeros(M, bool),
             "parent": np.full(M, -1), "want": defaultdict(list), "asked": np.zeros(M, bool), "check": np.zeros(M, bool),
             "checkers": 0, "reported": False}
        self.posts.append(m)
        pid = len(self.posts) - 1
        m["full"][author] = t
        m["env"][author] = t
        m["search"][author] = self._encode(author, t)  # the author's box indexes its own post (and, I3, ships it)
        self._forward(author, pid, -1, t)

    def _forward(self, box: int, pid: int, frm: int, t_ready: float) -> None:
        size = self.full_b if self.mode == "I1" else self.env_b
        for peer in self.mesh.get(box, ()):
            if peer != frm:
                self._send(box, int(peer), size, t_ready, self.arrive, int(peer), pid, box)

    # ---- receiving the post (I1: whole post, I2/I3: envelope)
    def arrive(self, box: int, pid: int, frm: int) -> None:
        m, t = self.posts[pid], self.eng.t
        if not np.isnan(m["env"][box]):
            self.dup_bytes[box] += self.full_b if self.mode == "I1" else self.env_b
            return
        m["env"][box], m["parent"][box] = t, frm
        t_ok = t + VERIFY_S
        self._forward(box, pid, frm, t_ok)
        if self.mode == "I1":
            m["full"][box] = t
            m["search"][box] = self._encode(box, t_ok)
        elif self.mode == "I2":
            m["check"][box] = True
            self._request(box, pid, t_ok)
        else:
            m["search"][box] = t_ok
            if self.rng.random() < self.q:
                m["checkers"] += 1
                m["check"][box] = True
                self._request(box, pid, t_ok)

    # ---- pulling the image along the gossip path
    def _request(self, box: int, pid: int, t_ready: float) -> None:
        m = self.posts[pid]
        if m["asked"][box]:
            return
        m["asked"][box] = True
        par = int(m["parent"][box])
        self.eng.at(t_ready + self.hop, self._serve_or_wait, par, box, pid)

    def _serve_or_wait(self, holder: int, box: int, pid: int) -> None:
        m = self.posts[pid]
        if not np.isnan(m["full"][holder]):
            self._send(holder, box, self.full_b, self.eng.t, self.got_full, box, pid)
        else:
            m["want"][holder].append(box)
            self._request(holder, pid, self.eng.t)

    def got_full(self, box: int, pid: int) -> None:
        m, t = self.posts[pid], self.eng.t
        m["full"][box] = t
        for b in m["want"].pop(box, []):
            self._send(box, b, self.full_b, t, self.got_full, b, pid)
        if not m["check"][box]:            # a relay that pulled only to pass the image on
            return
        done = self._encode(box, t)
        if self.mode == "I2":
            m["search"][box] = done
        elif m["false"] and not m["report_seen"][box]:
            self.eng.at(done, self._report, box, pid)

    # ---- I3 reports
    def _report(self, box: int, pid: int) -> None:
        m = self.posts[pid]
        m["reported"] = True
        self.report_arrive(box, pid, -1)

    def report_arrive(self, box: int, pid: int, frm: int) -> None:
        m, t = self.posts[pid], self.eng.t
        if m["report_seen"][box]:
            self.dup_bytes[box] += self.rep_b
            return
        m["report_seen"][box] = True
        m["hidden"][box] = t
        for peer in self.mesh.get(box, ()):
            if peer != frm:
                self._send(box, int(peer), self.rep_b, t + VERIFY_S, self.report_arrive, int(peer), pid, box)
        if not m["check"][box]:                # confirm: pull and re-encode (a checker has already done so)
            m["check"][box] = True
            if np.isnan(m["full"][box]):
                self._request(box, pid, t + VERIFY_S)
            else:
                self._encode(box, t)

    # ---- results
    def summary(self, members: np.ndarray) -> dict:
        honest = [m for m in self.posts if not m["false"]]
        d = np.concatenate([m["search"][members] - m["t0"] for m in honest])
        out = {"searchable_s": {"p50": float(np.nanpercentile(d, 50)), "p99": float(np.nanpercentile(d, 99)),
                                "max": float(np.nanmax(d)), "missing": int(np.isnan(d).sum())},
               "encodes_per_post_per_box": {"mean": float(self.encodes[members].mean() / max(len(self.posts), 1)),
                                            "p99_box": float(np.percentile(self.encodes[members], 99) /
                                                             max(len(self.posts), 1))},
               "tx_bytes_per_post_per_box": {"mean": float(self.tx_bytes[members].mean() / max(len(self.posts), 1)),
                                             "p99_box": float(np.percentile(self.tx_bytes[members], 99) /
                                                              max(len(self.posts), 1)),
                                             "max_box": float(self.tx_bytes[members].max() / max(len(self.posts), 1))},
               "dup_bytes_per_post_per_box": float(self.dup_bytes[members].mean() / max(len(self.posts), 1)),
               "holders": int(self.n), "q": self.q}
        if self.mode == "I3":
            ck = np.array([m["checkers"] for m in honest])
            out["checkers_per_post"] = {"mean": float(ck.mean()), "min": int(ck.min())}
            out["p_unchecked"] = float((1 - self.q) ** self.n)
            bad = [m for m in self.posts if m["false"]]
            if bad:
                h = np.array([np.nanmax(m["hidden"][members]) - m["t0"] if np.all(~np.isnan(m["hidden"][members]))
                              else np.inf for m in bad])
                out["false_hidden_everywhere_s"] = {"p50": float(np.percentile(h, 50)),
                                                    "p99": float(np.percentile(h, 99)), "max": float(h.max()),
                                                    "caught": int(np.sum([m["reported"] for m in bad])),
                                                    "n": len(bad)}
        return out
