"""Semantic neighbourhoods + vote-driven spread (CP39, journal/cp/CP39.md): who gets which public post, in a model.

Content: real Reddit 2010 posts (data/neighborhoods/posts.npz) with the subreddit as a hidden label; boxes never see
labels. A box's owner has 1-3 true interests (subreddits). Its box keeps an interest vector u (unit) that starts from a
few posts of those interests and moves toward posts its owner reads. Neighbours: D_SEM semantic ones (closest
advertised u) + D_RAND random ones, refreshed every refresh_s by asking one neighbour for its neighbour list.

A post reaches a box, the box verifies and encodes it (its own embedding, the routing value); the owner reads it if it
is of one of the owner's interests (a person's judgement, so the label) and may vote. The box forwards it to neighbours whose advertised u matches the post
(cos >= fwd_cos); a post with v votes also crosses up to floor(log2(1 + v)) non-matching hops (the rumour).
Baselines on the same posts: "members" (CP33: exactly the boxes whose owner holds the label) and "flood" (everyone).

Attack hooks (R5): spam posts whose claimed embedding is rotated toward a target subreddit by tol_deg (the honest
tolerance a relay must accept when devices differ; 0 = deterministic encoder), and bought votes.
Not modelled: byte sizes beyond a count, signatures and proofs (counted as one verification per receipt), vote messages
travelling with the post (a relay reads the post's current vote count), on/off beyond an always-on share.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim.engine import Engine

DAY = 86400.0
HOP_S = 0.05                     # per transfer (CP16)
D_SEM, D_RAND = 6, 2


@dataclass
class HoodScene:
    n: int = 300
    days: float = 2.0
    posts_per_box_day: float = 3.0
    interests: tuple = (1, 3)            # true interests per owner, inclusive
    seed_posts: int = 5                  # posts of its interests a box starts u from (cold start)
    alpha: float = 0.05                  # u <- unit(u + alpha e) per post read
    fwd_cos: float = 0.217               # forward to a neighbour if cos(e, u_neighbour) >= fwd_cos. CP39 calibration
                                         # (before the full runs): the 30th percentile of cos(post, interest) over
                                         # pairs where the post is of the owner's interest (out-of-interest: 13 % pass)
    vote_p: float = 0.2                  # an owner who reads a post of a true interest votes for it
    rumour: bool = True                  # votes buy non-matching hops: floor(log2(1 + v))
    refresh_s: float = 600.0
    always: float = 1.0                  # share of boxes always on; the rest on 8 h a day (laptops)
    advertise_sigma: float = 0.0         # noise on the advertised u (R4 mitigation)
    tol_deg: float = 0.0                 # R5: claimed-embedding tolerance relays accept (0 = deterministic)
    spam_boxes: int = 0                  # R5: attacker boxes
    spam_per_day: float = 0.0            # R5: spam posts per attacker box per day
    spam_target: int = -1                # R5: subreddit the spam is steered toward
    bought_votes: int = 0                # R5: votes the attacker adds to each spam post
    seed: int = 39


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def rotate(e, c, deg):
    cosang = float(np.clip(e @ c, -1, 1))
    ang = math.acos(cosang)
    step = min(math.radians(deg), ang)
    if ang < 1e-9:
        return e
    perp = unit(c - cosang * e)
    return unit(math.cos(step) * e + math.sin(step) * perp)


class Hood:
    def __init__(self, sc: HoodScene, world: dict):
        self.sc, self.w = sc, world
        self.rng = np.random.default_rng(sc.seed)
        self.eng = Engine()
        E, lab = world["emb"], world["label"]
        self.n_sub = int(lab.max()) + 1
        self.by_sub = [np.flatnonzero(lab == s) for s in range(self.n_sub)]
        size = np.array([len(x) for x in self.by_sub], float)
        self.cent = unit(np.stack([E[ix].mean(0) for ix in self.by_sub]))
        n = sc.n
        self.true = []
        for b in range(n):
            k = int(self.rng.integers(sc.interests[0], sc.interests[1] + 1))
            self.true.append(set(self.rng.choice(self.n_sub, k, replace=False, p=size / size.sum()).tolist()))
        self.u = np.zeros((n, E.shape[1]))
        for b in range(n):
            ix = np.concatenate([self.rng.choice(self.by_sub[s], sc.seed_posts) for s in self.true[b]])
            self.u[b] = unit(E[ix].mean(0))
        self.always = self.rng.random(n) < sc.always
        self.on_off = self.rng.uniform(0, DAY, n)                        # laptops: on from this time of day for 8 h
        self.spam = set(range(n - sc.spam_boxes, n)) if sc.spam_boxes else set()
        self.nb = [self._fresh(b) for b in range(n)]
        self.has: list[dict] = [dict() for _ in range(n)]             # post -> arrival time
        self.posts: list[dict] = []
        self.votes: list[int] = []
        self.c = {"receipts": 0, "encodes": 0, "forwards": 0, "reads": 0, "votes": 0}

    # ---- helpers
    def on(self, b, t):
        if self.always[b]:
            return True
        return ((t - self.on_off[b]) % DAY) < 8 * 3600.0

    def adv(self, b):
        """The interest vector box b advertises to its neighbours (R4: with noise)."""
        if not self.sc.advertise_sigma:
            return self.u[b]
        return unit(self.u[b] + self.rng.normal(0, self.sc.advertise_sigma, self.u.shape[1]))

    def _fresh(self, b, cands=None):
        n = self.sc.n
        pool = np.array(sorted(set(range(n)) - {b} if cands is None else set(cands) - {b}))
        if cands is None:
            pool = self.rng.choice(pool, min(60, len(pool)), replace=False)
        adv = np.stack([self.adv(x) for x in pool])
        sem = pool[np.argsort(-(adv @ self.u[b]))[:D_SEM]].tolist()
        rest = [x for x in range(n) if x != b and x not in sem]
        return sem + self.rng.choice(rest, D_RAND, replace=False).tolist()

    def refresh(self, b):
        t = self.eng.t
        if self.on(b, t):
            q = self.nb[b][int(self.rng.integers(len(self.nb[b])))]
            if self.on(q, t):
                self.nb[b] = self._fresh(b, set(self.nb[b]) | set(self.nb[q]))
        self.eng.after(self.sc.refresh_s, self.refresh, b)

    # ---- posts
    def write(self, b):
        t = self.eng.t
        sc = self.sc
        if self.on(b, t):
            if b in self.spam:
                ix = int(self.rng.integers(len(self.w["emb"])))
                claimed = rotate(self.w["emb"][ix], self.cent[sc.spam_target], sc.tol_deg)
                self._publish(b, ix, label=-1, claimed=claimed, votes=sc.bought_votes)
            else:
                s = int(self.rng.choice(sorted(self.true[b])))
                self._publish(b, int(self.rng.choice(self.by_sub[s])), label=s)
        rate = sc.spam_per_day if b in self.spam else sc.posts_per_box_day
        self.eng.after(self.rng.exponential(DAY / max(rate, 1e-9)), self.write, b)

    def _publish(self, b, ix, label, claimed=None, votes=0):
        pid = len(self.posts)
        e = self.w["emb"][ix]
        self.posts.append({"ix": ix, "label": label, "t": self.eng.t, "author": b,
                           "e": e if claimed is None else claimed})
        self.votes.append(votes)
        self.has[b][pid] = self.eng.t
        self._forward(b, pid, 0)

    def _receive(self, b, pid, frm, off_hops):
        t = self.eng.t
        if pid in self.has[b] or not self.on(b, t):
            return
        self.has[b][pid] = t
        self.c["receipts"] += 1
        self.c["encodes"] += 1
        p = self.posts[pid]
        e = p["e"]                                       # the routing value (claimed within tolerance, R5)
        if p["label"] in self.true[b] and b not in self.spam:
            self.c["reads"] += 1
            self.u[b] = unit(self.u[b] + self.sc.alpha * self.w["emb"][p["ix"]])
            if self.rng.random() < self.sc.vote_p:
                self.votes[pid] += 1
                self.c["votes"] += 1
        self._forward(b, pid, off_hops)

    def _forward(self, b, pid, off_hops):
        e = self.posts[pid]["e"]
        reach = int(math.log2(1 + self.votes[pid])) if self.sc.rumour else 0
        for q in self.nb[b]:
            match = float(e @ self.adv(q)) >= self.sc.fwd_cos
            if match or off_hops < reach:
                self.c["forwards"] += 1
                self.eng.after(HOP_S, self._receive, q, pid, b, off_hops + (0 if match else 1))

    def run(self) -> dict:
        sc = self.sc
        for b in range(sc.n):
            rate = sc.spam_per_day if b in self.spam else sc.posts_per_box_day
            if rate > 0:
                self.eng.after(self.rng.exponential(DAY / rate), self.write, b)
            self.eng.after(self.rng.uniform(0, sc.refresh_s), self.refresh, b)
        self.eng.run(sc.days * DAY)
        return self.result()

    # ---- results
    def result(self) -> dict:
        sc, n = self.sc, self.sc.n
        honest = [b for b in range(n) if b not in self.spam]
        small = set(self.w.get("small_ids", []))
        T = (60.0, 3600.0)
        rel = {T[0]: [0, 0], T[1]: [0, 0]}
        rel_small = {T[0]: [0, 0], T[1]: [0, 0]}
        waste_rx = rx = 0
        members_rx = 0
        for pid, p in enumerate(self.posts):
            if p["label"] < 0:
                continue
            for b in honest:
                if b == p["author"]:
                    continue
                want = p["label"] in self.true[b]
                got = self.has[b].get(pid)
                if got is not None:
                    rx += 1
                    waste_rx += not want
                if want and self.on(b, p["t"]):
                    members_rx += 1
                    for t in T:
                        ok = got is not None and got - p["t"] <= t
                        rel[t][0] += ok
                        rel[t][1] += 1
                        if p["label"] in small:
                            rel_small[t][0] += ok
                            rel_small[t][1] += 1
        out = {"n": n, "posts": len(self.posts), "counts": self.c,
               "recall_60s": rel[T[0]][0] / max(rel[T[0]][1], 1), "recall_1h": rel[T[1]][0] / max(rel[T[1]][1], 1),
               "recall_small_60s": rel_small[T[0]][0] / max(rel_small[T[0]][1], 1),
               "recall_small_1h": rel_small[T[1]][0] / max(rel_small[T[1]][1], 1),
               "waste_share": waste_rx / max(rx, 1), "receipts_per_box_day": rx / len(honest) / sc.days,
               "members_receipts_per_box_day": members_rx / len(honest) / sc.days,
               "flood_receipts_per_box_day": sum(1 for p in self.posts if p["label"] >= 0) * (len(honest) - 1)
               / len(honest) / sc.days}
        if self.spam:
            tgt = [b for b in honest if sc.spam_target in self.true[b]]
            sp = [pid for pid, p in enumerate(self.posts) if p["label"] < 0]
            out["spam_reach_target"] = float(np.mean([sum(pid in self.has[b] for b in tgt) / max(len(tgt), 1) for pid in sp])) if sp else None
            out["spam_reach_all"] = float(np.mean([sum(pid in self.has[b] for b in honest) / len(honest) for pid in sp])) if sp else None
        # R4: a neighbour guesses the owner's main interest from the advertised vector (nearest subreddit centre)
        guess = np.argmax(np.stack([self.adv(b) for b in honest]) @ self.cent.T, axis=1)
        out["exposure_top1"] = float(np.mean([g in self.true[b] for g, b in zip(guess, honest)]))
        return out
