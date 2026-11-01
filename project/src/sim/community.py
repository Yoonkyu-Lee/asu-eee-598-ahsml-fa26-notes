"""Users, topics (subs), subscriptions and message rates. Every parameter carries its source or '(추정)'."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

DAY = 86400.0


@dataclass(frozen=True)
class Params:
    users: int = 1000
    # activity, per daily active user per day (filled from cp/CP16.md sources)
    dau_frac: float = 0.25
    posts_per_dau: float = 0.05
    comments_per_dau: float = 0.5
    votes_per_dau: float = 5.0
    lurker_frac: float = 0.9          # users who never write (they may still vote)
    tail_sigma: float = 1.5           # lognormal spread of per-user rates (heavy tail)
    # topics
    users_per_topic: float = 100.0
    zipf_s: float = 1.0
    subs_per_user: float = 10.0
    # content
    image_frac: float = 0.3
    image_bytes: float = 300e3
    post_bytes: float = 600.0
    comment_bytes: float = 200.0
    vote_bytes: float = 40.0          # one vote inside a message (post id + direction)
    overhead_bytes: float = 400.0     # header + signature 64 + RLN proof 128 (compressed) + nullifier, share
    # time of day: activity per hour relative to the daily mean (mean of the 24 values = 1)
    diurnal: tuple = field(default_factory=lambda: tuple(1 + 0.6 * np.cos(2 * np.pi * (np.arange(24) - 20) / 24)))
    # boxes and links
    online: float = 0.95
    hop_s: float = 0.05
    uplink_bps: float = 20e6
    vote_bundle_s: float = 50.0
    sources: dict = field(default_factory=dict, compare=False, hash=False)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


@dataclass
class Community:
    p: Params
    n_topics: int
    topic_w: np.ndarray            # popularity weights (sum 1)
    subs: list                     # subs[u] = array of topic ids
    su: np.ndarray                 # flat (user, topic) pairs: user index
    st: np.ndarray                 # flat pairs: topic index
    share: np.ndarray              # flat pairs: share of the user's activity in that topic
    user_rates: dict               # per user per second, daily mean

    def pair_rate(self, kind: str) -> np.ndarray:
        return self.user_rates[kind][self.su] * self.share


def build(p: Params, seed: int = 16) -> Community:
    rng = np.random.default_rng(seed)
    n_topics = max(5, int(round(p.users / p.users_per_topic)))
    w = 1.0 / np.arange(1, n_topics + 1) ** p.zipf_s
    w /= w.sum()
    # per-user daily rates: lurkers write nothing; writers share the per-DAU mean with a lognormal tail (mean 1)
    writer = rng.random(p.users) >= p.lurker_frac
    ln = lambda: rng.lognormal(-p.tail_sigma ** 2 / 2, p.tail_sigma, p.users)
    wfrac = max(1 - p.lurker_frac, 1e-9)
    user_rates = {
        "post": np.where(writer, ln(), 0.0) * p.posts_per_dau * p.dau_frac / wfrac / DAY,
        "comment": np.where(writer, ln(), 0.0) * p.comments_per_dau * p.dau_frac / wfrac / DAY,
        "vote": ln() * p.votes_per_dau * p.dau_frac / DAY,
    }
    # subscriptions: Poisson(subs_per_user) popularity-weighted draws per user, duplicates merged
    # (so popular topics make the realized count a little smaller than the draws; the realized mean is reported)
    k = np.maximum(rng.poisson(p.subs_per_user, p.users), 1)
    du = np.repeat(np.arange(p.users), k)
    dt = rng.choice(n_topics, size=len(du), p=w)
    key = np.unique(du.astype(np.int64) * n_topics + dt)
    su, st = (key // n_topics).astype(np.int64), (key % n_topics).astype(np.int64)
    bounds = np.searchsorted(su, np.arange(p.users + 1))
    subs = [st[bounds[i]:bounds[i + 1]] for i in range(p.users)]
    ws = w[st]
    tot = np.bincount(su, weights=ws, minlength=p.users)
    share = ws / tot[su]        # a user's activity is spread over their subs in proportion to topic popularity
    return Community(p, n_topics, w, subs, su, st, share, user_rates)


def topic_rates(c: Community, vote_mode: str = "V2", hour_factor: float = 1.0) -> dict:
    """Messages per second per topic, by kind, at an hour whose activity is hour_factor x the daily mean.
    V1: one message per vote. V2: one message per (user, topic) per bundle window in which the user voted there."""
    T = c.p.vote_bundle_s
    bc = lambda x: np.bincount(c.st, weights=x, minlength=c.n_topics)
    lam = c.pair_rate("vote") * hour_factor
    return {"post": bc(c.pair_rate("post") * hour_factor), "comment": bc(c.pair_rate("comment") * hour_factor),
            "vote": bc(lam if vote_mode == "V1" else (1 - np.exp(-lam * T)) / T), "votes_inside": bc(lam)}


def bytes_per_msg(p: Params, kind: str, vote_mode: str = "V2", lam_vote: float = 0.0) -> float:
    if kind == "post":
        return p.overhead_bytes + p.post_bytes + p.image_frac * p.image_bytes
    if kind == "comment":
        return p.overhead_bytes + p.comment_bytes
    return p.overhead_bytes + p.vote_bytes * (1 if vote_mode == "V1" else max(1.0, lam_vote))
