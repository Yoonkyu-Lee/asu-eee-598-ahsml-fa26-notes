"""Replication policies: which topics each box holds. Availability = chance that, with the author's box off,
another box holding the topic is on (boxes on independently with probability `online`)."""
from __future__ import annotations

import math

import numpy as np

from .community import Community

AVAIL_TARGET = 0.99


def replicas_needed(online: float, target: float = AVAIL_TARGET) -> int:
    """Holders r so that 1 - (1 - online)^(r - 1) >= target (the author's own box counted as one holder)."""
    return 1 + math.ceil(math.log(1 - target) / math.log(1 - online))


def holders(c: Community, policy: str, seed: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Flat (box, topic) pairs held. P-all, P-sub, P-sub+k (topics short of replicas get extra boxes,
    picked among the least-loaded boxes)."""
    M, T = c.p.users, c.n_topics
    if policy == "P-all":  # every box holds every topic: load is the community total (handled without pairs)
        return None, None
    hb, ht = c.su.copy(), c.st.copy()
    if policy == "P-sub":
        return hb, ht
    rng = np.random.default_rng(seed)
    need = replicas_needed(c.p.online)
    count = np.bincount(ht, minlength=T)
    load = np.bincount(hb, minlength=M).astype(float)
    extra_b, extra_t = [], []
    order = np.argsort(ht, kind="stable")
    bounds = np.searchsorted(ht[order], np.arange(T + 1))
    for t in np.nonzero(count < need)[0]:
        have = set(hb[order[bounds[t]:bounds[t + 1]]].tolist())
        cand = rng.choice(M, size=min(M, 8 * need), replace=False)
        cand = [b for b in cand[np.argsort(load[cand], kind="stable")] if b not in have][: need - count[t]]
        for b in cand:
            extra_b.append(b)
            extra_t.append(t)
            load[b] += 1
    return np.concatenate([hb, np.array(extra_b, dtype=np.int64)]), np.concatenate([ht, np.array(extra_t, dtype=np.int64)])


def availability(c: Community, hb: np.ndarray, ht: np.ndarray, topic_post_rate: np.ndarray) -> dict:
    """Per topic 1 - (1 - online)^(holders - 1); min over topics that have posts, and post-weighted mean."""
    n = np.full(c.n_topics, c.p.users) if ht is None else np.bincount(ht, minlength=c.n_topics)
    a = 1 - (1 - c.p.online) ** np.maximum(n - 1, 0)
    a = np.where(n >= 2, a, 0.0)
    live = topic_post_rate > 0
    return {"min": float(a[live].min()) if live.any() else 1.0,
            "post_weighted": float((a * topic_post_rate).sum() / topic_post_rate.sum()),
            "topics_below_target": int((a[live] < AVAIL_TARGET).sum()), "replicas_needed": replicas_needed(c.p.online)}


def box_load(c: Community, hb: np.ndarray, ht: np.ndarray, rates: dict) -> dict:
    """Per box messages per second (and per kind) = sum of the held topics' rates."""
    M = c.p.users
    if hb is None:
        return {kind: np.full(M, r.sum()) for kind, r in rates.items()}
    return {kind: np.bincount(hb, weights=r[ht], minlength=M) for kind, r in rates.items()}
