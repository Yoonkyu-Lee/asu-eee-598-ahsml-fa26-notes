"""Set reconciliation between two boxes (CP29). Shared by the model (sim/sync.py) and the demo box.

A box's reconciled set is a sorted set of keys (t_key, id): t_key is an int second (a message's own author stamp, or
for a delete the target's stamp), id a 64-bit int (content hash). Two variants run on one engine:

  S2  range-based set reconciliation (RBSR, the idea of Negentropy): one fingerprint over the band; a range whose
      fingerprints differ is split into BRANCH (8, picked by the C1 gate in layer 3b) sub-ranges by the responder's own items, until a side holds at most LEAF
      items in it, which it then sends as a list of ids.
  S1  fixed time buckets: the band is cut into buckets of BUCKET_S seconds, one fingerprint each; a differing bucket
      is answered with its whole id list (LEAF = infinity, no recursion).

Fingerprint of a range = (sum of ids mod 2^64, count). A real box would hash ids to 128 bits first; the byte model
charges 16 B for it.

Protocol: messages are lists of ranges (lo, hi, mode, payload), lo/hi are keys, a range covers lo <= key < hi.
  FP  (fp, count)      "this is my fingerprint here"         -> equal: drop; else split or send IDS
  IDS [keys]           "these are all my keys here"          -> record what I need, answer HAVE with what they lack
  HAVE [keys]          "you lack these, I have them"         -> record need
Each side ends with `need` (keys the peer has and it lacks). Fetching them is the caller's business.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from heapq import merge

import numpy as np

M64 = (1 << 64) - 1
INF = (1 << 62, 0)
FP, IDS, HAVE = 0, 1, 2
# byte model (추정): a key on the wire 12 B (t 4 B + id 8 B), a range header = upper bound 12 B + mode 1 B,
# a fingerprint 16 B + count 4 B, a message header 40 B (CP26 HDR)
KEY_B, RANGE_B, FP_B, MSG_B = 12, 13, 20, 40


class KeySet:
    """Sorted keys with lazily rebuilt prefix sums: fingerprint of any range in O(log N)."""

    def __init__(self, keys=()):
        self.keys: list[tuple[int, int]] = sorted(keys)
        self._pre = None

    def __len__(self):
        return len(self.keys)

    def __contains__(self, k):
        i = bisect_left(self.keys, k)
        return i < len(self.keys) and self.keys[i] == k

    def add(self, k) -> bool:
        i = bisect_left(self.keys, k)
        if i < len(self.keys) and self.keys[i] == k:
            return False
        self.keys.insert(i, k)
        self._pre = None
        return True

    def discard(self, k) -> bool:
        i = bisect_left(self.keys, k)
        if i < len(self.keys) and self.keys[i] == k:
            del self.keys[i]
            self._pre = None
            return True
        return False

    def drop_before(self, t: int) -> list:
        """Remove keys with t_key < t; returns them."""
        i = bisect_left(self.keys, (t, -1))
        out, self.keys = self.keys[:i], self.keys[i:]
        if out:
            self._pre = None
        return out

    def _prefix(self):
        if self._pre is None:
            ids = np.fromiter((k[1] & M64 for k in self.keys), dtype=np.uint64, count=len(self.keys))
            self._pre = np.concatenate(([np.uint64(0)], np.cumsum(ids, dtype=np.uint64)))
        return self._pre

    def idx(self, k) -> int:
        return bisect_left(self.keys, k)

    def fp(self, lo, hi) -> tuple[int, int]:
        a, b = self.idx(lo), self.idx(hi)
        p = self._prefix()
        return (int(p[b]) - int(p[a])) & M64, b - a

    def items(self, lo, hi) -> list:
        return self.keys[self.idx(lo):self.idx(hi)]


class TopicKeySet:
    """CP33: a box's reconciled set split by sub (topic). A session reconciles the union of the subs both boxes hold:
    a range's fingerprint is the sum of the subs' fingerprints (the id sum mod 2^64 is additive) and its key list is
    the merge, so one session and the same wire format cover any set of subs. With one sub it is a KeySet."""

    def __init__(self):
        self.by: dict[int, KeySet] = {}
        self.topic: dict[int, int] = {}          # id -> sub

    def add(self, k, topic: int) -> bool:
        self.topic[k[1]] = topic
        return self.by.setdefault(topic, KeySet()).add(k)

    def discard(self, k) -> bool:
        t = self.topic.get(k[1])
        if t is None or t not in self.by:
            return False
        out = self.by[t].discard(k)
        if out:                                  # an id has one key (a post and its forgotten id share it)
            self.topic.pop(k[1], None)
        return out

    def __contains__(self, k):
        t = self.topic.get(k[1])
        return t is not None and t in self.by and k in self.by[t]

    def __len__(self):
        return sum(len(s) for s in self.by.values())

    def drop_before(self, t: int) -> list:
        out = []
        for s in self.by.values():
            out += s.drop_before(t)
        for k in out:
            self.topic.pop(k[1], None)
        return sorted(out)

    @property
    def keys(self) -> list:
        if len(self.by) == 1:
            return next(iter(self.by.values())).keys
        return list(merge(*(s.keys for s in self.by.values())))

    def view(self, topics=None) -> "TopicView":
        """The union of these subs (None: all) as one set for a Session."""
        sets = [s for t, s in self.by.items() if topics is None or t in topics]
        return TopicView(sets)


class TopicView:
    def __init__(self, sets: list):
        self.sets = sets

    def __len__(self):
        return sum(len(s) for s in self.sets)

    def fp(self, lo, hi) -> tuple[int, int]:
        f = c = 0
        for s in self.sets:
            a, b = s.fp(lo, hi)
            f, c = (f + a) & M64, c + b
        return f, c

    def items(self, lo, hi) -> list:
        if len(self.sets) == 1:
            return self.sets[0].items(lo, hi)
        return list(merge(*(s.items(lo, hi) for s in self.sets)))


class Session:
    """One side of a reconciliation. `variant` 'S1' or 'S2'. The band [lo, hi) is set by the initiator."""

    def __init__(self, ks: KeySet, variant: str = "S2", branch: int = 8, leaf: int = 16, bucket_s: int = 3600,
                 liar: str | None = None):
        self.ks, self.variant, self.branch, self.bucket_s = ks, variant, branch, bucket_s
        self.leaf = leaf if variant == "S2" else 10 ** 12
        self.liar = liar                      # None / 'echo' (claims every fingerprint matches) / 'withhold' (hides)
        self.need: set = set()
        self.have: set = set()
        self.sent = 0                         # bytes this side sent
        self.rounds = 0

    # ---- message size
    @staticmethod
    def size(msg) -> int:
        b = MSG_B
        for lo, hi, mode, pay in msg:
            b += RANGE_B + (FP_B if mode == FP else KEY_B * len(pay))
        return b

    def _send(self, msg):
        self.sent += self.size(msg)
        self.rounds += 1
        return msg

    def _keys(self, lo, hi):
        if self.liar == "withhold":
            return []
        return self.ks.items(lo, hi)

    def _fp(self, lo, hi):
        if self.liar == "withhold":
            return 0, 0
        return self.ks.fp(lo, hi)

    def initiate(self, lo, hi):
        if self.variant == "S1":
            assert hi[0] - lo[0] < 10 ** 8, "S1 needs a finite band"
            msg, t = [], lo[0] - lo[0] % self.bucket_s
            edges = [lo]
            while t + self.bucket_s < hi[0]:
                t += self.bucket_s
                edges.append((t, -1))
            edges.append(hi)
            for a, b in zip(edges[:-1], edges[1:]):
                if a < b:
                    msg.append((a, b, FP, self._fp(a, b)))
            return self._send(msg)
        return self._send([(lo, hi, FP, self._fp(lo, hi))])

    def _split(self, lo, hi):
        """Split [lo, hi) into up to `branch` sub-ranges at my own keys (equal counts)."""
        ks = self._keys(lo, hi)
        n = len(ks)
        cuts = [ks[(j * n) // self.branch] for j in range(1, self.branch)] if n else []
        edges = [lo] + [c for c in cuts if lo < c < hi] + [hi]
        out, seen = [], set()
        for a, b in zip(edges[:-1], edges[1:]):
            if a < b and (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b))
        return out

    def process(self, msg):
        """Handle the peer's message; returns my reply (empty list = nothing more to say)."""
        out = []
        for lo, hi, mode, pay in msg:
            if mode == FP:
                if self.liar == "echo":
                    continue
                mine = self._fp(lo, hi)
                if mine == tuple(pay):
                    continue
                if mine[1] <= self.leaf:
                    out.append((lo, hi, IDS, self._keys(lo, hi)))
                else:
                    for a, b in self._split(lo, hi):
                        out.append((a, b, FP, self._fp(a, b)))
            elif mode == IDS:
                theirs = set(pay)
                mine = set(self._keys(lo, hi))
                self.need |= theirs - mine
                lack = sorted(mine - theirs)
                self.have |= set(lack)
                if lack:
                    out.append((lo, hi, HAVE, lack))
            else:                                   # HAVE
                self.need |= set(pay)
        return self._send(out) if out else out


def reconcile(a: KeySet, b: KeySet, lo, hi, variant="S2", liar_b: str | None = None, max_rounds: int = 64,
              abort_after: int | None = None, **kw):
    """Run a whole session in memory (initiator a, responder b). Returns dict with need_a, need_b, bytes, rounds.
    abort_after: stop after that many messages (interrupted session)."""
    sa = Session(a, variant, **kw)
    sb = Session(b, variant, liar=liar_b, **kw)
    msg = sa.initiate(lo, hi)
    side, n = sb, 1
    while msg and n < max_rounds and (abort_after is None or n < abort_after):
        msg = side.process(msg)
        side = sa if side is sb else sb
        n += 1
    return {"need_a": sa.need, "need_b": sb.need, "bytes": sa.sent + sb.sent, "rounds": n,
            "done": not msg}
