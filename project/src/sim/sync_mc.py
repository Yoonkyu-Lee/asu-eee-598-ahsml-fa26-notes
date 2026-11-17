"""Small-scope exhaustive check of the sync rules (CP29 layer 3a): breadth-first search over every reachable state.

Scope (cp/CP29.md): 3 boxes (0 and 1 honest, 2 an adversary), posts P1 (by 0) and P2 (by 1), a delete D1 of P1 (by 0),
a comment C1 on P1 (by 1), and two forged envelopes the adversary can send at any time: XS (claims P1's id, other
content) and FD (a delete of P1 not signed by P1's key). The adversary also replays every valid message ever
published (worst case: it tries to resurrect P1 after D1). Events: any box on/off, any link up/down, publishing,
a push of one held item between two connected honest boxes, an adversary delivery, and an atomic reconciliation
between two connected honest boxes. Failed transfers and silent losses leave the state unchanged, so they are
covered by "that push did not happen". Windows and expiry are not modelled here (layer 2 covers them).

Claim (limited): with the accept rules of S1/S2 (content ids, validate first, tombstones in the set, comments kept
without their target), every reachable state satisfies S1-S4, and one reconciliation between two connected honest
boxes makes their sets equal (L1) and carries every delete (L2). The same search with the S0 accept rules
(store then validate, ids not content-checked) is run for contrast.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations

VALID = ("P1", "P2", "D1", "C1")
AUTHOR = {"P1": 0, "P2": 1, "D1": 0, "C1": 1}
PAIRS = ((0, 1), (0, 2), (1, 2))


def accept(held: frozenset, item: str, mode: str) -> frozenset:
    """One honest box receives one envelope. Returns its new held set."""
    if mode == "S0":                                   # store first, validate after (H8); id = claimed id
        ident = "P1" if item == "XS" else item
        if any(("P1" if h == "XS" else h) == ident for h in held):
            return held                                # duplicate id: dropped
        if ident == "P1" and "D1" in held:             # since CP28 a forgotten post is refused
            return held
        new = held | {item}
        if item == "D1" and "P1" in new:               # valid delete applies
            new = new - {"P1"}
        return frozenset(new)
    if item in held:
        return held
    if item == "XS":                                   # hash(header) != claimed id: rejected, nothing stored
        return held
    if item == "FD":                                   # signature does not match the target's key
        return held
    if item == "P1" and "D1" in held:                  # forgotten (its key stays as a tombstone)
        return held
    new = set(held) | {item}
    if item == "D1":
        new.discard("P1")
    return frozenset(new)                              # C1 is kept even without P1 (orphan, not shown)


def keyset(held: frozenset) -> frozenset:
    """What a box reconciles on: held messages plus the forgotten target of each delete it holds."""
    ks = set(h for h in held if h in VALID)
    if "D1" in ks:
        ks.add("P1")
    return frozenset(ks)


def reconcile(ha: frozenset, hb: frozenset, mode: str):
    """Atomic session: each side fetches what the other holds and it lacks (deletes first)."""
    def pull(dst, src):
        for it in sorted(src - dst, key=lambda x: x != "D1"):
            dst = accept(dst, it, mode)
        return dst
    return pull(ha, hb), pull(hb, ha)


def check(mode: str, max_states: int = 2_000_000) -> dict:
    start = (frozenset(), frozenset(), False, False, frozenset(), (False, False, False), (True, True, True))
    seen = {start}
    q = deque([start])
    viol: dict[str, list] = {}
    count: dict[str, int] = {}
    n_trans = 0

    def bad(p, s, note=""):
        viol.setdefault(p, [])
        count[p] = count.get(p, 0) + 1
        if len(viol[p]) < 3:
            viol[p].append({"state": str(s), "note": note})

    while q:
        s = q.popleft()
        h0, h1, had0, had1, pub, on, link = s
        H = (h0, h1)
        # ---- safety in this state
        for i, h in enumerate(H):
            if h - set(VALID):
                bad("S1", s, f"box {i} holds {sorted(h - set(VALID))}")
            if "P1" in h and "D1" in h:
                bad("S2", s, f"box {i}")
            had = (had0, had1)[i]
            if had and "P1" not in h and "D1" not in h and not (mode == "S0" and "XS" in h):
                bad("S3", s, f"box {i} lost P1 without D1")
        # S4: a box that lacks P1 and has no D1 must be able to take the valid P1
        if "P1" in pub:
            for i, h in enumerate(H):
                if "P1" not in h and "D1" not in h and "P1" not in accept(h, "P1", mode):
                    bad("S4", s, f"box {i} cannot accept P1")
        # ---- L1/L2: one session between connected honest boxes closes the difference
        if on[0] and on[1] and link[0]:
            a, b = reconcile(h0, h1, mode)
            if keyset(a) != keyset(b):
                bad("L1", s, f"after session {sorted(keyset(a))} vs {sorted(keyset(b))}")
            if ("D1" in a or "D1" in b) and not ("D1" in a and "D1" in b and "P1" not in a and "P1" not in b):
                bad("L2", s, "delete not carried")
        # ---- transitions
        nxt = []
        for i in range(3):
            o = list(on)
            o[i] = not o[i]
            nxt.append((h0, h1, had0, had1, pub, tuple(o), link))
        for j in range(3):
            l = list(link)
            l[j] = not l[j]
            nxt.append((h0, h1, had0, had1, pub, on, tuple(l)))
        for it in VALID:                                   # publishing (the author must be on)
            au = AUTHOR[it]
            if it in pub or not on[au]:
                continue
            ha = H[au]
            if it == "D1" and "P1" not in ha:
                continue
            if it == "C1" and "P1" not in ha:
                continue
            hh = list(H)
            hh[au] = accept(ha, it, mode)
            had = [had0 or "P1" in hh[0], had1 or "P1" in hh[1]]
            nxt.append((hh[0], hh[1], had[0], had[1], pub | {it}, on, link))
        if on[0] and on[1] and link[0]:                   # push one item, and a whole session
            for src, dst in ((0, 1), (1, 0)):
                for it in H[src]:
                    hh = list(H)
                    hh[dst] = accept(H[dst], it, mode)
                    if hh[dst] != H[dst]:
                        nxt.append((hh[0], hh[1], had0 or "P1" in hh[0], had1 or "P1" in hh[1], pub, on, link))
            a, b = reconcile(h0, h1, mode)
            nxt.append((a, b, had0 or "P1" in a, had1 or "P1" in b, pub, on, link))
        if on[2]:                                          # the adversary: forgeries and replays
            for i, j in ((0, 1), (1, 2)):
                if not (on[i] and link[j]):
                    continue
                for it in ("XS", "FD") + tuple(sorted(pub)):
                    hh = list(H)
                    hh[i] = accept(H[i], it, mode)
                    if hh[i] != H[i]:
                        nxt.append((hh[0], hh[1], had0 or "P1" in hh[0], had1 or "P1" in hh[1], pub, on, link))
        for t in nxt:
            n_trans += 1
            if t not in seen:
                seen.add(t)
                q.append(t)
                if len(seen) > max_states:
                    return {"mode": mode, "states": len(seen), "transitions": n_trans, "truncated": True,
                            "violations": dict(count), "examples": viol}
    return {"mode": mode, "states": len(seen), "transitions": n_trans, "truncated": False,
            "violations": dict(count), "examples": viol}
