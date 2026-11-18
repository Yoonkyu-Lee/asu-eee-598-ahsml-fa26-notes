"""Small-scope exhaustive check of the sync rules (CP29 layer 3a): breadth-first search over every reachable state.

Scope (cp/CP29.md): 3 boxes (0 and 1 honest, 2 an adversary), posts P1 (by 0) and P2 (by 1), a delete D1 of P1 (by 0),
a comment C1 on P1 (by 1), and two forged envelopes the adversary can send at any time: XS (claims P1's id, other
content) and FD (a delete of P1 not signed by P1's key). The adversary also replays every valid message ever
published (worst case: it tries to resurrect P1 after D1). Events: any box on/off, any link up/down, publishing,
a push of one held item between two connected honest boxes, an adversary delivery, and an atomic reconciliation
between two connected honest boxes. Failed transfers and silent losses leave the state unchanged, so they are
covered by "that push did not happen". Windows and expiry are not modelled here (layer 2 covers them).

Claim (limited): with the accept rules of S2 (content ids, validate first, tombstones in the set, comments kept
without their target), every reachable state satisfies S1-S4, and one reconciliation between two connected honest
boxes makes their sets equal (L1) and carries every delete (L2). The S0 contrast search was retired with S0 in recast R1b (tag pre-r1);
check(mutate=...) now shows the search can fail.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations

VALID = ("P1", "P2", "D1", "C1")
AUTHOR = {"P1": 0, "P2": 1, "D1": 0, "C1": 1}
PAIRS = ((0, 1), (0, 2), (1, 2))


def accept(held: frozenset, item: str, mutate: str | None = None) -> frozenset:
    """One honest box receives one envelope. Returns its new held set. mutate (to show check() can fail):
    "accept_forged" (no content-id check: XS is stored), "resurrect" (a deleted post is taken again)."""
    if item in held:
        return held
    if item == "XS" and mutate != "accept_forged":     # hash(header) != claimed id: rejected, nothing stored
        return held
    if item == "FD":                                   # signature does not match the target's key
        return held
    if item == "P1" and "D1" in held and mutate != "resurrect":   # forgotten (its key stays as a tombstone)
        return held
    new = set(held) | {item}
    if item == "D1":
        new.discard("P1")
    return frozenset(new)                              # C1 is kept even without P1 (orphan, not shown)


def keyset(held: frozenset) -> frozenset:
    """What a box reconciles on: held messages plus the forgotten target of each delete it holds."""
    ks = set(h for h in held if h in VALID or h == "P3")    # P3: CP30 check_clean only
    if "D1" in ks:
        ks.add("P1")
    return frozenset(ks)


def reconcile(ha: frozenset, hb: frozenset, mutate: str | None = None):
    """Atomic session: each side fetches what the other holds and it lacks (deletes first)."""
    def pull(dst, src):
        for it in sorted(src - dst, key=lambda x: x != "D1"):
            dst = accept(dst, it, mutate)
        return dst
    return pull(ha, hb), pull(hb, ha)


def check(max_states: int = 2_000_000, mutate: str | None = None) -> dict:
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
            if had and "P1" not in h and "D1" not in h:
                bad("S3", s, f"box {i} lost P1 without D1")
        # S4: a box that lacks P1 and has no D1 must be able to take the valid P1
        if "P1" in pub:
            for i, h in enumerate(H):
                if "P1" not in h and "D1" not in h and "P1" not in accept(h, "P1", mutate):
                    bad("S4", s, f"box {i} cannot accept P1")
        # ---- L1/L2: one session between connected honest boxes closes the difference
        if on[0] and on[1] and link[0]:
            a, b = reconcile(h0, h1, mutate)
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
            hh[au] = accept(ha, it, mutate)
            had = [had0 or "P1" in hh[0], had1 or "P1" in hh[1]]
            nxt.append((hh[0], hh[1], had[0], had[1], pub | {it}, on, link))
        if on[0] and on[1] and link[0]:                   # push one item, and a whole session
            for src, dst in ((0, 1), (1, 0)):
                for it in H[src]:
                    hh = list(H)
                    hh[dst] = accept(H[dst], it, mutate)
                    if hh[dst] != H[dst]:
                        nxt.append((hh[0], hh[1], had0 or "P1" in hh[0], had1 or "P1" in hh[1], pub, on, link))
            a, b = reconcile(h0, h1, mutate)
            nxt.append((a, b, had0 or "P1" in a, had1 or "P1" in b, pub, on, link))
        if on[2]:                                          # the adversary: forgeries and replays
            for i, j in ((0, 1), (1, 2)):
                if not (on[i] and link[j]):
                    continue
                for it in ("XS", "FD") + tuple(sorted(pub)):
                    hh = list(H)
                    hh[i] = accept(H[i], it, mutate)
                    if hh[i] != H[i]:
                        nxt.append((hh[0], hh[1], had0 or "P1" in hh[0], had1 or "P1" in hh[1], pub, on, link))
        for t in nxt:
            n_trans += 1
            if t not in seen:
                seen.add(t)
                q.append(t)
                if len(seen) > max_states:
                    return {"mode": "S12", "mutate": mutate, "states": len(seen), "transitions": n_trans, "truncated": True,
                            "violations": dict(count), "examples": viol}
    return {"mode": "S12", "mutate": mutate, "states": len(seen), "transitions": n_trans, "truncated": False,
            "violations": dict(count), "examples": viol}


def check_clean(max_states: int = 2_000_000, mutate: str | None = None) -> dict:
    """CP30 F/F': the same search (S2 accept rules) with the one-sided clean flags c01 (box 0 about box 1) and c10.

    A session started by box i sets c_ij when it finds no difference; any change to box i's set that did not come
    from box j clears c_ij (publishing, an adversary delivery, incl. P3: a valid post by the adversary, so that a
    third party can bring what the peer lacks); a box coming on clears its own flags, and a flag
    may be cleared at any time (a peer noticed as down, a restart). Links come back silently (flags untouched).
    Invariant: c_ij => keys(i) is a subset of keys(j). With both flags set the two sets are equal, so a pair whose
    floor checks are both skipped has no difference. Sessions are atomic here; a session that runs while the set
    changes is covered by the generation fence (layer 2 and 4 scenes).
    mutate (to show the check can fail): "keep_on_publish" (publishing does not clear the flag), "keep_on_adversary"
    (a third-party delivery does not clear it), "set_on_difference" (a session sets the flag even after fetching)."""
    start = (frozenset(), frozenset(), frozenset(), (False, False, False), (True, True, True), (False, False))
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
        h0, h1, pub, on, link, cl = s
        H = (h0, h1)
        k0, k1 = keyset(h0), keyset(h1)
        if cl[0] and not k0 <= k1:
            bad("clean01", s, f"{sorted(k0)} not in {sorted(k1)}")
        if cl[1] and not k1 <= k0:
            bad("clean10", s, f"{sorted(k1)} not in {sorted(k0)}")
        for i, h in enumerate(H):
            if h - set(VALID) - {"P3"}:
                bad("S1", s, f"box {i}")
            if "P1" in h and "D1" in h:
                bad("S2", s, f"box {i}")

        def st(hh0, hh1, c, p=pub, o=on, lk=link):
            return (hh0, hh1, p, o, lk, tuple(c))

        nxt = []
        for i in range(3):                                  # on/off; coming on clears that box's flags
            o = list(on)
            o[i] = not o[i]
            c = list(cl)
            if i < 2 and o[i]:
                c[i] = False
            nxt.append(st(h0, h1, c, o=tuple(o)))
        for j in range(3):                                  # links (silent return: flags untouched)
            lk = list(link)
            lk[j] = not lk[j]
            nxt.append(st(h0, h1, cl, lk=tuple(lk)))
        for i in range(2):                                  # a flag cleared at any time
            if cl[i]:
                c = list(cl)
                c[i] = False
                nxt.append(st(h0, h1, c))
        for it in VALID:                                    # publishing changes the author's set (from nobody)
            au = AUTHOR[it]
            if it in pub or not on[au]:
                continue
            if it in ("D1", "C1") and "P1" not in H[au]:
                continue
            hh = list(H)
            hh[au] = accept(H[au], it)
            c = list(cl)
            if hh[au] != H[au] and mutate != "keep_on_publish":
                c[au] = False
            nxt.append(st(hh[0], hh[1], c, p=pub | {it}))
        if on[0] and on[1] and link[0]:
            for src, dst in ((0, 1), (1, 0)):              # a push: dst's change came from src, its flag stays
                for it in H[src]:
                    hh = list(H)
                    hh[dst] = accept(H[dst], it)
                    if hh[dst] != H[dst]:
                        nxt.append(st(hh[0], hh[1], cl))
            same = k0 == k1
            a, b = reconcile(h0, h1)
            for ini in (0, 1):                              # a session started by either box
                c = list(cl)
                if same:
                    c[ini] = True
                nxt.append(st(a, b, c))
                if not same:                                # its fetches silently lost: nothing changes
                    c = list(cl)
                    if mutate == "set_on_difference":
                        c[ini] = True
                    nxt.append(st(h0, h1, c))
        if on[2]:                                           # adversary deliveries: a change from a third party
            for i, j in ((0, 1), (1, 2)):
                if not (on[i] and link[j]):
                    continue
                for it in ("XS", "FD", "P3") + tuple(sorted(pub)):
                    hh = list(H)
                    hh[i] = accept(H[i], it)
                    if hh[i] != H[i]:
                        c = list(cl)
                        if mutate != "keep_on_adversary":
                            c[i] = False
                        nxt.append(st(hh[0], hh[1], c))
        for t in nxt:
            n_trans += 1
            if t not in seen:
                seen.add(t)
                q.append(t)
                if len(seen) > max_states:
                    return {"mode": "S12c", "mutate": mutate, "states": len(seen), "transitions": n_trans, "truncated": True,
                            "violations": dict(count), "examples": viol}
    return {"mode": "S12c", "mutate": mutate, "states": len(seen), "transitions": n_trans, "truncated": False,
            "violations": dict(count), "examples": viol}


# ---------------------------------------------------------------------------------------------- CP33: subs
SUBS = ({"A", "B"}, {"A"}, {"B"})                      # box 0 in both subs, box 1 only A, box 2 only B
TOPIC = {"PA": "A", "PB": "B", "DB": "B", "CB": "B", "XB": "B"}
S_VALID = ("PA", "PB", "DB", "CB")
S_AUTHOR = {"PA": 0, "PB": 0, "DB": 0, "CB": 2}
S_PAIRS = ((0, 1), (0, 2), (1, 2))


def s_accept(held: frozenset, item: str, box: int, mutate: set) -> frozenset:
    """CP33 accept rules: refuse a sub the box does not hold (before anything else), then the CP29 rules."""
    if TOPIC[item] not in SUBS[box] and "accept_any" not in mutate:
        return held
    if item in held or item == "XB":                   # XB: claims PB's id, other content -> rejected
        return held
    if item == "PB" and "DB" in held:
        return held
    new = set(held) | {item}
    if item == "DB":
        new.discard("PB")
    return frozenset(new)


def s_keys(held: frozenset) -> frozenset:
    ks = set(held)
    if "DB" in ks:
        ks.add("PB")                                   # the forgotten target's key
    return frozenset(ks)


def check_subs(max_states: int = 3_000_000, mutate: str | None = None) -> dict:
    """CP33 layer 3a: three honest boxes with subs {A,B}, {A}, {B}; an adversary on every box's link that can deliver
    any published message (any sub) and a forged copy XB of PB at any time. Events: on/off, link up/down, publishing,
    a push of one held item to a connected box that holds its sub, and an atomic session between two connected boxes
    over the subs both hold. Properties: S5 no box holds a message of a sub it does not hold, S1 no forgery held,
    S2 no post held with its delete, L1/L2 after a session the two boxes' keys in their common subs are equal.
    mutate: "recon_all" (a session ignores subs: the CP29-CP32 box), "accept_any" (a box accepts any sub),
    "push_any" (a push goes to any connected box); several joined by "+". "recon_all+accept_any" is the CP29-CP32
    box (H12)."""
    mset = set((mutate or "").split("+")) - {""}
    start = ((frozenset(),) * 3, frozenset(), (True, True, True), (True, True, True))
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

    def common(i, j):
        return {"A", "B"} if "recon_all" in mset else SUBS[i] & SUBS[j]

    def session(H, i, j):
        cm = common(i, j)
        hi, hj = H[i], H[j]
        for src, dst in ((j, i), (i, j)):
            have = hi if dst == i else hj
            give = hj if dst == i else hi
            for it in sorted((give - have), key=lambda x: x != "DB"):
                if TOPIC[it] in cm:
                    have = s_accept(have, it, dst, mset)
            if dst == i:
                hi = have
            else:
                hj = have
        out = list(H)
        out[i], out[j] = hi, hj
        return tuple(out)

    while q:
        s = q.popleft()
        H, pub, on, link = s
        for i, h in enumerate(H):
            if any(TOPIC[x] not in SUBS[i] for x in h):
                bad("S5", s, f"box {i} holds {sorted(x for x in h if TOPIC[x] not in SUBS[i])}")
            if "XB" in h:
                bad("S1", s, f"box {i}")
            if "PB" in h and "DB" in h:
                bad("S2", s, f"box {i}")
        nxt = []
        for k, (i, j) in enumerate(S_PAIRS):
            if on[i] and on[j] and link[k]:
                after = session(H, i, j)
                cm = SUBS[i] & SUBS[j]
                ki = {x for x in s_keys(after[i]) if TOPIC[x] in cm}
                kj = {x for x in s_keys(after[j]) if TOPIC[x] in cm}
                if ki != kj:
                    bad("L1", s, f"pair {i},{j}: {sorted(ki)} vs {sorted(kj)}")
                if ("DB" in after[i] or "DB" in after[j]) and "B" in cm and not ("DB" in after[i] and "DB" in after[j]):
                    bad("L2", s, f"pair {i},{j}")
                nxt.append((after, pub, on, link))
                for src, dst in ((i, j), (j, i)):         # push one item
                    for it in H[src]:
                        if TOPIC[it] not in SUBS[dst] and "push_any" not in mset:
                            continue
                        hh = list(H)
                        hh[dst] = s_accept(H[dst], it, dst, mset)
                        if hh[dst] != H[dst]:
                            nxt.append((tuple(hh), pub, on, link))
        for i in range(3):
            o = list(on)
            o[i] = not o[i]
            nxt.append((H, pub, tuple(o), link))
        for k in range(3):
            lk = list(link)
            lk[k] = not lk[k]
            nxt.append((H, pub, on, tuple(lk)))
        for it in S_VALID:
            au = S_AUTHOR[it]
            if it in pub or not on[au]:
                continue
            if it in ("DB", "CB") and "PB" not in H[au]:
                continue
            hh = list(H)
            hh[au] = s_accept(H[au], it, au, mset)
            nxt.append((tuple(hh), pub | {it}, on, link))
        for i in range(3):                                 # the adversary delivers anything published, or XB
            if not on[i]:
                continue
            for it in ("XB",) + tuple(sorted(pub)):
                hh = list(H)
                hh[i] = s_accept(H[i], it, i, mset)
                if hh[i] != H[i]:
                    nxt.append((tuple(hh), pub, on, link))
        for t in nxt:
            n_trans += 1
            if t not in seen:
                seen.add(t)
                q.append(t)
                if len(seen) > max_states:
                    return {"mode": "subs", "mutate": mutate, "states": len(seen), "transitions": n_trans,
                            "truncated": True, "violations": dict(count), "examples": viol}
    return {"mode": "subs", "mutate": mutate, "states": len(seen), "transitions": n_trans, "truncated": False,
            "violations": dict(count), "examples": viol}
