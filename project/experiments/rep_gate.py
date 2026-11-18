"""CP37 (1), next piece (journal/cp/CP37.md, rules written before the run): how many receipts N, and which receipt
rule, keep spam out of feeds without keeping honest newcomers out for long. A day-level model, every number an estimate.

  .venv/Scripts/python.exe -W ignore experiments/rep_gate.py         -> results/chip/cp37_rep_gate.json
  .venv/Scripts/python.exe -W ignore experiments/rep_gate.py knobs   -> results/chip/cp37_rep_knobs.json

Behaviour: the CP16 BASE (sim/community.py). A post enters feeds if its author holds N counted receipts when posting;
otherwise it goes to a "new posts" lane that draws rho x the votes. Votes go to posts in proportion to quality; an
honest voter votes for spam at eps x the weight. Receipt rules: any (every voter counts), age (voters 30+ days old),
rep (founders, or voters who themselves hold N counted receipts). Attacks with K = 1% of users from day 0, each
posting at the CP22 limit: lone, ring (every attacker votes every other attacker's posts), ring + bought (each attacker
buys N counted votes once).
Mistaken honest votes on spam: "weight" (the rule as written: eps x a post's weight, so spam volume draws votes) and
"share" (marked after the smoke run: eps of all honest votes go to spam, split by visibility).
Newcomer wait (marked after the smoke run): from a newcomer's first post to the first post that entered feeds.
Knobs (CP37, the four remaining candidates): a per-author feed cap m (a feed and the new lane show at most m posts of
one author a day; only shown posts draw votes), a daily limit L per identity (honest posts above it are refused), the
attack share K, and the "viewer" rule (each viewer counts only votes from people it has voted for; an author it has
voted for is visible at once; 300 sampled viewers).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.community import Params  # noqa: E402

OUT = ROOT / "results" / "chip" / "cp37_rep_gate.json"
U0, DAYS, SEEDS = 20_000, 90, (1, 2, 3)
JOIN = 0.01                      # (assumption) new members per day, share of U0
RHO, EPS, Q_SIGMA = 0.2, 0.01, 1.0   # (assumptions) new-lane vote share, mistaken honest votes on spam, quality spread
HISTORY_DAYS = 30                # (assumption) founders start with this many days of receipts
AGE_DAYS = 30                    # rule "age"
ATTACK_SHARE = 0.01              # 10^4 of 10^6 (NORTHSTAR spam bar)
SPAM_PER_DAY = 6 * 144           # CP22 old-identity limit
NS = (0, 1, 3, 10, 30, 100)
EPS_MODES = ("weight", "share")
RULES, ATTACKS = ("any", "age", "rep"), ("lone", "ring", "ring_bought")
MEASURE = (60, 90)
OUT_K = ROOT / "results" / "chip" / "cp37_rep_knobs.json"
CAPS, LIMITS, SHARES, KNOB_NS = (None, 10, 3, 1), (864, 50, 10), (0.01, 0.001, 0.0001), (0, 1, 3, 10)
VIEWERS, VIEWER_NS = 300, (1, 3)
HONEST_REFUSE_MAX = 1e-3         # CP22: honest refusal bar


def _visible(n_items: np.ndarray, m) -> np.ndarray:
    """Per item: is it among the first m posts of its author that day (items grouped by author, in order)?"""
    if m is None:
        return np.ones(int(n_items.sum()), bool)
    starts = np.repeat(np.cumsum(n_items) - n_items, n_items)
    return (np.arange(int(n_items.sum())) - starts) < m


def run(rule: str, attack: str, N: int, seed: int, eps_mode: str = "weight", m=None, L: int = SPAM_PER_DAY,
        share: float = ATTACK_SHARE) -> dict:
    if rule == "viewer":
        return run_viewer(attack, N, seed, eps_mode, m, L, share)
    p = Params()
    rng = np.random.default_rng(seed)
    K = max(int(round(share * U0)), 1)
    cap = U0 + int(1.5 * JOIN * U0 * DAYS) + K + 100      # joins are Poisson: room above the mean
    join = np.full(cap, -1)
    founder, writer, attacker = (np.zeros(cap, bool) for _ in range(3))
    rate = np.zeros(cap)                    # items per day for writers
    rq = np.zeros(cap)                      # counted receipts
    first = np.full(cap, -1)                # day of the first post that entered feeds
    first_post = np.full(cap, -1)           # day of the first post
    item_mean = (p.posts_per_dau + p.comments_per_dau) * p.dau_frac / (1 - p.lurker_frac)
    votes_per_item = p.votes_per_dau / (p.posts_per_dau + p.comments_per_dau)

    def add_users(n, day, is_founder):
        lo = int((join >= 0).sum())
        hi = lo + n
        join[lo:hi] = day
        founder[lo:hi] = is_founder
        writer[lo:hi] = rng.random(n) >= p.lurker_frac
        ln = rng.lognormal(0, p.tail_sigma, n)
        rate[lo:hi] = np.where(writer[lo:hi], ln / np.exp(p.tail_sigma ** 2 / 2) * item_mean, 0.0)
        return lo, hi

    lo, hi = add_users(U0, 0, True)
    rq[lo:hi] = rng.poisson(rate[lo:hi] * HISTORY_DAYS * votes_per_item)
    a_lo = hi
    join[a_lo:a_lo + K] = 0
    attacker[a_lo:a_lo + K] = True
    if attack == "ring_bought":
        rq[a_lo:a_lo + K] += N
    feed = {"honest": 0, "spam": 0}
    refused = posted = 0
    spam_shown = L if m is None else min(L, m)
    for day in range(DAYS):
        if day:
            add_users(rng.poisson(JOIN * U0), day, False)
        live = join >= 0
        honest = live & ~attacker
        age = day - join
        if rule == "any":
            counted = live.copy()
        elif rule == "age":
            counted = live & (age >= AGE_DAYS)
        else:
            counted = live & (founder | (rq >= N))
        # honest posts
        wanted = rng.poisson(rate * honest)
        n_items = np.minimum(wanted, L)
        refused += int((wanted - n_items).sum())
        posted += int(wanted.sum())
        authors = np.repeat(np.arange(cap), n_items)
        q = rng.lognormal(0, Q_SIGMA, len(authors)) / np.exp(Q_SIGMA ** 2 / 2)
        shown = _visible(n_items, m)
        passed = rq[authors] >= N
        for a in np.unique(authors):
            if first_post[a] < 0:
                first_post[a] = day
        for a in np.unique(authors[passed]):
            if first[a] < 0:
                first[a] = day
        # spam posts (per attacker, all of a day's posts share the gate result)
        att = np.flatnonzero(attacker & live)
        s_pass = rq[att] >= N
        if MEASURE[0] <= day < MEASURE[1]:
            feed["honest"] += int((passed & shown).sum())
            feed["spam"] += int(s_pass.sum()) * spam_shown
        # honest votes, spread by weight over the posts people are shown
        w_h = q * np.where(passed, 1.0, RHO) * shown
        w_s = np.full(len(att), float(spam_shown)) * np.where(s_pass, 1.0, RHO)
        V = rng.poisson(p.votes_per_dau * p.dau_frac * honest.sum())
        p_counted = (counted & honest).sum() / max(honest.sum(), 1)
        if eps_mode == "weight":
            total_w = w_h.sum() + EPS * w_s.sum()
            v_h, v_s = (V * w_h / total_w, V * EPS * w_s / total_w) if total_w > 0 else (0 * w_h, 0 * w_s)
        else:
            v_h = (1 - EPS * (len(att) > 0)) * V * w_h / max(w_h.sum(), 1e-12)
            v_s = EPS * V * w_s / max(w_s.sum(), 1e-12)
        np.add.at(rq, authors, rng.binomial(rng.poisson(v_h), p_counted))
        if len(att):
            rq[att] += rng.binomial(rng.poisson(v_s), p_counted)
        # ring votes: every attacker votes every other attacker's posts
        if attack != "lone" and len(att) > 1:
            q_ring = counted[att]
            rq[att] += L * (q_ring.sum() - q_ring)
    newc = np.flatnonzero((join >= 1) & (join < MEASURE[0]) & ~attacker & (first_post >= 0)
                          & (first_post - join <= 30))
    wait = np.where(first[newc] >= 0, first[newc] - first_post[newc], np.inf)
    return {"spam_share_feed": feed["spam"] / max(feed["spam"] + feed["honest"], 1),
            "honest_feed_items_per_day": feed["honest"] / (MEASURE[1] - MEASURE[0]),
            "newcomer_wait_median_days": float(np.median(wait)),
            "newcomer_pass_30d": float(np.mean(wait <= 30)), "newcomers": int(len(newc)),
            "bought_votes_total": K * N if attack == "ring_bought" else 0,
            "honest_refused_share": refused / max(posted, 1)}


def run_viewer(attack: str, N: int, seed: int, eps_mode: str, m, L: int, share: float) -> dict:
    """Rule "viewer": viewer v sees author a if v voted for a before, or if at least N distinct people v voted for
    have voted for a. Votes land on posts by quality x (rho + (1 - rho) x share of sampled viewers who see the author)."""
    p = Params()
    rng = np.random.default_rng(seed)
    K = max(int(round(share * U0)), 1)
    cap = U0 + int(1.5 * JOIN * U0 * DAYS) + K + 100
    join = np.full(cap, -1)
    writer, attacker = np.zeros(cap, bool), np.zeros(cap, bool)
    rate = np.zeros(cap)
    first_post = np.full(cap, -1)
    item_mean = (p.posts_per_dau + p.comments_per_dau) * p.dau_frac / (1 - p.lurker_frac)
    votes_per_item = p.votes_per_dau / (p.posts_per_dau + p.comments_per_dau)
    rows, cols = [], []                      # who voted for whom (distinct pairs kept by the binary matrix)

    def add_users(n, day):
        lo = int((join >= 0).sum())
        hi = lo + n
        join[lo:hi] = day
        writer[lo:hi] = rng.random(n) >= p.lurker_frac
        ln = rng.lognormal(0, p.tail_sigma, n)
        rate[lo:hi] = np.where(writer[lo:hi], ln / np.exp(p.tail_sigma ** 2 / 2) * item_mean, 0.0)
        return lo, hi

    lo, hi = add_users(U0, 0)
    # founders' history: HISTORY_DAYS of votes from random founders
    hist = rng.poisson(rate[lo:hi] * HISTORY_DAYS * votes_per_item)
    rows += rng.integers(lo, hi, int(hist.sum())).tolist()
    cols += np.repeat(np.arange(lo, hi), hist).tolist()
    viewers = rng.choice(np.arange(lo, hi), VIEWERS, replace=False)
    a_lo = hi
    join[a_lo:a_lo + K] = 0
    attacker[a_lo:a_lo + K] = True
    att = np.arange(a_lo, a_lo + K)
    if attack != "lone" and K > 1:
        rr, cc = np.meshgrid(att, att)
        keep = rr != cc
        rows += rr[keep].tolist()
        cols += cc[keep].tolist()
    if attack == "ring_bought":
        for a in att:
            rows += rng.integers(lo, hi, N).tolist()
            cols += [int(a)] * N
    spam_shown = L if m is None else min(L, m)
    seen = {"honest": np.zeros(VIEWERS), "spam": np.zeros(VIEWERS)}
    newc_seen = {}
    refused = posted = 0
    vis = np.zeros((VIEWERS, cap), bool)
    for day in range(DAYS):
        if day:
            add_users(rng.poisson(JOIN * U0), day)
        B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(cap, cap))
        B.data[:] = 1.0
        Tv = B[viewers]
        vis = ((Tv @ B).toarray() >= N) | (Tv.toarray() > 0)
        f = vis.mean(0)                                    # share of sampled viewers who see each author
        live = join >= 0
        honest = live & ~attacker
        wanted = rng.poisson(rate * honest)
        n_items = np.minimum(wanted, L)
        refused += int((wanted - n_items).sum())
        posted += int(wanted.sum())
        authors = np.repeat(np.arange(cap), n_items)
        for a in np.unique(authors):
            if first_post[a] < 0:
                first_post[a] = day
        q = rng.lognormal(0, Q_SIGMA, len(authors)) / np.exp(Q_SIGMA ** 2 / 2)
        shown = _visible(n_items, m)
        if MEASURE[0] <= day < MEASURE[1]:
            per_author = np.bincount(authors[shown], minlength=cap)
            seen["honest"] += vis @ per_author
            seen["spam"] += vis[:, att].sum(1) * spam_shown
        w_h = q * (RHO + (1 - RHO) * f[authors]) * shown
        w_s = spam_shown * (RHO + (1 - RHO) * f[att])
        V = rng.poisson(p.votes_per_dau * p.dau_frac * honest.sum())
        voters_pool = np.flatnonzero(honest)
        if eps_mode == "weight":
            tot = w_h.sum() + EPS * w_s.sum()
            n_h = rng.binomial(V, w_h.sum() / tot) if tot > 0 else 0
        else:
            n_h = rng.binomial(V, 1 - EPS)
        n_s = V - n_h
        if n_h and w_h.sum() > 0:
            tgt = authors[rng.choice(len(authors), n_h, p=w_h / w_h.sum())]
            rows += rng.choice(voters_pool, n_h).tolist()
            cols += tgt.tolist()
        if n_s and w_s.sum() > 0:
            tgt = att[rng.choice(K, n_s, p=w_s / w_s.sum())]
            rows += rng.choice(voters_pool, n_s).tolist()
            cols += tgt.tolist()
    tot = seen["honest"] + seen["spam"]
    sh = np.where(tot > 0, seen["spam"] / np.maximum(tot, 1), 0.0)
    newc = np.flatnonzero((join >= 1) & (join < MEASURE[0]) & ~attacker & (first_post >= 0) & (first_post - join <= 30))
    newc_vis = vis[:, newc].mean(0) if len(newc) else np.zeros(0)
    return {"spam_share_feed": float(sh.mean()), "spam_share_feed_p90_viewer": float(np.quantile(sh, 0.9)),
            "viewers_seeing_any_spam": float(np.mean(seen["spam"] > 0)),
            "newcomer_visible_viewer_share_median": float(np.median(newc_vis)) if len(newc) else 0.0,
            "newcomers": int(len(newc)), "honest_refused_share": refused / max(posted, 1),
            "bought_votes_total": K * N if attack == "ring_bought" else 0}


def _knob_job(args):
    rule, attack, N, eps_mode, m, L, share = args
    rs = [run(rule, attack, N, s, eps_mode, m, L, share) for s in SEEDS]
    row = {"eps_mode": eps_mode, "rule": rule, "attack": attack, "N": N, "m": m, "L": L, "share": share,
           **{k: float(np.mean([r[k] for r in rs])) for k in rs[0]}}
    ok_refuse = row["honest_refused_share"] <= HONEST_REFUSE_MAX
    if rule == "viewer":
        row["works"] = row["spam_share_feed"] <= 0.05 and row["spam_share_feed_p90_viewer"] <= 0.05 and ok_refuse
    else:
        row["works"] = (row["spam_share_feed"] <= 0.05 and row["newcomer_wait_median_days"] <= 7
                        and row["newcomer_pass_30d"] >= 0.90 and ok_refuse)
    return row


def knobs():
    jobs = [(r, a, n, e, m, l, k) for e, r, a, n, m, l, k in
            product(EPS_MODES, RULES, ATTACKS, KNOB_NS, CAPS, LIMITS, SHARES)]
    jobs += [("viewer", a, n, e, m, l, k) for e, a, n, m, l, k in
             product(EPS_MODES, ATTACKS, VIEWER_NS, CAPS, LIMITS, SHARES)]
    res = {"what": "CP37 (1) four candidates (journal/cp/CP37.md)", "label": "estimate (model)",
           "params": {"caps": CAPS, "limits": LIMITS, "shares": SHARES, "ns": KNOB_NS, "viewer_ns": VIEWER_NS,
                      "viewers": VIEWERS, "honest_refuse_max": HONEST_REFUSE_MAX, "seeds": SEEDS},
           "rows": []}
    with ProcessPoolExecutor(18) as ex:
        for i, row in enumerate(ex.map(_knob_job, jobs, chunksize=4)):
            res["rows"].append(row)
            if row["works"]:
                print("works", {k: row[k] for k in ("eps_mode", "rule", "attack", "N", "m", "L", "share")}, flush=True)
            if i % 200 == 0:
                print(i, "of", len(jobs), flush=True)
    OUT_K.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT_K)


def main():
    res = {"what": "CP37 (1) receipt gate (journal/cp/CP37.md)", "label": "estimate (model)",
           "params": {"users": U0, "days": DAYS, "join": JOIN, "rho": RHO, "eps": EPS, "q_sigma": Q_SIGMA,
                      "history_days": HISTORY_DAYS, "age_days": AGE_DAYS, "attack_share": ATTACK_SHARE,
                      "spam_per_day": SPAM_PER_DAY, "seeds": SEEDS},
           "rows": []}
    for eps_mode in EPS_MODES:
      for rule in RULES:
        for attack in ATTACKS:
            for N in NS:
                rs = [run(rule, attack, N, s, eps_mode) for s in SEEDS]
                row = {"eps_mode": eps_mode, "rule": rule, "attack": attack, "N": N,
                       **{k: float(np.mean([r[k] for r in rs])) for k in rs[0]}}
                row["works"] = (row["spam_share_feed"] <= 0.05 and row["newcomer_wait_median_days"] <= 7
                                and row["newcomer_pass_30d"] >= 0.90)
                res["rows"].append(row)
                print(f"{eps_mode:6s} {rule:4s} {attack:12s} N={N:3d} spam {row['spam_share_feed']:.3f} wait "
                      f"{row['newcomer_wait_median_days']:.1f} d pass30 {row['newcomer_pass_30d']:.2f} "
                      f"works {row['works']}", flush=True)
    res["smallest_working_N_ring"] = {
        m: {rule: min((r["N"] for r in res["rows"] if r["eps_mode"] == m and r["attack"] == "ring" and r["works"]
                       and r["rule"] == rule), default=None) for rule in RULES} for m in EPS_MODES}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("smallest working N (ring):", res["smallest_working_N_ring"])


if __name__ == "__main__":
    knobs() if sys.argv[1:] == ["knobs"] else main()
