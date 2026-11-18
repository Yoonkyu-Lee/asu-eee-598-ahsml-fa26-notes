"""CP37 (1), next piece (journal/cp/CP37.md, rules written before the run): how many receipts N, and which receipt
rule, keep spam out of feeds without keeping honest newcomers out for long. A day-level model, every number an estimate.

  .venv/Scripts/python.exe -W ignore experiments/rep_gate.py   -> results/chip/cp37_rep_gate.json

Behaviour: the CP16 BASE (sim/community.py). A post enters feeds if its author holds N counted receipts when posting;
otherwise it goes to a "new posts" lane that draws rho x the votes. Votes go to posts in proportion to quality; an
honest voter votes for spam at eps x the weight. Receipt rules: any (every voter counts), age (voters 30+ days old),
rep (founders, or voters who themselves hold N counted receipts). Attacks with K = 1% of users from day 0, each
posting at the CP22 limit: lone, ring (every attacker votes every other attacker's posts), ring + bought (each attacker
buys N counted votes once).
Mistaken honest votes on spam: "weight" (the rule as written: eps x a post's weight, so spam volume draws votes) and
"share" (marked after the smoke run: eps of all honest votes go to spam, split by visibility).
Newcomer wait (marked after the smoke run): from a newcomer's first post to the first post that entered feeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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


def run(rule: str, attack: str, N: int, seed: int, eps_mode: str = "weight") -> dict:
    p = Params()
    rng = np.random.default_rng(seed)
    K = int(ATTACK_SHARE * U0)
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
        n_items = rng.poisson(rate * honest)
        authors = np.repeat(np.arange(cap), n_items)
        q = rng.lognormal(0, Q_SIGMA, len(authors)) / np.exp(Q_SIGMA ** 2 / 2)
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
            feed["honest"] += int(passed.sum())
            feed["spam"] += int(s_pass.sum()) * SPAM_PER_DAY
        # honest votes, spread by weight
        w_h = q * np.where(passed, 1.0, RHO)
        w_s = np.full(len(att), float(SPAM_PER_DAY)) * np.where(s_pass, 1.0, RHO)
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
            rq[att] += SPAM_PER_DAY * (q_ring.sum() - q_ring)
    newc = np.flatnonzero((join >= 1) & (join < MEASURE[0]) & ~attacker & (first_post >= 0)
                          & (first_post - join <= 30))
    wait = np.where(first[newc] >= 0, first[newc] - first_post[newc], np.inf)
    return {"spam_share_feed": feed["spam"] / max(feed["spam"] + feed["honest"], 1),
            "honest_feed_items_per_day": feed["honest"] / (MEASURE[1] - MEASURE[0]),
            "newcomer_wait_median_days": float(np.median(wait)),
            "newcomer_pass_30d": float(np.mean(wait <= 30)), "newcomers": int(len(newc)),
            "bought_votes_total": K * N if attack == "ring_bought" else 0}


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
    main()
