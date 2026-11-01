"""CP16: how much work reaches one node box, by community size, replication policy and vote design.
Rules are pre-registered in cp/CP16.md (Q1-Q4, S1-S3).

Run from project/:  .venv/Scripts/python.exe src/sim_cp16.py
Writes results/cp16/cp16_load.json and results/cp16/trace_peak_hour.json (open in ui.perfetto.dev).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import community as cm  # noqa: E402
from sim import policy as po  # noqa: E402
from sim.network import Gossip  # noqa: E402
from sim.trace import Trace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp16"
SIZES = (1_000, 10_000, 100_000, 1_000_000)
POLICIES = ("P-all", "P-sub", "P-sub+k")
VOTES = ("V1", "V2")
CHIP_VERIFY_PER_S = 13_429      # CP14, 6 x 6, worst mix
CP12_POSTS_PER_S = 1e6 / 86400  # the CP12 assumption: every box ingests 10^6 posts a day
ENC_MS_PER_POST = 8.05          # CP12, 32 x 32 at 500 MHz, batch 1

# Base parameters and their source ranges are filled from the CP16 survey (cp/CP16.md, "매개변수").
def diurnal(amp: float) -> tuple:
    """(추정) cosine day, peak at 20:00; peak/mean = 1 + amp. No sourced hourly curve was found."""
    return tuple(float(x) for x in 1 + amp * np.cos(2 * np.pi * (np.arange(24) - 20) / 24))


BASE = cm.Params(
    dau_frac=0.26,           # Reddit DAUq 121.4M / WAUq 471.6M (Q4'25 shareholder letter): daily share of weekly users
    posts_per_dau=0.0147,    # 0.10 posts+comments per DAUq-day (2.2B in H2'25) x posts share 1/(1+5.8) (2022 totals)
    comments_per_dau=0.0853,
    votes_per_dau=0.82,      # upvotes 24B vs posts+comments 2.93B (2022) = 8.2 per item, x 0.10
    lurker_frac=0.90,        # 90-9-1 (Nielsen Norman Group)
    tail_sigma=1.85,         # lognormal giving "20% of users contribute 84% of the comments" (arXiv 2503.02661)
    users_per_topic=4700,    # 121.4M DAUq / >100k active subreddits = 1,214 DAU per sub, / 0.26 (S-1, Q4'25 letter)
    zipf_s=1.3,              # 85% of (candidate-mention) comments in the top 500 of 100k subreddits (Pew 2016)
    subs_per_user=10.0,      # (추정); heavy posters post in 26-29 communities (arXiv 1503.01180)
    image_frac=0.30,         # ~30% of submissions are images, gifs, videos (arXiv 1703.01725, 2005-2014 data)
    image_bytes=177e3,       # web image p90 (HTTP Archive Web Almanac 2024) as a photo-upload proxy (추정 choice)
    post_bytes=650.0,        # (추정) CP4 sample: mean 164 tokens of English text ~ 4 bytes per token
    comment_bytes=180.0,     # 30.15 words mean (arXiv 2105.12006) x ~6 bytes per word
    diurnal=diurnal(0.6),    # (추정)
    online=0.95,             # (추정) below the broadband link's "at most 99%" (arXiv 1709.09349)
    hop_s=0.05, uplink_bps=20e6, vote_bundle_s=50.0,   # (추정)
)
RANGES: dict = {  # low, high; source-backed where a source exists, otherwise (추정)
    "dau_frac": (0.10, 0.50),
    "posts_per_dau": (0.0147, 0.0735),      # x5: DAUq counts logged-out readers, engaged users write more
    "comments_per_dau": (0.0853, 0.4265),
    "votes_per_dau": (0.82, 4.1),
    "lurker_frac": (0.75, 0.95),
    "tail_sigma": (1.2, 2.3),
    "users_per_topic": (500.0, 20000.0),
    "zipf_s": (1.0, 1.5),
    "subs_per_user": (3.0, 30.0),
    "image_frac": (0.3, 0.6),
    "online": (0.80, 0.99),
    "diurnal": (diurnal(0.3), diurnal(1.0)),
}


def load_table(c: cm.Community, policy: str, vote: str) -> dict:
    hb, ht = po.holders(c, policy)
    peak = max(c.p.diurnal)
    row = {}
    for label, f in (("mean", 1.0), ("peak", peak)):
        r = cm.topic_rates(c, vote, f)
        L = po.box_load(c, hb, ht, r)
        verify = L["post"] + L["comment"] + L["vote"]
        text = L["post"] + L["comment"]
        img = L["post"] * c.p.image_frac
        stats = lambda x: {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
                           "max": float(x.max())}
        row[label] = {"verify_per_s": stats(verify), "text_per_s": stats(text), "images_per_s": stats(img),
                      "vote_share_of_verify": float(L["vote"].sum() / max(verify.sum(), 1e-30)),
                      "votes_per_vote_msg": float(r["votes_inside"].sum() / max(r["vote"].sum(), 1e-30))}
    r = cm.topic_rates(c, vote, 1.0)
    row["availability"] = po.availability(c, hb, ht, r["post"])
    row["topics_held_per_box"] = float(c.n_topics if hb is None else len(hb) / c.p.users)
    return row


def analytic(p: cm.Params) -> dict:
    out = {}
    for M in SIZES:
        c = cm.build(p.with_(users=M))
        out[M] = {"n_topics": c.n_topics, "subs_realized": float(len(c.st) / M),
                  "community_msgs_per_day": {k: float(v.sum() * 86400) for k, v in cm.topic_rates(c, "V1").items()}}
        for pol in POLICIES:
            for v in VOTES:
                out[M][f"{pol} {v}"] = load_table(c, pol, v)
        print("analytic", M, f"{time.time() - T0:.0f}s", flush=True)
    return out


def pick(table: dict, M: int) -> tuple[str, str]:
    """Q1 then Q2: availability >= 99% everywhere, then the lowest P99-box verify rate at the peak hour."""
    ok = [(k, v) for k, v in table[M].items() if isinstance(v, dict) and "availability" in v
          and v["availability"]["min"] >= po.AVAIL_TARGET]
    k, _ = min(ok, key=lambda kv: kv[1]["peak"]["verify_per_s"]["p99"])
    pol, vote = k.split(" ")
    return pol, vote


def des(p: cm.Params, policy: str, vote: str, H: float = 3600.0) -> tuple[dict, Gossip]:
    c = cm.build(p)
    hb, ht = po.holders(c, policy)
    if hb is None:
        hb, ht = np.repeat(np.arange(c.p.users), c.n_topics), np.tile(np.arange(c.n_topics), c.p.users)
    peak = max(p.diurnal)
    # boxes to watch in the trace: the median and the busiest by expected load
    exp_load = po.box_load(c, hb, ht, cm.topic_rates(c, vote, peak))
    tot = exp_load["post"] + exp_load["comment"] + exp_load["vote"]
    order = np.argsort(tot)
    watch = (int(order[len(order) // 2]), int(order[-1]))
    g = Gossip(c, hb, ht, seed=17, watch=watch)
    g.schedule_hour(peak, H, vote)
    t = time.time()
    g.eng.run()
    wall = time.time() - t
    # expected deliveries: online boxes x held topics' rates from online authors (Q3)
    on = g.on
    lam = {k: np.bincount(c.st[on[c.su]], weights=c.pair_rate(k)[on[c.su]] * peak, minlength=c.n_topics)
           for k in ("post", "comment")}
    T = p.vote_bundle_s
    lv = c.pair_rate("vote")[on[c.su]] * peak
    lam["vote"] = np.bincount(c.st[on[c.su]], weights=lv if vote == "V1" else (1 - np.exp(-lv * T)) / T,
                              minlength=c.n_topics)
    hold_on = on[hb]
    exp = {k: float(lam[k][ht[hold_on]].sum() * H) for k in lam}
    got = {k: float(g.count[k].sum()) for k in ("post", "comment", "vote")}
    sp = g.spread()
    res = {"users": p.users, "policy": policy, "vote": vote, "hour_s": H, "events": g.eng.events, "wall_s": wall,
           "messages": len(g.msgs), "completeness": sp["completeness"], "expected": exp, "got": got,
           "rel_err_total": abs(sum(got.values()) - sum(exp.values())) / sum(exp.values()),
           "delay_s": {k: {"p50": float(np.nanpercentile(v, 50)), "p99": float(np.nanpercentile(v, 99)),
                           "max": float(np.nanmax(v)), "n": int(len(v))} for k, v in sp["delay_s"].items()},
           "dup_over_unique_bytes": float(g.dup_bytes.sum() / max(g.rx_bytes.sum(), 1)),
           "rx_Mbps_p99_box": float(np.percentile((g.rx_bytes + g.dup_bytes) * 8 / H / 1e6, 99)),
           "watch": watch}
    if vote == "V2":
        waits = [m["wait"] for m in g.msgs if m["kind"] == "vote"]
        res["vote_bundle_wait_s"] = {"p50": float(np.percentile(waits, 50)), "max": float(np.max(waits))}
    return res, g


def write_trace(g: Gossip, path: Path) -> None:
    tr = Trace()
    for box in g.watch:
        name = f"box {box}"
        log = [(t, k) for b, t, k in g.watch_log if b == box]
        for sec in range(int(max([t for t, _ in log], default=0)) + 1):
            ks = [k for t, k in log if sec <= t < sec + 1]
            if ks:
                tr.counter(name, "messages per second", sec, {k: ks.count(k) for k in ("post", "comment", "vote")})
        for t, k in log:
            tr.span(name, "verify", k, t, 1 / CHIP_VERIFY_PER_S)
    tr.write(path)


def sensitivity(policy: str, vote: str, M: int = 100_000) -> dict:
    out = {}
    for name, (lo, hi) in RANGES.items():
        out[name] = {}
        for tag, val in (("lo", lo), ("hi", hi)):
            p = BASE.with_(users=M, **{name: val})
            c = cm.build(p)
            tab = {f"{pol} {v}": load_table(c, pol, v) for pol in POLICIES for v in VOTES}
            k = pick({M: tab}, M)
            row = tab[f"{policy} {vote}"]
            out[name][tag] = {"value": val, "pick": " ".join(k), "p99_peak_verify": row["peak"]["verify_per_s"]["p99"],
                              "p99_peak_text": row["peak"]["text_per_s"]["p99"],
                              "avail_min": row["availability"]["min"]}
        print("sens", name, out[name], flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in BASE.__dict__.items()},
           "ranges": {k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in RANGES.items()}}
    res["analytic"] = analytic(BASE)
    pol, vote = pick(res["analytic"], 100_000)
    res["pick_Q1_Q2_at_1e5"] = {"policy": pol, "vote": vote}
    print("pick", pol, vote, flush=True)
    d, g = des(BASE.with_(users=10_000), pol, vote)
    res["des_10000"] = d
    print("des", {k: d[k] for k in ("events", "wall_s", "completeness", "rel_err_total")}, d["delay_s"], flush=True)
    write_trace(g, OUT / "trace_peak_hour.json")
    d_all, _ = des(BASE.with_(users=10_000), "P-all", vote, H=600.0)
    res["des_10000_P-all_600s"] = d_all
    res["sensitivity_1e5"] = sensitivity(pol, vote)
    (OUT / "cp16_load.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT / "cp16_load.json")


def generator_check(p: cm.Params, vote: str, hours: float = 2000.0) -> dict:
    """Q3, statistical half: messages the event generator makes over many peak hours vs the closed form.
    (The other half is delivery completeness in des(): every online holder gets every message.)"""
    c = cm.build(p)
    g = Gossip(c, c.su, c.st, seed=18)
    peak = max(p.diurnal)
    H = hours * 3600
    g.schedule_hour(peak, H, vote)
    made = {"post": 0, "comment": 0, "vote": 0}
    for _, _, fn, args in g.eng._q:
        made[args[2]] += 1
    on = g.on
    T = p.vote_bundle_s
    lv = c.pair_rate("vote")[on[c.su]] * peak
    exp = {k: float((c.pair_rate(k)[on[c.su]] * peak).sum() * H) for k in ("post", "comment")}
    exp["vote"] = float((lv if vote == "V1" else (1 - np.exp(-lv * T)) / T).sum() * H)
    return {"hours": hours, "made": made, "expected": exp,
            "rel_err": {k: abs(made[k] - exp[k]) / exp[k] for k in made}}


def propagation_test(p: cm.Params, policy: str, n: int = 300, uplinks=(5e6, 20e6, 100e6)) -> dict:
    """Q4 with enough samples: n text posts and n image posts injected into the largest topic."""
    out = {}
    for up in uplinks:
        q = p.with_(uplink_bps=up)
        c = cm.build(q)
        hb, ht = po.holders(c, policy)
        g = Gossip(c, hb, ht, seed=19)
        big = max(g.members, key=lambda t: len(g.members[t]))
        authors = g.rng.choice(g.members[big], size=2 * n)
        for i, a in enumerate(authors):   # spaced 10 s apart so uplinks are idle between messages
            img = i % 2 == 1
            size = q.overhead_bytes + q.post_bytes + (q.image_bytes if img else 0)
            g.eng.at(10.0 * i, g.publish, int(a), int(big), "post", size, {"image": img})
        g.eng.run()
        sp = g.spread()["delay_s"]
        out[f"{up / 1e6:g} Mbps"] = {"holders": int(len(g.members[big])),
                                    **{k: {"p50": float(np.nanpercentile(v, 50)), "p99": float(np.nanpercentile(v, 99))}
                                       for k, v in sp.items()}}
        print("prop", up, out[f"{up / 1e6:g} Mbps"], flush=True)
    return out


def extra() -> None:
    path = OUT / "cp16_load.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    pol, vote = res["pick_Q1_Q2_at_1e5"]["policy"], res["pick_Q1_Q2_at_1e5"]["vote"]
    res["generator_check_10000"] = generator_check(BASE.with_(users=10_000), vote)
    print("gen", res["generator_check_10000"]["rel_err"], flush=True)
    res["propagation_10000"] = propagation_test(BASE.with_(users=10_000), pol)
    path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


T0 = time.time()
if __name__ == "__main__":
    extra() if "--extra" in sys.argv else main()
