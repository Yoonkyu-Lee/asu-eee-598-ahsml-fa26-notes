"""CP39 B1 (journal/cp/CP39.md, rules written before the run): what does "every box receives every public post" (2)
cost one box at 10^6 and 10^7 members?

  .venv/Scripts/python.exe -W ignore experiments/hood_scale.py         -> results/neighborhoods/b_scale.json
  .venv/Scripts/python.exe -W ignore experiments/hood_scale.py 10m     -> results/neighborhoods/b_scale_10m.json (re-judged)
  .venv/Scripts/python.exe -W ignore experiments/hood_scale.py spam    -> results/neighborhoods/spam_all.json

Rates: the CP16 behaviour model (sim/community.py, sourced parameters) at 10^6 members; 10^7 = 10x (estimate: linear).
Times: chip/model/chip.py (chip), results/chip/cp6_crypto.json (measured CPU proof verification). Every number is an
estimate (model) unless marked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chip.model.chip import IMAGE_S, PROOF_S, SPEC, text_table  # noqa: E402
from sim.community import Params, build, topic_rates  # noqa: E402

OUT = ROOT / "results" / "neighborhoods" / "b_scale.json"
CP6_POSTS_PER_S = 1_000_000 / 86_400        # CP6 ordinary day: the chip's ordinary-day load
POST_TOKENS, COMMENT_TOKENS = 128, 48       # (assumption) text lengths for the chip time table
KEEP_DAYS = 30                              # (assumption) forgetting window
BUDGET = {"peak_Mbps": 1.0, "store_TB": 1.0, "x_cp6": 10.0}   # CP39 B1 rule (assumptions)
AUDIT_F = (1.0, 0.1, 0.01)
# CP39 10^7 re-judgement (journal/cp/CP39.md): sourced bases x assumed shares
SOURCES = {"line": "FCC 24-27 (2024 Section 706 report): fixed broadband benchmark 100/20 Mbps, "
                   "https://docs.fcc.gov/public/attachments/FCC-24-27A1.pdf",
           "cap": "Xfinity legacy plans: 1.2 TB per month, https://www.xfinity.com/support/articles/data-usage-plan",
           "disk": "Raspberry Pi SSD kit 512 GB, https://www.raspberrypi.com/news/raspberry-pi-ssds-and-ssd-kits/",
           "pi5": "CP34 necessity: Pi 5 = laptop P-core x 2.93 (Geekbench 6), 4 cores"}
BUDGET_10M = {"down_Mbps": 100 * 0.10, "up_Mbps": 20 * 0.10, "month_GB": 1200 * 0.25, "store_GB": 512 * 0.50,
              "cpu_cores": 1.0, "chip_busy": 1.0, "bootstrap_h": 24.0}
ID_FIRST = 1.2                              # CP27: lazy (ID-first) push bytes / payload
PI5_SLOW = 2.93                             # CP34 / CP20
ONLINE = 0.95                               # CP16 Params.online
JOIN_PER_DAY = (0.001, 0.01)                # (assumption) members joining or replacing a box per day
LINE_BPS = 100e6                            # the FCC benchmark download, for the bootstrap time
UNIT_BYTES = 400 + 256 * 4                  # spam unit: overhead + a 256-token text at ~4 bytes per token (assumption)
LIMITS_PER_DAY = {"old": 6 * 144, "new": 5 * 24}   # CP22: 6 units / 10 min, new groups 5 units / h


def main():
    p = Params(users=1_000_000)
    c = build(p)
    peak = float(max(p.diurnal))
    tr = topic_rates(c)                      # per topic, per second, daily mean
    cpu = json.loads((ROOT / "results" / "chip" / "cp6_crypto.json").read_text(encoding="utf-8"))
    lpe_batch_s = cpu["lp_e_core"]["groth16_bn254_batch64_verify_per_proof_us"] / 1e6 if \
        "groth16_bn254_batch64_verify_per_proof_us" in cpu["lp_e_core"] else None
    pc_batch_s = cpu["p_core"]["groth16_bn254_batch64_verify_per_proof_us"] / 1e6
    tt = text_table()
    post_s, comment_s = tt.text1(POST_TOKENS)[0], tt.text1(COMMENT_TOKENS)[0]

    # (1) per-box load if it holds only its subs: sum over its topics
    per_topic_msgs = tr["post"] + tr["comment"] + tr["vote"]
    box_sub = np.bincount(c.su, weights=per_topic_msgs[c.st], minlength=p.users)

    res = {"what": "CP39 B1 (journal/cp/CP39.md)", "label": "estimate (CP16 behaviour model x chip model)",
           "params": {"dau_frac": p.dau_frac, "posts_per_dau": p.posts_per_dau, "comments_per_dau": p.comments_per_dau,
                      "votes_per_dau": p.votes_per_dau, "peak_factor": peak, "keep_days": KEEP_DAYS,
                      "post_tokens": POST_TOKENS, "comment_tokens": COMMENT_TOKENS, "budget": BUDGET},
           "scales": {}}
    for users, k in ((1_000_000, 1.0), (10_000_000, 10.0)):
        post, comment, vote = (float(tr[x].sum()) * k for x in ("post", "comment", "vote"))
        inside = float(tr["votes_inside"].sum()) * k
        vote_bytes = vote * p.overhead_bytes + inside * p.vote_bytes
        text_Bps = post * (p.overhead_bytes + p.post_bytes) + comment * (p.overhead_bytes + p.comment_bytes) + vote_bytes
        image_Bps = post * p.image_frac * p.image_bytes
        msgs = post + comment + vote
        enc_s = post * post_s + comment * comment_s                     # chip seconds per second, all texts
        img_s = post * p.image_frac * IMAGE_S
        r = {"msgs_per_s_mean": msgs, "msgs_per_s_peak": msgs * peak,
             "posts_per_s_mean": post, "comments_per_s_mean": comment, "vote_msgs_per_s_mean": vote,
             "x_cp6_peak": msgs * peak / CP6_POSTS_PER_S,
             "text_votes_Mbps_peak": text_Bps * peak * 8 / 1e6, "images_Mbps_peak_if_all": image_Bps * peak * 8 / 1e6,
             "text_votes_GB_day": text_Bps * 86400 / 1e9, "images_GB_day_if_all": image_Bps * 86400 / 1e9,
             "store_TB_text_votes": text_Bps * 86400 * KEEP_DAYS / 1e12,
             "store_TB_with_images": (text_Bps + image_Bps) * 86400 * KEEP_DAYS / 1e12,
             "proofs_per_s_peak": msgs * peak,
             "cpu_p_core_share_verify_peak": msgs * peak * pc_batch_s,
             "cpu_lpe_core_share_verify_peak": msgs * peak * lpe_batch_s if lpe_batch_s else None,
             "chip_verify_busy_peak": msgs * peak * PROOF_S,
             "chip_text_busy_peak_by_audit_f": {str(f): enc_s * peak * f for f in AUDIT_F},
             "chip_image_busy_peak_by_audit_f": {str(f): img_s * peak * f for f in AUDIT_F},
             "sub_only_box_msgs_per_s_peak": {"median": float(np.median(box_sub)) * k * peak,
                                              "top1pct": float(np.quantile(box_sub, 0.99)) * k * peak},
             "lie_unseen_by_audit_f_over_boxes": {str(f): {str(n): (1 - f) ** n for n in (10, 100, 1000)}
                                                  for f in AUDIT_F}}
        r["affordable"] = {"bytes": r["text_votes_Mbps_peak"] <= BUDGET["peak_Mbps"],
                           "store": r["store_TB_text_votes"] <= BUDGET["store_TB"],
                           "msgs": r["x_cp6_peak"] <= BUDGET["x_cp6"]}
        r["affordable"]["all"] = all(r["affordable"].values())
        res["scales"][str(users)] = r
        print(users, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


def _rates(k):
    p = Params(users=1_000_000)
    c = build(p)
    tr = topic_rates(c)
    r = {x: float(tr[x].sum()) * k for x in ("post", "comment", "vote", "votes_inside")}
    r["top_topic_share"] = float(c.topic_w.max())
    return p, r


def _pi5_proof_s():
    cpu = json.loads((ROOT / "results" / "chip" / "cp6_crypto.json").read_text(encoding="utf-8"))
    return cpu["p_core"]["groth16_bn254_batch64_verify_per_proof_us"] / 1e6 * PI5_SLOW


def rejudge_10m():
    p, r = _rates(10.0)
    peak = float(max(p.diurnal))
    pi5_proof_s = _pi5_proof_s()
    chip_proof_s = 1 / SPEC["verifier"]["min_proofs_per_s"]          # worst-case throughput, CP14
    tt = text_table()
    post_s, comment_s = tt.text1(POST_TOKENS)[0], tt.text1(COMMENT_TOKENS)[0]
    msgs = r["post"] + r["comment"] + r["vote"]
    text_Bps = (r["post"] * (p.overhead_bytes + p.post_bytes) + r["comment"] * (p.overhead_bytes + p.comment_bytes)
                + r["vote"] * p.overhead_bytes + r["votes_inside"] * p.vote_bytes)
    image_Bps = r["post"] * p.image_frac * p.image_bytes
    raw_Bps = text_Bps + image_Bps
    down_Bps = raw_Bps * ID_FIRST
    store_B = raw_Bps * 86400 * KEEP_DAYS
    texts_s = r["post"] * post_s + r["comment"] * comment_s          # chip seconds per second, all texts
    imgs_s = r["post"] * p.image_frac * IMAGE_S
    boot = {"fetch_h": store_B * ID_FIRST * 8 / LINE_BPS / 3600,
            "verify_h_cpu_4cores": msgs * 86400 * KEEP_DAYS * pi5_proof_s / 4 / 3600,
            "verify_h_chip": msgs * 86400 * KEEP_DAYS * chip_proof_s / 3600,
            "encode_h_chip_f1": texts_s * 86400 * KEEP_DAYS / 3600,
            "encode_h_chip_f001": texts_s * 86400 * KEEP_DAYS * 0.01 / 3600,
            "images_h_chip_f1": imgs_s * 86400 * KEEP_DAYS / 3600}
    boot["ready_h_chip"] = max(boot["fetch_h"], boot["verify_h_chip"])
    boot["ready_h_cpu"] = max(boot["fetch_h"], boot["verify_h_cpu_4cores"])
    up = {}
    for phi in (ONLINE, 0.5, 0.3):
        for j in (0.0,) + JOIN_PER_DAY:
            serve_Bps = j * store_B * ID_FIRST / 86400                 # other members' bootstraps, spread over the day
            up[f"phi{phi}_join{j}"] = {"peak_Mbps": (down_Bps * peak + serve_Bps) / phi * 8 / 1e6,
                                       "mean_Mbps": (down_Bps + serve_Bps) / phi * 8 / 1e6}
    month_GB = {k: (down_Bps * 8 / 1e6 + v["mean_Mbps"]) * 1e6 / 8 * 86400 * 30 / 1e9 for k, v in up.items()}
    res = {"what": "CP39 B1 10^7 re-judgement (journal/cp/CP39.md)", "label": "estimate (CP16 model x10, chip model)",
           "sources": SOURCES, "budget": BUDGET_10M,
           "assumptions": {"id_first_bytes": ID_FIRST, "online": ONLINE, "join_per_day": JOIN_PER_DAY,
                           "keep_days": KEEP_DAYS, "pi5_slowdown": PI5_SLOW},
           "msgs_per_s_peak": msgs * peak, "down_Mbps_peak": down_Bps * peak * 8 / 1e6,
           "down_Mbps_peak_text_votes": text_Bps * ID_FIRST * peak * 8 / 1e6,
           "images_Mbps_peak": image_Bps * ID_FIRST * peak * 8 / 1e6, "upload": up, "month_GB": month_GB,
           "store_GB_30d": store_B / 1e9, "store_GB_30d_no_images": text_Bps * 86400 * KEEP_DAYS / 1e9,
           "cpu_cores_verify_peak": msgs * peak * pi5_proof_s,
           "chip_busy_peak_f1": (msgs * chip_proof_s + texts_s + imgs_s) * peak,
           "chip_busy_peak_f001": (msgs * chip_proof_s + 0.01 * (texts_s + imgs_s)) * peak,
           "bootstrap": boot}
    k0 = f"phi{ONLINE}_join{JOIN_PER_DAY[1]}"
    res["judged_at"] = k0
    res["pass"] = {"down": res["down_Mbps_peak"] <= BUDGET_10M["down_Mbps"],
                   "up": up[k0]["peak_Mbps"] <= BUDGET_10M["up_Mbps"],
                   "month": month_GB[k0] <= BUDGET_10M["month_GB"],
                   "store": res["store_GB_30d"] <= BUDGET_10M["store_GB"],
                   "cpu_no_chip": res["cpu_cores_verify_peak"] <= BUDGET_10M["cpu_cores"],
                   "chip_f1": res["chip_busy_peak_f1"] <= BUDGET_10M["chip_busy"],
                   "bootstrap_chip": boot["ready_h_chip"] <= BUDGET_10M["bootstrap_h"]}
    res["pass"]["all_with_chip"] = all(v for k, v in res["pass"].items() if k != "cpu_no_chip")
    (OUT.parent / "b_scale_10m.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def spam():
    out = {"what": "CP39 spam to every box (journal/cp/CP39.md)", "label": "estimate (CP22 limits x CP16 model)",
           "assumptions": {"unit_bytes": UNIT_BYTES, "limits_per_day": LIMITS_PER_DAY, "flat_rate": True}, "rows": []}
    pi5_proof_s = _pi5_proof_s()
    t256 = text_table().text1(256)[0]
    for users, k in ((1_000_000, 1.0), (10_000_000, 10.0)):
        p, r = _rates(k)
        honest_texts = (r["post"] + r["comment"]) * 86400
        for K in (100, 1_000, 10_000):
            for age, lim in LIMITS_PER_DAY.items():
                n = K * lim
                ps = n / 86400
                out["rows"].append({"users": users, "K": K, "identities": age, "spam_per_day": n,
                                    "store_GB_day": n * UNIT_BYTES / 1e9,
                                    "store_GB_30d": n * UNIT_BYTES * KEEP_DAYS / 1e9,
                                    "verify_per_s": ps, "cpu_cores_pi5": ps * pi5_proof_s,
                                    "chip_encode_busy_f1": ps * t256, "chip_encode_busy_f001": ps * t256 * 0.01,
                                    "feed_share_all": n / (n + honest_texts),
                                    "feed_share_top_tag": n / (n + honest_texts * r["top_topic_share"])})
    (OUT.parent / "spam_all.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for x in out["rows"]:
        print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in x.items()})


if __name__ == "__main__":
    {"10m": rejudge_10m, "spam": spam}.get(sys.argv[1] if sys.argv[1:] else "", main)()
