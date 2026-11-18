"""CP39 B1 (journal/cp/CP39.md, rules written before the run): what does "every box receives every public post" (2)
cost one box at 10^6 and 10^7 members?

  .venv/Scripts/python.exe -W ignore experiments/hood_scale.py   -> results/neighborhoods/b_scale.json

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
from chip.model.chip import IMAGE_S, PROOF_S, text_table  # noqa: E402
from sim.community import Params, build, topic_rates  # noqa: E402

OUT = ROOT / "results" / "neighborhoods" / "b_scale.json"
CP6_POSTS_PER_S = 1_000_000 / 86_400        # CP6 ordinary day: the chip's ordinary-day load
POST_TOKENS, COMMENT_TOKENS = 128, 48       # (assumption) text lengths for the chip time table
KEEP_DAYS = 30                              # (assumption) forgetting window
BUDGET = {"peak_Mbps": 1.0, "store_TB": 1.0, "x_cp6": 10.0}   # CP39 B1 rule (assumptions)
AUDIT_F = (1.0, 0.1, 0.01)


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


if __name__ == "__main__":
    main()
