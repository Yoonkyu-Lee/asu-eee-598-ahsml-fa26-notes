"""CP6 load model: what an always-on community node box spends, on an ordinary day and under a spam flood.

Run from project/:  .venv/Scripts/python.exe src/model_cp6.py
Reads measured results (CP4, CP5, CP6) and writes results/cp6/cp6_box.json. Every output is an estimate (추정)
built from those measurements, the assumptions marked below, and sourced numbers from cp/cp6_survey.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
POSTS_PER_DAY = 1_000_000  # assumed community rate, as in CP1-CP5
MSG_BYTES = 1024           # assumed size of one incoming message on the wire (envelope + proof + text)
LINKS_MBPS = (10, 100)     # assumed downstream bandwidth an attacker can fill toward one box
BOX_IDLE_W = {"Raspberry Pi 5 idle (출처, cp6_survey.md)": (2.4, 3.3), "2-bay NAS, disks asleep (출처)": (4.41, 4.41)}


def main() -> None:
    crypto = json.loads((RES / "cp6" / "cp6_crypto.json").read_text())
    cp5 = json.loads((RES / "cp5" / "cp5.json").read_text())
    cp4 = json.loads((RES / "cp4" / "cp4_npu.json").read_text())
    e = crypto["energy"]
    out: dict = {"assumptions": {"posts_per_day": POSTS_PER_DAY, "msg_bytes": MSG_BYTES, "links_mbps": LINKS_MBPS}}

    # Per post, active (above idle) joules: one anonymous-credential proof (Groth16) + Ed25519 + encoder.
    crypto_j = {"lp_e_core": e["lp_e_core/groth16"]["active_J_per_op"] + e["lp_e_core/ed25519"]["active_J_per_op"],
                "p_core": e["p_core/groth16"]["active_J_per_op"] + e["p_core/ed25519"]["active_J_per_op"]}
    encoders = {
        "LP-E core INT8 (CP4, 측정)": cp4["cpu_lp_e_int8"]["active_J_per_post"],
        "laptop NPU, per-head mask (CP5, 측정)": cp5["npu_fp16_heads_mask"]["speed"]["active_J_per_post"],
        "Hailo-8L batch 1 (출처 속도 x 비전 모델 전력, 추정)": 0.016,
        "design 16x16 INT8 (CP4 모델, 추정)": 0.0086,
    }
    rate = POSTS_PER_DAY / 86_400
    day = {}
    for name, ej in encoders.items():
        j = ej + crypto_j["lp_e_core"]
        day[name] = {"encoder_J": ej, "crypto_J": crypto_j["lp_e_core"], "avg_active_W": j * rate,
                     "crypto_share": crypto_j["lp_e_core"] / j}
    out["ordinary_day"] = {"posts_per_s": rate, "per_encoder": day, "box_idle_W": BOX_IDLE_W}

    # Spam flood: every message carries a proof that must be verified before it can be dropped.
    verify = {
        "P-core, one by one": (crypto["p_core"]["groth16_bn254_5_public_inputs"]["2^12"]["verify_us"], e["p_core/groth16"]["active_J_per_op"]),
        "P-core, batch of 64": (crypto["p_core"]["groth16_bn254_batch64_verify_per_proof_us"], None),
        "LP-E core, one by one": (crypto["lp_e_core"]["groth16_bn254_5_public_inputs"]["2^12"]["verify_us"], e["lp_e_core/groth16"]["active_J_per_op"]),
        "LP-E core, batch of 64": (crypto["lp_e_core"]["groth16_bn254_batch64_verify_per_proof_us"], None),
    }
    flood = {}
    for mbps in LINKS_MBPS:
        msgs = mbps * 1e6 / 8 / MSG_BYTES
        rows = {}
        for name, (us, j) in verify.items():
            if j is None:  # batch: energy scaled by time from the one-by-one run on the same core (추정)
                single_us, single_j = verify[name.replace("batch of 64", "one by one")]
                j = single_j * us / single_us
            rows[name] = {"us_per_proof": us, "cores_needed": msgs * us / 1e6, "active_W": msgs * j, "active_mJ_per_proof": j * 1e3}
        flood[f"{mbps}Mbps"] = {"msgs_per_s": msgs, "verifiers": rows,
                                 "mJ_per_proof_for_2W": 2.0 / msgs * 1e3}  # what fits a 2 W box budget (가정)
    out["spam_flood"] = flood
    (RES / "cp6" / "cp6_box.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
