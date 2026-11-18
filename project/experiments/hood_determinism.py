"""CP39 R1 (journal/cp/CP39.md): if boxes run slightly different encoders, how far does semantic routing diverge?

  .venv/Scripts/python.exe -W ignore experiments/hood_determinism.py   -> results/neighborhoods/r1_determinism.json

The same 20,000 Reddit posts embedded by four encoder variants (CP3 cache: FP32, FP16, dynamic INT8, per-channel INT8)
stand for boxes with different devices. Neighbourhoods are k-means centres of the FP32 embeddings; a box is an
interest vector mixing 1-3 centres; a box forwards a post to the 3 boxes whose interest is closest to its embedding.
Measured on the model: next-hop agreement between variants, neighbourhood agreement, and how often an attacker who may
send any embedding within the honest tolerance can move a post into another neighbourhood.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cp3_cache"
OUT = ROOT / "results" / "neighborhoods" / "r1_determinism.json"
VARIANTS = ("fp32", "fp16", "int8_dyn_hf", "int8_pc")
K, BOXES, HOPS, SEED, ATTACK_N = 50, 200, 3, 39, 2000
Q_HONEST = 0.001            # tolerance accepts 99.9 % of honest re-encodings (CP17, CP27 convention)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def kmeans(x, k, rng, iters=25):
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(x @ c.T, axis=1)
        c = unit(np.stack([x[a == j].mean(0) if np.any(a == j) else c[j] for j in range(k)]))
    return c


def top(e, interests, k=HOPS):
    return np.argsort(-(e @ interests.T), axis=1)[:, :k]


def toward(e, c, theta):
    """Rotate each unit e toward unit c by at most theta (stop at c)."""
    cosang = np.clip(np.sum(e * c, axis=1), -1, 1)
    ang = np.arccos(cosang)
    step = np.minimum(theta, ang)
    perp = unit(c - cosang[:, None] * e + 1e-12)
    return unit(np.cos(step)[:, None] * e + np.sin(step)[:, None] * perp)


def attack(e, cent, theta, rng):
    """Share of posts an attacker can move into some other neighbourhood with an embedding within theta of e."""
    idx = rng.choice(len(e), ATTACK_N, replace=False)
    ee = e[idx]
    own = np.argmax(ee @ cent.T, axis=1)
    moved = np.zeros(len(ee), bool)
    for t in range(len(cent)):
        y = toward(ee, np.broadcast_to(cent[t], ee.shape), theta)
        moved |= (np.argmax(y @ cent.T, axis=1) == t) & (own != t)
    return float(moved.mean())


def main():
    rng = np.random.default_rng(SEED)
    E = {v: unit(np.load(CACHE / f"{v}.npz")["c"].astype(np.float64)) for v in VARIANTS}
    cent = kmeans(E["fp32"], K, rng)
    mix = np.zeros((BOXES, K))
    for b in range(BOXES):
        js = rng.choice(K, rng.integers(1, 4), replace=False)
        mix[b, js] = rng.random(len(js)) + 0.2
    interests = unit(mix @ cent)
    hops = {v: top(E[v], interests) for v in VARIANTS}
    near = {v: np.argmax(E[v] @ cent.T, axis=1) for v in VARIANTS}
    pairs, coss = {}, []
    for a, b in itertools.combinations(VARIANTS, 2):
        cos = np.sum(E[a] * E[b], axis=1)
        coss.append(cos)
        inter = np.array([len(set(x) & set(y)) for x, y in zip(hops[a], hops[b])])
        tau = float(np.quantile(cos, Q_HONEST))
        pairs[f"{a}|{b}"] = {"cos_min": float(cos.min()), "cos_q001": tau, "cos_median": float(np.median(cos)),
                             "next_hop_overlap_mean": float(inter.mean() / HOPS), "next_hop_all_same": float((inter == HOPS).mean()),
                             "neighbourhood_same": float((near[a] == near[b]).mean()),
                             "attack_move_share": attack(E[b], cent, float(np.arccos(tau)), rng)}
    tau_all = float(np.quantile(np.concatenate(coss), Q_HONEST))      # a relay does not know the author's device
    res = {"what": "CP39 R1 determinism (journal/cp/CP39.md)", "label": "estimate (model) on measured embeddings (CP3 cache)",
           "params": {"K": K, "boxes": BOXES, "hops": HOPS, "seed": SEED, "attack_posts": ATTACK_N, "q_honest": Q_HONEST,
                      "variants": VARIANTS, "posts": int(len(E["fp32"]))},
           "pairs": pairs,
           "mixed_devices": {"tau": tau_all, "theta_deg": float(np.degrees(np.arccos(tau_all))),
                             "attack_move_share": attack(E["fp32"], cent, float(np.arccos(tau_all)), rng)},
           "deterministic": {"tau": 1.0, "attack_move_share": 0.0},
           "centre_spacing_deg": {"nearest_median": float(np.median(np.degrees(np.arccos(np.clip(
               np.sort(cent @ cent.T, axis=1)[:, -2], -1, 1)))))}}
    worst_same = min(p["next_hop_all_same"] for p in pairs.values())
    worst_attack = max(p["attack_move_share"] for p in pairs.values())
    res["verdict"] = {"needed": worst_same < 0.90 or worst_attack >= 0.10, "worst_next_hop_all_same": worst_same,
                      "worst_attack_move_share": worst_attack}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, p in pairs.items():
        print(f"{k:24s} cos q0.1% {p['cos_q001']:.4f}  next hop all same {p['next_hop_all_same']:.3f}  "
              f"overlap {p['next_hop_overlap_mean']:.3f}  hood same {p['neighbourhood_same']:.3f}  attack {p['attack_move_share']:.3f}")
    print("mixed devices", res["mixed_devices"], "centre spacing", res["centre_spacing_deg"])
    print("verdict", res["verdict"])


if __name__ == "__main__":
    main()
