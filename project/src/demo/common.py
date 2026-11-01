"""Shared by the box processes and the scenario runner (CP21): the small community, message wire format, COCO pixels.

Community: boxes = users 0..n-1. Topic 0 = public sub A (everyone), 1 = sub B, 2 = private room R (room key),
3 = slashing notices (everyone). Meshes come from sim.network.meshes with a fixed seed, so the runner, the boxes
and the simulator all use the same links.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sim.box import Msg  # noqa: E402
from sim.network import meshes  # noqa: E402

A, B, ROOM, MEMB = 0, 1, 2, 3
TOPIC_NAMES = {A: "sub A", B: "sub B", ROOM: "room R", MEMB: "notices"}
COCO = ROOT / "data" / "coco"
IMG_BYTES = 224 * 224 * 3


def community(n: int = 5, D: int = 2, seed: int = 21, base_port: int = 8601) -> dict:
    holders = {A: list(range(n)), B: [0, 2, 4][: max(1, min(3, n))], ROOM: [1, 2, 3][: max(1, min(3, n - 1))],
               MEMB: list(range(n))}
    mesh = meshes({t: np.array(h) for t, h in holders.items()}, D, np.random.default_rng(seed))
    return {"n": n, "D": D, "seed": seed, "ports": [base_port + i for i in range(n)],
            "holders": {str(t): h for t, h in holders.items()},
            "mesh": {str(t): {str(b): [int(x) for x in p] for b, p in m.items()} for t, m in mesh.items()},
            "room_key": os.urandom(32).hex()}


def save_cfg(cfg: dict, path: Path) -> None:
    path.write_text(json.dumps(cfg), encoding="utf-8")


def msg_to_json(m: Msg) -> dict:
    return {"mid": m.mid, "kind": m.kind, "topic": m.topic, "t0": m.t0, "size": m.size, "epub": m.epub.hex(),
            "sig": m.sig.hex(), "nonce": m.nonce.hex(), "ct": m.ct.hex(), "epoch": m.epoch, "x": str(m.x),
            "y": str(m.y), "nul": str(m.nul), "vnul": None if m.vnul is None else str(m.vnul), "proof_ok": m.proof_ok,
            "image": m.image, "tokens": m.tokens, "pull_bytes": m.pull_bytes}


def msg_from_json(d: dict) -> Msg:
    return Msg(int(d["mid"]), d["kind"], int(d["topic"]), float(d["t0"]), float(d["size"]), bytes.fromhex(d["epub"]),
               bytes.fromhex(d["sig"]), bytes.fromhex(d["nonce"]), bytes.fromhex(d["ct"]), int(d["epoch"]),
               int(d["x"]), int(d["y"]), int(d["nul"]), None if d["vnul"] is None else int(d["vnul"]),
               bool(d["proof_ok"]), d["image"], int(d["tokens"]), float(d["pull_bytes"]))


class Coco:
    """COCO val2017 pixels (224 x 224, data/coco/val2017_224.npy) and captions; images 0-299 were CP20's calibration."""

    def __init__(self):
        self.px = np.load(COCO / "val2017_224.npy", mmap_mode="r")
        self.caps = json.loads((COCO / "val2017_index.json").read_text(encoding="utf-8"))["captions"]

    def pixels(self, i: int) -> np.ndarray:
        return np.asarray(self.px[i])
