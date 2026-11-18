"""One box's state (the one object the box modules share) and its small helpers: the clock, the test link cut, HTTP to
a peer. Every other box module is functions that take this object; the lock guards all of it."""
from __future__ import annotations

import itertools
import json
import os
import queue
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOM, Coco  # noqa: E402
from sim.box import Msg, Shared  # noqa: E402
from sim.recon import TopicKeySet  # noqa: E402
from sim import proto24 as p24  # noqa: E402
from store import Store  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Node:
    def __init__(self, cfg: dict, bid: int, state: str | None = None):
        self.cfg, self.bid, self.n = cfg, bid, cfg["n"]
        self.holders = {int(t): v for t, v in cfg["holders"].items()}
        self.mesh = {int(t): {int(b): p for b, p in m.items()} for t, m in cfg["mesh"].items()}
        self.addr = cfg.get("addr") or [f"127.0.0.1:{p}" for p in cfg["ports"]]
        self.peer_ips = {a.rsplit(":", 1)[0] for a in self.addr}
        rk = cfg.get("room_key")
        self.room_key = bytes.fromhex(rk) if rk and bid in self.holders[ROOM] else None
        self.S = Shared()
        topics = [t for t, hs in self.holders.items() if bid in hs]
        self.box = p24.Box24(bid, topics, {ROOM: self.room_key} if self.room_key else {}, self.S, {"text": 0, "image": 0})
        # this box's identity (user bid) is drawn here and never leaves this process
        self.tier = cfg.get("tiers", ["new"] * self.n)[bid]
        self.W = p24.World24(self.n, seed=int.from_bytes(os.urandom(8), "big"), tiers=cfg.get("tiers"))
        self.commit = self.W.ids[bid].commit
        self.lock = threading.RLock()
        self.got: set[int] = set()
        self.msgs: dict[int, Msg] = {}
        self.own: set[int] = set()
        self.parent: dict[int, int] = {}
        self.full: dict[int, bytes] = {}
        self.full_ev: dict[int, threading.Event] = {}
        self.asked: set[int] = set()
        self.slash_sent: set[int] = set()
        self.verified: list[int] = []
        self.count: Counter = Counter()
        self.chip_s = 0.0
        self.pending = 0
        self.slashed = False
        self.notices: list[dict] = []
        self.enc_tokens_max = 0
        self.peer_seen: dict[int, float] = {}
        self.st = Store(Path(state)) if state else None
        self.up: set[int] = set()                  # peers known to be on: they said hello, answered or sent (rule 2;
                                                   # CP28: the mesh is the boxes that are on, as in sim.exchange)
        self.announcers: dict[int, list] = {}      # CP28 lazy push: mid -> boxes that announced it, in order
        self.replaying = False
        self.replayed: set[int] = set()
        self.restored = 0                        # messages taken back from disk at start (CP25)
        self.tomb: dict[int, set] = {}           # deleted post -> the author's one-time public keys (hex)
        self.gone: set[int] = set()              # posts held here and then forgotten (CP28)
        # CP29 "recon": the reconciled set (t_key, id), windows, triggers, test-only faults
        self.keys = TopicKeySet()                # CP33: by sub; a session covers the subs both boxes hold
        self.R = float(cfg.get("R_s", 7 * 86400.0))
        self.sigma = float(cfg.get("sigma_s", 120.0))
        self.pae = tuple(cfg.get("pae_s", (600.0, 1800.0)))
        self.probe = {}                          # peer -> [backoff s, next probe time]
        self.dirty: set[int] = set()
        self.last_recon: dict[int, float] = {}
        self.next_floor: dict[int, float] = {}
        self.recon_busy: set[int] = set()
        self.gone_t: dict[int, float] = {}       # forgotten id -> its t_key (kept until now - R - 2 sigma)
        self.fault = {"drop": set(), "offset": 0.0, "silent": 0, "stall": None, "no_fence": False}
        self.silent_mids: set[int] = set()       # test-only: envelopes this box "took" and lost (nobody notices)
        # CP30 (cp/CP30.md): K one fetch per id at a time, F floor only for pairs that changed (with the fence)
        self.fetching: dict[int, int] = {}       # id -> the peer a session is fetching it from
        self.skipped: dict[int, set] = {}        # id -> peers whose session skipped it (in flight elsewhere)
        self.again: set[int] = set()             # reconcile again with these when the running session ends
        self.clean: set[int] = set()             # peers p with clean(p): no difference at my last session with p
        self.ver = 0                             # mutation generation of my set (the fence)
        self.lazy_pending: set[int] = set()      # ids a lazy push is still bringing (CP28 IHAVE1/IWANT)
        self.pmax = float(cfg.get("probe_max_s", 300.0))
        self.pool = ThreadPoolExecutor(16)
        self.coco = Coco() if Coco.available() else None
        self.encq: queue.PriorityQueue = queue.PriorityQueue()   # (class, seq, job): encoder.py
        self.seq = itertools.count()

    def now(self) -> float:
        """This box's clock (a test-only offset can be set through /fault)."""
        return time.time() + self.fault["offset"]

    def forgotten(self, mid: int) -> bool:
        """Held here, then deleted by its author (or dropped on arrival by a matching tombstone). A tombstone alone is
        not enough: a forged delete leaves one too (another key), and the real post must still come in. An id alone
        has no key to compare, so only posts known gone are refused (CP28: ids and picture pulls checked the mid)."""
        return mid in self.gone

    def inc(self, k: int = 1) -> None:
        with self.lock:
            self.pending += k

    def cut(self, peer: int) -> None:
        """Test-only link cut (/fault drop): calls to that peer fail like a dead link."""
        if peer in self.fault["drop"]:
            raise ConnectionError(f"link to box {peer} is cut (test)")

    def url(self, peer: int, path: str) -> str:
        return f"http://{self.addr[peer]}{path}"

    def post(self, peer: int, path: str, payload: dict) -> None:
        self.cut(peer)
        req = urllib.request.Request(self.url(peer, path), data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
