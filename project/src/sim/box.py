"""Box software (CP17): what one home node box does with a message, run for real.

Author side (`World.make`): fresh Ed25519 key per message (a delete is signed with the same key), body sealed with
ChaCha20-Poly1305 under the sub key (public square) or the room key (private room), RLN signal, vote nullifier.
Box side (`Box.validate` then `Box.apply`): signature, proof, RLN share and nullifier check (slashing on a reused
nullifier), one vote per identity per post, decrypt, then index / tally / delete / ban.

The pairing check itself is not run: a message carries `proof_ok` and the verify unit is charged its CP14 time.
The nullifiers and RLN shares are real arithmetic in the BN254 scalar field, so double votes and over-rate senders
are caught by the same logic a real box would run. Expensive pure functions (signature check, decryption,
embedding) are computed once per message and shared by every box (same input, same output).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

R = 21888242871839275222246405745257275088548364400416034343698204186575808495617  # BN254 scalar field


def h(*parts) -> int:
    b = hashlib.blake2b(digest_size=32)
    for x in parts:
        b.update(x if isinstance(x, bytes) else str(x).encode())
        b.update(b"|")
    return int.from_bytes(b.digest(), "big") % R


def sub_key(topic: int) -> bytes:
    return hashlib.blake2b(f"sub|{topic}".encode(), digest_size=32).digest()


def recover(x1: int, y1: int, x2: int, y2: int) -> int:
    """Two RLN shares on the same line (same epoch and message index) give the identity secret a0."""
    a1 = (y1 - y2) * pow(x1 - x2, -1, R) % R
    return (y1 - a1 * x1) % R


class Identity:
    def __init__(self, rng: np.random.Generator):
        self.a0 = int.from_bytes(rng.bytes(32), "big") % R
        self.commit = h("id", self.a0)

    def rln(self, epoch: int, k: int, x: int) -> tuple[int, int]:
        """RLN v2: a1 = H(a0, epoch, k), share y = a0 + a1 x, nullifier = H(a1). k < limit is enforced in-circuit."""
        a1 = h("a1", self.a0, epoch, k)
        return (self.a0 + a1 * x) % R, h("nul", a1)

    def vote_nullifier(self, target: int) -> int:
        return h("vote", self.a0, target)


@dataclass
class Msg:
    mid: int
    kind: str                 # post | comment | vote | delete | room | slash
    topic: int
    t0: float
    size: float
    epub: bytes
    sig: bytes
    nonce: bytes
    ct: bytes
    epoch: int
    x: int
    y: int
    nul: int
    vnul: int | None = None
    proof_ok: bool = True
    image: int | None = None  # synthetic image seed (image post)
    tokens: int = 0
    pull_bytes: float = 0.0   # image body fetched after the envelope (I2)
    truth: dict = field(default_factory=dict)   # world-only ground truth, never read by boxes

    def header(self) -> bytes:
        return f"{self.kind}|{self.topic}|{self.epoch}|{self.x}|{self.y}|{self.nul}|{self.vnul}".encode()


class Shared:
    """Results of pure functions, computed the first time any box asks (real code), reused after."""

    def __init__(self):
        self.sig: dict[int, bool] = {}
        self.plain: dict[tuple, dict | None] = {}
        self.emb: dict[int, np.ndarray] = {}      # text embedding (posts, comments, image captions)
        self.img: dict[int, np.ndarray] = {}      # image embedding

    def sig_ok(self, m: Msg) -> bool:
        if m.mid not in self.sig:
            try:
                Ed25519PublicKey.from_public_bytes(m.epub).verify(m.sig, m.header() + m.nonce + m.ct)
                self.sig[m.mid] = True
            except InvalidSignature:
                self.sig[m.mid] = False
        return self.sig[m.mid]

    def open(self, m: Msg, key: bytes) -> dict | None:
        k = (m.mid, key)
        if k not in self.plain:
            try:
                self.plain[k] = json.loads(ChaCha20Poly1305(key).decrypt(m.nonce, m.ct, m.header()))
            except InvalidTag:
                self.plain[k] = None
        return self.plain[k]


class World:
    """Authors: identities, message construction, the membership group (which commitments may prove)."""

    def __init__(self, n_users: int, epoch_s: float = 600.0, limit: int = 20, seed: int = 17):
        self.rng = np.random.default_rng(seed)
        self.ids = [Identity(self.rng) for _ in range(n_users)]
        self.members = {i.commit for i in self.ids}
        self.epoch_s, self.limit = epoch_s, limit
        self.counter: dict[tuple, int] = {}       # (user, epoch) -> messages sent
        self.over_limit = 0                        # honest messages that found the epoch's budget used up
        self.keys: dict[int, Ed25519PrivateKey] = {}
        self.msgs: list[Msg] = []

    def make(self, user: int, kind: str, topic: int, t: float, body: dict, size: float, key: bytes, *,
             target: int | None = None, reuse_k: int | None = None, proof_ok: bool = True,
             image: int | None = None, tokens: int = 0, pull_bytes: float = 0.0, truth: dict | None = None) -> Msg:
        ident = self.ids[user]
        epoch = int(t // self.epoch_s)
        if reuse_k is None:
            k = self.counter.get((user, epoch), 0)
            self.counter[(user, epoch)] = k + 1
            if k >= self.limit:
                self.over_limit += 1
                k = self.limit - 1   # an honest client would wait; counted, the message still goes out
        else:
            k = reuse_k
        if kind == "delete":      # signed with the target's own one-time key (only its author holds it)
            sk = self.keys[target]
        else:
            sk = Ed25519PrivateKey.generate()
        nonce = os.urandom(12)
        mid = len(self.msgs)
        vnul = ident.vote_nullifier(target) if kind == "vote" else None
        plain = json.dumps(body).encode()
        x = h("x", nonce, plain)                   # the signal the proof binds
        y, nul = ident.rln(epoch, k, x)
        m = Msg(mid, kind, topic, t, size, sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw), b"", nonce,
                b"", epoch, x, y, nul, vnul, proof_ok and ident.commit in self.members, image, tokens, pull_bytes,
                {"user": user, "commit": ident.commit, "target": target, **(truth or {})})
        m.ct = ChaCha20Poly1305(key).encrypt(nonce, plain, m.header())
        m.sig = sk.sign(m.header() + nonce + m.ct)
        self.keys[mid] = sk
        self.msgs.append(m)
        return m

    def make_slash(self, t: float, a0: int, topic: int) -> Msg:
        """A slashing notice: anyone can check H(a0) is in the group, then remove it."""
        sk = Ed25519PrivateKey.generate()
        nonce = os.urandom(12)
        m = Msg(len(self.msgs), "slash", topic, t, 400.0, sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
                b"", nonce, b"", int(t // self.epoch_s), 0, 0, 0, None, True, truth={"a0": a0})
        m.ct = json.dumps({"a0": str(a0)}).encode()          # public on purpose
        m.sig = sk.sign(m.header() + nonce + m.ct)
        self.members.discard(h("id", a0))
        self.msgs.append(m)
        return m


def hot(score: int, t_created: float) -> float:
    """Reddit's open-source 'hot' rank: log10 of |votes| plus age in units of 45,000 s."""
    s = 1 if score > 0 else -1 if score < 0 else 0
    return s * math.log10(max(abs(score), 1)) + t_created / 45000.0


class Box:
    """Full state of one box. validate() is what gossip runs before forwarding; apply() the rest of ingest."""

    def __init__(self, bid: int, topics, room_keys: dict, shared: Shared, emb_dim: dict):
        self.bid, self.topics = bid, set(int(t) for t in topics)
        self.room_keys = dict(room_keys)          # topic -> key
        self.S = shared
        self.rln: dict[tuple, tuple] = {}
        self.vnul: set[int] = set()
        self.banned: set[int] = set()
        self.items: dict[int, dict] = {}          # mid -> {kind, topic, t, epub, title, score, parent}
        self.deleted: set[int] = set()
        self.idx_ids = {k: [] for k in emb_dim}   # "text" / "image" -> list of mids (deleted ones skipped)
        self.idx_vec = {k: [] for k in emb_dim}
        self.log: list = []                       # (t, mid, verdict)

    # ---- before forwarding
    def validate(self, m: Msg) -> tuple[str, int | None]:
        if not self.S.sig_ok(m):
            return "bad_sig", None
        if m.kind == "slash":
            a0 = int(json.loads(m.ct)["a0"])
            return "ok", a0
        if not m.proof_ok:
            return "bad_proof", None
        key = (m.epoch, m.nul)
        if key in self.rln:
            x, y = self.rln[key]
            if x == m.x:
                return "dup", None
            return "slash", recover(x, y, m.x, m.y)
        self.rln[key] = (m.x, m.y)
        if m.kind == "vote":
            if m.vnul in self.vnul:
                return "double_vote", None
            self.vnul.add(m.vnul)
        return "ok", None

    # ---- the rest of ingest
    def key_for(self, topic: int) -> bytes | None:
        if topic in self.room_keys:
            return self.room_keys[topic]
        return sub_key(topic) if topic in self.topics else None

    def apply(self, m: Msg, t: float, verdict: tuple) -> str:
        v, a0 = verdict
        if v == "slash" or m.kind == "slash":
            self.banned.add(h("id", a0))
            self.log.append((t, m.mid, "slash"))
            return "slash"
        if v != "ok":
            self.log.append((t, m.mid, v))
            return v
        key = self.key_for(m.topic)
        body = self.S.open(m, key) if key else None
        if body is None:
            self.log.append((t, m.mid, "no_key"))
            return "no_key"
        if m.kind in ("post", "comment", "room"):
            self.items[m.mid] = {"kind": m.kind, "topic": m.topic, "t": m.t0, "epub": m.epub,
                                 "title": body.get("title"), "score": 0, "parent": body.get("parent")}
            if m.mid in self.S.emb:          # image posts: the caption here, the picture once it is pulled
                self.index("text", m.mid)
        elif m.kind == "vote":
            tgt = body["target"]
            if tgt in self.items:
                self.items[tgt]["score"] += body["dir"]
        elif m.kind == "delete":
            tgt = body["target"]
            if tgt in self.items and self.items[tgt]["epub"] == m.epub:   # only the author's one-time key
                self.deleted.add(tgt)
            else:
                self.log.append((t, m.mid, "bad_delete"))
                return "bad_delete"
        self.log.append((t, m.mid, "ok"))
        return "ok"

    def index(self, kind: str, mid: int) -> None:
        self.idx_ids[kind].append(mid)
        self.idx_vec[kind].append((self.S.emb if kind == "text" else self.S.img)[mid])

    def n_index(self) -> int:
        return sum(len(v) for v in self.idx_ids.values())

    # ---- queries
    def search(self, q: np.ndarray, kind: str = "text", k: int = 10) -> list[int]:
        ids = self.idx_ids[kind]
        if not ids:
            return []
        V = np.asarray(self.idx_vec[kind])
        s = V @ q
        out = []
        for i in np.argsort(-s):
            if ids[i] not in self.deleted:
                out.append(ids[i])
                if len(out) == k:
                    break
        return out

    def feed(self, n: int = 25) -> list[int]:
        posts = [(hot(v["score"], v["t"]), mid) for mid, v in self.items.items()
                 if v["kind"] == "post" and mid not in self.deleted]
        return [mid for _, mid in sorted(posts, reverse=True)[:n]]
