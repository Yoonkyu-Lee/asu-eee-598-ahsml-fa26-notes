"""One home node box as a real process (CP21 E1, CP24): HTTP, the box software (sim.box, or sim.proto24 when the
config says proto "cp24"), real Ed25519, ChaCha20-Poly1305, RLN shares and nullifiers, MiniLM INT8 and CLIP
embeddings, search, feed, votes, deletes, room.
The proof's pairing check is not run (the message carries proof_ok, as in the simulator); its time on the chip is
shown from the CP19/CP20 chip model.

CP24 box (proto "cp24"): O-b check order, I5 + F1 limits (the owner's client waits when its indices are used up),
an encoder queue by priority (query, old text, new text, old pictures, new pictures; text up to 128 tokens), its own
identity drawn in this process, photos uploaded from the screen, and several computers (cfg "addr"): the box listens
on --bind, answers owner calls only from this computer and peer calls only from the listed addresses, and with
cfg "sync" asks its peers for what it missed (start, then every 20 s).

CP25 (--state DIR, set by lan.py): the box keeps its identity, the indices it used, its one-time keys, the messages it
took and the pictures on disk (store.py), and takes them back in arrival order when it starts again. A deleted post is
forgotten on disk and in /sync too, and a copy that comes back later from a stale box is dropped. events.jsonl logs
times for remote_report.py (no plaintext).

CP26 exchange (cfg "exchange"): "x2" = event-driven, as chosen on the model (sim.exchange X2): push to the mesh; when
the box comes on (start, or a clock jump after the laptop slept) it offers the ids it took in its last session to one
peer (that peer asks only for what it lacks) and asks that peer for everything published since it went off minus
60 s (GET /since). The answer carries the peer's coverage start; a gap is asked of another peer. A peer that has not
caught up yet only claims its own session. Nothing runs on a timer except a local clock check. "poll" = the CP24
exchange (every 20 s every listed box is asked for its whole list), kept to compare.

CP28 (cfg "push" = "lazy", cfg "text" = "fp32"): ids first (CP27 P2): a box tells its mesh peers a message's id
(POST /ihave1); a peer that lacks it answers "want" to the first announcer only, and if the envelope has not come
1 s later it fetches it from the next announcer (GET /env). Text is encoded by FP32 MiniLM (the CP3 contract; FP16
grade, CP27 B2).

  .venv/Scripts/python.exe src/demo/box_server.py --cfg results/cp21/cfg.json --id 0
Peers:   POST /gossip {msg, frm}    GET /pull?mid=&frm=    GET /hello    GET /sync    POST /inject {msg}
         GET /since?lo=&hi=&frm=   POST /ihave {ids, frm}   (CP26 x2)
Owner:   GET / (screen)  /api/stats  /api/feed  /api/search?q=  /api/search_image?q=  /api/img?mid=  /api/debug
         POST /api/post {topic, title, body, photo | rgb}  /api/vote {mid, dir}  /api/delete {mid}  /api/room {body}
              /api/misbehave {}  (CP24: two posts on one message index -> the identity is revealed)
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import queue
import sys
import threading
import time
import traceback
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import A, B, IMG_BYTES, MEMB, ROOM, TOPIC_NAMES, Coco, msg_from_json, msg_to_json  # noqa: E402
from sim.box import Box, Msg, Shared, World, h, sub_key  # noqa: E402
from sim import proto24 as p24  # noqa: E402
from store import Store  # noqa: E402

UI = Path(__file__).resolve().parent / "ui.html"
CHIP = {"proof_s": 89.6e-6, "image_s": 0.552}     # CP19 verify unit (single proof), CP20 FP16 image tower (추정)
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
REQ_TIMEOUT_S = 1.0      # CP28 lazy push: ask the next announcer after this (sim.exchange Scene.req_timeout_s)
W_S = 60.0               # CP26: catch-up margin and the model's window (sim.exchange.Scene.w_s)
TICK_S, JUMP_S = 5.0, 30.0   # local clock check: a jump over JUMP_S means the computer slept


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class BoxNode:
    def __init__(self, cfg: dict, bid: int, load_models: bool = True, state: str | None = None):
        self.cfg, self.bid, self.n = cfg, bid, cfg["n"]
        self.p24 = cfg.get("proto") == "cp24"
        self.holders = {int(t): v for t, v in cfg["holders"].items()}
        self.mesh = {int(t): {int(b): p for b, p in m.items()} for t, m in cfg["mesh"].items()}
        self.addr = cfg.get("addr") or [f"127.0.0.1:{p}" for p in cfg["ports"]]
        self.peer_ips = {a.rsplit(":", 1)[0] for a in self.addr}
        rk = cfg.get("room_key")
        self.room_key = bytes.fromhex(rk) if rk and bid in self.holders[ROOM] else None
        self.S = Shared()
        topics = [t for t, hs in self.holders.items() if bid in hs]
        self.box = (p24.Box24 if self.p24 else Box)(bid, topics, {ROOM: self.room_key} if self.room_key else {}, self.S,
                                                    {"text": 0, "image": 0})
        if self.p24:     # this box's identity (user bid) is drawn here and never leaves this process
            self.tier = cfg.get("tiers", ["new"] * self.n)[bid]
            self.W = p24.World24(self.n, seed=int.from_bytes(os.urandom(8), "big"), tiers=cfg.get("tiers"))
        else:            # CP21: every identity from the shared seed (one computer)
            self.tier = "old"
            self.W = World(self.n, seed=cfg["seed"])
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
        self.st = Store(Path(state)) if state and self.p24 else None
        self.exchange = cfg.get("exchange", "poll" if cfg.get("sync") else None)
        meta = self.st.meta() if self.st else {}
        self.cov = meta.get("cov")                 # coverage start (None: holds nothing yet)
        self.last_off = meta.get("alive")          # last time this box was seen alive (end of the last session)
        self.prev_session = meta.get("session")    # start of the last session
        self.session = time.time()
        self.ready = self.exchange != "x2"         # caught up in this session
        self.cov_before, self.gap_lo, self.offered, self.stalled = self.cov, 0.0, False, False
        self.t_come_on = self.session
        self.catch_lock = threading.Lock()
        self.up: set[int] = set()                  # peers known to be on: they said hello, answered or sent (rule 2;
                                                   # CP28: the mesh is the boxes that are on, as in sim.exchange)
        self.announcers: dict[int, list] = {}      # CP28 lazy push: mid -> boxes that announced it, in order
        self.replaying = False
        self.replayed: set[int] = set()
        self.tomb: dict[int, set] = {}           # deleted post -> the author's one-time public keys (hex)
        self.gone: set[int] = set()              # posts held here and then forgotten (CP28)
        if self.st:
            self._restore_identity()
            self.commit = self.W.ids[bid].commit
        self.pool = ThreadPoolExecutor(16)
        self.coco = Coco() if Coco.available() else None
        if load_models:
            from bench_cp18 import ClipText
            from sim import ml
            import sim_cp19
            if cfg.get("text") == "fp32":                      # CP28: contract precision (CP3, CP27)
                self.text_enc = ml.Encoder("fp32", threads=1)
            else:
                try:
                    self.text_enc = ml.Encoder("int8", threads=1)
                except Exception:                              # e.g. an ONNX Runtime build without the INT8 kernels
                    log("MiniLM INT8 failed, using FP32:", traceback.format_exc(limit=1).strip().splitlines()[-1])
                    self.text_enc = ml.Encoder("fp32", threads=1)
            self.img_enc = ml.ImageEncoder("fp32", threads=1)    # FP32 = the FP16 tower's quality (CP20)
            self.clip_text = ClipText(threads=1)
            self.chip_table = sim_cp19.EncTable(8, 1024)
        self.encq: queue.PriorityQueue = queue.PriorityQueue()
        self.seq = itertools.count()
        if self.p24:
            threading.Thread(target=self._encoder, daemon=True).start()
        self.restored = self._replay() if self.st else 0
        self.ready = True

    # ---- the box's state on disk (CP25)
    def _restore_identity(self) -> None:
        d = self.st.identity()
        if d is None:                                          # first start: keep the identity drawn above
            self._save_identity()
            return
        me = self.W.ids[self.bid]
        self.W.members.discard(me.commit)
        me.a0 = int(d["a0"])
        me.commit = h("id", me.a0)
        self.W.members.add(me.commit)
        if d["tier"] != self.tier:
            log(f"this identity is in the {d['tier']} group (kept; the address list says {self.tier})")
        self.tier = self.W.tier[self.bid] = d["tier"]
        for sp, e, k in d["used"]:
            self.W.used[(self.bid, sp, int(e))] = int(k)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        for mid, raw in self.st.keys().items():
            self.W.keys[int(mid)] = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw))
            self.own.add(int(mid))
        self.tomb = self.st.tomb()

    def _save_identity(self) -> None:
        used = {(sp, e): k for (u, sp, e), k in self.W.used.items() if u == self.bid}
        self.st.save_identity(self.W.ids[self.bid].a0, self.tier, used)

    def _save_keys(self) -> None:
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
        self.st.save_keys({str(mid): sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
                           for mid, sk in self.W.keys.items()})

    def _replay(self) -> int:
        """Take the stored messages again, in arrival order: same checks, same index; nothing is sent."""
        t0 = time.perf_counter()
        rows = self.st.messages()
        self.replaying = True
        try:
            for d in rows:
                m = msg_from_json(d)
                pic = self.st.picture(m.mid)
                if pic is not None:
                    self.full[m.mid] = pic
                self.replayed.add(m.mid)
                self.deliver(m, int(d.get("_frm", -1)))
        finally:
            self.replaying = False
        ms = (time.perf_counter() - t0) * 1e3
        self.st.ev("start", restored=len(rows), ms=ms, identity=hex(self.commit)[2:10], tier=self.tier)
        if rows:
            log(f"restored {len(rows)} messages from {self.st.dir} in {ms / 1e3:.1f} s (encoding continues in the background)")
        return len(rows)

    def forgotten(self, mid: int) -> bool:
        """Held here, then deleted by its author (or dropped on arrival by a matching tombstone). A tombstone alone is
        not enough: a forged delete leaves one too (another key), and the real post must still come in. An id alone
        has no key to compare, so only posts known gone are refused (CP28: ids and picture pulls checked the mid)."""
        return mid in self.gone

    def _have_picture(self, mid: int, data: bytes) -> None:
        self.full[mid] = data
        if self.st:
            self.st.save_picture(mid, data)

    def _after_delete(self, m: Msg) -> None:
        """Forgetting (CP25): a delete by the author removes the post's envelope and picture here, on disk and from
        /sync. A delete that comes before its post leaves a tombstone, so the post is dropped when it comes."""
        body = self.S.open(m, self.key_for(m.topic)) if self.key_for(m.topic) else None
        if body is None:
            return
        tgt = int(body["target"])
        it = self.box.items.get(tgt)
        if it is not None and it["epub"] != m.epub:            # someone else's delete: Box.apply refused it
            return
        self.tomb.setdefault(tgt, set()).add(m.epub.hex())
        self.st.save_tomb(self.tomb)
        if it is None:
            return
        self.gone.add(tgt)
        self.msgs.pop(tgt, None)
        self.full.pop(tgt, None)
        self.S.plain = {k: v for k, v in self.S.plain.items() if k[0] != tgt}
        self.S.emb.pop(tgt, None)
        self.S.img.pop(tgt, None)
        it["title"] = None
        for kind, ids in self.box.idx_ids.items():
            keep = [i for i, x in enumerate(ids) if x != tgt]
            self.box.idx_ids[kind] = [ids[i] for i in keep]
            self.box.idx_vec[kind] = [self.box.idx_vec[kind][i] for i in keep]
        if not self.replaying:
            self.st.forget(tgt)
            self.st.ev("forget", mid=str(tgt))

    # ---- helpers
    def _inc(self, k: int = 1) -> None:
        with self.lock:
            self.pending += k

    def _url(self, peer: int, path: str) -> str:
        return f"http://{self.addr[peer]}{path}"

    def _post(self, peer: int, path: str, payload: dict) -> None:
        req = urllib.request.Request(self._url(peer, path), data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()

    def send(self, peer: int, m: Msg, direct: bool = False) -> None:
        """Push m to peer: the envelope (CP17), or with cfg "push" = "lazy" its id first (CP28), unless the peer
        already asked for it (direct)."""
        self.count["send"] += 1
        self._inc()
        payload = {"msg": msg_to_json(m), "frm": self.bid}
        lazy = self.cfg.get("push") == "lazy" and not direct

        # ids first: no retries. A peer that missed an id hears it from another announcer (REQ_TIMEOUT_S) or gets it
        # when it comes on (catch-up); retrying an off box only lands late after it restarts (CP28 D, run 3)
        tries = 1 if lazy else 3

        def go():
            try:
                for i in range(tries):
                    try:
                        if lazy:
                            self.count["ihave_sent"] += 1
                            req = urllib.request.Request(self._url(peer, "/ihave1"),
                                                         data=json.dumps({"mid": str(m.mid), "frm": self.bid}).encode(),
                                                         headers={"Content-Type": "application/json"})
                            want = json.loads(urllib.request.urlopen(req, timeout=60).read())["want"]
                            if want:
                                self._post(peer, "/gossip", payload)
                        else:
                            self._post(peer, "/gossip", payload)
                        self.up.add(peer)
                        return
                    except Exception as e:
                        if i == tries - 1:
                            log(f"gossip to box {peer} ({self.addr[peer]}) failed:", e)
                            self.up.discard(peer)
                            return
                        time.sleep(0.5 * (i + 1))
            finally:
                self._inc(-1)
        self.pool.submit(go)

    def ihave1(self, mid: int, frm: int) -> bool:
        """An id was announced. Want it from the first announcer only; check again after REQ_TIMEOUT_S."""
        with self.lock:
            self.count["ihave_rx"] += 1
            self.up.add(frm)
            if mid in self.got or self.forgotten(mid):
                return False
            ann = self.announcers.setdefault(mid, [])
            ann.append(frm)
            if len(ann) > 1:
                return False
        threading.Timer(REQ_TIMEOUT_S, self._lazy_check, (mid, 1)).start()
        return True

    def _lazy_check(self, mid: int, i: int) -> None:
        with self.lock:
            if mid in self.got:
                return
            ann = self.announcers.get(mid, [])
            if i >= len(ann):
                return
            frm = ann[i]
        try:
            d = json.loads(urllib.request.urlopen(self._url(frm, f"/env?mid={mid}&frm={self.bid}"), timeout=60).read())
            if d.get("msg"):
                self.count["env_rx"] += 1
                self.deliver(msg_from_json(d["msg"]), frm)
                return
        except Exception as e:
            log(f"envelope {mid} from box {frm} failed:", type(e).__name__)
        threading.Timer(REQ_TIMEOUT_S, self._lazy_check, (mid, i + 1)).start()

    def targets(self, topic: int) -> list[int]:
        """The topic's mesh peers. With the x2 exchange (rule 2, graft): only peers known to be on, filled up to D
        from the topic's other holders that are on, most recently heard from first."""
        mesh = [int(p) for p in self.mesh.get(topic, {}).get(self.bid, ())]
        if self.exchange != "x2":
            return mesh
        out = [p for p in mesh if p in self.up]
        D = self.cfg.get("D", 2)
        if len(out) < D:
            more = [int(p) for p in self.holders.get(topic, ()) if int(p) != self.bid and int(p) not in out
                    and int(p) in self.up]
            more.sort(key=lambda p: -self.peer_seen.get(p, 0.0))
            out += more[:D - len(out)]
        return out

    def key_for(self, topic: int) -> bytes | None:
        return self.box.key_for(topic)

    def text_of(self, m: Msg) -> str | None:
        key = self.key_for(m.topic)
        body = self.S.open(m, key) if key else None
        if body is None:
            return None
        return ((body.get("title") or "") + "\n\n" + (body.get("body") or "")).strip()

    # ---- the ingest path (mirrors sim.stack.DayNet.deliver / sim.proto24.DayNet24.deliver, same operation names)
    def deliver(self, m: Msg, frm: int, via: str = "gossip") -> None:
        with self.lock:
            if m.mid in self.got:
                self.count["dup"] += 1
                return
            self.got.add(m.mid)
            if m.epub.hex() in self.tomb.get(m.mid, ()):           # deleted by its author: forgotten (CP25)
                self.count["forgotten"] += 1
                self.gone.add(m.mid)
                return
            self.msgs[m.mid] = m
            if self.st and not self.replaying:
                self.st.add_message(msg_to_json(m), frm)
            if frm >= 0 and not self.replaying:
                self.peer_seen[frm] = time.time()
                self.up.add(frm)
            t = time.time()
            if self.p24:
                v = self.box.validate(m)
                for op in p24.stages(m, v[0]):
                    self.count[op] += 1
                    if op == "proof":
                        self.verified.append(m.mid)
                        self.chip_s += CHIP["proof_s"]
            else:
                self.count["ed25519"] += 1
                if m.kind != "slash":
                    self.count["proof"] += 1
                    self.count["rln"] += 1
                    self.verified.append(m.mid)
                    self.chip_s += CHIP["proof_s"]
                v = self.box.validate(m)
            if self.st and not self.replaying:
                self.st.ev("recv", mid=str(m.mid), frm=frm, kind=m.kind, via=via, verdict=v[0])
            if v[0] != "ok" and frm >= 0:                       # a relay drops it; the author's box sends anyway
                if v[0] == "slash" and not self.replaying:
                    self.publish_slash(v[1])
                self.box.apply(m, t, v)
                return
            for peer in self.targets(m.topic):
                if peer != frm and not self.replaying and via != "catch":    # caught-up messages are not news
                    self.send(peer, m)
            if v[0] != "ok":
                self.box.apply(m, t, v)
                return
            if m.kind == "slash":
                self.box.apply(m, t, v)
                self._notice(m)
                return
            self.count["decrypt"] += 1
            if self.p24:
                return self._ingest24(m, frm, t, v)
            if m.kind in ("post", "comment", "room"):
                txt = self.text_of(m)
                if txt is not None:
                    feed = self.text_enc.tokenize([txt])
                    self.S.emb[m.mid] = self.text_enc.encode_tokens(feed)[0]
                    self.chip_s += self.chip_table.text1(int(feed["attention_mask"].sum()))[0]
                self.count["text"] += 1
                self.count["insert"] += 1
            self.box.apply(m, t, v)
            if m.kind == "vote":
                self.count["tally"] += 1
            elif m.kind == "delete":
                self.count["delete"] += 1
            if m.image is not None and m.kind == "post":
                self.full_ev.setdefault(m.mid, threading.Event())
                if frm < 0:
                    self.full[m.mid] = self.coco.pixels(m.image).tobytes()
                    self.full_ev[m.mid].set()
                    self._index_image(m.mid)
                else:
                    self.parent[m.mid] = frm
                    self._request(m.mid)

    def _ingest24(self, m: Msg, frm: int, t: float, v: tuple) -> None:
        """After decryption (CP24): count check, then the encoder queue (text first, pictures last; old group first)."""
        if m.kind in p24.TEXT_KINDS:
            txt = self.text_of(m)
            if txt is None:
                self.box.apply(m, t, v)
                return
            feed = self.text_enc.tokenize([txt])
            ntok = int(feed["attention_mask"].sum())
            if self.box.apply(m, t, v, tokens=ntok) != "ok":     # underweight: not indexed
                return
            feed = {k: a[:, :p24.F1_TOKENS] for k, a in feed.items()}
            self._enqueue(1 if m.tier == "old" else 3, ("text", m.mid, feed))
            if m.image is not None and m.kind == "post":
                self.full_ev.setdefault(m.mid, threading.Event())
                if frm < 0 and m.mid not in self.full and self.coco is not None:
                    self._have_picture(m.mid, self.coco.pixels(m.image).tobytes())
                if m.mid in self.full:                          # mine, or stored (CP25)
                    self.full_ev[m.mid].set()
                    self._enqueue(4 if m.tier == "old" else 5, ("image", m.mid))
                elif frm >= 0:
                    self.parent[m.mid] = frm
                    self._request(m.mid)
            return
        self.box.apply(m, t, v)
        if m.kind == "vote":
            self.count["tally"] += 1
        elif m.kind == "delete":
            self.count["delete"] += 1
            if self.st:
                self._after_delete(m)

    def _enqueue(self, cls: int, job: tuple) -> None:
        self._inc()
        self.encq.put((cls, next(self.seq), job))

    def _encoder(self) -> None:
        """One encoder, lowest class first (CP22 T2/T3). The owner's queries do not wait here (class 0 = run now)."""
        while True:
            cls, _, job = self.encq.get()
            try:
                if job[0] == "text":
                    mid, feed = job[1], job[2]
                    t0 = time.perf_counter()
                    e = self.text_enc.encode_tokens(feed)[0]
                    ms = (time.perf_counter() - t0) * 1e3
                    nt = int(feed["attention_mask"].sum())
                    with self.lock:
                        self.S.emb[mid] = e
                        self.enc_tokens_max = max(self.enc_tokens_max, nt)
                        self.chip_s += self.chip_table.text1(nt)[0]
                        self.count["text"] += 1
                        self.count["insert"] += 1
                        if mid in self.box.items and mid not in self.box.deleted:
                            self.box.index("text", mid)
                            if self.st:
                                self.st.ev("index", mid=str(mid), what="text", ms=ms, tokens=nt,
                                           replay=mid in self.replayed)
                else:
                    self._index_image(job[1])
            except Exception:
                log("encoder job failed:", traceback.format_exc())
            finally:
                self._inc(-1)

    def _notice(self, m: Msg) -> None:
        a0 = int(json.loads(m.ct)["a0"])
        c = h("id", a0)
        mine = c == self.commit
        self.notices.append({"t": time.time(), "identity": hex(c)[2:10], "mine": mine})
        self.W.members.discard(c)
        if mine:
            self.slashed = True

    def publish_slash(self, a0: int) -> None:
        """A reused RLN share gives the identity secret a0: publish it once (content-addressed, so the notices of
        boxes that found the same reuse are one message)."""
        mid = (1 << 40) + h("slash-id", a0) % (1 << 30)
        if a0 in self.slash_sent or mid in self.got:           # already published, or already got someone's notice
            return
        self.slash_sent.add(a0)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        sk = Ed25519PrivateKey.generate()
        nonce = os.urandom(12)
        s = Msg(mid, "slash", MEMB, time.time(), 400.0, sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
                b"", nonce, b"", 0, 0, 0, 0, None, True)
        s.ct = json.dumps({"a0": str(a0)}).encode()
        s.sig = sk.sign(s.header() + nonce + s.ct)
        self.count["slash_published"] += 1
        self.deliver(s, -1)

    # ---- image pulls (I2)
    def _request(self, mid: int) -> None:
        if mid in self.asked:
            return
        self.asked.add(mid)
        self._inc()

        def go():
            try:
                order = [self.parent[mid]] + [p for p in range(self.n) if p not in (self.parent[mid], self.bid)]
                for peer in order if self.cfg.get("sync") else order[:1]:
                    try:
                        data = urllib.request.urlopen(self._url(peer, f"/pull?mid={mid}&frm={self.bid}"),
                                                      timeout=300).read()
                    except Exception as e:
                        log(f"pull {mid} from box {peer} failed:", e)
                        continue
                    with self.lock:
                        if self.forgotten(mid):
                            return
                        self._have_picture(mid, data)
                        self.full_ev[mid].set()
                        if self.p24:
                            self._enqueue(4 if self.msgs[mid].tier == "old" else 5, ("image", mid))
                        else:
                            self._index_image(mid)
                    return
            finally:
                self._inc(-1)
        self.pool.submit(go)

    def serve_pull(self, mid: int) -> bytes | None:
        with self.lock:
            if self.forgotten(mid):
                return None
            ev = self.full_ev.setdefault(mid, threading.Event())
            if mid not in self.full and mid in self.parent:
                self._request(mid)
        ev.wait(300)
        with self.lock:
            self.count["pull"] += 1
            return self.full.get(mid)

    def _index_image(self, mid: int) -> None:
        data = self.full.get(mid)
        if data is None:                                        # forgotten while it waited (CP25)
            return
        px = np.frombuffer(data, np.uint8).reshape(1, 224, 224, 3)
        t0 = time.perf_counter()
        e = self.img_enc.encode(px)[0]
        ms = (time.perf_counter() - t0) * 1e3
        with self.lock:
            self.S.img[mid] = e
            self.chip_s += CHIP["image_s"]
            self.count["image"] += 1
            self.count["insert"] += 1
            if mid in self.box.items and mid not in self.box.deleted:
                self.box.index("image", mid)
                if self.st:
                    self.st.ev("index", mid=str(mid), what="image", ms=ms, replay=mid in self.replayed)

    # ---- catching up (several computers): every message a peer holds, in its arrival order
    def sync_from(self, peer: int) -> int:
        raw = urllib.request.urlopen(self._url(peer, f"/sync?frm={self.bid}"), timeout=60).read()
        data = json.loads(raw)
        self.count["sync_req"] += 1
        self.count["sync_env"] += len(data)
        self.count["sync_bytes"] += len(raw)
        new = 0
        for d in data:
            m = msg_from_json(d)
            if m.mid in self.got:
                continue
            new += 1
            self.deliver(m, peer, via="sync")
        self.peer_seen[peer] = time.time()
        return new

    def sync_loop(self) -> None:
        while True:
            for peer in range(self.n):
                if peer == self.bid:
                    continue
                try:
                    if self.st:                                  # round trip and clock offset (NTP style, CP25)
                        t0 = time.time()
                        r = json.loads(urllib.request.urlopen(self._url(peer, "/hello"), timeout=10).read())
                        t1 = time.time()
                        self.st.ev("peer", peer=peer, rtt_ms=(t1 - t0) * 1e3,
                                   offset_ms=(r.get("t", (t0 + t1) / 2) - (t0 + t1) / 2) * 1e3)
                    k = self.sync_from(peer)
                    if self.st:
                        self.st.ev("sync", peer=peer, new=k)
                    if k:
                        log(f"sync: {k} new messages from box {peer}")
                except Exception as e:
                    log(f"sync from box {peer} ({self.addr[peer]}) failed:", type(e).__name__, e)
            time.sleep(20)

    def sync_list(self) -> list[dict]:
        with self.lock:
            return [msg_to_json(m) for m in self.msgs.values()]

    # ---- CP26 x2: event-driven exchange (sim.exchange X2)
    def claim(self) -> float:
        """The coverage start this box may claim: its own once it has caught up in this session, else only this
        session (it has everything pushed since it came on). A box that could not fill its gap stays at that."""
        return self.cov if self.ready and self.cov is not None else self.session

    def hello_info(self) -> dict:
        return {"box": self.bid, "proto": self.cfg.get("proto", "cp21"), "n": self.n, "t": time.time(),
                "cov": self.claim(), "exchange": self.exchange}

    def since(self, lo: float, hi: float) -> dict:
        with self.lock:
            rows = [msg_to_json(m) for m in self.msgs.values() if lo <= m.t0 < hi]
            return {"cov": self.claim(), "msgs": rows}

    def ihave(self, ids: list) -> list[str]:
        with self.lock:
            return [x for x in ids if int(x) not in self.got]

    def _hello(self, peer: int) -> dict | None:
        try:
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(self._url(peer, f"/hello?frm={self.bid}"), timeout=10).read())
            t1 = time.time()
        except Exception as e:
            log(f"box {peer} ({self.addr[peer]}) not reachable:", type(e).__name__)
            self.up.discard(peer)
            return None
        self.up.add(peer)
        self.peer_seen[peer] = time.time()
        self.count["hello"] += 1
        if self.st:
            self.st.ev("peer", peer=peer, rtt_ms=(t1 - t0) * 1e3, offset_ms=(r.get("t", (t0 + t1) / 2) - (t0 + t1) / 2) * 1e3)
        return r

    def _save_meta(self) -> None:
        if self.st:
            self.st.save_meta({"cov": self.cov if self.ready else self.cov_before, "session": self.session,
                               "alive": time.time()})

    def come_on(self, last_off: float | None, prev_session: float | None) -> None:
        """Rule 3: this box came on (start, or the computer woke). The gap to fill starts W_S before it went off."""
        with self.lock:
            self.cov_before = self.cov
            self.gap_lo = 0.0 if self.cov is None or last_off is None else max(last_off - W_S, 0.0)
            self.prev_session, self.last_off = prev_session, last_off
            self.offered, self.stalled, self.t_come_on = False, False, time.time()
        self.catch_up()

    def peer_appeared(self) -> None:
        """A box said hello: if this box had nobody to ask when it came on, it asks now."""
        if self.stalled:
            self.stalled = False
            threading.Thread(target=self.catch_up, daemon=True).start()

    def catch_up(self) -> None:
        """Offer my last session's ids to one peer, then get what was published in [gap_lo, now) from a peer whose
        claim reaches back that far; if its claim starts later, ask another peer for the part before it."""
        if not self.catch_lock.acquire(blocking=False):
            return
        try:
            lo = self.gap_lo
            mesh = sorted({int(p) for m in self.mesh.values() for p in m.get(self.bid, ())})
            order = mesh + [p for p in range(self.n) if p != self.bid and p not in mesh]
            with ThreadPoolExecutor(8) as ex:
                hello = {p: r for p, r in zip(order, ex.map(self._hello, order)) if r is not None}
                if self.st:          # measurement only (events.jsonl): 4 more round trips per peer, so the clock
                    list(ex.map(self._hello, [q for q in hello for _ in range(4)]))   # offset uses the shortest
            claim = lambda q: hello[q]["cov"]
            tried, hi, reach = set(), float("inf"), None
            while True:
                cands = [q for q in order if q in hello and q not in tried]
                if not cands:
                    break
                good = [q for q in cands if q in mesh and claim(q) <= lo] or [q for q in cands if claim(q) <= lo]
                p = good[0] if good else min(cands, key=claim)
                if not self.offered:
                    self.offered = True
                    self._offer(p, self.prev_session, self.last_off)
                tried.add(p)
                try:
                    raw = urllib.request.urlopen(self._url(p, f"/since?lo={lo}&hi={hi}&frm={self.bid}"), timeout=300).read()
                except Exception as e:
                    log(f"catch-up from box {p} failed:", type(e).__name__)
                    continue
                d = json.loads(raw)
                self.count["catch_req"] += 1
                self.count["catch_env"] += len(d["msgs"])
                self.count["catch_bytes"] += len(raw)
                for row in d["msgs"]:
                    self.deliver(msg_from_json(row), p, via="catch")
                reach = d["cov"] if reach is None else min(reach, d["cov"])
                if reach > lo and any(q not in tried and claim(q) < reach for q in hello):
                    hi = reach
                    continue
                break
            if reach is None:                                  # nobody answered: wait for a box to appear
                self.stalled = True
                log("nobody to catch up from; claiming this session only until a box appears")
                return
            with self.lock:
                if reach <= lo:
                    self.cov = lo if self.cov_before is None else self.cov_before
                else:                                          # [lo, reach) is nowhere on: claim what is held
                    self.cov = min(reach, self.session)
                self.ready = True
                missing = [mid for mid, m in self.msgs.items() if m.image is not None and m.kind == "post"
                           and mid not in self.full and mid in self.parent]
                for mid in missing:                            # picture bodies a failed pull left behind (rule 4)
                    self.asked.discard(mid)
                    self._request(mid)
            self._save_meta()
            ms = (time.time() - self.t_come_on) * 1e3
            if self.st:
                self.st.ev("catch_done", lo=lo, ms=ms, asked=sorted(tried), cov=self.cov)
            log(f"caught up {ms / 1e3:.1f} s after coming on, from {sorted(tried)}; coverage from {self.cov}")
        finally:
            self.catch_lock.release()

    def _offer(self, p: int, prev_session: float | None, last_off: float | None) -> None:
        if prev_session is None or last_off is None or not self.st:
            return
        # arrivals in the last session = from its start until this session began (nothing arrives while off; the
        # last-alive time is saved only every TICK_S, so it would cut the last seconds)
        ids = [r["mid"] for r in self.st.messages_meta() if prev_session <= r["_at"] < self.session]
        if not ids:
            return
        self.count["offer_ids"] += len(ids)
        try:
            req = urllib.request.Request(self._url(p, "/ihave"), data=json.dumps({"ids": ids, "frm": self.bid}).encode(),
                                         headers={"Content-Type": "application/json"})
            want = json.loads(urllib.request.urlopen(req, timeout=60).read())["want"]
        except Exception as e:
            log(f"offer to box {p} failed:", type(e).__name__)
            return
        with self.lock:
            rows = [self.msgs[int(x)] for x in want if int(x) in self.msgs]
        for m in rows:                                          # it asked for these: the envelopes (it forwards them on)
            self.send(p, m, direct=True)

    def alive_loop(self) -> None:
        """Local only: keep the last-alive time, and treat a clock jump (the computer slept) as coming on again."""
        last = time.time()
        while True:
            time.sleep(TICK_S)
            now = time.time()
            if now - last > JUMP_S:
                log(f"the clock jumped {now - last:.0f} s (asleep?): catching up")
                with self.lock:
                    prev, self.session, self.ready = self.session, now, False
                self.come_on(last, prev)
            last = now
            self._save_meta()

    # ---- the owner's side
    def query_text(self, q: str, k: int = 10) -> list[dict]:
        v = self.text_enc.encode([q])[0]
        with self.lock:
            self.count["query"] += 1
            self.chip_s += self.chip_table.text1(12)[0]
            return [self.item(mid) for mid in self.box.search(v, "text", k)]

    def query_image(self, q: str, k: int = 10) -> list[dict]:
        v = self.clip_text.encode([q])[0]
        with self.lock:
            self.count["query_clip"] += 1
            return [self.item(mid) for mid in self.box.search(v, "image", k)]

    def item(self, mid: int) -> dict:
        it = self.box.items.get(mid, {})
        m = self.msgs.get(mid)
        body = self.S.open(m, self.key_for(m.topic)) if m is not None and self.key_for(m.topic) else None
        return {"mid": str(mid), "kind": it.get("kind"), "topic": TOPIC_NAMES.get(it.get("topic"), it.get("topic")),
                "title": it.get("title"), "body": (body or {}).get("body", "")[:300], "score": it.get("score", 0),
                "own": mid in self.own, "photo": m is not None and m.image is not None and mid in self.full,
                "deleted": mid in self.box.deleted}

    def write(self, kind: str, topic: int, body: dict, *, target: int | None = None, photo: int | None = None,
              rgb: bytes | None = None, reuse: bool = False) -> str:
        key = self.room_key if topic == ROOM else sub_key(topic)
        image = photo if photo is not None else (-1 if rgb is not None else None)
        size = 400 + len(json.dumps(body)) + (IMG_BYTES if image is not None else 0)
        kw = {}
        if self.p24:
            text = ((body.get("title") or "") + "\n\n" + (body.get("body") or "")).strip()
            # attention mask, not len(ids): tokenizer.json pads to 128 until the first tokenize() (CP25 fix: a fresh
            # box's first post was counted as 128 tokens and spent two text indices)
            ntok = int(self.text_enc.tokenize([text])["attention_mask"].sum()) if kind in p24.TEXT_KINDS else 0
            kw = {"reuse": reuse, "tokens": ntok}
        with self.lock:
            m = self.W.make(self.bid, kind, topic, time.time(), body, size, key, target=target, image=image,
                            pull_bytes=IMG_BYTES if image is not None else 0.0, **kw)
            new = int.from_bytes(os.urandom(6), "big") + (1 << 48)        # this process's ids must not collide
            self.W.keys[new] = self.W.keys.pop(m.mid)
            m.mid = new
            self.own.add(new)
            if rgb is not None:
                self._have_picture(new, rgb)
            if self.st:                                     # before it leaves: a restart never reuses an index
                self._save_identity()
                self._save_keys()
                self.st.ev("write", mid=str(new), kind=kind, topic=topic, tokens=kw.get("tokens", 0),
                           photo=image is not None)
        self.deliver(m, -1)
        return str(new)

    def stats(self) -> dict:
        with self.lock:
            now = time.time()
            out = {"box": self.bid, "pending": self.pending, "count": dict(self.count),
                   "chip_virtual_ms": self.chip_s * 1e3, "items": len(self.box.items), "deleted": len(self.box.deleted),
                   "index": {k: len(v) for k, v in self.box.idx_ids.items()}, "banned": len(self.box.banned),
                   "topics": sorted(self.box.topics), "room": ROOM in self.box.room_keys, "proto": self.cfg.get("proto", "cp21"),
                   "tier": self.tier, "identity": hex(self.commit)[2:10], "slashed": self.slashed,
                   "notices": self.notices[-10:], "enc_tokens_max": self.enc_tokens_max, "coco": self.coco is not None,
                   "addr": self.addr, "peers_seen_s_ago": {str(p): round(now - t, 1) for p, t in self.peer_seen.items()},
                   "kept_on_disk": self.st is not None, "restored": self.restored}
            if self.p24:
                out["budget"] = self.W.budget(self.bid, now)
            return out


def handler(node: BoxNode):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, obj, code=200, ctype="application/json"):
            data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def allowed(self, path: str) -> bool:
            """Owner calls only from this computer; peer calls only from the listed addresses."""
            ip = self.client_address[0]
            if ip in LOOPBACK:
                return True
            return path in ("/gossip", "/pull", "/hello", "/sync", "/since", "/ihave", "/ihave1", "/env") \
                and ip in node.peer_ips

        def do_GET(self):
            u = urlparse(self.path)
            if not self.allowed(u.path):
                return self.reply({"error": "owner calls are answered only on this computer"}, 403)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self.reply(UI.read_bytes(), ctype="text/html; charset=utf-8")
            if u.path == "/hello":
                if "frm" in q:                                  # a box that came on says hello
                    node.up.add(int(q["frm"]))
                    node.peer_seen[int(q["frm"])] = time.time()
                    node.peer_appeared()
                return self.reply(node.hello_info())
            if u.path == "/env":
                with node.lock:
                    m = node.msgs.get(int(q["mid"]))
                    return self.reply({"msg": msg_to_json(m) if m is not None else None})
            if u.path == "/since":
                return self.reply(node.since(float(q.get("lo", 0)), float(q.get("hi", "inf"))))
            if u.path == "/sync":
                return self.reply(node.sync_list())
            if u.path == "/pull":
                data = node.serve_pull(int(q["mid"]))
                return self.reply(data, ctype="application/octet-stream") if data else self.reply({"error": "none"}, 404)
            if u.path == "/api/stats":
                return self.reply(node.stats())
            if u.path == "/api/feed":
                with node.lock:
                    return self.reply([node.item(mid) for mid in node.box.feed(25)])
            if u.path == "/api/search":
                return self.reply(node.query_text(q.get("q", "")))
            if u.path == "/api/search_image":
                return self.reply(node.query_image(q.get("q", "")))
            if u.path == "/api/img":
                mid = int(q["mid"])
                with node.lock:
                    data = node.full.get(mid)
                return self.reply({"rgb": base64.b64encode(data).decode() if data else None})
            if u.path == "/api/room":
                with node.lock:
                    if ROOM not in node.box.room_keys:
                        return self.reply({"member": False, "items": []})
                    rows = [node.item(mid) for mid, it in node.box.items.items() if it["kind"] == "room"]
                return self.reply({"member": True, "items": rows})
            if u.path == "/api/debug":
                with node.lock:
                    return self.reply({"verified": [str(x) for x in node.verified], "log": [[t, str(m), v] for t, m, v in node.box.log],
                                       "scores": {str(k): v["score"] for k, v in node.box.items.items()},
                                       "deleted": [str(x) for x in node.box.deleted],
                                       "banned": [str(x) for x in node.box.banned], "index_text": [str(x) for x in node.box.idx_ids["text"]]})
            self.reply({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            if not self.allowed(u.path):
                return self.reply({"error": "owner calls are answered only on this computer"}, 403)
            d = self.body()
            try:
                return self.post(u.path, d)
            except p24.Over as e:
                left = e.until - time.time()
                what = {"t": "text", "p": "picture", "a": "vote/delete"}[e.space]
                return self.reply({"error": f"no {what} index left in this window: your client waits {left / 60:.0f} min "
                                            f"({node.tier} group limits)", "wait_s": left}, 429)

        def post(self, path: str, d: dict):
            if path == "/gossip":
                node.count["env_rx"] += 1
                node.deliver(msg_from_json(d["msg"]), int(d["frm"]))
                return self.reply({"ok": True})
            if path == "/ihave1":
                return self.reply({"want": node.ihave1(int(d["mid"]), int(d["frm"]))})
            if path == "/ihave":
                node.count["want_ids"] += len(want := node.ihave(d["ids"]))
                return self.reply({"want": want})
            if path == "/inject":
                m = msg_from_json(d["msg"])
                node.own.add(m.mid)
                node.deliver(m, -1)
                return self.reply({"ok": True})
            if path == "/api/post":
                rgb = base64.b64decode(d["rgb"]) if d.get("rgb") else None
                if rgb is not None and len(rgb) != IMG_BYTES:
                    return self.reply({"error": "picture must be 224 x 224 RGB"}, 400)
                photo = None
                if d.get("photo") and rgb is None:
                    if node.coco is None:
                        return self.reply({"error": "no COCO pictures on this computer: pick a file instead"}, 400)
                    photo = int(np.random.default_rng().integers(300, len(node.coco.caps)))
                title = d.get("title") or (node.coco.caps[photo][0] if photo is not None else "")
                return self.reply({"mid": node.write("post", int(d.get("topic", A)), {"title": title, "body": d.get("body", "")},
                                                     photo=photo, rgb=rgb)})
            if path == "/api/vote":
                mid = int(d["mid"])
                topic = node.box.items[mid]["topic"]
                return self.reply({"mid": node.write("vote", topic, {"target": mid, "dir": int(d.get("dir", 1))}, target=mid)})
            if path == "/api/delete":
                mid = int(d["mid"])
                if mid not in node.W.keys:
                    return self.reply({"error": "only the author holds this post's key"}, 403)
                topic = node.box.items[mid]["topic"]
                return self.reply({"mid": node.write("delete", topic, {"target": mid}, target=mid)})
            if path == "/api/room":
                if ROOM not in node.box.room_keys:
                    return self.reply({"error": "not a room member"}, 403)
                return self.reply({"mid": node.write("room", ROOM, {"body": d.get("body", "")})})
            if path == "/api/misbehave":
                if not node.p24:
                    return self.reply({"error": "CP24 boxes only"}, 400)
                try:
                    a = node.write("post", A, {"title": "an ordinary post", "body": "(the next one reuses its message index)"})
                except p24.Over:                                   # the window is used up: reuse its last index
                    a = None
                b = node.write("post", A, {"title": "a second post on the same message index", "body": "breaks the limit"},
                               reuse=True)
                return self.reply({"mids": [a, b]})
            self.reply({"error": "not found"}, 404)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to take peers from other computers")
    ap.add_argument("--state", help="keep the box's state in this folder (CP25; CP24 boxes only)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    node = BoxNode(cfg, a.id, state=a.state)
    port = int(node.addr[a.id].rsplit(":", 1)[1])
    srv = ThreadingHTTPServer((a.bind, port), handler(node))
    srv.daemon_threads = True
    log(f"box {a.id} ({cfg.get('proto', 'cp21')}, {node.tier} group, identity {hex(node.commit)[2:10]}) "
        f"listening on {a.bind}:{port}; open http://127.0.0.1:{port}/ on this computer")
    if node.exchange == "poll":
        threading.Thread(target=node.sync_loop, daemon=True).start()
    elif node.exchange == "x2":
        def come_on():
            node.come_on(node.last_off, node.prev_session)
            node.alive_loop()
        threading.Thread(target=come_on, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
