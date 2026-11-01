"""One home node box as a real process (CP21 E1): HTTP on 127.0.0.1, the CP17 box software (sim.box), real Ed25519,
ChaCha20-Poly1305, RLN shares and nullifiers, MiniLM INT8 and CLIP embeddings, search, feed, votes, deletes, room.
The proof's pairing check is not run (the message carries proof_ok, as in the simulator); its time on the chip is
shown from the CP19/CP20 chip model.

  .venv/Scripts/python.exe src/demo/box_server.py --cfg results/cp21/cfg.json --id 0
Peers:   POST /gossip {msg, frm}    GET /pull?mid=&frm=    POST /inject {msg}  (a message written on this box)
Owner:   GET / (screen)  /api/stats  /api/feed  /api/search?q=  /api/search_image?q=  /api/img?mid=  /api/debug
         POST /api/post {topic, title, body, photo}  /api/vote {mid, dir}  /api/delete {mid}  /api/room {body}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
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

UI = Path(__file__).resolve().parent / "ui.html"
CHIP = {"proof_s": 89.6e-6, "image_s": 0.552}     # CP19 verify unit (single proof), CP20 FP16 image tower (추정)


class BoxNode:
    def __init__(self, cfg: dict, bid: int, load_models: bool = True):
        self.cfg, self.bid, self.n = cfg, bid, cfg["n"]
        self.holders = {int(t): v for t, v in cfg["holders"].items()}
        self.mesh = {int(t): {int(b): p for b, p in m.items()} for t, m in cfg["mesh"].items()}
        self.room_key = bytes.fromhex(cfg["room_key"])
        self.S = Shared()
        topics = [t for t, hs in self.holders.items() if bid in hs]
        self.box = Box(bid, topics, {ROOM: self.room_key} if bid in self.holders[ROOM] else {}, self.S,
                       {"text": 0, "image": 0})
        self.W = World(self.n, seed=cfg["seed"])         # this box's own identity (user bid) for the screen
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
        self.pool = ThreadPoolExecutor(16)
        self.coco = Coco()
        if load_models:
            from bench_cp18 import ClipText
            from sim import ml
            import sim_cp19
            self.text_enc = ml.Encoder("int8", threads=1)
            self.img_enc = ml.ImageEncoder("fp32", threads=1)    # FP32 = the FP16 tower's quality (CP20)
            self.clip_text = ClipText(threads=1)
            self.chip_table = sim_cp19.EncTable(8, 1024)
        self.ready = True

    # ---- helpers
    def _inc(self, k: int = 1) -> None:
        with self.lock:
            self.pending += k

    def _post(self, peer: int, path: str, payload: dict) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self.cfg['ports'][peer]}{path}",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()

    def send(self, peer: int, m: Msg) -> None:
        self.count["send"] += 1
        self._inc()
        payload = {"msg": msg_to_json(m), "frm": self.bid}

        def go():
            try:
                self._post(peer, "/gossip", payload)
            finally:
                self._inc(-1)
        self.pool.submit(go)

    def key_for(self, topic: int) -> bytes | None:
        return self.box.key_for(topic)

    def text_of(self, m: Msg) -> str | None:
        key = self.key_for(m.topic)
        body = self.S.open(m, key) if key else None
        if body is None:
            return None
        return ((body.get("title") or "") + "\n\n" + (body.get("body") or "")).strip()

    # ---- the ingest path (mirrors sim.stack.DayNet.deliver, counted by the same operation names)
    def deliver(self, m: Msg, frm: int) -> None:
        with self.lock:
            if m.mid in self.got:
                self.count["dup"] += 1
                return
            self.got.add(m.mid)
            self.msgs[m.mid] = m
            t = time.time()
            self.count["ed25519"] += 1
            if m.kind != "slash":
                self.count["proof"] += 1
                self.count["rln"] += 1
                self.verified.append(m.mid)
                self.chip_s += CHIP["proof_s"]
            v = self.box.validate(m)
            if v[0] != "ok" and frm >= 0:                       # a relay drops it; the author's box sends anyway
                if v[0] == "slash":
                    self.publish_slash(v[1])
                self.box.apply(m, t, v)
                return
            for peer in self.mesh.get(m.topic, {}).get(self.bid, ()):
                if peer != frm:
                    self.send(peer, m)
            if v[0] != "ok":
                self.box.apply(m, t, v)
                return
            if m.kind == "slash":
                self.box.apply(m, t, v)
                return
            self.count["decrypt"] += 1
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
                url = f"http://127.0.0.1:{self.cfg['ports'][self.parent[mid]]}/pull?mid={mid}&frm={self.bid}"
                data = urllib.request.urlopen(url, timeout=300).read()
                with self.lock:
                    self.full[mid] = data
                    self.full_ev[mid].set()
                    self._index_image(mid)
            finally:
                self._inc(-1)
        self.pool.submit(go)

    def serve_pull(self, mid: int) -> bytes:
        with self.lock:
            ev = self.full_ev.setdefault(mid, threading.Event())
            if mid not in self.full and mid in self.parent:
                self._request(mid)
        ev.wait(300)
        with self.lock:
            self.count["pull"] += 1
            return self.full[mid]

    def _index_image(self, mid: int) -> None:
        px = np.frombuffer(self.full[mid], np.uint8).reshape(1, 224, 224, 3)
        self.S.img[mid] = self.img_enc.encode(px)[0]
        self.chip_s += CHIP["image_s"]
        self.count["image"] += 1
        self.count["insert"] += 1
        if mid in self.box.items and mid not in self.box.deleted:
            self.box.index("image", mid)

    # ---- the owner's side
    def query_text(self, q: str, k: int = 10) -> list[dict]:
        with self.lock:
            v = self.text_enc.encode([q])[0]
            self.count["query"] += 1
            self.chip_s += self.chip_table.text1(12)[0]
            return [self.item(mid) for mid in self.box.search(v, "text", k)]

    def query_image(self, q: str, k: int = 10) -> list[dict]:
        with self.lock:
            v = self.clip_text.encode([q])[0]
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

    def write(self, kind: str, topic: int, body: dict, *, target: int | None = None, photo: int | None = None) -> str:
        key = self.room_key if topic == ROOM else sub_key(topic)
        size = 400 + len(json.dumps(body)) + (IMG_BYTES if photo is not None else 0)
        m = self.W.make(self.bid, kind, topic, time.time(), body, size, key, target=target, image=photo,
                        pull_bytes=IMG_BYTES if photo is not None else 0.0)
        new = int.from_bytes(os.urandom(6), "big") + (1 << 48)        # this process's ids must not collide
        self.W.keys[new] = self.W.keys.pop(m.mid)
        m.mid = new
        self.own.add(new)
        self.deliver(m, -1)
        return str(new)

    def stats(self) -> dict:
        with self.lock:
            return {"box": self.bid, "pending": self.pending, "count": dict(self.count),
                    "chip_virtual_ms": self.chip_s * 1e3, "items": len(self.box.items), "deleted": len(self.box.deleted),
                    "index": {k: len(v) for k, v in self.box.idx_ids.items()}, "banned": len(self.box.banned),
                    "topics": sorted(self.box.topics), "room": ROOM in self.box.room_keys}


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

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self.reply(UI.read_bytes(), ctype="text/html; charset=utf-8")
            if u.path == "/pull":
                return self.reply(node.serve_pull(int(q["mid"])), ctype="application/octet-stream")
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
            d = self.body()
            if u.path == "/gossip":
                node.deliver(msg_from_json(d["msg"]), int(d["frm"]))
                return self.reply({"ok": True})
            if u.path == "/inject":
                m = msg_from_json(d["msg"])
                node.own.add(m.mid)
                node.deliver(m, -1)
                return self.reply({"ok": True})
            if u.path == "/api/post":
                photo = int(np.random.default_rng().integers(300, len(node.coco.caps))) if d.get("photo") else None
                title = d.get("title") or (node.coco.caps[photo][0] if photo is not None else "")
                return self.reply({"mid": node.write("post", int(d.get("topic", A)), {"title": title, "body": d.get("body", "")},
                                                     photo=photo)})
            if u.path == "/api/vote":
                mid = int(d["mid"])
                topic = node.box.items[mid]["topic"]
                return self.reply({"mid": node.write("vote", topic, {"target": mid, "dir": int(d.get("dir", 1))}, target=mid)})
            if u.path == "/api/delete":
                mid = int(d["mid"])
                if mid not in node.W.keys:
                    return self.reply({"error": "only the author holds this post's key"}, 403)
                topic = node.box.items[mid]["topic"]
                return self.reply({"mid": node.write("delete", topic, {"target": mid}, target=mid)})
            if u.path == "/api/room":
                if ROOM not in node.box.room_keys:
                    return self.reply({"error": "not a room member"}, 403)
                return self.reply({"mid": node.write("room", ROOM, {"body": d.get("body", "")})})
            self.reply({"error": "not found"}, 404)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--id", type=int, required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    node = BoxNode(cfg, a.id)
    srv = ThreadingHTTPServer(("127.0.0.1", cfg["ports"][a.id]), handler(node))
    srv.daemon_threads = True
    print(f"box {a.id} on http://127.0.0.1:{cfg['ports'][a.id]}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
