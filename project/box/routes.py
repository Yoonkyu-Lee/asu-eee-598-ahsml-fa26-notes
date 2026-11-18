"""The box's HTTP side: one table from (method, path) to (who may call it, handler).

Who may call (checked before the request is read):
  PEER    the boxes in the config's address list, and this computer
  OWNER   this computer only (the owner's screen and API, and the test hooks /sync, /api/holds, /inject)
  TEST    this computer only, and only when cfg "faults" is on
Every call that names a peer (frm) whose link is cut by /fault gets 503. A handler returns the reply body, or
(body, code) or (body, code, content type); bytes go out as they are, anything else as JSON.
"""
from __future__ import annotations

import base64
import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import ingest
import owner
import peer_sync
from box_state import Node
from common import A, IMG_BYTES, ROOM, msg_from_json, msg_to_json
from sim import proto24 as p24

UI = Path(__file__).resolve().parent / "ui.html"
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
PEER, OWNER, TEST = "peer", "owner", "test"


# ---- peers
def hello(node: Node, q: dict):
    if "frm" in q:                                  # a box that came on says hello
        peer_sync.peer_appeared(node, int(q["frm"]))
    return peer_sync.hello_info(node)


def env(node: Node, q: dict):
    with node.lock:
        m = node.msgs.get(int(q["mid"]))
        return {"msg": msg_to_json(m) if m is not None else None}


def pull(node: Node, q: dict):
    data = ingest.serve_pull(node, int(q["mid"]))
    return (data, 200, "application/octet-stream") if data else ({"error": "none"}, 404)


def gossip(node: Node, d: dict):
    node.count["env_rx"] += 1
    if node.fault["silent"] > 0:                       # test-only: acked, then lost (no retry either)
        node.fault["silent"] -= 1
        node.count["silent_dropped"] += 1
        node.silent_mids.add(int(d["msg"]["mid"]))
        return {"ok": True}
    ingest.deliver(node, msg_from_json(d["msg"]), int(d["frm"]))
    return {"ok": True}


def offer(node: Node, d: dict):
    node.count["offer_want"] += len(want := peer_sync.offer(node, int(d["frm"]), d["ids"]))
    return {"want": want}


# ---- the owner's screen and API
def sync_list(node: Node, q: dict):
    with node.lock:
        return [msg_to_json(m) for m in node.msgs.values()]


def feed(node: Node, q: dict):
    with node.lock:
        return [owner.item(node, mid) for mid in node.box.feed(25)]


def img(node: Node, q: dict):
    mid = int(q["mid"])
    with node.lock:
        data = node.full.get(mid)
    return {"rgb": base64.b64encode(data).decode() if data else None}


def room_read(node: Node, q: dict):
    with node.lock:
        if ROOM not in node.box.room_keys:
            return {"member": False, "items": []}
        rows = [owner.item(node, mid) for mid, it in node.box.items.items() if it["kind"] == "room"]
    return {"member": True, "items": rows}


def debug(node: Node, q: dict):
    with node.lock:
        return {"verified": [str(x) for x in node.verified], "log": [[t, str(m), v] for t, m, v in node.box.log],
                "scores": {str(k): v["score"] for k, v in node.box.items.items()},
                "deleted": [str(x) for x in node.box.deleted],
                "banned": [str(x) for x in node.box.banned], "index_text": [str(x) for x in node.box.idx_ids["text"]]}


def inject(node: Node, d: dict):
    m = msg_from_json(d["msg"])
    node.own.add(m.mid)
    ingest.deliver(node, m, -1)
    return {"ok": True}


def post(node: Node, d: dict):
    rgb = base64.b64decode(d["rgb"]) if d.get("rgb") else None
    if rgb is not None and len(rgb) != IMG_BYTES:
        return {"error": "picture must be 224 x 224 RGB"}, 400
    photo = None
    if d.get("photo") and rgb is None:
        if node.coco is None:
            return {"error": "no COCO pictures on this computer: pick a file instead"}, 400
        photo = int(np.random.default_rng().integers(300, len(node.coco.caps)))
    title = d.get("title") or (node.coco.caps[photo][0] if photo is not None else "")
    return {"mid": owner.write(node, "post", int(d.get("topic", A)), {"title": title, "body": d.get("body", "")},
                               photo=photo, rgb=rgb)}


def vote(node: Node, d: dict):
    mid = int(d["mid"])
    topic = node.box.items[mid]["topic"]
    return {"mid": owner.write(node, "vote", topic, {"target": mid, "dir": int(d.get("dir", 1))}, target=mid)}


def delete(node: Node, d: dict):
    mid = int(d["mid"])
    if mid not in node.W.keys:
        return {"error": "only the author holds this post's key"}, 403
    topic = node.box.items[mid]["topic"]
    return {"mid": owner.write(node, "delete", topic, {"target": mid}, target=mid)}


def room_write(node: Node, d: dict):
    if ROOM not in node.box.room_keys:
        return {"error": "not a room member"}, 403
    return {"mid": owner.write(node, "room", ROOM, {"body": d.get("body", "")})}


def misbehave(node: Node, d: dict):
    """CP24: two posts on one message index -> the identity is revealed."""
    try:
        a = owner.write(node, "post", A, {"title": "an ordinary post", "body": "(the next one reuses its message index)"})
    except p24.Over:                                   # the window is used up: reuse its last index
        a = None
    b = owner.write(node, "post", A, {"title": "a second post on the same message index", "body": "breaks the limit"},
                    reuse=True)
    return {"mids": [a, b]}


ROUTES = {
    ("GET", "/hello"):             (PEER, hello),
    ("GET", "/env"):               (PEER, env),
    ("GET", "/pull"):              (PEER, pull),
    ("POST", "/gossip"):           (PEER, gossip),
    ("POST", "/ihave1"):           (PEER, lambda node, d: {"want": peer_sync.ihave1(node, int(d["mid"]), int(d["frm"]))}),
    ("POST", "/recon"):            (PEER, lambda node, d: peer_sync.recon_serve(node, int(d["frm"]), d["msg"], d.get("topics"))),
    ("POST", "/envs"):             (PEER, lambda node, d: peer_sync.envs(node, d["mids"])),
    ("POST", "/offer"):            (PEER, offer),
    ("GET", "/"):                  (OWNER, lambda node, q: (UI.read_bytes(), 200, "text/html; charset=utf-8")),
    ("GET", "/api/stats"):         (OWNER, lambda node, q: owner.stats(node)),
    ("GET", "/api/feed"):          (OWNER, feed),
    ("GET", "/api/search"):        (OWNER, lambda node, q: owner.query_text(node, q.get("q", ""))),
    ("GET", "/api/search_image"):  (OWNER, lambda node, q: owner.query_image(node, q.get("q", ""))),
    ("GET", "/api/img"):           (OWNER, img),
    ("GET", "/api/room"):          (OWNER, room_read),
    ("GET", "/api/debug"):         (OWNER, debug),
    ("POST", "/api/post"):         (OWNER, post),
    ("POST", "/api/vote"):         (OWNER, vote),
    ("POST", "/api/delete"):       (OWNER, delete),
    ("POST", "/api/room"):         (OWNER, room_write),
    ("POST", "/api/misbehave"):    (OWNER, misbehave),
    ("GET", "/sync"):              (OWNER, sync_list),
    ("GET", "/api/holds"):         (OWNER, lambda node, q: peer_sync.holds(node)),
    ("POST", "/inject"):           (OWNER, inject),
    ("POST", "/fault"):            (TEST, lambda node, d: peer_sync.set_fault(node, d)),
}
PEER_PATHS = {path for (_, path), (who, _) in ROUTES.items() if who == PEER}


def handler(node: Node):
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

        def do_GET(self):
            self.serve("GET")

        def do_POST(self):
            self.serve("POST")

        def serve(self, method: str):
            u = urlparse(self.path)
            ip = self.client_address[0]
            if not (ip in LOOPBACK or (u.path in PEER_PATHS and ip in node.peer_ips)):
                return self.reply({"error": "owner calls are answered only on this computer"}, 403)
            if method == "GET":
                d = {k: v[0] for k, v in parse_qs(u.query).items()}
            else:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if "frm" in d and int(d["frm"]) in node.fault["drop"]:
                return self.reply({"error": "link cut (test)"}, 503)
            who, fn = ROUTES.get((method, u.path), (None, None))
            if fn is None:
                return self.reply({"error": "not found"}, 404)
            if who == TEST and not node.cfg.get("faults"):
                return self.reply({"error": "faults are off"}, 403)
            try:
                out = fn(node, d)
            except p24.Over as e:
                left = e.until - time.time()
                what = {"t": "text", "p": "picture", "a": "vote/delete"}[e.space]
                return self.reply({"error": f"no {what} index left in this window: your client waits {left / 60:.0f} min "
                                            f"({node.tier} group limits)", "wait_s": left}, 429)
            return self.reply(*out) if isinstance(out, tuple) else self.reply(out)
    return H
