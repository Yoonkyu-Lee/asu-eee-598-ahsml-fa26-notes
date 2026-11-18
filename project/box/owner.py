"""The owner's side of the box: search (the owner's queries run now, not in the encoder queue), what the screen shows of
a post, writing (posts, votes, deletes, the room), and the box's stats."""
from __future__ import annotations

import json
import time

import disk
import ingest
from box_state import Node
from common import IMG_BYTES, ROOM, TOPIC_NAMES
from sim.box import content_id, sub_key, target_ref
from sim import proto24 as p24


def query_text(node: Node, q: str, k: int = 10) -> list[dict]:
    v = node.text_enc.encode([q])[0]
    with node.lock:
        node.count["query"] += 1
        node.chip_s += node.chip_table.text1(12)[0]
        return [item(node, mid) for mid in node.box.search(v, "text", k)]


def query_image(node: Node, q: str, k: int = 10) -> list[dict]:
    v = node.clip_text.encode([q])[0]
    with node.lock:
        node.count["query_clip"] += 1
        return [item(node, mid) for mid in node.box.search(v, "image", k)]


def item(node: Node, mid: int) -> dict:
    it = node.box.items.get(mid, {})
    m = node.msgs.get(mid)
    body = node.S.open(m, node.box.key_for(m.topic)) if m is not None and node.box.key_for(m.topic) else None
    return {"mid": str(mid), "kind": it.get("kind"), "topic": TOPIC_NAMES.get(it.get("topic"), it.get("topic")),
            "title": it.get("title"), "body": (body or {}).get("body", "")[:300], "score": it.get("score", 0),
            "own": mid in node.own, "photo": m is not None and m.image is not None and mid in node.full,
            "deleted": mid in node.box.deleted}


def write(node: Node, kind: str, topic: int, body: dict, *, target: int | None = None, photo: int | None = None,
          rgb: bytes | None = None, reuse: bool = False) -> str:
    key = node.room_key if topic == ROOM else sub_key(topic)
    image = photo if photo is not None else (-1 if rgb is not None else None)
    size = 400 + len(json.dumps(body)) + (IMG_BYTES if image is not None else 0)
    text = ((body.get("title") or "") + "\n\n" + (body.get("body") or "")).strip()
    # attention mask, not len(ids): tokenizer.json pads to 128 until the first tokenize() (CP25 fix: a fresh
    # box's first post was counted as 128 tokens and spent two text indices)
    ntok = int(node.text_enc.tokenize([text])["attention_mask"].sum()) if kind in p24.TEXT_KINDS else 0
    kw = {"reuse": reuse, "tokens": ntok}
    with node.lock:
        if kind == "delete":
            tm = node.msgs.get(target)
            if tm is None:
                raise KeyError("the post is not held here")
            kw["ref"] = target_ref(tm)
        m = node.W.make(node.bid, kind, topic, node.now(), body, size, key, target=target, image=image,
                        pull_bytes=IMG_BYTES if image is not None else 0.0, **kw)
        new = content_id(m)                                    # CP29: the id is the content's hash
        node.W.keys[new] = node.W.keys.pop(m.mid)
        m.mid = new
        node.own.add(new)
        if rgb is not None:
            disk.have_picture(node, new, rgb)
        if node.st:                                     # before it leaves: a restart never reuses an index
            disk.save_identity(node)
            disk.save_keys(node)
            node.st.ev("write", mid=str(new), kind=kind, topic=topic, tokens=kw.get("tokens", 0),
                       photo=image is not None)
    ingest.deliver(node, m, -1)
    return str(new)


def stats(node: Node) -> dict:
    with node.lock:
        now = time.time()
        out = {"box": node.bid, "pending": node.pending, "count": dict(node.count),
               "chip_virtual_ms": node.chip_s * 1e3, "items": len(node.box.items), "deleted": len(node.box.deleted),
               "index": {k: len(v) for k, v in node.box.idx_ids.items()}, "banned": len(node.box.banned),
               "topics": sorted(node.box.topics), "room": ROOM in node.box.room_keys, "proto": node.cfg["proto"],
               "tier": node.tier, "identity": hex(node.commit)[2:10], "slashed": node.slashed,
               "notices": node.notices[-10:], "enc_tokens_max": node.enc_tokens_max, "coco": node.coco is not None,
               "addr": node.addr, "peers_seen_s_ago": {str(p): round(now - t, 1) for p, t in node.peer_seen.items()},
               "kept_on_disk": node.st is not None, "restored": node.restored,
               "budget": node.W.budget(node.bid, now)}
        return out
