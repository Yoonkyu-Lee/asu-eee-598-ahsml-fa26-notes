"""The box's state on disk (CP25), between the node and store.py: the identity and the indices it used, the one-time
keys, pictures, and forgetting a deleted post. Taking the stored messages back at start is ingest.replay."""
from __future__ import annotations

from box_state import Node, log
from sim.box import Msg, h, ref_target


def restore_identity(node: Node) -> None:
    d = node.st.identity()
    if d is None:                                          # first start: keep the identity drawn in Node
        save_identity(node)
        return
    me = node.W.ids[node.bid]
    node.W.members.discard(me.commit)
    me.a0 = int(d["a0"])
    me.commit = h("id", me.a0)
    node.W.members.add(me.commit)
    if d["tier"] != node.tier:
        log(f"this identity is in the {d['tier']} group (kept; the address list says {node.tier})")
    node.tier = node.W.tier[node.bid] = d["tier"]
    for sp, e, k in d["used"]:
        node.W.used[(node.bid, sp, int(e))] = int(k)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    for mid, raw in node.st.keys().items():
        node.W.keys[int(mid)] = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw))
        node.own.add(int(mid))
    node.tomb = node.st.tomb()


def save_identity(node: Node) -> None:
    used = {(sp, e): k for (u, sp, e), k in node.W.used.items() if u == node.bid}
    node.st.save_identity(node.W.ids[node.bid].a0, node.tier, used)


def save_keys(node: Node) -> None:
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    node.st.save_keys({str(mid): sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
                       for mid, sk in node.W.keys.items()})


def have_picture(node: Node, mid: int, data: bytes) -> None:
    node.full[mid] = data
    if node.st:
        node.st.save_picture(mid, data)


def after_delete(node: Node, m: Msg) -> None:
    """Forgetting (CP25): a delete by the author removes the post's envelope and picture here, on disk and from
    /sync. A delete that comes before its post leaves a tombstone, so the post is dropped when it comes."""
    rt = ref_target(m)                                     # the target from the signed header (relays too)
    if rt is None:
        return
    tgt = rt[0]
    node.keys.add((int(rt[1]), tgt), m.topic)              # its key stays in the set as a forgotten id
    node.gone_t[tgt] = rt[1]
    node.gone.add(tgt)
    it = node.box.items.get(tgt)
    if it is not None and it["epub"] != m.epub:            # someone else's delete: Box.apply refused it
        return
    node.tomb.setdefault(tgt, set()).add(m.epub.hex())
    node.st.save_tomb(node.tomb)
    if it is None:
        return
    node.gone.add(tgt)
    node.msgs.pop(tgt, None)
    node.full.pop(tgt, None)
    node.S.plain = {k: v for k, v in node.S.plain.items() if k[0] != tgt}
    node.S.emb.pop(tgt, None)
    node.S.img.pop(tgt, None)
    it["title"] = None
    for kind, ids in node.box.idx_ids.items():
        keep = [i for i, x in enumerate(ids) if x != tgt]
        node.box.idx_ids[kind] = [ids[i] for i in keep]
        node.box.idx_vec[kind] = [node.box.idx_vec[kind][i] for i in keep]
    if not node.replaying:
        node.st.forget(tgt)
        node.st.ev("forget", mid=str(tgt))
