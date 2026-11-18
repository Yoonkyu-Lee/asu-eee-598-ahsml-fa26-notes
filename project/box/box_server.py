"""One home node box as a real process (CP21 E1, CP24): HTTP, the box software (sim.proto24), real Ed25519,
ChaCha20-Poly1305, RLN shares and nullifiers, MiniLM and CLIP embeddings, search, feed, votes, deletes, room.
The proof's pairing check is not run (the message carries proof_ok, as in the simulator); its time on the chip is
shown from the CP19/CP20 chip model.

CP24: O-b check order, I5 + F1 limits (the owner's client waits when its indices are used up), an encoder queue by
priority (query, old text, new text, old pictures, new pictures; text up to 128 tokens), its own identity drawn in
this process, photos uploaded from the screen, and several computers (cfg "addr"): the box listens on --bind, answers
owner calls only from this computer and peer calls only from the listed addresses.

CP25 (--state DIR, set by lan.py): the box keeps its identity, the indices it used, its one-time keys, the messages it
took and the pictures on disk (store.py), and takes them back in arrival order when it starts again. A deleted post is
forgotten on disk and in /sync too, and a copy that comes back later from a stale box is dropped. events.jsonl logs
times for remote_report.py (no plaintext).

CP28 (cfg "text" = "fp32"): ids first (CP27 P2): a box tells its mesh peers a message's id (POST /ihave1); a peer
that lacks it answers "want" to the first announcer only, and if the envelope has not come 1 s later it fetches it
from the next announcer (GET /env). Text is encoded by FP32 MiniLM (the CP3 contract; FP16 grade, CP27 B2).

CP29, CP30: set reconciliation on every contact (sim/recon.py S2), one fetch per id (K), cfg "floor_skip" = F'.
The CP21 box, the poll and x2 exchanges and envelope-first push are at tag pre-r1.

  .venv/Scripts/python.exe box/lan.py --hosts ... --me 0      (lan.py writes the config and starts this box)

The box's code (R4a): box_state.py the state, ingest.py the ingest path, peer_sync.py push and reconciliation,
encoder.py the encoders and the virtual chip, disk.py the state on disk, owner.py the owner's side, routes.py the
HTTP routes and who may call each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import disk  # noqa: E402
import encoder  # noqa: E402
import ingest  # noqa: E402
import peer_sync  # noqa: E402
import routes  # noqa: E402
from box_state import Node, log  # noqa: E402


def start(cfg: dict, bid: int, state: str | None = None) -> Node:
    """The box: its state, the identity kept on disk (CP25), the encoders and their thread, then the stored messages
    taken again."""
    node = Node(cfg, bid, state)
    if node.st:
        disk.restore_identity(node)
        node.commit = node.W.ids[bid].commit
    encoder.load(node)
    threading.Thread(target=encoder.run, args=(node,), daemon=True).start()
    node.restored = ingest.replay(node) if node.st else 0
    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to take peers from other computers")
    ap.add_argument("--state", help="keep the box's state in this folder (CP25)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    node = start(cfg, a.id, state=a.state)
    port = int(node.addr[a.id].rsplit(":", 1)[1])
    srv = ThreadingHTTPServer((a.bind, port), routes.handler(node))
    srv.daemon_threads = True
    log(f"box {a.id} ({cfg['proto']}, {node.tier} group, identity {hex(node.commit)[2:10]}) "
        f"listening on {a.bind}:{port}; open http://127.0.0.1:{port}/ on this computer")

    def come_on():
        peer_sync.come_on(node)
        peer_sync.alive_loop(node)
    threading.Thread(target=come_on, daemon=True).start()
    threading.Thread(target=peer_sync.recon_loop, args=(node,), daemon=True).start()
    threading.Thread(target=peer_sync.expiry_loop, args=(node,), daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
