"""A box's state on disk (CP25), so a box that is stopped and started again is the same member of the community.

  identity.json   the identity secret a0, the age group, the indices used per (space, window)   (written before a
                  message leaves, so a restarted box never spends an index twice and reveals itself)
  keys.json       the one-time Ed25519 key of each of my messages (only the author can delete a post)
  msgs.jsonl      every message envelope this box took, in arrival order (ciphertext as received; no plaintext)
  img/<mid>.rgb   picture bodies (224 x 224 RGB)
  tomb.json       deleted posts: mid -> the author's one-time public keys. A deleted post's envelope and picture are
                  removed from disk and from /sync, and a copy that arrives later from a stale box is dropped.
  events.jsonl    measurement log (times, ids, sizes; no plaintext) for remote_report.py

The secret is a plain file (CP25 K-a: owner-only permissions where the OS has them); fine for the team's run, not for
strangers' boxes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.dir = Path(path)
        (self.dir / "img").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._ev = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._msgs = open(self.dir / "msgs.jsonl", "a", encoding="utf-8")

    # ---- small JSON files, replaced atomically
    def _read(self, name: str, default):
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name: str, obj, secret: bool = False) -> None:
        p = self.dir / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        if secret:
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        os.replace(tmp, p)

    def identity(self) -> dict | None:
        return self._read("identity.json", None)

    def save_identity(self, a0: int, tier: str, used: dict) -> None:
        self._write("identity.json", {"a0": str(a0), "tier": tier,
                                      "used": [[sp, e, k] for (sp, e), k in used.items()]}, secret=True)

    def keys(self) -> dict:
        return self._read("keys.json", {})

    def save_keys(self, keys: dict) -> None:
        self._write("keys.json", keys, secret=True)

    def tomb(self) -> dict:
        return {int(k): set(v) for k, v in self._read("tomb.json", {}).items()}

    def save_tomb(self, tomb: dict) -> None:
        self._write("tomb.json", {str(k): sorted(v) for k, v in tomb.items()})

    # ---- messages
    def messages(self) -> list[dict]:
        p = self.dir / "msgs.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    def add_message(self, d: dict, frm: int) -> None:
        with self.lock:
            self._msgs.write(json.dumps({**d, "_frm": frm}) + "\n")
            self._msgs.flush()

    def forget(self, mid: int) -> None:
        """Drop a deleted post's envelope (rewrite the log without it) and its picture."""
        with self.lock:
            self._msgs.close()
            p = self.dir / "msgs.jsonl"
            keep = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip() and int(json.loads(x)["mid"]) != mid]
            tmp = p.with_suffix(".tmp")
            tmp.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
            os.replace(tmp, p)
            self._msgs = open(p, "a", encoding="utf-8")
            (self.dir / "img" / f"{mid}.rgb").unlink(missing_ok=True)

    def picture(self, mid: int) -> bytes | None:
        p = self.dir / "img" / f"{mid}.rgb"
        return p.read_bytes() if p.exists() else None

    def save_picture(self, mid: int, data: bytes) -> None:
        p = self.dir / "img" / f"{mid}.rgb"
        if not p.exists():
            p.write_bytes(data)

    # ---- measurement log
    def ev(self, name: str, /, **f) -> None:
        with self.lock:
            self._ev.write(json.dumps({"ev": name, "t": time.time(), **f}) + "\n")
            self._ev.flush()
