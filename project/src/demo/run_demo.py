"""Start the demo: five CP24 boxes on 127.0.0.1:8601-8605 (boxes 0-2 in the old group, 3-4 new), a few posts and
photos already in, until Ctrl+C. --cp21 starts CP21's boxes instead (CP17 box software, one shared seed).

  .venv/Scripts/python.exe src/demo/run_demo.py            then open http://127.0.0.1:8601/ (and 8602-8605)
  .venv/Scripts/python.exe src/demo/run_demo.py --empty    no starting posts
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import community, save_cfg  # noqa: E402

LOGS = HERE.parents[1] / "logs"


def call(port: int, path: str, obj: dict | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if obj is None else json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main() -> None:
    cfg = community() if "--cp21" in sys.argv else community(proto="cp24", tiers=["old", "old", "old", "new", "new"])
    LOGS.mkdir(exist_ok=True)
    path = LOGS / "cp21_demo_cfg.json"
    save_cfg(cfg, path)
    procs = [subprocess.Popen([sys.executable, str(HERE / "box_server.py"), "--cfg", str(path), "--id", str(i)],
                              stdout=open(LOGS / f"cp21_demo_box{i}.log", "w"), stderr=subprocess.STDOUT)
             for i in range(cfg["n"])]
    try:
        for p in cfg["ports"]:
            for _ in range(600):
                try:
                    call(p, "/api/stats")
                    break
                except Exception:
                    time.sleep(0.5)
        if "--empty" not in sys.argv:
            for i in range(5):      # one photo per box (the old group may post one per 10 min)
                call(cfg["ports"][i % cfg["n"]], "/api/post", {"topic": 0, "title": "", "body": "", "photo": True})
            for i, (t, b) in enumerate([("Welcome to sub A", "Every box here keeps its own copy and its own index."),
                                        ("No operator", "Delete your post and every box drops it."),
                                        ("Sub B is smaller", "Only boxes 0, 2 and 4 hold sub B.")]):
                call(cfg["ports"][i], "/api/post", {"topic": 0 if i < 2 else 1, "title": t, "body": b})
            call(cfg["ports"][1], "/api/room", {"body": "Only boxes 1, 2 and 3 can read this."})
        for i, p in enumerate(cfg["ports"]):
            print(f"box {i}: http://127.0.0.1:{p}/", flush=True)
        print("Ctrl+C to stop", flush=True)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
