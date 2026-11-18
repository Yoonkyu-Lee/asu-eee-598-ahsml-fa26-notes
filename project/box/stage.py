"""CP32: the one-screen demo. Five users, one box each, every client on one page (journal/cp/CP32.md).

  .venv/Scripts/python.exe box/stage.py                 then open http://127.0.0.1:8620/
  .venv/Scripts/python.exe box/stage.py --headless      the same story without a browser -> results/demo/story_headless.json
  --real-timers   the box defaults (floor 10-30 min, probe cap 5 min) instead of the test timers (20-40 s, 20 s)
  --keep          keep the boxes' state from the last run (default: start from empty boxes)

The page shows each user's own client (the box's real web UI) side by side. "Play" runs the story: each user's
client types and clicks by itself (the stage tells it what to do; the client calls only its own box), and after
each step the stage reads every box and shows whether what the protocol promises is what each screen shows.
You can also use any client by hand, switch boxes off and on, and split the network.
Everything listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import story as ST  # noqa: E402

STAGE_PORT = 8620
PAGE = HERE / "stage.html"


class Stage:
    def __init__(self, real_timers: bool, keep: bool):
        self.F = ST.Fleet(real_timers)
        if not keep:
            self.F.fresh()
        self.story = ST.Story(self.F)
        self.lock = threading.Lock()
        self.status = "starting boxes"
        self.t_play = None

    def boot(self) -> None:
        self.F.start_all()
        self.status = "ready"

    def reset(self) -> None:
        self.status = "resetting"
        self.F.fresh()
        self.story = ST.Story(self.F)
        self.boot()

    def state(self) -> dict:
        boxes = []
        for i in ST.ALL:
            row = {"i": i, "on": self.F.on(i), "port": ST.PORTS[i], "drop": sorted(self.F.drop[i])}
            if row["on"]:
                try:
                    s = ST.call(ST.PORTS[i], "/api/stats", timeout=3)[1]
                    c = s["count"]
                    row.update(items=s["items"], heard={k: v for k, v in s["peers_seen_s_ago"].items()},
                               notices=len(s["notices"]), slashed=s["slashed"], chip_ms=round(s["chip_virtual_ms"], 1),
                               recon=c.get("recon_sessions", 0), env_in=c.get("env_rx", 0),
                               rejected=sum(v for k, v in c.items() if k.startswith("rej_")),
                               verified=c.get("proof", 0), encoded=c.get("text", 0) + c.get("image", 0))
                except OSError:
                    row["starting"] = True
            boxes.append(row)
        return {"status": self.status, "boxes": boxes, "results": self.story.results}


def handler(stage: Stage):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, obj, code: int = 200, ctype: str = "application/json") -> None:
            data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self.reply(PAGE.read_bytes(), ctype="text/html; charset=utf-8")
            if u.path == "/api/story":
                return self.reply({"personas": ST.PERSONAS, "ports": ST.PORTS, "steps": ST.STEPS,
                                   "ui_acts": sorted(ST.UI_ACTS), "timers": {k: list(v) if isinstance(v, tuple) else v
                                                                             for k, v in stage.F.timers.items()},
                                   "real_timers": stage.F.real})
            if u.path == "/api/state":
                return self.reply(stage.state())
            self.reply({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if u.path == "/api/begin":
                    if d["n"] == 1:
                        stage.t_play = time.time()
                    stage.story.begin(d["n"])
                    return self.reply({"ok": True})
                if u.path == "/api/act":            # given = what the user's client got; absent = the stage does it
                    return self.reply(stage.story.act(d["n"], d["k"], d.get("given")))
                if u.path == "/api/check":
                    return self.reply(stage.story.check(d["n"]))
                if u.path == "/api/finish":
                    t = round(time.time() - stage.t_play, 1) if stage.t_play else None
                    path = stage.story.report(d.get("mode", "browser"), t)
                    return self.reply({"saved": str(path.relative_to(ST.ROOT)).replace("\\", "/")})
                if u.path == "/api/power":
                    i = int(d["i"])
                    if d["on"]:
                        return self.reply({"up_s": stage.F.start(i)})
                    stage.F.kill(i)
                    return self.reply({"off": True})
                if u.path == "/api/net":
                    stage.F.set_drop(d.get("sides"))
                    return self.reply({"drop": {str(k): sorted(v) for k, v in stage.F.drop.items()}})
                if u.path == "/api/reset":
                    stage.reset()
                    return self.reply({"ok": True})
            except Exception as e:                    # show it on the page; the stage keeps running
                return self.reply({"error": f"{type(e).__name__}: {e}"}, 500)
            self.reply({"error": "not found"}, 404)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--real-timers", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--port", type=int, default=STAGE_PORT)
    a = ap.parse_args()
    if a.headless:
        sys.exit(0 if ST.run_headless(a.real_timers) else 1)
    stage = Stage(a.real_timers, a.keep)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(stage))
    srv.daemon_threads = True
    threading.Thread(target=stage.boot, daemon=True).start()
    print(f"open http://127.0.0.1:{a.port}/  (the five boxes take about a minute to load their models; Ctrl+C stops all)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stage.F.stop()


if __name__ == "__main__":
    main()
