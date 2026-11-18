"""CP26 rule D: five lan.py boxes on this computer, box 0 always on (except one outage), boxes 1-4 coming and going,
run once with the event-driven exchange (x2) and once with CP24's polling, and the same scene replayed on the model
(sim.exchange, script mode).

  .venv/Scripts/python.exe src/demo/exchange_check.py [address]   -> results/cp26/exchange_check.json   (about 15 min)

Per box: catch-up requests, envelopes in catch-up answers, ids offered, ids wanted (x2); polling requests, envelopes
and bytes (poll). In the end every box holds every message (completeness), and every box's feed is the same.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from lan import addresses  # noqa: E402
import lan_check  # noqa: E402
from sim.exchange import Exchange, Scene  # noqa: E402

ROOT = HERE.parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "cp26"
PORTS = [8631, 8632, 8633, 8634, 8635]
lan_check.PORTS = PORTS
own, up = lan_check.own, lan_check.up
KEYS = ("catch_req", "catch_env", "offer_ids", "want_ids", "sync_req", "sync_env", "sync_bytes", "catch_bytes", "hello",
        "send", "dup", "ihave_rx", "env_rx", "ihave_sent")
PUSH = "--eager" not in sys.argv          # CP28: ids first (lan.py default); --eager = CP26's run

# (action, box): on / off / post. Box 0 is the always-on box; it has one outage while boxes 1 and 3 write.
SCENE = [("on", 0), ("on", 1), ("post", 0), ("post", 1), ("off", 1),
         ("on", 2), ("post", 0), ("post", 2), ("off", 2),
         ("on", 1), ("on", 3), ("post", 3),
         ("off", 0), ("post", 1), ("post", 3), ("off", 1),
         ("on", 0), ("post", 0), ("off", 3),
         ("on", 4), ("post", 4), ("on", 2), ("on", 1), ("on", 3)]


def run(ip: str, exchange: str) -> dict:
    hosts = ",".join(f"{ip}:{p}" for p in PORTS)
    state = LOGS / f"cp26_check_{exchange}"
    shutil.rmtree(state, ignore_errors=True)
    py, procs, t0, script, posts = sys.executable, {}, None, [], 0

    def start(i):
        args = [py, str(HERE / "lan.py"), "--hosts", hosts, "--me", str(i), "--state", str(state / f"box{i}"),
                "--exchange", exchange, "--push", "lazy" if PUSH else "eager"]
        procs[i] = subprocess.Popen(args, stdout=open(LOGS / f"cp26_check_{exchange}_box{i}.log", "a"),
                                    stderr=subprocess.STDOUT)
        up(i)

    totals = {i: {} for i in range(5)}

    def bank(i):                                                # counters restart with the process: keep a running sum
        c = own(i, "/api/stats")["count"]
        for k in KEYS:
            totals[i][k] = totals[i].get(k, 0) + c.get(k, 0)

    try:
        for act, i in SCENE:              # times are taken right after each action (run 3; runs 1-2 took them after
            if act == "on":               # the pauses, so the model saw a post and the next "off" at once)
                start(i)
            elif act == "off":
                bank(i)
                procs[i].terminate()
                procs[i].wait(30)
            else:
                posts += 1
                own(i, "/api/post", {"topic": 0, "title": f"post {posts} by box {i}", "body": ""})
            t0 = t0 or time.time()
            script.append((time.time() - t0, act if act != "post" else "pub", i))
            time.sleep(3 if act == "on" else 2 if act == "post" else 0)
        time.sleep(25 if exchange == "poll" else 8)
        lan_check.settle(range(5))
        feeds = {i: sorted(lan_check.titles(i)) for i in range(5)}
        for i in range(5):
            bank(i)
    finally:
        for pr in procs.values():
            pr.terminate()
    want = sorted(f"post {k} by box {i}" for k, (a, i) in enumerate([x for x in SCENE if x[0] == "post"], 1))
    return {"exchange": exchange, "per_box": totals, "feeds_complete": [feeds[i] == want for i in range(5)],
            "posts": posts, "script": script}


def model(script: list) -> dict:
    end = script[-1][0] + 60
    x = Exchange(Scene(n=5, days=end / 86400, p_loss=0.0, grace_s=1.0, push="lazy" if PUSH else "eager"), "X2",
                 script=[(round(t, 3), k, b) for t, k, b in script])
    r = x.run()
    return {"per_box": {b: dict(x.ops[b]) for b in range(5)}, "missing_pairs": r["missing_pairs"]}


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else addresses()[0]
    LOGS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"address": "this computer's network address", "push": "lazy" if PUSH else "eager", "x2": run(ip, "x2")}
    res["model"] = model(res["x2"]["script"])
    res["poll"] = run(ip, "poll") if "--no-poll" not in sys.argv else None
    diff = []
    for b in range(5):
        for k in ("catch_req", "catch_env", "offer_ids", "want_ids") + (("ihave_rx", "env_rx") if PUSH else ()):
            demo, mod = res["x2"]["per_box"][b].get(k, 0), res["model"]["per_box"][b].get(k, 0)
            if demo != mod:
                diff.append([b, k, demo, mod])
    res["x2_vs_model_differences"] = diff
    res["pass"] = {"x2_complete": all(res["x2"]["feeds_complete"]),
                   "poll_complete": res["poll"] is None or all(res["poll"]["feeds_complete"]),
                   "model_complete": res["model"]["missing_pairs"] == 0, "x2_equals_model": not diff}
    out = Path(next((a for a in sys.argv[2:] if a.endswith(".json")), OUT / "exchange_check.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in ("x2",) + (("poll",) if res["poll"] else ()):
        print(k, res[k]["feeds_complete"], {b: {kk: v for kk, v in c.items() if v} for b, c in res[k]["per_box"].items()})
    print("model", res["model"])
    print("differences", diff)
    print("pass", res["pass"])


if __name__ == "__main__":
    main()
