"""CP24 R3: three lan.py boxes on this computer, called over its network address as if on three computers.

  .venv/Scripts/python.exe src/demo/lan_check.py [address]     -> results/cp24/lan_check.json   (about 2 min)

Boxes 0 and 1 know the room password, box 2 only relays the room. Box 2 starts late (catch-up by sync). Checks:
L1 a post and an uploaded photo reach every box, L2 votes and the author's delete reach every box, L3 room R only
for the password holders, L4 owner calls from the network address are refused and peer calls answered, L5 the late
box catches up, L6 breaking the limit gets the identity revealed on every box and its later posts dropped,
L7 the honest client is told to wait once its picture slots are used up.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lan import addresses  # noqa: E402

ROOT = HERE.parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "cp24"
PORTS = [8611, 8612, 8613]


def call(url: str, obj: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=None if obj is None else json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def own(i: int, path: str, obj: dict | None = None) -> dict:
    code, r = call(f"http://127.0.0.1:{PORTS[i]}{path}", obj)
    return r if code == 200 else {"_code": code, **r}


def up(i: int) -> None:
    for _ in range(600):
        try:
            own(i, "/api/stats")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"box {i} did not start")


def settle(boxes, s: float = 4.0) -> None:
    time.sleep(s)
    for _ in range(200):
        if all(own(i, "/api/stats")["pending"] == 0 for i in boxes):
            return
        time.sleep(0.2)


def titles(i: int) -> list[str]:
    return [x["title"] for x in own(i, "/api/feed")]


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else addresses()[0]
    hosts = ",".join(f"{ip}:{p}" for p in PORTS)
    LOGS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    def start(i, room):
        args = [py, str(HERE / "lan.py"), "--hosts", hosts, "--me", str(i)] + (["--room-pass", "said in person"] if room else [])
        return subprocess.Popen(args, stdout=open(LOGS / f"cp24_lan_box{i}.log", "w"), stderr=subprocess.STDOUT)

    procs = [start(0, True), start(1, True)]
    res = {"address": "private network address of this computer", "checks": {}}
    C = res["checks"]
    try:
        up(0), up(1)
        rng = os.urandom(224 * 224 * 3)
        a = own(0, "/api/post", {"topic": 0, "title": "hello from box 0", "body": "first post on the network"})
        p = own(1, "/api/post", {"topic": 0, "title": "a photo from box 1", "body": "", "rgb": base64.b64encode(rng).decode()})
        own(0, "/api/room", {"body": "a room note only the password holders read"})
        settle([0, 1])
        procs.append(start(2, False))                           # the late joiner
        up(2)
        settle([0, 1, 2], 25.0)                                 # first sync pass + pulls
        C["L5_late_box_caught_up"] = {"feed": titles(2),
                                      "pass": {"hello from box 0", "a photo from box 1"} <= set(titles(2))}
        img = [own(i, f"/api/img?mid={p['mid']}")["rgb"] for i in range(3)]
        C["L1_post_and_photo_everywhere"] = {"pass": all("hello from box 0" in titles(i) for i in range(3))
                                             and all(x == base64.b64encode(rng).decode() for x in img)}
        own(2, "/api/vote", {"mid": a["mid"], "dir": 1})
        own(1, "/api/vote", {"mid": a["mid"], "dir": 1})
        settle([0, 1, 2])
        scores = [next(x["score"] for x in own(i, "/api/feed") if x["mid"] == a["mid"]) for i in range(3)]
        not_author = own(2, "/api/delete", {"mid": a["mid"]})
        own(0, "/api/delete", {"mid": a["mid"]})
        settle([0, 1, 2])
        C["L2_votes_and_delete"] = {"scores": scores, "non_author_delete": not_author.get("_code"),
                                    "pass": scores == [2, 2, 2] and not_author.get("_code") == 403
                                    and all("hello from box 0" not in titles(i) for i in range(3))}
        rooms = [own(i, "/api/room") for i in range(3)]
        C["L3_room"] = {"members": [r["member"] for r in rooms], "notes": [len(r["items"]) for r in rooms],
                        "box2_send": own(2, "/api/room", {"body": "x"}).get("_code"),
                        "pass": [r["member"] for r in rooms] == [True, True, False] and [len(r["items"]) for r in rooms][:2] == [1, 1]}
        c1, _ = call(f"http://{ip}:{PORTS[0]}/api/stats")
        c2, _ = call(f"http://{ip}:{PORTS[0]}/api/post", {"topic": 0, "title": "x"})
        c3, h = call(f"http://{ip}:{PORTS[0]}/hello")
        C["L4_access"] = {"owner_get_from_network": c1, "owner_post_from_network": c2, "peer_hello": c3,
                          "pass": c1 == 403 and c2 == 403 and c3 == 200}
        own(2, "/api/misbehave", {})
        settle([0, 1, 2], 6.0)
        st = [own(i, "/api/stats") for i in range(3)]
        own(2, "/api/post", {"topic": 0, "title": "box 2 after being revealed", "body": ""})
        settle([0, 1, 2])
        C["L6_slashing"] = {"notices": [len(s["notices"]) for s in st], "slashed": [s["slashed"] for s in st],
                            "after_seen_by": [i for i in (0, 1) if "box 2 after being revealed" in titles(i)],
                            "pass": all(len(s["notices"]) == 1 for s in st) and [s["slashed"] for s in st] == [False, False, True]
                            and not any("box 2 after being revealed" in titles(i) for i in (0, 1))}
        waits = [own(0, "/api/post", {"topic": 0, "title": f"photo {k}", "body": "",
                                      "rgb": base64.b64encode(os.urandom(224 * 224 * 3)).decode()}).get("_code") for k in range(3)]
        C["L7_picture_limit"] = {"codes": waits, "budget": own(0, "/api/stats")["budget"]["p"],
                                 "pass": waits == [None, None, 429]}
        res["stats"] = {i: {k: v for k, v in s.items() if k in ("count", "tier", "proto", "items")} for i, s in enumerate(st)}
        res["pass"] = all(v["pass"] for v in C.values())
        for k, v in C.items():
            print(k, v["pass"], {x: y for x, y in v.items() if x != "pass"})
        (OUT / "lan_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("pass" if res["pass"] else "FAIL")
    finally:
        for pr in procs:
            pr.terminate()


if __name__ == "__main__":
    main()
