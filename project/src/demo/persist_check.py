"""CP25 P1-P4: three lan.py boxes on this computer (private address), each keeping its state on disk.

  .venv/Scripts/python.exe src/demo/persist_check.py [address]     -> results/cp25/persist_check.json   (about 5 min)

P1 box 0 is stopped and started again: same identity, posts, index and remaining slots; it can still delete its old
   posts (and --new-identity on box 2 draws a new identity but keeps the posts).
P2 a post after the restart in the same window reveals nothing (the used indices were kept).
P3 forgetting: a deleted post's envelope and picture are on no box's disk and in no /sync; a stale copy sent to a
   restarted box is dropped; a delete that comes before its post drops the post; a forged delete does not.
P4 remote_report.py reads the three event logs (one computer: clock offsets under 5 ms).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lan import addresses  # noqa: E402
from lan_check import call, settle, titles  # noqa: E402
import lan_check  # noqa: E402
from common import A, msg_from_json, msg_to_json  # noqa: E402
from remote_report import load, report  # noqa: E402
from sim import proto24 as p24  # noqa: E402
from sim.box import sub_key  # noqa: E402

ROOT = HERE.parents[1]
LOGS = ROOT / "logs"
STATE = LOGS / "cp25_check"
OUT = ROOT / "results" / "cp25"
PORTS = [8621, 8622, 8623]
lan_check.PORTS = PORTS
own, up = lan_check.own, lan_check.up


def stored_mids(i: int) -> set[str]:
    p = STATE / f"box{i}" / "msgs.jsonl"
    return {str(json.loads(x)["mid"]) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}


def synced_mids(i: int) -> set[str]:
    return {str(d["mid"]) for d in call(f"http://127.0.0.1:{PORTS[i]}/sync")[1]}


def gossip(i: int, m, frm: int) -> None:
    call(f"http://127.0.0.1:{PORTS[i]}/gossip", {"msg": msg_to_json(m), "frm": frm})


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else addresses()[0]
    hosts = ",".join(f"{ip}:{p}" for p in PORTS)
    shutil.rmtree(STATE, ignore_errors=True)
    STATE.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    procs: dict[int, subprocess.Popen] = {}

    def start(i, *extra):
        args = [py, str(HERE / "lan.py"), "--hosts", hosts, "--me", str(i), "--room-pass", "said on the phone",
                "--state", str(STATE / f"box{i}"), *extra]
        procs[i] = subprocess.Popen(args, stdout=open(LOGS / f"cp25_persist_box{i}.log", "a"), stderr=subprocess.STDOUT)

    def restart(i, *extra):
        procs[i].terminate()
        procs[i].wait(30)
        start(i, *extra)
        up(i)

    res = {"address": "private network address of this computer", "checks": {}}
    C = res["checks"]
    try:
        for i in range(3):
            start(i)
        for i in range(3):
            up(i)
        rgb = os.urandom(224 * 224 * 3)
        a1 = own(0, "/api/post", {"topic": A, "title": "before the restart", "body": "box 0 writes, then stops"})["mid"]
        p1 = own(0, "/api/post", {"topic": A, "title": "a photo before the restart", "body": "",
                                  "rgb": base64.b64encode(rgb).decode()})["mid"]
        b1 = own(1, "/api/post", {"topic": A, "title": "box 1 says hi", "body": ""})["mid"]
        settle([0, 1, 2], 6.0)
        own(0, "/api/vote", {"mid": b1, "dir": 1})
        settle([0, 1, 2])
        a1_env = next(d for d in call(f"http://127.0.0.1:{PORTS[2]}/sync")[1] if str(d["mid"]) == a1)
        keys = ("identity", "items", "index", "tier")
        before = own(0, "/api/stats")
        restart(0)
        settle([0, 1, 2], 6.0)
        after = own(0, "/api/stats")
        same = {k: (before[k], after[k]) for k in keys}
        left = (before["budget"]["t"]["left"], after["budget"]["t"]["left"], before["budget"]["p"]["left"],
                after["budget"]["p"]["left"])
        d1 = own(0, "/api/delete", {"mid": a1})
        d2 = own(0, "/api/delete", {"mid": p1})
        settle([0, 1, 2])
        gone = all(t not in titles(i) for i in range(3) for t in ("before the restart", "a photo before the restart"))
        C["P1_restart"] = {"same": same, "slots_left_t_p_before_after": left, "restored": after["restored"],
                           "delete_codes": [d1.get("_code"), d2.get("_code")], "old_posts_gone_everywhere": gone,
                           "pass": all(a == b for a, b in same.values()) and left[0] == left[1] and left[2] == left[3]
                           and after["restored"] > 0 and d1.get("_code") is None and d2.get("_code") is None and gone}
        w_before = int(time.time() // p24.window("new", "t"))
        a2 = own(0, "/api/post", {"topic": A, "title": "after the restart", "body": "same identity, next slot"})
        settle([0, 1, 2], 4.0)
        st = [own(i, "/api/stats") for i in range(3)]
        C["P2_no_index_reused"] = {"same_window": w_before == int(time.time() // p24.window("new", "t")),
                                   "notices": [len(s["notices"]) for s in st],
                                   "after_seen": [("after the restart" in titles(i)) for i in range(3)],
                                   "pass": "mid" in a2 and all(len(s["notices"]) == 0 for s in st)
                                   and all("after the restart" in titles(i) for i in range(3))}
        disk = [bool({a1, p1} & stored_mids(i)) for i in range(3)]
        pics = [(STATE / f"box{i}" / "img" / f"{p1}.rgb").exists() for i in range(3)]
        sync = [bool({a1, p1} & synced_mids(i)) for i in range(3)]
        restart(1)                                              # its log no longer holds a1: only the tombstone knows
        gossip(1, msg_from_json(a1_env), 2)
        settle([0, 1, 2])
        stale = {"seen": "before the restart" in titles(1), "forgotten": own(1, "/api/stats")["count"].get("forgotten", 0)}
        W = p24.World24(1, seed=int.from_bytes(os.urandom(8), "big"))
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        def post(title):
            m = W.make(0, "post", A, time.time(), {"title": title, "body": ""}, 500.0, sub_key(A), tokens=6)
            new = int.from_bytes(os.urandom(6), "big") + (1 << 48)
            W.keys[new] = W.keys.pop(m.mid)
            m.mid = new
            return m

        def delete(target, forged=False):
            real = W.keys[target]
            if forged:
                W.keys[target] = Ed25519PrivateKey.generate()
            m = W.make(0, "delete", A, time.time(), {"target": target}, 400.0, sub_key(A), target=target)
            W.keys[target] = real
            m.mid = int.from_bytes(os.urandom(6), "big") + (1 << 48)
            return m
        t1, t2 = post("a forged delete came first"), post("the author's delete came first")
        gossip(2, delete(t1.mid, forged=True), 0)
        gossip(2, delete(t2.mid), 0)
        settle([0, 1, 2])
        gossip(2, t1, 0)
        gossip(2, t2, 0)
        settle([0, 1, 2])
        early = {"forged_then_post_seen": [("a forged delete came first" in titles(i)) for i in range(3)],
                 "deleted_then_post_seen": [("the author's delete came first" in titles(i)) for i in range(3)]}
        C["P3_forgetting"] = {"on_disk": disk, "picture_on_disk": pics, "in_sync": sync, "stale_copy": stale, **early,
                              "pass": not any(disk + pics + sync) and not stale["seen"] and stale["forgotten"] >= 1
                              and all(early["forged_then_post_seen"]) and not any(early["deleted_then_post_seen"])}
        old_id, old_feed = own(2, "/api/stats"), sorted(titles(2))
        restart(2, "--new-identity")
        settle([0, 1, 2], 6.0)
        new_id, new_feed = own(2, "/api/stats"), sorted(titles(2))
        # the feed, not stats "items": items also counts deleted posts held in memory until a restart (first run)
        C["P1b_new_identity"] = {"identity": [old_id["identity"], new_id["identity"]],
                                 "items": [old_id["items"], new_id["items"]], "feed": [old_feed, new_feed],
                                 "pass": old_id["identity"] != new_id["identity"] and old_feed == new_feed}
        time.sleep(22)                                          # one more sync round: round trips for every pair
        rep = report(load([str(STATE / f"box{i}" / "events.jsonl") for i in range(3)]))
        offs = [abs(v["used"]) for v in rep["clock_offset_ms"].values()]
        C["P4_report"] = {"pairs_with_offset": len(offs), "max_abs_offset_ms": max(offs) if offs else None,
                          "text_settle_n": rep["W1_settle_text_s"]["all_other_boxes"]["n"],
                          "pass": len(offs) == 6 and max(offs) < 5 and rep["W1_settle_text_s"]["all_other_boxes"]["n"] > 0}
        res["report"] = rep
        res["pass"] = all(v["pass"] for v in C.values())
        for k, v in C.items():
            print(k, v["pass"], {x: y for x, y in v.items() if x != "pass"})
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "persist_check.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("pass" if res["pass"] else "FAIL")
    finally:
        for pr in procs.values():
            pr.terminate()


if __name__ == "__main__":
    main()
