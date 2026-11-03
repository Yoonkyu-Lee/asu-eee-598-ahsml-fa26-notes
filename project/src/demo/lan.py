"""Run one CP24 box on this computer, as one member of a community spread over several computers (CP24).

Everyone runs the same command with the same address list; only --me (and the room password) differ:

  python src/demo/lan.py --whoami                                     # this computer's addresses on the network
  python src/demo/lan.py --hosts 172.20.10.2,172.20.10.3,172.20.10.4 --me 0 --room-pass "said in person"
  python src/demo/lan.py --hosts ... --me 0 --check                   # can I reach the others? (boxes must be running)

Then open http://127.0.0.1:8601/ on this computer. The mesh and the subs come from the address list (no file to pass
around). Your identity secret is drawn by your box and never leaves it. The room key is derived from the room password:
whoever was told the password can read room R, the other boxes only relay it. Everyone starts in the new group
(the community was formed today); --old 0,2 puts those boxes in the old group (must be the same on every computer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import community, save_cfg  # noqa: E402

ROOT = HERE.parents[1]
LOGS = ROOT / "logs"


def addresses() -> list[str]:
    out = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # no packet is sent; picks the interface of the default route
        out.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        out |= {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        pass
    return sorted(a for a in out if not a.startswith("127."))


def check(cfg: dict, me: int) -> bool:
    ok = True
    for i, a in enumerate(cfg["addr"]):
        if i == me:
            continue
        try:
            r = json.loads(urllib.request.urlopen(f"http://{a}/hello", timeout=5).read())
            print(f"box {i} at {a}: reachable ({r['proto']}, says it is box {r['box']})")
            ok &= r["box"] == i
        except Exception as e:
            ok = False
            print(f"box {i} at {a}: NOT reachable ({type(e).__name__}: {e})")
    if not ok:
        print("hints: same Wi-Fi or hotspot? campus Wi-Fi often blocks laptop-to-laptop traffic (use a phone hotspot);"
              " allow Python through the firewall (Windows: Private networks; macOS: accept incoming connections);"
              " is that box running, and is its --hosts list the same as yours?")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", help="everyone's address, same order on every computer")
    ap.add_argument("--me", type=int, help="my position in --hosts")
    ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--room-pass", default="", help="room R password (empty: this box only relays the room)")
    ap.add_argument("--old", default="", help="boxes in the old group, e.g. 0,2")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--whoami", action="store_true")
    a = ap.parse_args()
    if a.whoami:
        print("this computer:", ", ".join(addresses()) or "no network address found")
        return
    if a.hosts is None or a.me is None:
        ap.error("--hosts and --me are needed")
    entries = [x.strip() for x in a.hosts.split(",") if x.strip()]      # "ip" or "ip:port"
    hosts = [x.rsplit(":", 1)[0] if ":" in x else x for x in entries]
    ports = [int(x.rsplit(":", 1)[1]) if ":" in x else a.port for x in entries]
    n = len(hosts)
    if not 0 <= a.me < n:
        ap.error(f"--me must be 0..{n - 1}")
    old = {int(x) for x in a.old.split(",") if x.strip()}
    cfg = community(n, D=2, seed=24, base_port=a.port, proto="cp24", hosts=hosts, room=list(range(n)),
                    tiers=["old" if i in old else "new" for i in range(n)])
    cfg["ports"], cfg["addr"] = ports, [f"{x}:{p}" for x, p in zip(hosts, ports)]
    cfg["room_key"] = hashlib.blake2b(b"room-R|" + a.room_pass.encode(), digest_size=32).hexdigest() if a.room_pass else None
    cfg["sync"] = True
    if a.check:
        sys.exit(0 if check(cfg, a.me) else 1)
    mine = set(addresses())
    if hosts[a.me] not in mine and not hosts[a.me].startswith("127."):
        print(f"warning: {hosts[a.me]} is not one of this computer's addresses ({', '.join(sorted(mine))})")
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f"lan_cfg_box{a.me}.json"           # holds the room key: kept out of git
    save_cfg(cfg, path)
    print(f"box {a.me} of {n}, {cfg['tiers'][a.me]} group, room R: {'member' if a.room_pass else 'relay only'}")
    print(f"open http://127.0.0.1:{ports[a.me]}/ on this computer (Ctrl+C stops the box)", flush=True)
    import box_server                                  # same process, so Ctrl+C (or closing the window) stops the box
    sys.argv = [sys.argv[0], "--cfg", str(path), "--id", str(a.me), "--bind", "0.0.0.0"]
    try:
        box_server.main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
