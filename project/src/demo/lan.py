"""Run one CP24 box on this computer, as one member of a community spread over several computers (CP24).

Everyone runs the same command with the same address list; only --me (and the room password) differ:

  python src/demo/lan.py --whoami                                     # this computer's addresses on the network
  python src/demo/lan.py --hosts 172.20.10.2,172.20.10.3,172.20.10.4 --me 0 --room-pass "said in person"
  python src/demo/lan.py --hosts ... --me 0 --check                   # can I reach the others? (boxes must be running)

Then open http://127.0.0.1:8601/ on this computer. The mesh and the subs come from the address list (no file to pass
around). Your identity secret is drawn by your box and never leaves it. The room key is derived from the room password:
whoever was told the password can read room R, the other boxes only relay it. Everyone starts in the new group
(the community was formed today); --old 0,2 puts those boxes in the old group (must be the same on every computer).

CP25: the box keeps its state in state/box<me>/ (identity, keys, messages, pictures; git-ignored), so stopping and
starting it again keeps the same identity and posts. --new-identity retires the stored identity (e.g. after it was
revealed for breaking the limit); the posts it holds stay. Over Tailscale, use everyone's 100.x.y.z address
(--whoami lists it when Tailscale is running).

CP26: boxes exchange by events (--exchange x2, default): a box that comes on offers what it took last time and gets
what it missed from one box; nothing is polled. --exchange poll is the CP24 way (every 20 s, every box, everything).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
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
    return sorted((a for a in out | set(tailscale_ip()) if not a.startswith("127.")), key=lambda a: (is_tailscale(a), a))


def is_tailscale(a: str) -> bool:
    """Tailscale hands out addresses in 100.64.0.0/10."""
    p = a.split(".")
    return len(p) == 4 and p[0] == "100" and 64 <= int(p[1]) <= 127


def tailscale_ip() -> list[str]:
    for exe in ("tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                r"C:\Program Files\Tailscale\tailscale.exe"):
        try:
            r = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            return [x.strip() for x in r.stdout.split() if x.strip()]
    return []


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
    ap.add_argument("--state", help="state folder (default state/box<me>)")
    ap.add_argument("--new-identity", action="store_true", help="retire the stored identity and draw a new one")
    ap.add_argument("--push", choices=("lazy", "eager"), default="lazy",
                    help="lazy: ids first, envelopes only on request (CP27 P2, CP28); eager: envelopes to every peer")
    ap.add_argument("--text", choices=("fp32", "int8"), default="fp32",
                    help="fp32: contract precision (CP3, CP27); int8: CP21-CP26 boxes")
    ap.add_argument("--exchange", choices=("x2", "poll"), default="x2",
                    help="x2: catch up when the box comes on (CP26); poll: ask every box for everything every 20 s (CP24)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--whoami", action="store_true")
    a = ap.parse_args()
    if a.whoami:
        found = addresses()
        print("this computer:", ", ".join(x + (" (Tailscale)" if is_tailscale(x) else "") for x in found)
              or "no network address found")
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
    cfg = community(n, D=6, seed=24, base_port=a.port, proto="cp24", hosts=hosts, room=list(range(n)),
                    tiers=["old" if i in old else "new" for i in range(n)])
    cfg["ports"], cfg["addr"] = ports, [f"{x}:{p}" for x, p in zip(hosts, ports)]
    cfg["room_key"] = hashlib.blake2b(b"room-R|" + a.room_pass.encode(), digest_size=32).hexdigest() if a.room_pass else None
    cfg["sync"] = True                                  # several computers: a picture is pulled from any box
    cfg["exchange"] = a.exchange
    cfg["push"], cfg["text"] = a.push, a.text          # CP28; mesh D = 6 as in DayNet (all peers for n <= 7)
    if a.check:
        sys.exit(0 if check(cfg, a.me) else 1)
    mine = set(addresses())
    if hosts[a.me] not in mine and not hosts[a.me].startswith("127."):
        print(f"warning: {hosts[a.me]} is not one of this computer's addresses ({', '.join(sorted(mine))})")
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f"lan_cfg_box{a.me}.json"           # holds the room key: kept out of git
    save_cfg(cfg, path)
    state = Path(a.state) if a.state else ROOT / "state" / f"box{a.me}"
    if a.new_identity:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("identity.json", "keys.json"):
            if (state / name).exists():
                (state / name).rename(state / f"retired-{stamp}-{name}")
        print(f"retired the stored identity in {state}; a new one is drawn now")
    print(f"box {a.me} of {n}, {cfg['tiers'][a.me]} group, room R: {'member' if a.room_pass else 'relay only'}, "
          f"state kept in {state}")
    print(f"open http://127.0.0.1:{ports[a.me]}/ on this computer (Ctrl+C stops the box)", flush=True)
    import box_server                                  # same process, so Ctrl+C (or closing the window) stops the box
    sys.argv = [sys.argv[0], "--cfg", str(path), "--id", str(a.me), "--bind", "0.0.0.0", "--state", str(state)]
    try:
        box_server.main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
