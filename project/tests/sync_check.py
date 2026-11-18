"""CP29 layer 4: the recon exchange on real box processes, with faults injected (link cuts, a lost envelope nobody
notices, clock offsets, process kills, forged envelopes, a delete before its post, a partition).
CP30 adds scenes 9-12 (fetch dedupe on come-on, the clean-flag fence, floor skips, the fence switched off).
CP33 adds scenes 13-14 (sub B stays in sub B; sub B's two members cut apart with only non-members between them) and
judges the end state per sub (journal/cp/CP33.md). CP33_ONLY=1 runs 13-14 only -> results/tests/sync_check_33.json.

  .venv/Scripts/python.exe tests/sync_check.py      -> results/tests/sync_check.json   (about 25 min)
  CP30_SCENES=0 ...                                    -> scenes 1-8 only (the CP29 run)
  CP30_ONLY=1 ...                                      -> scenes 9-12 only            -> results/tests/sync_check_30.json

Four lan.py boxes on this computer (ports 8641-8644), --exchange recon --faults, floor period 20-40 s and probe cap
20 s so that a scene takes a minute, not half an hour (the rules are the same; cp/CP29.md layer 4). Each scene says
what must hold at the end; the same scene on the model (sim/sync.py, layer 2) is named next to it. Per box: every
verified message was verified once (no message is verified twice), recon sessions and their triggers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "box"))
sys.path.insert(0, str(HERE.parent))
import lan_check  # noqa: E402

ROOT = HERE.parent
LOGS = ROOT / "logs"
CP30 = os.environ.get("CP30_SCENES", "1") == "1"
OUT = ROOT / "results" / "tests"
PORTS = [8641, 8642, 8643, 8644]
lan_check.PORTS = PORTS
own = lan_check.own
N = len(PORTS)
PAE = (20.0, 40.0)
PMAX = 20.0
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Run:
    def __init__(self):
        self.state = LOGS / ("cp30_check" if CP30 else "cp29_check")
        shutil.rmtree(self.state, ignore_errors=True)
        self.procs = {}
        self.drop = {i: set() for i in range(N)}
        self.hosts = ",".join(f"127.0.0.1:{p}" for p in PORTS)

    def start(self, i):
        args = [sys.executable, str(ROOT / "box" / "lan.py"), "--hosts", self.hosts, "--me", str(i),
                "--state", str(self.state / f"box{i}"), "--exchange", "recon", "--push", "lazy", "--faults",
                "--pae", f"{PAE[0]},{PAE[1]}", "--probe-max", str(PMAX), "--room-pass", "cp29-test"] +                (["--floor-skip"] if CP30 else [])
        self.procs[i] = subprocess.Popen(args, stdout=open(LOGS / f"{'cp30' if CP30 else 'cp29'}_check_box{i}.log", "a"),
                                         stderr=subprocess.STDOUT)
        lan_check.up(i)
        if self.drop[i]:
            own(i, "/fault", {"drop": sorted(self.drop[i])})

    def kill(self, i):
        self.procs[i].kill()                   # a crash: no goodbye
        self.procs[i].wait(30)

    def cut(self, a, b, on=True):
        for x, y in ((a, b), (b, a)):
            (self.drop[x].add if on else self.drop[x].discard)(y)
            if self.procs.get(x) and self.procs[x].poll() is None:
                own(x, "/fault", {"drop": sorted(self.drop[x])})

    def post(self, i, title, topic=0):
        r = own(i, "/api/post", {"topic": topic, "title": title, "body": "cp29 check"})
        return r["mid"]

    def holds(self, i):
        return own(i, "/api/holds")

    def has(self, i, mid):
        return str(mid) in self.holds(i)["msgs"]

    def gone(self, i, mid):
        return str(mid) in self.holds(i)["gone"]

    def wait(self, pred, timeout=120.0):
        t = time.time()
        while time.time() - t < timeout:
            try:
                if pred():
                    return round(time.time() - t, 1)
            except Exception:
                pass
            time.sleep(1.0)
        return None

    def env(self, i, mid):
        return own(i, f"/env?mid={mid}")["msg"]

    def stop(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.kill()


def scenes(R: Run) -> list[dict]:
    out = []

    def rec(name, model, ok, **info):
        out.append({"scene": name, "model_scene": model, "pass": bool(ok), **info})
        log(f"{name:44s} {'PASS' if ok else 'FAIL'} {info}")

    # 1. a box cut off from everyone while a post goes out; the links come back without any traffic
    for j in (0, 2, 3):
        R.cut(1, j)
    x = R.post(0, "1 cut off, then back")
    time.sleep(5)
    for j in (0, 2, 3):
        R.cut(1, j, on=False)
    t = R.wait(lambda: R.has(1, x), 3 * (PMAX + PAE[1]))
    rec("silent link return (H1 class)", "silent link return / H1", t is not None, seconds=t)

    # 2. an envelope acked and lost (box 1 believes it has it; nobody notices) -> only the floor repairs it
    own(1, "/fault", {"silent": 1})
    y = R.post(0, "2 lost after ack")
    t = R.wait(lambda: R.has(1, y), 3 * PAE[1] + 30)
    rec("silent loss, then idle", "silent loss, then idle", t is not None, seconds=t,
        floor_sessions_box1=R.holds(1)["count"].get("recon_f", 0))

    # 3. author clock 110 s behind; the receiver is off when the post goes out (H3 class)
    R.kill(1)
    own(0, "/fault", {"offset": -110.0})
    z = R.post(0, "3 stamped 110 s behind")
    own(0, "/fault", {"offset": 0.0})
    time.sleep(2)
    R.start(1)
    t = R.wait(lambda: R.has(1, z), 60)
    rec("skewed author, receiver was off (H3 class)", "H3 skew beyond the catch-up margin", t is not None, seconds=t)

    # 4. a forged envelope under a real id comes first; an envelope with a broken signature comes too
    for j in (0, 2, 3):
        R.cut(1, j)
    w = R.post(0, "4 forged copies first")
    time.sleep(2)
    real = R.env(0, w)
    forged = dict(real, ct=("00" + real["ct"][2:]))               # other content, same claimed id
    badsig = dict(real, sig=("00" + real["sig"][2:]))             # same content and id, signature broken
    own(1, "/gossip", {"msg": forged, "frm": -1})
    own(1, "/gossip", {"msg": badsig, "frm": -1})
    held_bad = R.has(1, w)
    rej = R.holds(1)["count"]
    for j in (0, 2, 3):
        R.cut(1, j, on=False)
    t = R.wait(lambda: R.has(1, w), 3 * (PMAX + PAE[1]))
    rec("forged and bad-signature copies first (H8)", "H8 squatting liar, invalid first",
        t is not None and not held_bad and rej.get("rej_id", 0) >= 1 and rej.get("rej_bad_sig", 0) >= 1,
        seconds=t, rej_id=rej.get("rej_id", 0), rej_bad_sig=rej.get("rej_bad_sig", 0), held_forged=held_bad)

    # 5. a delete reaches box 1 before its post
    for j in (0, 2, 3):
        R.cut(1, j)
    p = R.post(0, "5 deleted before box 1 sees it")
    time.sleep(1)
    d = own(0, "/api/delete", {"mid": p})["mid"]
    time.sleep(1)
    own(1, "/gossip", {"msg": R.env(0, d), "frm": -1})
    gone_now = R.gone(1, p)
    for j in (0, 2, 3):
        R.cut(1, j, on=False)
    time.sleep(3 * (PMAX + PAE[1]))
    rec("delete before its post", "delete arrives before its post",
        gone_now and not R.has(1, p) and all(not R.has(i, p) for i in range(N)) and R.has(1, d),
        gone_on_arrival=gone_now, held_anywhere=[R.has(i, p) for i in range(N)])

    # 6. partition {0,1} | {2,3}, a post on each side, then heal silently
    for a in (0, 1):
        for b in (2, 3):
            R.cut(a, b)
    u = R.post(0, "6 left side")
    v = R.post(2, "6 right side")
    time.sleep(5)
    for a in (0, 1):
        for b in (2, 3):
            R.cut(a, b, on=False)
    t = R.wait(lambda: all(R.has(i, u) and R.has(i, v) for i in range(N)), 3 * (PMAX + PAE[1]))
    rec("partition heals", "partition heals", t is not None, seconds=t)

    # 7. the author's box crashes right after posting (nobody fetched it yet), comes back later
    q = R.post(3, "7 author crashed right away")
    R.kill(3)
    time.sleep(10)
    R.start(3)
    t = R.wait(lambda: all(R.has(i, q) for i in range(N)), 60)
    rec("author crashes after posting, restarts", "H5 class (holder off, then on)", t is not None, seconds=t)

    # 8. a box crashes, misses posts, restarts (state on disk)
    R.kill(2)
    r1 = R.post(0, "8 while box 2 is down")
    r2 = R.post(1, "8 while box 2 is down, too")
    time.sleep(3)
    R.start(2)
    t = R.wait(lambda: R.has(2, r1) and R.has(2, r2), 60)
    rec("crash and restart", "H2 class", t is not None, seconds=t)
    if not CP30:
        return out
    return out + scenes30(R) + scenes33(R)


def scenes33(R: Run) -> list[dict]:
    """CP33: sub B (topic 1) is held by boxes 0 and 2 only (common.community)."""
    out = []

    def rec(name, model, ok, **info):
        out.append({"scene": name, "model_scene": model, "pass": bool(ok), **info})
        log(f"{name:44s} {'PASS' if ok else 'FAIL'} {info}")

    rj = {i: R.holds(i)["count"].get("rej_sub", 0) for i in (1, 3)}
    w = R.post(0, "13 in sub B", topic=1)
    t = R.wait(lambda: R.has(2, w), 10)
    time.sleep(2 * PAE[1])
    outside = [i for i in (1, 3) if R.has(i, w)]
    rec("sub B post stays in sub B", "D1 a foreign sub never arrives", t is not None and not outside,
        seconds=t, held_outside=outside, rej_sub_delta={i: R.holds(i)["count"].get("rej_sub", 0) - rj[i] for i in (1, 3)})

    R.cut(0, 2)
    x = R.post(0, "14 sub B while 0-2 is cut", topic=1)
    time.sleep(30)
    reached = R.has(2, x)
    R.cut(0, 2, on=False)
    t = R.wait(lambda: R.has(2, x), 3 * (PMAX + PAE[1]))
    outside = [i for i in (1, 3) if R.has(i, x)]
    rec("sub B members cut apart, non-members between", "D2/D5 (members only)",
        not reached and t is not None and not outside, reached_during_cut=reached, after_link_back_s=t,
        held_outside=outside)
    return out


def scenes30(R: Run) -> list[dict]:
    out = []

    def rec(name, model, ok, **info):
        out.append({"scene": name, "model_scene": model, "pass": bool(ok), **info})
        log(f"{name:44s} {'PASS' if ok else 'FAIL'} {info}")

    # 9. CP30 K: a box comes on while three peers hold six posts it lacks; every one comes once
    R.kill(3)
    new = [R.post(i, f"9 while box 3 is down ({i}.{j})") for i in (0, 1, 2) for j in (0, 1)]
    time.sleep(2)
    R.start(3)
    c0 = R.holds(3)["count"]
    t = R.wait(lambda: all(R.has(3, x) for x in new), 60)
    time.sleep(5)
    c1 = R.holds(3)["count"]
    rx = c1.get("env_rx", 0)
    rec("come on with 3 peers: each missing post comes once (K)", "K, come-on duplicates",
        t is not None and c1.get("dup", 0) == c0.get("dup", 0), seconds=t, dup_after_start=c1.get("dup", 0),
        fetched=c1.get("recon_fetched", 0), envelopes_in=rx, offer_want=c1.get("offer_want", 0))

    # 10/12. CP30 F', the fence: box 1 is cut from 2 and 3 (and has settled); box 0's session with box 1 is held after
    # round 1 ("equal"); meanwhile box 0 takes W from box 2 (and its push to box 1 is lost). Judged on box 0's state:
    # with the fence the session must not set clean(1) (fence_held + 1); with the fence off it sets it (the mutation).
    def held_session(tag, no_fence):
        own(0, "/fault", {"no_fence": no_fence})
        for j in (2, 3):
            R.cut(1, j)
        time.sleep(2 * PAE[1] + 2 * PMAX)                  # (e) sessions after the cut are over; pairs settle
        R.wait(lambda: 1 not in R.holds(0)["busy"], 60)
        h0 = R.holds(0)["count"].get("fence_held", 0)
        own(1, "/fault", {"silent": 1})
        own(0, "/fault", {"stall": {"peer": 1, "s": 8}, "recon_now": 1})
        time.sleep(2)
        w = R.post(2, f"{tag} during a held session")
        time.sleep(10)                                     # the held session has ended
        h = R.holds(0)
        st = {"fence_held_delta": h["count"].get("fence_held", 0) - h0, "box0_clean_has_1": 1 in h["clean"]}
        t = R.wait(lambda: R.has(1, w), 3 * PAE[1] + PMAX)
        st["box1_got_W_s"] = t
        own(0, "/fault", {"no_fence": False})
        for j in (2, 3):
            R.cut(1, j, on=False)
        R.wait(lambda: R.has(1, w), 3 * (PMAX + PAE[1]))
        return st

    st = held_session("10", False)
    rec("held session, change meanwhile: fence keeps clean(1) off", "F fence (layer 2: third-party change)",
        st["fence_held_delta"] >= 1 and not st["box0_clean_has_1"], **st)
    st = held_session("12", True)
    rec("same with the fence off (mutation): clean(1) set wrongly", "mutation of F",
        st["fence_held_delta"] == 0 and st["box0_clean_has_1"], **st)

    # 11. CP30 F': a quiet spell: floor checks are skipped for pairs that spoke lately (sessions only for quiet pairs)
    time.sleep(2 * PAE[1])
    f0 = {i: R.holds(i)["count"].get("recon_f", 0) for i in range(N)}
    s0 = {i: R.holds(i)["count"].get("floor_skip", 0) for i in range(N)}
    time.sleep(3 * PAE[1])
    f1 = {i: R.holds(i)["count"].get("recon_f", 0) - f0[i] for i in range(N)}
    s1 = {i: R.holds(i)["count"].get("floor_skip", 0) - s0[i] for i in range(N)}
    rec("quiet: some floor checks skipped (F')", "F clean, then quiet (F' re-checks silent pairs)",
        sum(s1.values()) > 0, floor_sessions=f1, floor_skips=s1)
    return out


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    R = Run()
    try:
        for i in range(N):
            R.start(i)
        time.sleep(5)
        res = (scenes30(R) if os.environ.get("CP30_ONLY") else scenes33(R) if os.environ.get("CP33_ONLY")
               else scenes(R))
        time.sleep(PAE[1] + 5)
        hs = {i: R.holds(i) for i in range(N)}         # CP33: per sub, members equal and non-members hold none
        subs = sorted({s for i in range(N) for s in hs[i]["subs"]})
        same = True
        for s in subs:
            mem = [i for i in range(N) if s in hs[i]["subs"]]
            ks = {i: sorted(map(tuple, hs[i]["keys_by_sub"].get(str(s), []))) for i in range(N)}
            same &= all(ks[i] == ks[mem[0]] for i in mem) and all(not ks[i] for i in range(N) if i not in mem)
        per_box = {}
        for i in range(N):
            dbg = own(i, "/api/debug")
            ver = dbg["verified"]
            c = R.holds(i)["count"]
            per_box[i] = {"verified": len(ver), "verified_twice": len(ver) - len(set(ver)),
                          "recon": {k: v for k, v in c.items() if k.startswith("recon_")},
                          "rejected": {k: v for k, v in c.items() if k.startswith("rej_")},
                          "dup": c.get("dup", 0)}
    finally:
        R.stop()
    out = {"scenes": res, "final_sets_equal": same, "per_box": per_box,
           "all_pass": all(s["pass"] for s in res) and same and all(b["verified_twice"] == 0 for b in per_box.values()),
           "settings": {"boxes": N, "pae_s": PAE, "probe_max_s": PMAX}}
    name = ("sync_check_30.json" if os.environ.get("CP30_ONLY") else "sync_check_33.json" if os.environ.get("CP33_ONLY")
            else "sync_check.json")
    (OUT / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log("final sets equal:", same, "| verified twice:", {i: b["verified_twice"] for i, b in per_box.items()})
    log("ALL PASS" if out["all_pass"] else "SOME FAILED")


if __name__ == "__main__":
    main()
