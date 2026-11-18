"""CP32: the story the one-screen demo plays, its checks, and the five box processes (journal/cp/CP32.md).

Five users, one box each, on 127.0.0.1:8621-8625 (lan.py, the current design: recon S2+KE, ids-first push).
A step is a list of acts and one check. An act is either something a user does in their own client
(post, search, room, vote, delete, delete_other, misbehave) or something done to the world (power, split, heal,
isolate, forge). The browser stage (stage.py) plays the user acts through the real client UI; the headless run
calls the same owner API the client would call. Checks only read each box's owner API (an observer, not protocol
traffic).

  .venv/Scripts/python.exe box/stage.py --headless      -> results/demo/story_headless.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "demo"
STATE = ROOT / "state" / "stage"
BASE_PORT = 8621
N = 5
ROOM_PASS = "said at the stage"           # demo only: the room password the room members were told
FAST = {"pae_s": (20.0, 40.0), "probe_max_s": 20.0}     # the test timers of tests/sync_check.py (labelled on screen)
REAL = {"pae_s": (600.0, 1800.0), "probe_max_s": 300.0}  # the box defaults
PORTS = [BASE_PORT + i for i in range(N)]

PERSONAS = [
    {"name": "Ana", "subs": "A, B", "room": True,
     "role": {"ko": "평범한 작성자. 나중에 자기 글을 지움", "en": "an ordinary writer; later deletes a post"}},
    {"name": "Ben", "subs": "A", "room": True,
     "role": {"ko": "방에 쓰고 투표함", "en": "writes in the room, votes"}},
    {"name": "Cho", "subs": "A, B", "room": False,
     "role": {"ko": "검색하고 사진 올림. 방은 중계만 (암호 모름)", "en": "searches, posts a photo; only relays the room"}},
    {"name": "Dev", "subs": "A", "room": True,
     "role": {"ko": "노트북. 꺼졌다 켜짐", "en": "a laptop that goes off and on"}},
    {"name": "Eve", "subs": "A, B", "room": False,
     "role": {"ko": "한도를 어기는 사람", "en": "breaks the rate limit"}},
]
UI_ACTS = {"post", "search", "room", "vote", "delete", "delete_other", "misbehave"}
ALL = list(range(N))


def S(ko: str, en: str) -> dict:
    return {"ko": ko, "en": en}


STEPS = [
    {"say": S("Ana가 서브 A에 글을 씀. 글에는 작성자 이름이 없음: 다른 화면에는 그냥 '글'로 뜸.",
              "Ana posts in sub A. Posts carry no author name: on the other screens it is just a post."),
     "acts": [{"who": 0, "do": "post", "topic": 0, "title": "Bike lanes on Mill Ave are finally open",
               "body": "Rode to campus today without a single car door.", "label": "p1"}],
     "check": {"fn": "held", "label": "p1", "on": ALL, "t": 10}},
    {"say": S("Cho가 '자전거 출근'으로 검색. 검색은 Cho의 박스가 자기 색인에서 함 (서버 없음).",
              "Cho searches 'cycling to school'. Cho's own box searches its own index (no server)."),
     "acts": [{"who": 2, "do": "search", "q": "cycling to school", "label": "s2"}],
     "check": {"fn": "search", "act": "s2", "label": "p1", "top": 1}},
    {"say": S("Ana가 서브 B에 글. 서브 B를 가진 박스(Ana·Cho·Eve)만 받고, Ben·Dev 박스는 아예 안 받음.",
              "Ana posts in sub B. Only the boxes holding sub B (Ana, Cho, Eve) get it; Ben's and Dev's never do."),
     "acts": [{"who": 0, "do": "post", "topic": 1, "title": "Sub B only: board game night on Friday",
               "body": "Bring your own snacks.", "label": "p3"}],
     "check": {"fn": "held", "label": "p3", "on": [0, 2, 4], "off": [1, 3], "t": 10, "grace": 5},
     "known": S("알려진 구멍 H12 (CP32에서 찾음): 집합 맞추기(CP29~)가 서브를 안 가리고 모든 id를 맞춰서, 서브 B에 없는 박스도 "
                "서브 B 글을 받아 검증·저장함 (열쇠가 없어 화면엔 안 뜸). 설계(CP16 P-sub)는 구독한 서브만 복제.",
                "Known hole H12 (found in CP32): set reconciliation (since CP29) reconciles every id regardless of sub, so "
                "boxes outside sub B also receive, verify and store sub-B posts (without the key they never show them). "
                "The design (CP16 P-sub) replicates only the subs a box joined.")},
    {"say": S("Ben이 방 R에 씀. 암호를 들은 Ana·Ben·Dev만 읽음. Cho·Eve 박스는 암호문 봉투를 들고 전달만 함.",
              "Ben writes in room R. Only Ana, Ben and Dev (told the password) can read it. Cho's and Eve's boxes "
              "carry the encrypted envelope and cannot open it."),
     "acts": [{"who": 1, "do": "room", "body": "Room: meet at 7 by the library", "label": "r4"}],
     "check": {"fn": "room", "label": "r4", "body": "Room: meet at 7 by the library", "read": [0, 1, 3],
               "relay": [2, 4], "t": 10}},
    {"say": S("Dev가 노트북을 닫음. 박스 3이 꺼짐 (작별 인사 없이, 프로세스를 죽임).",
              "Dev closes the laptop. Box 3 goes off (no goodbye: the process is killed)."),
     "acts": [{"who": 3, "do": "power", "on": False}],
     "check": {"fn": "down", "who": 3}},
    {"say": S("Dev가 없는 동안 Cho가 사진 글을 올림 (COCO 사진). 켜진 네 박스가 받음.",
              "While Dev is away, Cho posts a photo (a COCO picture). The four boxes that are on get it."),
     "acts": [{"who": 2, "do": "post", "topic": 0, "title": "A photo while Dev is away", "body": "", "photo": True,
               "label": "p6"}],
     "check": {"fn": "held", "label": "p6", "on": [0, 1, 2, 4], "t": 10}},
    {"say": S("Ben이 그 사진에 ▲. 투표도 메시지라 켜진 박스마다 점수를 스스로 셈.",
              "Ben upvotes the photo. A vote is a message too: every box that is on counts the score itself."),
     "acts": [{"who": 1, "do": "vote", "target": "p6", "dir": 1, "label": "v7"}],
     "check": {"fn": "score", "label": "p6", "value": 1, "on": [0, 1, 2, 4], "t": 10}},
    {"say": S("Dev가 노트북을 엶. 박스 3이 켜지자마자 이웃들과 집합을 맞춰 놓친 사진 글과 투표를 받음.",
              "Dev opens the laptop. As soon as box 3 is on it reconciles with its neighbours and gets the missed "
              "photo post and vote."),
     "acts": [{"who": 3, "do": "power", "on": True}],
     "check": {"fn": "caught", "who": 3, "labels": ["p6"], "score": ["p6", 1], "t": 30}},
    {"say": S("Ana가 1번 글을 지움. 다섯 박스 모두가 원본을 버리고, 다시 와도 안 받도록 id를 기억함 (망각).",
              "Ana deletes the first post. All five boxes drop the original and remember its id so it never "
              "comes back (forgetting)."),
     "acts": [{"who": 0, "do": "delete", "target": "p1", "label": "d9"}],
     "check": {"fn": "gone", "label": "p1", "on": ALL, "t": 10}},
    {"say": S("Eve가 Cho의 사진을 지우려 함. 남의 글엔 지우기 버튼이 없고, 박스에 직접 요청해도 거절 (작성자 키 없음). "
              "운영자도 없으니 누구도 남의 글을 못 지움.",
              "Eve tries to delete Cho's photo. There is no delete button on someone else's post, and asking the box "
              "directly is refused (no author key). With no operator, nobody can delete someone else's post."),
     "acts": [{"who": 4, "do": "delete_other", "target": "p6", "label": "x10"}],
     "check": {"fn": "refused", "act": "x10", "label": "p6", "on": ALL}},
    {"say": S("망이 둘로 쪼개짐: {Ana, Ben} | {Cho, Dev, Eve}. 양쪽에서 한 명씩 글을 씀. 반대쪽엔 안 감.",
              "The network splits: {Ana, Ben} | {Cho, Dev, Eve}. One person on each side posts. Nothing crosses."),
     "acts": [{"do": "split", "sides": [[0, 1], [2, 3, 4]]},
              {"who": 0, "do": "post", "topic": 0, "title": "Left side: the library closes early today", "body": "",
               "label": "p11a"},
              {"who": 3, "do": "post", "topic": 0, "title": "Right side: rain until noon", "body": "",
               "label": "p11b"}],
     "check": {"fn": "split", "parts": [["p11a", [0, 1], [2, 3, 4]], ["p11b", [2, 3, 4], [0, 1]]], "t": 10,
               "grace": 3}},
    {"say": S("망이 다시 붙음. 아무도 알려 주지 않아도 박스들이 다시 만나 서로 없는 것만 주고받음.",
              "The network heals. Nobody announces it; the boxes meet again and exchange only what the other lacks."),
     "acts": [{"do": "heal"}],
     "check": {"fn": "held_all", "labels": ["p11a", "p11b"], "on": ALL, "t": 180}},
    {"say": S("공격자: Ben 박스가 잠깐 떨어진 사이 Dev가 글을 쓰고, 공격자가 그 글의 위조본(같은 id, 다른 내용)을 "
              "Ben 박스에 먼저 넣음. Ben 박스는 id가 내용의 해시가 아니라서 거절하고, 다시 붙으면 진짜를 받음.",
              "An attacker: while Ben's box is cut off, Dev posts, and the attacker hands Ben's box a forged copy "
              "first (same id, other content). Ben's box refuses it (the id is not the content's hash) and gets the "
              "real one when the links come back."),
     "acts": [{"do": "isolate", "who": 1},
              {"who": 3, "do": "post", "topic": 0, "title": "Dev: the meet-up moved to 8 pm", "body": "",
               "label": "p13"},
              {"do": "forge", "victim": 1, "author": 3, "target": "p13", "label": "f13"},
              {"do": "heal"}],
     "check": {"fn": "forged", "act": "f13", "label": "p13", "victim": 1, "t": 180}},
    {"say": S("Eve가 한도를 어김: 같은 메시지 번호로 글 둘. 둘을 받은 박스는 누구든 Eve의 신원 비밀을 계산해 공개하고, "
              "모든 박스가 Eve를 그룹에서 뺌. 운영자의 차단이 아니라 수학.",
              "Eve breaks the limit: two posts on the same message index. Any box holding both can compute Eve's "
              "identity secret and publish it, and every box removes Eve from the group. Not an operator's ban: math."),
     "acts": [{"who": 4, "do": "misbehave", "label": "m14"}],
     "check": {"fn": "slashed", "who": 4, "on": ALL, "t": 10}},
    {"say": S("Eve가 다시 글을 씀. 다른 네 박스는 증명이 더 이상 통과하지 않아 버림.",
              "Eve posts again. The other four boxes drop it: the proof no longer verifies."),
     "acts": [{"who": 4, "do": "post", "topic": 0, "title": "Eve again, after being caught", "body": "",
               "label": "p15"}],
     "check": {"fn": "dropped", "label": "p15", "author": 4, "off": [0, 1, 2, 3], "grace": 5}},
]
for _n, _s in enumerate(STEPS, 1):
    _s["n"] = _n


def call(port: int, path: str, obj: dict | None = None, timeout: float = 120.0) -> tuple[int, object]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if obj is None else json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


class Fleet:
    """The five box processes (lan.py), their state folders and the test-only link cuts."""

    def __init__(self, real_timers: bool = False):
        self.timers = REAL if real_timers else FAST
        self.real = real_timers
        self.procs: dict[int, subprocess.Popen] = {}
        self.drop = {i: set() for i in ALL}
        self.hosts = ",".join(f"127.0.0.1:{p}" for p in PORTS)

    def fresh(self) -> None:
        self.stop()
        shutil.rmtree(STATE, ignore_errors=True)
        self.drop = {i: set() for i in ALL}

    def start(self, i: int, wait: bool = True) -> float:
        LOGS.mkdir(exist_ok=True)
        args = [sys.executable, str(HERE / "lan.py"), "--hosts", self.hosts, "--me", str(i),
                "--state", str(STATE / f"box{i}"), "--faults"]
        if not self.real:
            args += ["--pae", "{},{}".format(*self.timers["pae_s"]), "--probe-max", str(self.timers["probe_max_s"])]
        if PERSONAS[i]["room"]:
            args += ["--room-pass", ROOM_PASS]
        self.procs[i] = subprocess.Popen(args, stdout=open(LOGS / f"cp32_stage_box{i}.log", "a"),
                                         stderr=subprocess.STDOUT)
        return self.wait_up(i) if wait else 0.0

    def wait_up(self, i: int, timeout: float = 300.0) -> float:
        t = time.time()
        while time.time() - t < timeout:
            if self.procs[i].poll() is not None:
                raise RuntimeError(f"box {i} exited (see logs/cp32_stage_box{i}.log)")
            try:
                call(PORTS[i], "/api/stats", timeout=5)
                if self.drop[i]:
                    call(PORTS[i], "/fault", {"drop": sorted(self.drop[i])})
                return round(time.time() - t, 1)
            except (OSError, ValueError):
                time.sleep(0.3)
        raise RuntimeError(f"box {i} did not start")

    def start_all(self) -> None:
        for i in ALL:
            self.start(i, wait=False)
        for i in ALL:
            self.wait_up(i)

    def on(self, i: int) -> bool:
        return i in self.procs and self.procs[i].poll() is None

    def kill(self, i: int) -> None:
        if self.on(i):
            self.procs[i].kill()
            self.procs[i].wait(30)

    def stop(self) -> None:
        for i in list(self.procs):
            self.kill(i)

    def set_drop(self, groups: list[list[int]] | None) -> None:
        """groups = sides of a split (no link between sides); None heals every link."""
        self.drop = {i: set() for i in ALL}
        if groups:
            for side in groups:
                for i in side:
                    self.drop[i] = {j for j in ALL if j not in side}
        for i in ALL:
            if self.on(i):
                call(PORTS[i], "/fault", {"drop": sorted(self.drop[i])})


class Story:
    """Plays STEPS against a Fleet. User acts come either from the browser (record) or from here (perform)."""

    def __init__(self, fleet: Fleet):
        self.F = fleet
        self.mids: dict[str, str] = {}       # label -> message id
        self.titles: dict[str, str] = {}     # label -> title (what the client looks for)
        self.acts: dict[str, dict] = {}      # label -> act result
        self.results: dict[int, dict] = {}
        self.t_act: dict[int, float] = {}

    # ---- reading the boxes (owner API on loopback: an observer)
    def own(self, i: int, path: str, obj: dict | None = None):
        return call(PORTS[i], path, obj)[1]

    def holds(self, i: int) -> dict:
        return self.own(i, "/api/holds") if self.F.on(i) else {"msgs": [], "gone": [], "count": {}}

    def has(self, i: int, label: str) -> bool:
        return self.mids.get(label) in self.holds(i)["msgs"]

    def score(self, i: int, label: str):
        for it in self.own(i, "/api/feed"):
            if it["mid"] == self.mids.get(label):
                return it["score"]
        return None

    # ---- acts
    def ui_act(self, a: dict) -> dict:
        """What the client would send to its own box (the headless way; the browser does it through the UI)."""
        i = a["who"]
        if a["do"] == "post":
            code, r = call(PORTS[i], "/api/post", {"topic": a["topic"], "title": a["title"], "body": a.get("body", ""),
                                                   "photo": bool(a.get("photo"))})
        elif a["do"] == "search":
            code, r = call(PORTS[i], "/api/search?q=" + urllib.request.quote(a["q"]))
        elif a["do"] == "room":
            code, r = call(PORTS[i], "/api/room", {"body": a["body"]})
        elif a["do"] == "vote":
            code, r = call(PORTS[i], "/api/vote", {"mid": self.mids[a["target"]], "dir": a.get("dir", 1)})
        elif a["do"] in ("delete", "delete_other"):
            code, r = call(PORTS[i], "/api/delete", {"mid": self.mids[a["target"]]})
        elif a["do"] == "misbehave":
            code, r = call(PORTS[i], "/api/misbehave", {})
        else:
            raise ValueError(a["do"])
        return {"status": code, "result": r}

    def world_act(self, a: dict) -> dict:
        if a["do"] == "power":
            if a["on"]:
                return {"up_s": self.F.start(a["who"])}
            self.F.kill(a["who"])
            return {"off": True}
        if a["do"] == "split":
            self.F.set_drop(a["sides"])
            return {"sides": a["sides"]}
        if a["do"] == "isolate":
            self.F.set_drop([[a["who"]], [j for j in ALL if j != a["who"]]])
            return {"isolated": a["who"]}
        if a["do"] == "heal":
            self.F.set_drop(None)
            return {"healed": True}
        if a["do"] == "forge":                    # tests/sync_check.py scene 4, by an attacker that holds a copy
            v, mid = a["victim"], self.mids[a["target"]]
            real = self.own(a["author"], f"/env?mid={mid}")["msg"]
            forged = dict(real, ct="00" + real["ct"][2:])          # other content, same claimed id
            before = self.holds(v)["count"].get("rej_id", 0)
            self.own(v, "/gossip", {"msg": forged, "frm": -1})
            h = self.holds(v)
            return {"held_forged": mid in h["msgs"], "rej_id_delta": h["count"].get("rej_id", 0) - before}
        raise ValueError(a["do"])

    def begin(self, n: int) -> None:
        self.results.pop(n, None)

    def act(self, n: int, k: int, given: dict | None = None) -> dict:
        """given: the result the browser's client got (user act). None: perform it here."""
        a = STEPS[n - 1]["acts"][k]
        if given is not None:
            r = given
        elif a["do"] in UI_ACTS:
            r = self.ui_act(a)
        else:
            r = self.world_act(a)
        if a["do"] == "power" and a["on"]:
            self.acts["_"] = r                    # when the box answered (the catch-up check counts from here)
        if "label" in a:
            self.acts[a["label"]] = r
            res = r.get("result")
            if isinstance(res, dict) and res.get("mid"):
                self.mids[a["label"]] = str(res["mid"])
                self.titles[a["label"]] = a.get("title") or a.get("body", "")
        self.t_act[n] = time.time()
        return r

    # ---- checks
    def _wait(self, pred, t: float, poll: float = 0.3):
        t0 = time.time()
        while True:
            try:
                if pred():
                    return round(time.time() - t0, 2)
            except (OSError, ValueError, KeyError):
                pass
            if time.time() - t0 > t:
                return None
            time.sleep(poll)

    def _arrivals(self, label: str, boxes: list[int], t: float) -> dict:
        """When each box first held the message, from the end of the act (s, one PC, loopback)."""
        t0, seen = time.time(), {}
        while time.time() - t0 <= t and len(seen) < len(boxes):
            for i in boxes:
                if i not in seen:
                    try:
                        if self.has(i, label):
                            seen[i] = round(time.time() - t0, 2)
                    except (OSError, ValueError):
                        pass
            time.sleep(0.25)
        return seen

    def check(self, n: int) -> dict:
        c = STEPS[n - 1]["check"]
        fn, info, ok = c["fn"], {}, False
        if fn == "held":
            seen = self._arrivals(c["label"], c["on"], c["t"])
            ok = len(seen) == len(c["on"])
            info["arrival_s"] = {str(k): v for k, v in sorted(seen.items())}
            if c.get("off"):
                time.sleep(c.get("grace", 0))
                wrong = [i for i in c["off"] if self.has(i, c["label"])]
                info["held_where_it_should_not"] = wrong
                ok = ok and not wrong
        elif fn == "search":
            rows = (self.acts.get(c["act"]) or {}).get("result") or []
            top = [r["mid"] for r in rows[:c["top"]]] if isinstance(rows, list) else []
            ok = self.mids.get(c["label"]) in top
            info["top"] = [r.get("title") for r in rows[:3]] if isinstance(rows, list) else rows
        elif fn == "room":
            mid = self.mids.get(c["label"])

            def readers_ok():
                return all(any(it["mid"] == mid and c["body"] in (it.get("body") or "") for it in self.own(i, "/api/room")["items"])
                           for i in c["read"])
            t = self._wait(readers_ok, c["t"])
            relay = {str(i): {"holds_envelope": self.has(i, c["label"]), "member": self.own(i, "/api/room")["member"]}
                     for i in c["relay"]}
            ok = t is not None and all(v["holds_envelope"] and not v["member"] for v in relay.values())
            info.update(readers_s=t, relays=relay)
        elif fn == "down":
            i = c["who"]
            try:
                call(PORTS[i], "/api/stats", timeout=2)
                ok = False
            except OSError:
                ok = not self.F.on(i)
        elif fn == "score":
            def all_scored():
                return all(self.score(i, c["label"]) == c["value"] for i in c["on"])
            t = self._wait(all_scored, c["t"])
            ok, info["seconds"] = t is not None, t
        elif fn == "caught":
            i = c["who"]

            def caught():
                return all(self.has(i, lb) for lb in c["labels"]) and self.score(i, c["score"][0]) == c["score"][1]
            t = self._wait(caught, c["t"])
            ok, info["seconds_after_box_answered"] = t is not None, t
            info["box_start_s"] = (self.acts.get("_") or {}).get("up_s")
            cnt = self.holds(i)["count"]
            info["recon"] = {k: cnt.get(k, 0) for k in ("recon_sessions", "recon_fetched", "recon_bytes", "offer_want")}
        elif fn == "gone":
            mid = self.mids.get(c["label"])

            def gone_all():
                return all(mid not in (h := self.holds(i))["msgs"] and mid in h["gone"] for i in c["on"])
            t = self._wait(gone_all, c["t"])
            ok, info["seconds"] = t is not None, t
        elif fn == "refused":
            r = self.acts.get(c["act"]) or {}
            still = [i for i in c["on"] if self.has(i, c["label"])]
            ok = r.get("status") == 403 and len(still) == len(c["on"])
            info.update(status=r.get("status"), error=(r.get("result") or {}).get("error"), still_held_on=still)
        elif fn == "split":
            for label, on, off in c["parts"]:
                seen = self._arrivals(label, on, c["t"])
                info[label] = {"arrival_s": {str(k): v for k, v in seen.items()}}
            time.sleep(c.get("grace", 0))
            ok = True
            for label, on, off in c["parts"]:
                wrong = [i for i in off if self.has(i, label)]
                info[label]["crossed_to"] = wrong
                ok = ok and len(info[label]["arrival_s"]) == len(on) and not wrong
        elif fn == "held_all":
            def everywhere():
                return all(self.has(i, lb) for lb in c["labels"] for i in c["on"])
            t = self._wait(everywhere, c["t"], poll=0.5)
            ok, info["seconds"] = t is not None, t
        elif fn == "forged":
            f = self.acts.get(c["act"]) or {}
            t = self._wait(lambda: self.has(c["victim"], c["label"]), c["t"], poll=0.5)
            ok = t is not None and not f.get("held_forged", True) and f.get("rej_id_delta", 0) >= 1
            info.update(held_forged=f.get("held_forged"), rej_id_delta=f.get("rej_id_delta"), real_after_heal_s=t)
        elif fn == "slashed":
            def everyone():
                return all(len(self.own(i, "/api/stats")["notices"]) >= 1 for i in c["on"])
            t = self._wait(everyone, c["t"])
            st = self.own(c["who"], "/api/stats")
            ok = t is not None and st["slashed"]
            info.update(seconds=t, offender_sees_slashed=st["slashed"])
        elif fn == "dropped":
            time.sleep(c["grace"])
            wrong = [i for i in c["off"] if self.has(i, c["label"])]
            ok = bool(self.mids.get(c["label"])) and not wrong
            info.update(held_by=wrong, author_holds=self.has(c["author"], c["label"]),
                        rejected={str(i): {k: v for k, v in self.holds(i)["count"].items() if k.startswith("rej_")}
                                  for i in c["off"]})
        res = {"n": n, "pass": bool(ok), "check": fn, **info}
        self.results[n] = res
        return res

    def held_at_end(self) -> dict:
        """Which boxes hold each story message when the story ends (e.g. a sub-B post on boxes outside sub B)."""
        held = {i: set(self.holds(i)["msgs"]) if self.F.on(i) else None for i in ALL}
        return {lb: [i for i in ALL if held[i] is not None and mid in held[i]] for lb, mid in self.mids.items()}

    def report(self, mode: str, t_total: float | None = None) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        boxes = {}
        for i in ALL:
            try:
                s = self.own(i, "/api/stats")
                boxes[str(i)] = {"count": s["count"], "chip_virtual_ms": round(s["chip_virtual_ms"], 1),
                                 "items": s["items"], "slashed": s["slashed"]}
            except OSError:
                boxes[str(i)] = None
        out = {"cp": 32, "mode": mode, "what": "one-screen demo story on five lan.py boxes (journal/cp/CP32.md)",
               "label": "measured on one PC over loopback; times are not network latency",
               "settings": {"boxes": N, "ports": PORTS, "exchange": "recon (default)", "push": "lazy (default)",
                            "timers": "box defaults" if self.F.real else "test timers (tests/sync_check.py)",
                            **{k: list(v) if isinstance(v, tuple) else v for k, v in self.F.timers.items()}},
               "pass": all(self.results.get(s["n"], {}).get("pass") for s in STEPS),
               "steps": [{"n": s["n"], "say_en": s["say"]["en"], **self.results.get(s["n"], {"pass": None})} for s in STEPS],
               "total_s": t_total, "boxes_at_end": boxes, "held_at_end": self.held_at_end()}
        path = OUT / f"story_{mode}.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        return path


def run_headless(real_timers: bool = False) -> bool:
    F = Fleet(real_timers)
    F.fresh()
    t0 = time.time()
    print("starting five boxes ...", flush=True)
    F.start_all()
    print(f"boxes up in {time.time() - t0:.0f} s", flush=True)
    st = Story(F)
    t1 = time.time()
    try:
        for s in STEPS:
            st.begin(s["n"])
            for k in range(len(s["acts"])):
                st.act(s["n"], k)
            res = st.check(s["n"])
            print(f"[{time.time() - t1:6.1f}s] step {s['n']:2d} {'PASS' if res['pass'] else 'FAIL'} "
                  f"{json.dumps({k: v for k, v in res.items() if k not in ('n', 'pass')})}", flush=True)
        path = st.report("headless", round(time.time() - t1, 1))
        ok = all(r["pass"] for r in st.results.values()) and len(st.results) == len(STEPS)
        print(("ALL PASS" if ok else "SOME FAIL") + f" -> {path.relative_to(ROOT)}", flush=True)
        return ok
    finally:
        F.stop()
