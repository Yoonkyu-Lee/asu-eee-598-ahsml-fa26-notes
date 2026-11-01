"""CP19 scenarios on one box: the busiest box of a 10^6-user community (CP16 load, P-all upper bound) with the CP18
chip (encoder 8x8, 1 MB; verify unit 6x6) and a Pi 5-class host, driven through five scenarios (cp/CP19.md):
S1 ordinary day, S2 evening peak, S3 catch-up after h hours offline, S4 invalid-proof flood, S5 valid spam.

Resources (discrete events on src/sim/engine.py):
  host     4 cores, FIFO to the earliest free core; CP17 per-operation costs (laptop x Pi 5 factor)
  verify   one unit, batches whatever is queued (up to 128) when no flood is seen, one proof at a time in a flood;
           times and energies from the CP14 6x6 curve (V4, w8) plus the CP17 W3 public input
  encoder  one unit shared by queries, new items and the backlog, scheduled by Q0 / Q1 / Q2 (cp/CP19.md);
           times and energies from model_cp18.Encoder
  downlink 100 Mbps FIFO for image bodies and the catch-up download (small live messages are not queued on it)

  .venv/Scripts/python.exe src/sim_cp19.py          -> results/cp19/cp19.json, results/cp19/trace_*.json
Chip and host times are models (추정); the handshake cost is a laptop 예비 측정 x Pi 5 factor.
"""
from __future__ import annotations

import bisect
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_cp18 as m18  # noqa: E402
from model_cp12 import PI5_IDLE_W, PI5_STRESS_W  # noqa: E402
from model_cp4 import token_lengths  # noqa: E402
from sim import community as cm  # noqa: E402
from sim.engine import Engine  # noqa: E402
from sim.trace import Trace  # noqa: E402
from sim_cp16 import BASE  # noqa: E402
from sim_cp17 import PI5, VERIFY_SINGLE_S, W3_EXTRA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "cp19"
T0 = time.time()

LINK_BPS, PARENT_UP_BPS, HOP_S = 100e6, 20e6, 0.05      # CP12 flood design point; CP16 uplink and hop
HOST_CORES = 4
CORE_W = (PI5_STRESS_W[0] - PI5_IDLE_W[0]) / HOST_CORES  # active watts per loaded Pi 5 core (CP12 source)
FLOOD_PER_S = 100e6 / 8 / 1024                           # 1 KB messages filling 100 Mbps (CP6, CP12)
INVALID_PER_CONN = 10        # P4 weight -100, graylist -10,000: 100 n^2 >= 10,000 (nwaku, CP13 sources)
CONN_PER_IP_S = 0.2          # go-libp2p new connections per IPv4 /32 per second (CP13 sources)
SIZE = {"vote": 440.0, "post": 1050.0, "comment": 580.0, "image": 177e3, "flood": 1024.0}   # CP16 params
LIMITS = {"K1_text_box_s": 5.0 - 0.36, "K2_query_s": 0.050, "K3_backlog_8h_s": 3600.0, "K4_host_cores": 2.0,
          "K5_chip_W": 2.0}   # K1: 5 s minus CP16 text propagation P99 0.36 s
K_REF_SHARE = 0.01           # rule S: 1% of members (추정)
OVER_LIMIT_MAX = 1e-3        # honest messages refused by the rate limit

# verify unit (CP14 V4 w8, machine 6x6; +W3 public input). Valid batch: t(k) = VF + VC k from the single proof and
# the 128 batch; energy likewise. Invalid proofs one at a time (f = 1 point); the worst mix f = 0.15 per proof.
W3 = 1 + W3_EXTRA
V_SINGLE, V_128 = VERIFY_SINGLE_S * W3, 128 / 29_704 * W3
VC = (V_128 - V_SINGLE) / 127
VF = V_SINGLE - VC
E_SINGLE = 0.0685e-3 * (V_SINGLE / (1 / 14_633))          # valid single ~ invalid single energy x time ratio (추정)
EC = (0.0321e-3 * 128 * W3 - E_SINGLE) / 127
EF = E_SINGLE - EC
V_BAD, E_BAD = 1 / 14_633 * W3, 0.0685e-3 * W3
V_WORST, E_WORST = 1 / 13_429 * W3, 0.0773e-3 * W3


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ inputs

def inputs() -> dict:
    c16 = json.loads((RES / "cp16" / "cp16_load.json").read_text(encoding="utf-8"))
    a = c16["analytic"]["1000000"]["P-all V2"]
    day = {k: a["mean"][f"{k}_per_s"]["mean"] * 86_400 for k in ("verify", "text", "images")}
    di = np.array(c16["params"]["diurnal"], float)
    cmd = c16["analytic"]["1000000"]["community_msgs_per_day"]
    cost = json.loads((RES / "cp17" / "cp17.json").read_text(encoding="utf-8"))["cost_box_s"]
    return {"day": day, "diurnal": di / di.mean(), "post_share": cmd["post"] / (cmd["post"] + cmd["comment"]),
            "cost": cost}


def handshake_cost(n: int = 2000) -> dict:
    """예비 측정 (power preflight not confirmed): one inbound Noise XX + libp2p identity check on the responder."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    def per(fn, k=n):
        t = time.perf_counter()
        for _ in range(k):
            fn()
        return (time.perf_counter() - t) / k

    a, b = X25519PrivateKey.generate(), X25519PrivateKey.generate().public_key()
    sk = Ed25519PrivateKey.generate()
    msg = os.urandom(96)
    sig = sk.sign(msg)
    pk = sk.public_key()
    lap = {"x25519_keygen": per(X25519PrivateKey.generate), "x25519_dh": per(lambda: a.exchange(b)),
           "ed25519_sign": per(lambda: sk.sign(msg)), "ed25519_verify": per(lambda: pk.verify(sig, msg))}
    lap["handshake"] = lap["x25519_keygen"] + 3 * lap["x25519_dh"] + lap["ed25519_sign"] + lap["ed25519_verify"]
    return {"laptop_s": lap, "box_s": lap["handshake"] * PI5}


def honest_limits(p: cm.Params, peak: float) -> dict:
    """Share of honest messages refused by a per-10-minute limit, at the peak hour, over 10^6 users (CP16 model).
    Count per window ~ Poisson(rate x 600). Returns the smallest limit with share <= OVER_LIMIT_MAX per kind group."""
    c = cm.build(p.with_(users=1_000_000), seed=19)
    groups = {"all (L-a)": c.user_rates["post"] + c.user_rates["comment"] + c.user_rates["vote"],
              "text (L-b)": c.user_rates["post"] + c.user_rates["comment"], "vote (L-b)": c.user_rates["vote"]}
    out = {}
    nmax = 200
    for g, r in groups.items():
        lam = r * peak * 600
        lam = lam[lam > 0]
        pmf = np.empty((nmax + 1, len(lam)))
        pmf[0] = np.exp(-lam)
        for k in range(1, nmax + 1):
            pmf[k] = pmf[k - 1] * lam / k
        n = np.arange(nmax + 1)[:, None]
        total = lam.sum()
        share = {}
        for L in range(1, 61):
            share[L] = float((np.maximum(n - L, 0) * pmf).sum() / total)
        pick = min(L for L, s in share.items() if s <= OVER_LIMIT_MAX)
        out[g] = {"pick": pick, "share_at_pick": share[pick], "share_at_20": share[20],
                  "max_user_per_window": float(lam.max())}
    return out


# ------------------------------------------------------------------ encoder tables

class EncTable:
    def __init__(self, R: int, glb: int):
        parts = json.loads((RES / "cp12" / "cp12_parts.json").read_text(encoding="utf-8"))
        self.enc = m18.Encoder({"minilm": m18.load(f"minilm_glb{glb}"), "clipv": m18.load(f"clipv_glb{glb}")},
                               parts, R, glb)
        self.R, self.glb = R, glb
        self.text = [self.enc.run(m18.MINILM, np.array([L])) for L in range(1, 257)]
        self.img = self.enc.run(m18.CLIPV, np.array([50]))
        self.cache: dict = {}
        self.bmax_img = max(1, self.enc.b_max(m18.CLIPV, 50))

    def text1(self, L: int) -> tuple[float, float]:
        return self.text[min(max(L, 1), 256) - 1]

    def batch(self, lens: tuple) -> tuple[float, float]:
        if lens not in self.cache:
            self.cache[lens] = self.enc.run(m18.MINILM, np.array(lens))
        return self.cache[lens]

    def imgs(self, b: int) -> tuple[float, float]:
        k = ("img", b)
        if k not in self.cache:
            self.cache[k] = self.enc.run(m18.CLIPV, np.full(b, 50))
        return self.cache[k]


# ------------------------------------------------------------------ one box

class Box:
    def __init__(self, et: EncTable, cost: dict, *, order="O-b", sched="Q2", backlog_B=8, vmode="batch",
                 h_conn=0.0, trace=None, abort_s=None, live_first=False, seed=19):
        self.et, self.c = et, cost
        self.order, self.sched, self.B, self.vmode, self.h_conn = order, sched, backlog_B, vmode, h_conn
        self.eng = Engine()
        self.rng = np.random.default_rng(seed)
        self.host_free = [0.0] * HOST_CORES
        self.host_s = 0.0
        self.host_hour = np.zeros(24 * 4)
        self.link_free = 0.0
        self.vq: deque = deque()
        self.vq_back: deque = deque()
        self.live_first = live_first     # backlog work: host cores 2-3 only, verify fills batches after live proofs
        self.v_busy = False
        self.v_s = self.v_J = 0.0
        self.v_n = 0
        self.q = {0: deque(), 1: deque()}
        self.fifo: deque = deque()
        self.pool_t: list = []
        self.pool_i: deque = deque()
        self.resume2 = None
        self.e_busy = False
        self.e_s = self.e_J = 0.0
        self.e_hour = np.zeros(24 * 4)
        self.chip_J_hour = np.zeros(24 * 4)
        self.seq = 0
        self.lat = {"text": {}, "image": {}, "query": {}}     # arrival time -> latency (completed only)
        self.sub = {"text": [], "image": [], "query": []}     # arrival times of everything that is timed
        self.backlog_left = 0
        self.backlog_done = None
        self.flood_n = 0
        self.tr, self.abort_s, self.aborted = trace, abort_s, False

    # ---- resources
    def _span(self, track: str, name: str, s: float, d: float) -> None:
        if self.tr is not None and self.tr[1] <= s < self.tr[2]:
            self.tr[0].span("box", track, name, s, d)

    def host(self, t: float, dur: float, name: str = "", back: bool = False) -> float:
        cores = range(HOST_CORES // 2, HOST_CORES) if (back and self.live_first) else range(HOST_CORES)
        i = min(cores, key=self.host_free.__getitem__)
        s = max(t, self.host_free[i])
        self.host_free[i] = s + dur
        self.host_s += dur
        h = int(s // 3600)
        if 0 <= h < len(self.host_hour):
            self.host_hour[h] += dur
        self._span(f"host {i}", name, s, dur)
        return s + dur

    def link(self, t: float, nbytes: float) -> float:
        s = max(t, self.link_free)
        self.link_free = s + nbytes * 8 / LINK_BPS
        return self.link_free

    def _chip(self, t: float, J: float) -> None:
        h = int(t // 3600)
        if 0 <= h < len(self.chip_J_hour):
            self.chip_J_hour[h] += J

    # ---- arrivals
    def message(self, t: float, kind: str, *, L: int = 0, image: bool = False, cls: int = 1, record: bool = True,
                valid: bool = True, replay: bool = False) -> None:
        if record and cls != 2 and kind in ("post", "comment"):
            self.sub["text"].append(t)
            if image:
                self.sub["image"].append(t)
        self.eng.at(t, self._arrive, {"kind": kind, "L": L, "image": image, "cls": cls, "rec": record,
                                      "valid": valid, "replay": replay, "t_box": t})

    def query(self, t: float) -> None:
        self.sub["query"].append(t)
        self.eng.at(t, self._query)

    def _arrive(self, m: dict) -> None:
        t, c = self.eng.t, self.c
        if m["kind"] == "flood":
            self.flood_n += 1
            if self.h_conn and self.flood_n % INVALID_PER_CONN == 1:
                self.host(t, self.h_conn, "handshake")
            if self.order == "O-b" and m["replay"]:        # same (epoch, nullifier, x) seen: dropped by a lookup
                self.host(t, c["nullifier"], "dedupe")
                return
        pre = c["sig"] if self.order == "O-a" else c["nullifier"]
        t1 = self.host(t, pre, "ed25519" if self.order == "O-a" else "dedupe", m["cls"] == 2)
        self.eng.at(t1, self._to_verify, m)

    # ---- verify unit
    def _to_verify(self, m: dict) -> None:
        (self.vq_back if (self.live_first and m["cls"] == 2) else self.vq).append(m)
        if not self.v_busy:
            self._v_start()

    def _v_start(self) -> None:
        if not self.vq and not self.vq_back:
            self.v_busy = False
            return
        self.v_busy = True
        if self.vmode == "batch":
            k = min(len(self.vq), 128)
            batch = [self.vq.popleft() for _ in range(k)]
            while self.vq_back and len(batch) < 128:
                batch.append(self.vq_back.popleft())
            k = len(batch)
            dur, J = VF + VC * k, EF + EC * k
        else:
            m = self.vq.popleft()
            batch = [m]
            if m["kind"] != "flood":
                dur, J = V_SINGLE, E_SINGLE
            elif self.vmode == "worst":
                dur, J = V_WORST, E_WORST
            else:
                dur, J = V_BAD, E_BAD
        self.v_s += dur
        self.v_J += J
        self.v_n += len(batch)
        self._chip(self.eng.t, J)
        self._span("verify", f"{len(batch)} proof", self.eng.t, dur)
        self.eng.after(dur, self._v_done, batch)

    def _v_done(self, batch: list) -> None:
        t, c = self.eng.t, self.c
        for m in batch:
            if m["kind"] == "flood":
                if m["replay"]:                              # O-a: a replay passes the proof, caught as a duplicate
                    self.host(t, c["nullifier"], "dup")
                continue
            post = c["nullifier"] if self.order == "O-a" else c["sig"] + c["nullifier"]
            size = SIZE["vote"] if m["kind"] == "vote" else SIZE["post"] if m["image"] or m["kind"] == "post" \
                else SIZE["comment"]
            back = m["cls"] == 2
            t1 = self.host(t, post + c["decrypt_fixed"] + c["decrypt_per_b"] * size, "check+decrypt", back)
            if m["kind"] == "vote":
                self.host(t1, c["tally"], "tally", back)
                continue
            self.eng.at(t1, self._enc_submit, {"kind": "text", "L": m["L"], "cls": m["cls"], "m": m})
            if m["image"]:
                t2 = self.link(t1, SIZE["image"]) + HOP_S + SIZE["image"] * 8 / PARENT_UP_BPS
                t3 = self.host(t2, c["decrypt_per_b"] * SIZE["image"], "decrypt image", back)
                self.eng.at(t3, self._enc_submit, {"kind": "image", "cls": m["cls"], "m": m})
        self._v_start()

    # ---- encoder
    def _query(self) -> None:
        s, J = self.et.text1(12)
        self._enc_submit({"kind": "query", "cls": 0, "t_sub": self.eng.t, "s": s, "J": J, "layers": 6})

    def _enc_submit(self, job: dict) -> None:
        if job["kind"] != "query":
            if job["kind"] == "text":
                job["s"], job["J"] = self.et.text1(job["L"])
                job["layers"] = 6
            else:
                job["s"], job["J"] = self.et.img
                job["layers"] = 12
        if job["cls"] == 2:
            self.backlog_left += 1
        if self.sched == "Q0":
            self.fifo.append(job)
        elif job["cls"] == 2:
            if job["kind"] == "text":
                self.seq += 1
                bisect.insort(self.pool_t, (job["L"], self.seq, job))
            else:
                self.pool_i.append(job)
        else:
            self.q[job["cls"]].append(job)
        if not self.e_busy:
            self._e_next()

    def _form_backlog(self) -> dict | None:
        if self.pool_t:
            take = []
            while self.pool_t and len(take) < self.B:
                L = self.pool_t[0][0]
                if len(take) + 1 > max(1, self.et.enc.b_max(m18.MINILM, L)):
                    break
                take.append(self.pool_t.pop(0)[2])
            s, J = self.et.batch(tuple(min(j["L"], 256) for j in take))
            return {"kind": "text", "cls": 2, "members": take, "s": s, "J": J, "layers": 6}
        if self.pool_i:
            b = min(self.B, self.et.bmax_img, len(self.pool_i))
            take = [self.pool_i.popleft() for _ in range(b)]
            s, J = self.et.imgs(b)
            return {"kind": "image", "cls": 2, "members": take, "s": s, "J": J, "layers": 12}
        return None

    def _e_next(self) -> None:
        job = None
        if self.sched == "Q0":
            if self.fifo:
                job = self.fifo.popleft()
        else:
            for k in (0, 1):
                if self.q[k]:
                    job = self.q[k].popleft()
                    break
            if job is None:
                job, self.resume2 = (self.resume2, None) if self.resume2 else (self._form_backlog(), None)
        if job is None:
            self.e_busy = False
            return
        self.e_busy = True
        if "left" not in job:
            job["left"] = job["layers"] if self.sched == "Q2" else 1
            job["chunk"] = job["s"] / job["left"]
        dur = job["chunk"]
        J = job["J"] * dur / job["s"]
        self.e_s += dur
        self.e_J += J
        h = int(self.eng.t // 3600)
        if 0 <= h < len(self.e_hour):
            self.e_hour[h] += dur
        self._chip(self.eng.t, J)
        self._span("encoder", f"{job['kind']} c{job['cls']}", self.eng.t, dur)
        self.eng.after(dur, self._e_done, job)

    def _e_done(self, job: dict) -> None:
        t = self.eng.t
        job["left"] -= 1
        if job["left"] > 0:
            if job["cls"] == 2:
                self.resume2 = job
            elif self.sched == "Q0":
                self.fifo.appendleft(job)
            else:
                self.q[job["cls"]].appendleft(job)
            self._e_next()
            return
        if job["kind"] == "query":
            self.lat["query"][job["t_sub"]] = t - job["t_sub"]
        else:
            for j in job.get("members", [job]):
                t1 = self.host(t, self.c["insert"], "insert", j["cls"] == 2)
                if j["cls"] == 2:
                    self.backlog_left -= 1
                    if self.backlog_left == 0:
                        self.backlog_done = t1
                elif j["m"]["rec"]:
                    self.lat[job["kind"]][j["m"]["t_box"]] = t1 - j["m"]["t_box"]
                    if self.abort_s and t1 - j["m"]["t_box"] > self.abort_s:
                        self.aborted = True
                        self.eng._q.clear()
        self._e_next()

    def run(self, until: float) -> None:
        self.eng.run(until)


# ------------------------------------------------------------------ streams

def legit(box: Box, inp: dict, lens: np.ndarray, t0: float, t1: float, hour_of, *, cls: int = 1,
          probe_dt: float | None = None, query_dt: float | None = None, rng=None, at=None) -> int:
    """Poisson arrivals of the busiest box's honest traffic between t0 and t1 (hour_of(t) -> clock hour).
    at(t) maps the creation time to the arrival time at the box (default: same)."""
    rng = rng or box.rng
    d, di = inp["day"], inp["diurnal"]
    n_all = 0
    for h0 in np.arange(np.floor(t0 / 3600) * 3600, t1, 3600):
        a, b = max(h0, t0), min(h0 + 3600, t1)
        if b <= a:
            continue
        f = di[int(hour_of(a)) % 24]
        for kind, per_day in (("text", d["text"]), ("vote", d["verify"] - d["text"])):
            n = rng.poisson(per_day / 86_400 * f * (b - a))
            ts = np.sort(rng.uniform(a, b, n))
            for t in ts:
                tt = at(t) if at else t
                if kind == "vote":
                    box.message(tt, "vote", cls=cls)
                else:
                    img = rng.random() < d["images"] / d["text"]
                    post = img or rng.random() < (inp["post_share"] - d["images"] / d["text"]) / (1 - d["images"] / d["text"])
                    box.message(tt, "post" if post else "comment", L=int(rng.choice(lens)), image=img, cls=cls)
            n_all += n
    if probe_dt:
        for t in np.arange(t0 + probe_dt / 2, t1, probe_dt):
            box.message(float(t), "comment", L=int(rng.choice(lens)))
    if query_dt:
        for t in np.arange(t0 + query_dt / 3, t1, query_dt):
            box.query(float(t))
    return n_all


def pct(x, q=99) -> float | None:
    return float(np.percentile(x, q)) if len(x) else None


def lat(box: Box, k: str, a: float = -np.inf, b: float = np.inf) -> np.ndarray:
    """Latencies of everything timed that arrived in [a, b); not finished by the end of the run = infinite."""
    return np.array([box.lat[k].get(t, np.inf) for t in box.sub[k] if a <= t < b])


def summary(box: Box, dur: float, a: float = -np.inf, b: float = np.inf) -> dict:
    chip_W = (box.v_J + box.e_J) / dur
    cores = box.host_s / dur
    lt, li, lq = lat(box, "text", a, b), lat(box, "image", a, b), lat(box, "query", a, b)
    return {"text_p99_s": pct(lt), "text_p50_s": pct(lt, 50), "n_text": len(lt), "unfinished_text": int(np.isinf(lt).sum()),
            "image_p99_s": pct(li), "query_p99_ms": (pct(lq) or 0) * 1e3,
            "query_max_ms": float(lq.max() * 1e3) if len(lq) else None,
            "n_query": len(lq), "host_cores": cores, "verify_busy": box.v_s / dur,
            "encoder_busy": box.e_s / dur, "chip_W": chip_W,
            "box_W": [i + min(cores, HOST_CORES) * CORE_W + chip_W for i in PI5_IDLE_W], "aborted": box.aborted}


# ------------------------------------------------------------------ scenarios

def s1_day(et, inp, lens, sched="Q2") -> dict:
    box = Box(et, inp["cost"], sched=sched, seed=191)
    legit(box, inp, lens, 0, 86_400, lambda t: t // 3600, query_dt=60)
    box.run(86_400 + 600)
    r = summary(box, 86_400)
    r["chip_J_day"] = box.v_J + box.e_J
    r["verify_J_day"], r["encoder_J_day"] = box.v_J, box.e_J
    r["chip_W_by_hour"] = (box.chip_J_hour[:24] / 3600).tolist()
    r["host_cores_by_hour"] = (box.host_hour[:24] / 3600).tolist()
    r["encoder_busy_by_hour"] = (box.e_hour[:24] / 3600).tolist()
    r["box_idle_share_chip"] = r["chip_W"] / PI5_IDLE_W[0]
    return r


def s2_peak(et, inp, lens, sched) -> dict:
    box = Box(et, inp["cost"], sched=sched, seed=192)
    legit(box, inp, lens, 0, 3600, lambda t: 20, probe_dt=5, query_dt=5)
    box.run(3600 + 300)
    return summary(box, 3600)


def s3_backlog(et, inp, lens, hours, sched, B, trace=None, live_first=True) -> dict:
    """Off for `hours` before 20:00; at t = 0 the box downloads what it missed (in creation order, 100 Mbps) and
    processes it at backlog priority while the peak hour's live traffic continues."""
    box = Box(et, inp["cost"], sched=sched, backlog_B=B, seed=193, trace=trace, live_first=live_first)
    rng = np.random.default_rng(1930 + hours)
    missed = []

    class Rec:   # collects the backlog stream instead of scheduling it
        def message(self, t, kind, cls=1, **kw):
            missed.append((t, kind, kw))

        def query(self, t):
            pass
    rec = Rec()
    rec.rng = rng
    legit(rec, inp, lens, -hours * 3600.0, 0.0, lambda t: (20 + t // 3600) % 24, rng=rng)
    missed.sort(key=lambda x: x[0])
    t_link = 0.0
    nbytes = 0.0
    for _, kind, kw in missed:
        size = SIZE[kind] if kind in SIZE else SIZE["comment"]
        t_link += size * 8 / LINK_BPS
        nbytes += size
        box.message(t_link, kind, cls=2, **kw)
    box.link_free = t_link
    legit(box, inp, lens, 0, 4 * 3600, lambda t: 20, probe_dt=5, query_dt=5)
    box.run(4 * 3600 + 600)
    end = box.backlog_done or float("inf")
    r = summary(box, min(end, 4 * 3600), 0.0, min(end, 4 * 3600))
    r.update({"hours": hours, "sched": sched, "B": B, "live_first": live_first, "missed_msgs": len(missed), "download_MB": nbytes / 1e6,
              "download_s": t_link, "catch_up_s": end, "catch_up_min": end / 60})
    return r


def s4_flood(et, inp, lens, order, policy, mix, h_conn, seeds=3, dur=60.0, trace=None) -> dict:
    """mix: 'invalid' (every flood proof is wrong) or 'replay' (CP14 worst mix: 15% wrong, 85% replayed valid)."""
    rows = []
    for s in range(seeds):
        vmode = "worst" if (mix == "replay" and order == "O-a") else "single"
        box = Box(et, inp["cost"], order=order, sched="Q2", vmode=vmode, h_conn=h_conn if policy == "N-b" else 0.0,
                  seed=194 + s, trace=trace if s == 0 else None)
        rng = np.random.default_rng(1940 + s)
        n = rng.poisson(FLOOD_PER_S * dur)
        for t in np.sort(rng.uniform(0, dur, n)):
            box.message(float(t), "flood", valid=False, record=False,
                        replay=(mix == "replay" and rng.random() < 0.85))
        legit(box, inp, lens, 0, dur, lambda t: 20, probe_dt=0.5, query_dt=1.0, rng=rng)
        box.run(dur + 120)
        rows.append((box, summary(box, dur)))
    lat_t = np.concatenate([lat(b, "text") for b, _ in rows])
    lat_q = np.concatenate([lat(b, "query") for b, _ in rows])
    r = {k: float(np.mean([x[k] for _, x in rows])) for k in ("host_cores", "verify_busy", "encoder_busy", "chip_W")}
    r.update({"order": order, "policy": policy, "mix": mix, "text_p99_s": pct(lat_t), "n_text": len(lat_t),
              "unfinished_text": int(np.isinf(lat_t).sum()),
              "query_p99_ms": pct(lat_q) * 1e3, "flood_per_s": FLOOD_PER_S,
              "box_W": [i + r["host_cores"] * CORE_W + r["chip_W"] for i in PI5_IDLE_W],
              "host_offered_cores": float(np.mean([b.host_s for b, _ in rows]) / dur)})
    r["ips_needed_N-b"] = FLOOD_PER_S / (CONN_PER_IP_S * INVALID_PER_CONN)
    return r


def s5_spam(et, inp, lens, K, per_id_s, dur=3600.0, seed=195) -> dict:
    """K identities each sending image posts with a 256-token body at the limit (valid proofs)."""
    box = Box(et, inp["cost"], sched="Q2", seed=seed, abort_s=60.0)
    rng = np.random.default_rng(seed)
    n = rng.poisson(K * per_id_s * dur)
    for t in np.sort(rng.uniform(0, dur, n)):
        box.message(float(t), "post", L=256, image=True, record=False)
    legit(box, inp, lens, 0, dur, lambda t: 20, probe_dt=5, query_dt=5, rng=rng)
    box.run(dur + 300)
    r = summary(box, dur)
    r.update({"K": K, "spam_per_s": K * per_id_s})
    return r


def k_crit(et, inp, lens, per_id_s) -> dict:
    """Largest K with K1 held for an hour (bisection on K around the capacity estimate)."""
    s_spam = et.text1(256)[0] + et.img[0]
    d = inp["day"]
    f = inp["diurnal"][20]
    t_mean = float(np.mean([et.text1(int(L))[0] for L in lens[:2000]]))
    legit_busy = f * (d["text"] * t_mean + d["images"] * et.img[0]) / 86_400 + et.text1(12)[0] / 5
    k0 = max(1.0, (1 - legit_busy) / (per_id_s * s_spam))
    ok = lambda r: (not r["aborted"]) and r["text_p99_s"] is not None and r["text_p99_s"] <= LIMITS["K1_text_box_s"]
    lo, hi = max(1, int(k0 / 4)), int(k0 * 2) + 2
    runs = {}
    r = s5_spam(et, inp, lens, lo, per_id_s)
    runs[lo] = r
    if not ok(r):
        lo = 0
    while True:
        r = s5_spam(et, inp, lens, hi, per_id_s)
        runs[hi] = r
        if not ok(r):
            break
        lo, hi = hi, hi * 2
    while hi - lo > max(1, int(0.02 * hi)):
        mid = (lo + hi) // 2
        r = s5_spam(et, inp, lens, mid, per_id_s)
        runs[mid] = r
        lo, hi = (mid, hi) if ok(r) else (lo, mid)
    best = runs.get(lo)
    return {"K_crit": lo, "K_estimate": k0, "spam_s_each": s_spam, "legit_encoder_busy": legit_busy,
            "at_K_crit": best and {k: best[k] for k in ("text_p99_s", "query_p99_ms", "encoder_busy", "chip_W",
                                                          "host_cores", "spam_per_s")},
            "tried": {int(k): {"ok": ok(v), "p99": v["text_p99_s"], "aborted": v["aborted"]} for k, v in runs.items()}}


# ------------------------------------------------------------------ main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    quick = "--quick" in sys.argv
    inp = inputs()
    lens = np.minimum(token_lengths(), 256)
    res = {"inputs": {"day": inp["day"], "peak_factor": float(inp["diurnal"][20]), "limits": LIMITS,
                      "verify_model": {"V_single_us": V_SINGLE * 1e6, "V_batch128_us_per": V_128 / 128 * 1e6,
                                       "V_bad_us": V_BAD * 1e6, "V_worst_us": V_WORST * 1e6,
                                       "E_single_mJ": E_SINGLE * 1e3, "E_batch128_mJ_per": (EF + EC * 128) / 128 * 1e3},
                      "host_box_s": inp["cost"]}}
    hs = handshake_cost()
    res["handshake"] = hs
    log("handshake", hs)
    pick = EncTable(8, 1024)
    res["encoder_pick"] = {"text_164_ms": pick.text1(164)[0] * 1e3, "text_256_ms": pick.text1(256)[0] * 1e3,
                           "query_ms": pick.text1(12)[0] * 1e3, "image_ms": pick.img[0] * 1e3}
    log("encoder", res["encoder_pick"])

    # S2 / S3 first: they decide the schedule (rule Q)
    res["S2"] = {q: s2_peak(pick, inp, lens, q) for q in ("Q0", "Q1", "Q2")}
    for q, r in res["S2"].items():
        log("S2", q, {k: r[k] for k in ("text_p99_s", "query_p99_ms", "encoder_busy", "host_cores", "chip_W")})
    order_q = [("Q0", 1), ("Q1", 8), ("Q1", 1), ("Q2", 8), ("Q2", 1)]
    res["S3"] = []
    for lf in (False, True):     # False = as planned (host and verify FIFO); True = added after the first run
        for h in ((8,) if quick else (1, 8, 24)):
            for q, B in order_q:
                r = s3_backlog(pick, inp, lens, h, q, B, live_first=lf)
                res["S3"].append(r)
                log("S3", lf, h, q, B, {k: r[k] for k in ("catch_up_min", "text_p99_s", "query_p99_ms",
                                                           "query_max_ms", "host_cores", "chip_W", "download_s")})

    def passes(q, B, lf):
        s2 = res["S2"][q]
        s3 = [r for r in res["S3"] if r["sched"] == q and r["B"] == B and r["live_first"] == lf]
        k3 = all(r["catch_up_s"] <= LIMITS["K3_backlog_8h_s"] for r in s3 if r["hours"] == 8)
        k12 = all(r["text_p99_s"] <= LIMITS["K1_text_box_s"] and r["query_p99_ms"] <= LIMITS["K2_query_s"] * 1e3
                  for r in s3 + [s2])
        return k3 and k12
    res["rule_Q"] = {}
    for lf in (False, True):
        tag = "live_first" if lf else "fifo_host_verify"
        d = {f"{q} B={B}": passes(q, B, lf) for q, B in order_q}
        sel = next(((q, B) for q, B in order_q if passes(q, B, lf)), None)
        d["pick"] = sel and f"{sel[0]} B={sel[1]}"
        res["rule_Q"][tag] = d
    log("rule Q", res["rule_Q"])
    sched, B = sel if sel else ("Q2", 1)     # sel: live-first pick (the last loop pass)

    # S1 ordinary day with the picked schedule
    res["S1"] = s1_day(pick, inp, lens, sched)
    log("S1", {k: res["S1"][k] for k in ("text_p99_s", "query_p99_ms", "chip_W", "chip_J_day", "host_cores", "box_W")})

    # S4 floods
    res["S4"] = []
    tr4 = (Trace(), 10.0, 10.02)
    for order, policy, mix in (("O-a", "N-a", "invalid"), ("O-a", "N-b", "invalid"), ("O-a", "N-a", "replay"),
                               ("O-b", "N-a", "invalid"), ("O-b", "N-b", "invalid"), ("O-b", "N-b", "replay")):
        r = s4_flood(pick, inp, lens, order, policy, mix, hs["box_s"], seeds=1 if quick else 3,
                     trace=tr4 if (order, policy, mix) == ("O-b", "N-b", "invalid") else None)
        res["S4"].append(r)
        log("S4", order, policy, mix, {k: r[k] for k in ("host_offered_cores", "verify_busy", "text_p99_s",
                                                          "query_p99_ms", "chip_W")})
    tr4[0].write(OUT / "trace_flood_20ms.json")
    ok4 = lambda r: (r["host_offered_cores"] <= LIMITS["K4_host_cores"] and r["text_p99_s"] <= LIMITS["K1_text_box_s"]
                     and r["query_p99_ms"] <= LIMITS["K2_query_s"] * 1e3 and r["chip_W"] <= LIMITS["K5_chip_W"]
                     and r["verify_busy"] < 1.0)
    pick_o = None
    for o in ("O-a", "O-b"):
        if all(ok4(r) for r in res["S4"] if r["order"] == o and r["policy"] == "N-b"):
            pick_o = o
            break
    res["rule_C"] = {"pass": {f"{r['order']} {r['policy']} {r['mix']}": ok4(r) for r in res["S4"]},
                     "pick": pick_o or "Ed25519 on chip (candidate)"}
    log("rule C", res["rule_C"])

    # S5 valid spam: limits from the honest model, then K_crit per encoder and limit
    lim = honest_limits(BASE, float(inp["diurnal"][20]))
    res["honest_limits"] = lim
    log("limits", lim)
    rates = {"L-a": 20 / 600, "L-b": lim["text (L-b)"]["pick"] / 600}
    res["S5"] = {}
    encs = {"8x8 1MB": pick} if quick else {"8x8 1MB": pick, "16x16 2MB": EncTable(16, 2048),
                                            "32x32 2MB": EncTable(32, 2048)}
    for name, et in encs.items():
        for L, r_id in rates.items():
            k = k_crit(et, inp, lens, r_id)
            res["S5"][f"{name} {L}"] = k
            log("S5", name, L, k["K_crit"], k["K_estimate"])
    members = 1_000_000
    k_ref = K_REF_SHARE * members
    use_L = "L-b" if res["S5"]["8x8 1MB L-b"]["K_crit"] > res["S5"]["8x8 1MB L-a"]["K_crit"] else "L-a"
    verdict = None
    for name in encs:
        if res["S5"][f"{name} {use_L}"]["K_crit"] >= k_ref:
            verdict = name
            break
    res["rule_S"] = {"limit": use_L, "K_ref": k_ref, "K_crit_share": {k: v["K_crit"] / members for k, v in
                                                                      res["S5"].items()},
                     "pick": verdict or "8x8 1MB kept; protocol requirement (identity cost x limit)"}
    log("rule S", res["rule_S"])
    (OUT / "cp19.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # trace: first 2 s of the 8 h catch-up with the picked schedule
    tr3 = (Trace(), 0.0, 2.0)
    s3_backlog(pick, inp, lens, 8, sched, B, trace=tr3, live_first=True)
    tr3[0].write(OUT / "trace_catchup_2s.json")
    log("wrote")


if __name__ == "__main__":
    main()
