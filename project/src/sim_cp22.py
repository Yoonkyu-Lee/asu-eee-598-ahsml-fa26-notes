"""CP22: valid-proof spam without an operator. The CP19 box (CP20 chip: encoder 8x8, image tower FP16, GEMM-boundary
yield, verify 6x6) with three protocol knobs that keep messages unlinkable (cp/CP22.md):
  T1 weight limit   a message spends w encoder units (256-token text = 1, picture + img/text) from W per epoch
  T2 pictures last  picture embeddings wait behind every live text
  T3 age groups     two membership trees (joined > 30 days ago or not); the box serves the old group first and the
                    new group has its own limit
Rule S2: honest refusals <= 1e-3, K1 for both groups' honest posts and K2 while 10^4 new-group identities spam for an
hour; fewest changes wins; K_crit of the old group is the residual attack.

  .venv/Scripts/python.exe src/sim_cp22.py          -> results/cp22/cp22.json
Every number is a model (추정) on CP16 behaviour and CP18-CP20 chip numbers.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_cp19 as c19  # noqa: E402
import sim_cp20 as c20  # noqa: E402
from functools import partial  # noqa: E402
from sim import community as cm  # noqa: E402
from sim_cp16 import BASE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp22"
T0 = time.time()
EPOCH = 600.0
K_REF = 10_000
S_NEW = {"base": 0.05, "low": 0.01, "high": 0.20}      # share of honest writing from identities < 30 days (추정)
OVER_MAX = 1e-3
L_OLD = 6                                             # CP19 L-b text limit per 10 min
COMMENT_TOKENS = 40                                   # 30 words (CP16 source) -> about 40 tokens (추정)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ honest refusals

class Honest:
    """Per-user text rates at the peak hour (CP16 model, 10^6 users) and message weights."""

    def __init__(self, et, lens: np.ndarray, peak: float, seed: int = 22):
        c = cm.build(BASE.with_(users=1_000_000), seed=19)
        lam = (c.user_rates["post"] + c.user_rates["comment"]) * peak      # per second at the peak hour
        self.rate = lam[lam > 0]
        self.post_share = float(c.user_rates["post"].sum() / (c.user_rates["post"] + c.user_rates["comment"]).sum())
        unit = et.text1(256)[0]
        self.img_w = et.img[0] / unit
        rng = np.random.default_rng(seed)
        n = 200_000
        is_post = rng.random(n) < self.post_share
        img = is_post & (rng.random(n) < BASE.image_frac)
        L = np.where(is_post, rng.choice(lens, n), COMMENT_TOKENS)
        tw = np.array([et.text1(int(x))[0] for x in range(1, 257)]) / unit
        self.w = tw[np.minimum(L, 256) - 1] + img * self.img_w      # weight per honest message (sampled)
        self.unit = unit

    def refused_count(self, limit: int, epoch: float = EPOCH) -> float:
        """Share of honest text messages beyond `limit` per epoch (Poisson counts per user, peak-hour rates)."""
        from scipy.stats import poisson
        lam = self.rate * epoch
        n = np.arange(0, 400)
        pm = poisson.pmf(n[:, None], lam[None, :])
        return float((np.maximum(n[:, None] - limit, 0) * pm).sum() / lam.sum())

    def refused_weight(self, W: float, epoch: float = EPOCH, draws: int = 400, seed: int = 7) -> float:
        """Share of honest messages refused when each identity may spend W units per epoch (messages in arrival
        order; one that does not fit is refused, later smaller ones may still fit)."""
        from scipy.stats import poisson
        rng = np.random.default_rng(seed)
        nmax = 120
        exp_ref = np.zeros(nmax + 1)
        for k in range(1, nmax + 1):
            ws = rng.choice(self.w, (draws, k))
            used = np.zeros(draws)
            ref = np.zeros(draws)
            for j in range(k):
                fit = used + ws[:, j] <= W
                used = np.where(fit, used + ws[:, j], used)
                ref += ~fit
            exp_ref[k] = ref.mean()
        lam = self.rate * epoch
        pm = poisson.pmf(np.arange(nmax + 1)[:, None], lam[None, :])
        return float((exp_ref[:, None] * pm).sum() / lam.sum())


NEW_EPOCHS = (600.0, 3600.0)     # the new group's window: 10 min (as now) or 1 h (a second RLN epoch length)


def pick_limits(H: Honest, s_new: float, t1: bool, t3: bool) -> dict:
    """Old group: 6 posts per 10 min or, with T1, the smallest W per 10 min meeting 1e-3 on its own.
    New group (T3): over windows of 10 min and 1 h, the limit allowing the lowest spam rate per identity
    (limit / window) such that total honest refusals stay <= 1e-3."""
    if t1:
        old = next(W for W in np.arange(1.0, 60.0, 0.5) if H.refused_weight(W) <= OVER_MAX)
        r_old = H.refused_weight(old)
    else:
        old, r_old = L_OLD, H.refused_count(L_OLD)
    out = {"old": float(old), "old_epoch_s": EPOCH, "unit": "units" if t1 else "posts"}
    if not t3:
        return {**out, "new": float(old), "new_epoch_s": EPOCH, "refused_total": r_old}
    budget = (OVER_MAX - (1 - s_new) * r_old) / s_new
    best = None
    for E in NEW_EPOCHS:
        f = (lambda x, E=E: H.refused_weight(x, E)) if t1 else (lambda x, E=E: H.refused_count(int(x), E))
        grid = np.arange(0.5, 120.0, 0.5) if t1 else np.arange(1, 121)
        x = next((g for g in grid if f(g) <= budget), None)
        if x is not None and (best is None or x / E < best[0] / best[1]):
            best = (float(x), E, f(x))
    return {**out, "new": best[0], "new_epoch_s": best[1], "refused_old": r_old, "refused_new": best[2],
            "refused_total": (1 - s_new) * r_old + s_new * best[2]}


# ------------------------------------------------------------------ the box

class Box22(c20.Box20):
    """Encoder priority: query 0, old-group text 1, new-group text 3, then (T2) old pictures 4, new pictures 5;
    without T2 a picture shares its text's priority. Without T3 everyone is in the old group. Backlog stays last."""

    def __init__(self, et, cost, *, t2=False, t3=False, **kw):
        self.t2, self.t3, self.s_new = t2, t3, S_NEW["base"]
        super().__init__(et, cost, **kw)
        self.q = defaultdict(deque)
        for k in ("text_old", "text_new", "image_old", "image_new"):
            self.lat[k], self.sub[k] = {}, []

    def message(self, t, kind, *, tier=None, **kw):
        if tier is None:
            tier = "new" if (self.t3 and self.rng.random() < self.s_new) else "old"
        if kw.get("record", True) and kw.get("cls", 1) != 2 and kind in ("post", "comment"):
            self.sub[f"text_{tier}"].append(t)
            if kw.get("image"):
                self.sub[f"image_{tier}"].append(t)
        self.eng.at(t, self._arrive, {"kind": kind, "L": kw.get("L", 0), "image": kw.get("image", False),
                                      "cls": kw.get("cls", 1), "rec": kw.get("record", True), "valid": True,
                                      "replay": False, "t_box": t, "tier": tier})
        if kw.get("record", True) and kw.get("cls", 1) != 2 and kind in ("post", "comment"):
            self.sub["text"].append(t)
            if kw.get("image"):
                self.sub["image"].append(t)

    def host(self, t, dur, name="", back=False):
        """Work dated in the future (the picture's decryption, booked when its pull is scheduled) used to take a core
        slot ahead of time and so delayed everything arriving before it (sim_cp19.Box books slots eagerly). Under a
        flood of picture posts that built a queue that is not real. Future-dated work is counted but books no core."""
        if t > self.eng.t + 1e-9:
            self.host_s += dur
            return t + dur
        return super().host(t, dur, name, back)

    def _enc_submit(self, job):
        if job["kind"] != "query" and job["cls"] != 2:
            tier = job["m"].get("tier", "old")
            if job["kind"] == "text":
                job["cls"] = 1 if tier == "old" else 3
            elif self.t2:
                job["cls"] = 4 if tier == "old" else 5
            else:
                job["cls"] = 1 if tier == "old" else 3
        super()._enc_submit(job)

    def _e_next(self):
        """sim_cp19.Box._e_next with any number of priority classes (lowest number first)."""
        best = next((k for k in sorted(self.q) if self.q[k]), None)
        if best is not None:
            job = self.q[best].popleft()
        elif self.resume2:
            job, self.resume2 = self.resume2, None
        else:
            job = self._form_backlog()
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
        self._chip(self.eng.t, J)
        self.eng.after(dur, self._e_done, job)

    def _e_done(self, job):
        if job["left"] == 1 and job["kind"] in ("text", "image"):
            for j in job.get("members", [job]):
                m = j.get("m")
                if m and m["rec"] and j["cls"] != 2:
                    self.lat[f"{job['kind']}_{m['tier']}"][m["t_box"]] = self.eng.t + self.c["insert"] - m["t_box"]
        super()._e_done(job)


def attack(et, inp, lens, *, K, tier, per_id_s, weight_units, H, t2, t3, s_new, vm, dur=3600.0, seed=221):
    """K identities of one group at their limit, each message the most expensive one allowed."""
    make = partial(Box22, t2=t2, t3=t3, vkind="chip", vm=vm)
    box = make(et, inp["cost"], sched="Q2", seed=seed)
    box.s_new = s_new
    rng = np.random.default_rng(seed)
    n = rng.poisson(K * per_id_s * dur)
    for t in np.sort(rng.uniform(0, dur, n)):
        box.message(float(t), "post", L=256, image=True, record=False, tier=tier)
    c19.legit(box, inp, lens, 0, dur, lambda t: 20, query_dt=5, rng=rng)
    for t in np.arange(2.5, dur, 5.0):                  # probes: one honest comment per group every 5 s
        box.message(float(t), "comment", L=int(rng.choice(lens)), tier="old")
        if t3:
            box.message(float(t) + 1.0, "comment", L=int(rng.choice(lens)), tier="new")
    box.run(dur + 300)
    p = lambda k: float(np.percentile(c19.lat(box, k), 99, method="higher")) if box.sub[k] else None
    r = {"K": K, "tier": tier, "spam_per_s": K * per_id_s, "text_old_p99_s": p("text_old"),
         "text_new_p99_s": p("text_new"), "image_old_p99_s": p("image_old"), "query_p99_ms": (p("query") or 0) * 1e3,
         "encoder_busy": box.e_s / box.eng.t, "chip_W": (box.v_J + box.e_J) / dur, "host_cores": box.host_s / dur}
    lim = c19.LIMITS["K1_text_box_s"]
    r["K1_old"] = r["text_old_p99_s"] is not None and r["text_old_p99_s"] <= lim
    r["K1_new"] = (not t3) or (r["text_new_p99_s"] is not None and r["text_new_p99_s"] <= lim)
    r["K2"] = r["query_p99_ms"] <= c19.LIMITS["K2_query_s"] * 1e3
    return r


def k_crit(fn, k0: int) -> tuple[int, dict]:
    """Largest K for which fn(K) keeps K1 (both groups) and K2."""
    ok = lambda r: r["K1_old"] and r["K1_new"] and r["K2"]
    runs = {}
    lo, hi = 0, max(1, k0)
    while True:
        runs[hi] = fn(hi)
        if not ok(runs[hi]):
            break
        lo, hi = hi, hi * 2
    while hi - lo > max(1, int(0.03 * hi)):
        mid = (lo + hi) // 2
        runs[mid] = fn(mid)
        lo, hi = (mid, hi) if ok(runs[mid]) else (lo, mid)
    return lo, runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inp = c19.inputs()
    lens = np.minimum(c19.token_lengths(), 256)
    et = c20.Fp16Image(c19.EncTable(8, 1024), 12 * 4 + 1)      # CP20 X3g
    vm = c20.machines()["6x6"]
    peak = float(inp["diurnal"][20])
    H = Honest(et, lens, peak)
    spam_w = (et.text1(256)[0] + et.img[0]) / H.unit             # units of the most expensive post
    res = {"inputs": {"K_ref": K_REF, "s_new": S_NEW, "unit_s": H.unit, "image_units": H.img_w,
                      "spam_post_units": spam_w, "honest_post_share": H.post_share,
                      "honest_weight_p50_p99": [float(np.percentile(H.w, 50)), float(np.percentile(H.w, 99))]}}
    log("inputs", res["inputs"])
    rows = {}
    for t1, t2, t3 in itertools.product((False, True), repeat=3):
        name = "+".join(n for n, on in (("T1", t1), ("T2", t2), ("T3", t3)) if on) or "T0"
        lim = pick_limits(H, S_NEW["base"], t1, t3)
        # spam rate per identity: count limit -> that many posts; weight limit -> W / units per post
        rate = lambda x, E: (x / spam_w if t1 else x) / E
        r_new, r_old = rate(lim["new"], lim["new_epoch_s"]), rate(lim["old"], lim["old_epoch_s"])
        go = lambda K, tier, r: attack(et, inp, lens, K=K, tier=tier, per_id_s=r, weight_units=t1, H=H, t2=t2, t3=t3,
                                       s_new=S_NEW["base"], vm=vm)
        ref = go(K_REF, "new" if t3 else "old", r_new)
        k0 = max(1, int(0.9 / (r_old * (et.text1(256)[0] + (0 if t2 else et.img[0])))))
        kc_old, runs = k_crit(lambda K: go(K, "old", r_old), max(1, k0 // 2))
        row = {"limits": lim, "spam_rate_per_id": {"new": r_new, "old": r_old}, "at_K_ref": ref,
               "K_crit_old": kc_old, "K_crit_old_runs": {str(k): {kk: v[kk] for kk in ("K1_old", "K1_new", "K2",
                                                                                    "text_old_p99_s")}
                                                         for k, v in runs.items()},
               "changes": int(t1) + int(t2) + int(t3)}
        row["S2_pass"] = (lim["refused_total"] <= OVER_MAX and ref["K1_old"] and ref["K1_new"] and ref["K2"])
        rows[name] = row
        log(name, lim, "K_ref:", {k: ref[k] for k in ("K1_old", "K1_new", "K2", "text_old_p99_s", "text_new_p99_s",
                                                     "query_p99_ms")}, "K_crit_old", kc_old)
        (OUT / "cp22.json").write_text(json.dumps({**res, "rows": rows}, indent=1, default=float), encoding="utf-8")
    order = ["T2", "T1", "T3", "T1+T2", "T2+T3", "T1+T3", "T1+T2+T3"]
    pick = next((n for n in order if rows[n]["S2_pass"]), None)
    res["rows"] = rows
    res["rule_S2"] = {"order": order, "pick": pick}
    # sensitivity of the pick's new-group limit to s_new
    if pick and "T3" in pick:
        t1 = "T1" in pick
        sens = {}
        for k, v in S_NEW.items():
            lim = pick_limits(H, v, t1, True)
            r = (lim["new"] / spam_w if t1 else lim["new"]) / lim["new_epoch_s"]
            a = attack(et, inp, lens, K=K_REF, tier="new", per_id_s=r, weight_units=t1, H=H, t2="T2" in pick, t3=True,
                       s_new=v, vm=vm)
            sens[k] = {"limits": lim, "at_K_ref": a, "pass": lim["refused_total"] <= OVER_MAX and a["K1_old"]
                       and a["K1_new"] and a["K2"]}
        res["s_new_sensitivity"] = sens
    log("rule S2", res["rule_S2"], res.get("s_new_sensitivity"))
    (OUT / "cp22.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("wrote")


if __name__ == "__main__":
    main()
