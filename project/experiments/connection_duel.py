"""CP35 (journal/cp/CP35.md): does a connection policy make the verify unit unnecessary under an invalid-proof flood?

Two worlds with the same box, attacker and connection policy N-d; only where proofs are checked differs:
  W-cpu   proofs on the Pi 5 host CPU (sim_cp20.CPU_SINGLE)          W-chip  proofs on the verify unit 6x6 (CP14)
N-d: known neighbours pass; unknown connections get a token bucket of A per second (the rest are dropped before the
handshake at c_rej host seconds each); an admitted unknown connection may send one message until its first proof
checks, and an invalid proof closes it, so the attacker gets one invalid proof per admitted connection.
W-chip may instead keep CP19's N-b inside the cap (10 invalid proofs per admitted connection); the better is used.

  .venv/Scripts/python.exe -W ignore experiments/connection_duel.py   -> results/chip/cp35_connection_duel.json

The box model is CP19/CP20 at tag cp30 (as chip/model/chip_w16.py), text times from the CP34 16-bit mapping.
Every number is an estimate (model); c_rej, the attacker's attempt rate and the join model are assumptions (CP35.md).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "chip" / "model"))
import encoder_resize_w16 as rz  # noqa: E402

OUT = ROOT / "results" / "chip" / "cp35_connection_duel.json"
T0 = time.time()

A_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000, 10_000]     # unknown connections admitted per second
LAMBDAS = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5]                   # attacker connection attempts per second
PHIS = [0.5, 0.9, 1.0]                                     # share of a newcomer's candidates under attack
C_REJ = {"base": 10e-6, "low": 2e-6, "high": 50e-6}       # host s per attempt dropped before the handshake (assumption)
LINK_SYN_PER_S = 100e6 / (74 * 8)                          # attempts that fill 100 Mbps with 74 B SYNs (estimate)
K_CAND, D_JOIN, TRY_S, J_S = 10, 3, 0.1, 60.0              # newcomer: candidates, neighbours needed, s per try, J
DUR, SEEDS = 60.0, 3
CHIP_M = "6x6"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ newcomer join (Monte Carlo)

def join_p99(A: float, lam: float, phi: float, n: int = 10_000, seed: int = 35) -> float:
    """P99 seconds until a newcomer holds D_JOIN neighbours among K_CAND candidates, round(phi * K_CAND) of them under
    attack. A free candidate accepts on the first try; an attacked one with probability p = A / (A + lam) per try.
    Each try takes TRY_S. Free candidates are reached within one round of the list; the rest come from attacked
    candidates, so tries = (free ones used) + NegBin(r, p) (all attacked candidates have the same p)."""
    k_att = round(phi * K_CAND)
    k_free = K_CAND - k_att
    used_free = min(k_free, D_JOIN)
    r = D_JOIN - used_free
    rng = np.random.default_rng(seed)
    if r == 0:                      # upper bound: attacked candidates passed on the way may also accept
        free_seen = np.cumsum(rng.random((n, K_CAND)).argsort(axis=1) < k_free, axis=1)
        pos = np.argmax(free_seen >= D_JOIN, axis=1) + 1
        return float(np.percentile(pos * TRY_S, 99))
    p = A / (A + lam)
    tries = used_free + rng.negative_binomial(r, p, n) + r
    return float(np.percentile(tries * TRY_S, 99))


def a_star(lam: float, phi: float) -> int | None:
    for A in A_GRID:
        if join_p99(A, lam, phi) <= J_S:
            return A
    return None


# ------------------------------------------------------------------ box model (worker processes)

_W = {}


def _init(wt: str) -> None:
    wt = Path(wt)
    sys.path.insert(0, str(wt / "src"))
    import bench_cp1
    bench_cp1.DATA = rz.ROOT / "data" / "reddit" / bench_cp1.DATA.name
    import model_cp4
    model_cp4.MODEL_DIR = rz.ROOT / "models" / model_cp4.MODEL_DIR.name
    import sim_cp19 as c19
    import sim_cp20 as c20
    from sim_cp24 import fixed
    c19.Box.host = fixed(c19.Box.host)
    c20.Box20.host = fixed(c20.Box20.host)
    t1, bt = c19.EncTable.text1, c19.EncTable.batch

    def text1(self, n):
        s, J = t1(self, n)
        f, g = rz.fac(self.R, n, "w16")
        return s * f, J * g

    def batch(self, ls):
        s, J = bt(self, ls)
        f, g = rz.fac(self.R, len(ls) * max(ls), "w16")
        return s * f, J * g
    c19.EncTable.text1, c19.EncTable.batch = text1, batch

    h_conn = json.loads((wt / "results" / "cp19" / "cp19.json").read_text(encoding="utf-8"))["handshake"]["box_s"]
    _W.update(c19=c19, c20=c20, et=c19.EncTable(16, 2048), inp=c19.inputs(),
              lens=np.minimum(c19.token_lengths(), 256), h_conn=h_conn, vm=c20.machines()[CHIP_M])


def box_cell(args: tuple) -> dict:
    """One world with cap A against lam attempts/s: min(A, lam) connections/s admitted, the rest dropped at c_rej."""
    world, A, lam, per_conn, c_rej = args
    adm, rej = min(A, lam), max(0.0, lam - A)
    c19, et, inp, lens = _W["c19"], _W["et"], _W["inp"], _W["lens"]
    rows = []
    for s in range(SEEDS):
        kw = dict(vkind="cpu") if world == "W-cpu" else dict(vkind="chip", vm=_W["vm"])
        box = _W["c20"].Box20(et, inp["cost"], order="O-b", sched="Q2", vmode="single", h_conn=0.0, seed=350 + s, **kw)
        rng = np.random.default_rng(3500 + s)
        # every admitted connection costs a handshake; it then carries per_conn invalid proofs, but never more proofs
        # in total than the link carries (the attacker takes every token and spreads the proofs over them)
        for t in np.sort(rng.uniform(0, DUR, rng.poisson(adm * DUR))):
            box.eng.at(float(t), lambda _x, b=box, d=_W["h_conn"]: b.host(b.eng.t, d, "handshake"), None)
        n = rng.poisson(min(adm * per_conn, c19.FLOOD_PER_S) * DUR)
        for t in np.sort(rng.uniform(0, DUR, n)):
            box.message(float(t), "flood", valid=False, record=False)
        if rej > 0:
            chunk = c_rej * rej * 0.01
            for t in np.arange(0.0, DUR, 0.01):
                box.eng.at(float(t), lambda _x, b=box, d=chunk: b.host(b.eng.t, d, "reject"), None)
        c19.legit(box, inp, lens, 0, DUR, lambda t: 20, probe_dt=0.5, query_dt=1.0, rng=rng)
        box.run(DUR + 120)
        rows.append(box)
    lt = np.concatenate([c19.lat(b, "text") for b in rows])
    lq = np.concatenate([c19.lat(b, "query") for b in rows])
    lim = c19.LIMITS
    r = {"world": world, "A": A, "lam": lam, "admitted_per_s": adm, "per_conn": per_conn, "rejected_per_s": rej,
         "c_rej_s": c_rej,
         "host_cores": float(np.mean([b.host_s for b in rows]) / DUR),
         "chip_W": float(np.mean([(b.v_J + b.e_J) for b in rows]) / DUR),
         "text_p99_s": c19.pct(lt), "unfinished_text": int(np.isinf(lt).sum()), "query_p99_ms": c19.pct(lq) * 1e3}
    r["K"] = {"K1": r["text_p99_s"] is not None and r["text_p99_s"] <= lim["K1_text_box_s"],
              "K2": r["query_p99_ms"] <= lim["K2_query_s"] * 1e3,
              "K4": r["host_cores"] <= lim["K4_host_cores"], "K5": r["chip_W"] <= lim["K5_chip_W"]}
    r["pass"] = all(r["K"].values())
    return r


def same_frame(_x=None) -> dict:
    """R0: CP19 N-b on the chip at link rate, which CP34 cp34_chip.json reports (S4 chip W, verify busy)."""
    c19 = _W["c19"]
    s4 = c19.s4_flood(_W["et"], _W["inp"], _W["lens"], "O-b", "N-b", "invalid", _W["h_conn"])
    return {"chip_W": s4["chip_W"], "verify_busy": s4["verify_busy"], "host_cores": s4["host_offered_cores"]}


# ------------------------------------------------------------------ main

def main() -> None:
    from at_tag import tag_tree
    wt = str(tag_tree("cp30"))
    res = {"what": "CP35 connection policy vs chip (journal/cp/CP35.md)", "label": "estimate (model)",
           "assumptions": {"c_rej_s": C_REJ, "lambdas": LAMBDAS, "phis": PHIS, "A_grid": A_GRID,
                           "link_syn_per_s": LINK_SYN_PER_S, "K_cand": K_CAND, "D_join": D_JOIN, "try_s": TRY_S,
                           "J_s": J_S, "dur_s": DUR, "seeds": SEEDS, "chip_machine": CHIP_M}}
    res["join"] = {f"{lam:g}|{phi}": {"A_star": a_star(lam, phi),
                                      "p99_s_by_A": {A: join_p99(A, lam, phi) for A in A_GRID}}
                   for lam in LAMBDAS for phi in PHIS}
    log("join", {k: v["A_star"] for k, v in res["join"].items()})

    jobs = []
    for world in ("W-cpu", "W-chip"):
        for A in A_GRID:
            for lam in LAMBDAS:
                jobs.append((world, A, lam, 1, C_REJ["base"]))
                if world == "W-chip":
                    jobs.append((world, A, lam, 10, C_REJ["base"]))
    with ProcessPoolExecutor(max_workers=14, initializer=_init, initargs=(wt,)) as ex:
        r0 = ex.submit(same_frame).result()
        res["same_frame"] = {"cp35": r0, "cp34": json.loads((rz.RES / "cp34_chip.json").read_text())["power"]["w16 16x16 2MB"]}
        log("same frame", r0)
        cells = []
        for r in ex.map(box_cell, jobs):
            cells.append(r)
            log(r["world"], "A", r["A"], "lam", r["lam"], "per_conn", r["per_conn"], "cores",
                round(r["host_cores"], 3), "pass", r["pass"])
        res["box"] = cells
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

        verdict = {}
        sens = []
        for lam in LAMBDAS:
            for phi in PHIS:
                A = res["join"][f"{lam:g}|{phi}"]["A_star"]
                v = {"A_star": A}
                for world in ("W-cpu", "W-chip"):
                    if A is None:
                        v[world] = False
                        continue
                    ok = [c["pass"] for c in cells if c["world"] == world and c["A"] == A and c["lam"] == lam]
                    v[world] = any(ok)
                    for k in ("low", "high"):
                        sens.append((world, A, lam, 1, C_REJ[k]))
                verdict[f"{lam:g}|{phi}"] = v
        res["verdict"] = verdict
        res["sensitivity"] = list(ex.map(box_cell, sorted(set(sens))))
    only_chip = [k for k, v in verdict.items() if v.get("W-chip") and not v.get("W-cpu")]
    res["D1"] = ("not needed" if not only_chip else "narrowed") + (
        "" if any(v.get("W-chip") for v in verdict.values()) else " (chip passes no cell)")
    res["only_chip_cells"] = only_chip
    res["neither_cells"] = [k for k, v in verdict.items() if not v.get("W-chip") and not v.get("W-cpu")]
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("verdict", verdict, "D1", res["D1"])
    log("wrote", OUT)


if __name__ == "__main__":
    main()
