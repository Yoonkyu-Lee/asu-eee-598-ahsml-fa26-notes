"""CP24 R1: CP19 and CP20 again with the fixed host (CP22's Box22.host), then every decision compared with the
committed results.

  .venv/Scripts/python.exe src/sim_cp24.py rerun     # about 15 min CPU: sim_cp19.main + sim_cp20.main -> results/cp24/rerun/
  .venv/Scripts/python.exe src/sim_cp24.py compare   # -> results/cp24/rerun/compare.json
  ... rerun --gap / compare --gap                       # R1b (added after R1's first look): -> results/cp24/rerun_gap/

The bug: sim_cp19.Box.host (and sim_cp20.Box20.host) booked work dated in the future (a picture's decryption, booked
when its pull is scheduled) on a core at once, so everything arriving before that time queued behind it. The fix
counts future work (total and hourly busy time) and books no core, as CP22 did.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
GAP = "--gap" in sys.argv
RR = RES / "cp24" / ("rerun_gap" if GAP else "rerun")


def fixed(orig):
    def host(self, t, dur, name="", back=False):
        if t > self.eng.t + 1e-9:
            self.host_s += dur
            h = int(t // 3600)
            if 0 <= h < len(self.host_hour):
                self.host_hour[h] += dur
            return t + dur
        return orig(self, t, dur, name, back)
    return host


def gapfill(cores_of):
    """R1b: every piece of host work, dated now or later, takes the earliest gap of its length on any allowed core
    (work-conserving; work arriving earlier fills idle time before later-dated work). CP22's fix instead lets
    later-dated work skip the cores, which also drops the contention of every chained step booked ahead."""
    from bisect import bisect_left, bisect_right
    eps = 1e-12

    def host(self, t, dur, name="", back=False):
        n = cores_of(self)
        cores = range(n // 2, n) if (back and self.live_first) else range(n)
        cal = self.__dict__.setdefault("_cal", [([], []) for _ in range(n)])   # per core: starts, ends (merged)
        now = self.eng.t
        best = None
        for i in cores:
            S, E = cal[i]
            k = bisect_right(E, now)
            if k:
                del S[:k], E[:k]
            st = t
            for j in range(bisect_right(E, t), len(S)):
                if S[j] >= st + dur:
                    break
                st = max(st, E[j])
            if best is None or st < best[0]:
                best = (st, i)
        st, i = best
        S, E = cal[i]
        en = st + dur
        p = bisect_left(S, st)
        if p > 0 and E[p - 1] >= st - eps:              # touches the previous busy block
            p -= 1
            st = S[p]
            del S[p], E[p]
        if p < len(S) and S[p] <= en + eps:             # touches the next one
            en = max(en, E[p])
            del S[p], E[p]
        S.insert(p, st)
        E.insert(p, en)
        self.host_free[i] = max(self.host_free[i], best[0] + dur)
        self.host_s += dur
        h = int(best[0] // 3600)
        if 0 <= h < len(self.host_hour):
            self.host_hour[h] += dur
        return best[0] + dur
    return host


def rerun() -> None:
    import sim_cp19 as c19
    import sim_cp20 as c20
    if GAP:
        c19.Box.host = gapfill(lambda b: c19.HOST_CORES)
        c20.Box20.host = gapfill(lambda b: b.nc)
    else:
        c19.Box.host = fixed(c19.Box.host)
        c20.Box20.host = fixed(c20.Box20.host)
    # sim_cp20 reads RES/cp14, RES/cp19 (the rerun) and OUT/cp20_bench.json: a mirror with copies of the inputs
    for sub in ("cp14", "cp19", "cp20"):
        (RR / sub).mkdir(parents=True, exist_ok=True)
    copies = [RR / "cp14" / "cp14_fixed.json", RR / "cp20" / "cp20_bench.json"]
    shutil.copy(RES / "cp14" / "cp14_fixed.json", copies[0])
    shutil.copy(RES / "cp20" / "cp20_bench.json", copies[1])
    try:
        c19.OUT = RR / "cp19"
        c19.main()
        c20.RES, c20.OUT = RR, RR / "cp20"
        c20.main()
    finally:
        for p in copies:
            p.unlink(missing_ok=True)


def flat(x, pre=""):
    """Every number in a result tree, keyed by its path."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{pre}/{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flat(v, f"{pre}[{i}]"))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out[pre] = float(x)
    else:
        out[pre] = x
    return out


def compare() -> None:
    res = {}
    for cp, picks in (("cp19", ("rule_Q", "rule_C", "rule_S")), ("cp20", ("rule_V2", "rule_F", "rule_A"))):
        old = json.loads((RES / cp / f"{cp}.json").read_text(encoding="utf-8"))
        new = json.loads((RR / cp / f"{cp}.json").read_text(encoding="utf-8"))
        r = {"decisions": {}, "moved_10pct": [], "n_numbers": 0}
        for k in picks:
            fo, fn = flat(old.get(k)), flat(new.get(k))
            nonnum = {p: (fo.get(p), fn.get(p)) for p in set(fo) | set(fn)
                      if not isinstance(fo.get(p), float) or not isinstance(fn.get(p), float)}
            r["decisions"][k] = {"same": all(a == b for a, b in nonnum.values()),
                                 "changed": {p: v for p, v in nonnum.items() if v[0] != v[1]}}
        fo, fn = flat(old), flat(new)
        for p in sorted(set(fo) & set(fn)):
            a, b = fo[p], fn[p]
            if isinstance(a, float) and isinstance(b, float):
                r["n_numbers"] += 1
                if abs(b - a) > 0.1 * max(abs(a), 1e-12) and abs(b - a) > 1e-9:
                    r["moved_10pct"].append([p, a, b])
        res[cp] = r
        print(cp, {k: v["same"] for k, v in r["decisions"].items()}, len(r["moved_10pct"]), "of", r["n_numbers"],
              "numbers moved > 10%")
    (RR / "compare.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    {"rerun": rerun, "compare": compare}[sys.argv[1]]()
