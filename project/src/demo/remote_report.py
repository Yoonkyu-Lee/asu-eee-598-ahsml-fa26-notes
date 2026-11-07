"""CP25: numbers from the boxes' events.jsonl files (each laptop sends its own; no plaintext in them).

  .venv/Scripts/python.exe src/demo/remote_report.py state/box0/events.jsonl box1.jsonl box2.jsonl [--out results/cp25/x.json]

The box number of a file is taken from a "box<i>" in its path, else from its position in the list.
- W1 settle: a text post's write time on the author's box -> the time it is searchable on another box that was on
  (took the post by gossip). Clock offsets come from each box's /hello round trips (offset = peer clock - my clock,
  NTP style: the shortest round trip within half an hour); both corrected and raw values are reported.
- W2 round trips per pair; picture settle (write -> picture searchable).
- W3 catching up: after each restart, the messages taken by sync in the first 5 min and the time until the last of
  them is searchable.
- W4 encoding time on each box's CPU (text per post, picture per picture).
- W5 what was written: counts, text tokens, picture share (no content).
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WINDOW = 1800.0          # s: clock offsets are taken from samples within half an hour (laptop clocks drift slowly)


def pct(xs, q):
    return float(np.percentile(xs, q, method="higher")) if len(xs) else None


def summ(xs) -> dict:
    xs = list(xs)
    return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99), "max": max(xs) if xs else None}


def load(paths: list[str]) -> dict[int, list[dict]]:
    out = {}
    for i, p in enumerate(paths):
        m = re.search(r"box(\d+)", p.replace("\\", "/"))
        b = int(m.group(1)) if m else i
        out[b] = [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def report(ev: dict[int, list[dict]]) -> dict:
    # clock offsets: off[(j, b)] = sorted (t, peer b's clock - j's clock, round trip) seen from j
    off = defaultdict(list)
    rtt = defaultdict(list)
    for j, rows in ev.items():
        for r in rows:
            if r["ev"] == "peer":
                off[(j, r["peer"])].append((r["t"], r["offset_ms"] / 1e3, r["rtt_ms"] / 1e3))
                rtt[(j, r["peer"])].append(r["rtt_ms"])
    for k in off:
        off[k].sort()

    def best(s: list, t: float) -> tuple | None:
        """The sample with the shortest round trip within WINDOW of t (its error is at most half its round trip;
        NTP picks the same way). Added after P4's first run, where single samples were off by up to rtt / 2."""
        lo, hi = bisect.bisect_left(s, (t - WINDOW,)), bisect.bisect_right(s, (t + WINDOW,))
        near = s[lo:hi] or [min(s, key=lambda x: abs(x[0] - t))]
        return min(near, key=lambda x: x[2])

    def offset(j: int, b: int, t: float) -> float | None:
        """b's clock - j's clock near time t (j's clock)."""
        if j == b:
            return 0.0
        for key, sign in (((j, b), 1), ((b, j), -1)):
            s = off.get(key)
            if s:
                return sign * best(s, t)[1]
        return None

    writes = {}
    for b, rows in ev.items():
        for r in rows:
            if r["ev"] == "write":
                writes[r["mid"]] = (b, r)
    gossip = {(j, r["mid"]) for j, rows in ev.items() for r in rows if r["ev"] == "recv" and r["via"] == "gossip"}
    settle = {"text": defaultdict(list), "image": defaultdict(list)}
    raw = {"text": defaultdict(list), "image": defaultdict(list)}
    no_offset = 0
    for j, rows in ev.items():
        for r in rows:
            if r["ev"] != "index" or r.get("replay") or r["mid"] not in writes:
                continue
            b, w = writes[r["mid"]]
            if j != b and (j, r["mid"]) not in gossip:          # it came by sync: catching up, not live (W3)
                continue
            o = offset(j, b, r["t"])
            if o is None:
                no_offset += 1
                o = 0.0
            raw[r["what"]][f"{b}->{j}"].append(r["t"] - w["t"])
            settle[r["what"]][f"{b}->{j}"].append(r["t"] + o - w["t"])
    out = {"boxes": sorted(ev), "no_offset_samples": no_offset}
    for what in ("text", "image"):
        others = [x for k, v in settle[what].items() if k.split("->")[0] != k.split("->")[1] for x in v]
        out[f"W1_settle_{what}_s"] = {"all_other_boxes": summ(others),
                                      "all_other_boxes_raw": summ(x for k, v in raw[what].items()
                                                                  if k.split("->")[0] != k.split("->")[1] for x in v),
                                      "by_pair": {k: summ(v) for k, v in sorted(settle[what].items())}}
    out["W1_pass_text_p99_le_5s"] = (out["W1_settle_text_s"]["all_other_boxes"]["p99"] or 1e9) <= 5.0
    out["W2_rtt_ms"] = {f"{j}->{b}": summ(v) for (j, b), v in sorted(rtt.items())}
    out["clock_offset_ms"] = {f"{j}->{b}": {"used": best(v, v[-1][0])[1] * 1e3, "bound": best(v, v[-1][0])[2] * 1e3 / 2,
                                            "every_sample": summ([x[1] * 1e3 for x in v])}
                              for (j, b), v in sorted(off.items())}
    # W3: catching up after each restart
    caught = []
    for j, rows in ev.items():
        starts = [r for r in rows if r["ev"] == "start"]
        for s in starts:
            t0 = s["t"]
            got = {r["mid"] for r in rows if r["ev"] == "recv" and r["via"] == "sync" and t0 <= r["t"] <= t0 + 300}
            idx = [r["t"] for r in rows if r["ev"] == "index" and r["mid"] in got and r["t"] >= t0]
            caught.append({"box": j, "restored": s["restored"], "replay_ms": s["ms"], "synced": len(got),
                           "caught_up_s": (max(idx) - t0) if idx else None})
    out["W3_catch_up"] = caught
    out["W4_encode_ms"] = {}
    for j, rows in ev.items():
        out["W4_encode_ms"][j] = {w: summ(r["ms"] for r in rows if r["ev"] == "index" and r["what"] == w)
                                  for w in ("text", "image")}
    ws = [w for _, w in writes.values()]
    posts = [w for w in ws if w["kind"] in ("post", "comment", "room")]
    out["W5_written"] = {"by_kind": dict(sorted(Counter(w["kind"] for w in ws).items())),
                         "text_tokens": summ(w["tokens"] for w in posts),
                         "photo_share": (sum(w["photo"] for w in posts) / len(posts)) if posts else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = report(load(a.files))
    s = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(s, encoding="utf-8")
    print(s if len(s) < 6000 else json.dumps({k: r[k] for k in r if not k.startswith("W2")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
