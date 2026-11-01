"""CP17: box functional stack + virtual accelerator + image posts.

  .venv/Scripts/python.exe src/sim_cp17.py            # everything (about 30 min, CPU only)
  .venv/Scripts/python.exe src/sim_cp17.py --decide   # host costs, CLIP tolerance, image delivery, vote cost only

Writes results/cp17/cp17.json and results/cp17/trace_box_peak_hour.json (Perfetto).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only (project/CLAUDE.md 5)

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_cp16 as s16  # noqa: E402
from sim import community as cm, ml, policy as po  # noqa: E402
from sim.box import Box, Shared, World, h, sub_key  # noqa: E402
from sim.delivery import ImageDelivery  # noqa: E402
from sim.stack import Accel, Cost, DayNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cp17"
DATA = ROOT / "data" / "reddit" / "reddit_title_text_2010.jsonl.gz"
PI5 = 2.93               # Cortex-A76 = Core Ultra 9 185H / 2.93 (Geekbench 6, CP12 출처 기반 추정)
VERIFY_SINGLE_S = 26_542 / 300e6   # CP14: V4 single proof on 6 x 6, 26,542 cycles at 300 MHz (= 1 / 11,303 s)
W3_EXTRA = 32 * 11 / 26_785        # one more public input: <= 32 window additions x 11 Fp mults (추정)
THUMB_B = 10e3                     # (추정) feed thumbnail
EMB_B = 516                        # INT8 512-d + scale
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ host costs (laptop, x PI5)

def host_costs(enc: ml.Encoder) -> dict:
    """예비 측정: the power preflight of project/CLAUDE.md 5 was not confirmed for this run."""
    def per(fn, n):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n

    sk = Ed25519PrivateKey.generate()
    pk, msg = sk.public_key(), os.urandom(1100)
    sig = sk.sign(msg)
    key, nonce = os.urandom(32), os.urandom(12)
    ct_s = ChaCha20Poly1305(key).encrypt(nonce, os.urandom(650), None)
    ct_l = ChaCha20Poly1305(key).encrypt(nonce, os.urandom(177_000), None)
    d_s = per(lambda: ChaCha20Poly1305(key).decrypt(nonce, ct_s, None), 3000)
    d_l = per(lambda: ChaCha20Poly1305(key).decrypt(nonce, ct_l, None), 300)
    seen = {}
    rln = per(lambda: seen.setdefault((1, h("nul", 12345, os.urandom(4))), 1), 3000)
    V = np.random.default_rng(0).standard_normal((8000, 384)).astype(np.float32)
    q = V[0]
    srch = per(lambda: np.argpartition(V @ q, -10)[-10:], 300)
    items = [(float(i), i) for i in np.random.default_rng(1).random(8000)]
    feed = per(lambda: sorted(items, reverse=True)[:25], 100)
    lst = []
    ins = per(lambda: lst.append(q), 10000)
    tok = per(lambda: enc.tokenize(["a short title about the community and what people said there"]), 300)
    laptop = {"ed25519_verify": per(lambda: pk.verify(sig, msg), 3000), "decrypt_650B": d_s, "decrypt_177KB": d_l,
              "rln_check": rln, "search_8000": srch, "feed_8000": feed, "insert": ins, "tokenize": tok}
    return laptop


def cost_model(laptop: dict) -> Cost:
    per_b = (laptop["decrypt_177KB"] - laptop["decrypt_650B"]) / (177_000 - 650)
    return Cost(sig=laptop["ed25519_verify"] * PI5, nullifier=laptop["rln_check"] * PI5,
                decrypt_fixed=max(laptop["decrypt_650B"] - per_b * 650, 0) * PI5, decrypt_per_b=per_b * PI5,
                insert=(laptop["insert"] + laptop["tokenize"]) * PI5, tally=laptop["rln_check"] * PI5,
                search_fixed=20e-6 * PI5, search_per_vec=laptop["search_8000"] / 8000 * PI5,
                feed_fixed=20e-6 * PI5, feed_per_item=laptop["feed_8000"] / 8000 * PI5,
                verify=VERIFY_SINGLE_S * (1 + W3_EXTRA), image_enc=ml.chip_image_s())


# ------------------------------------------------------------------ decision 1 inputs: CLIP tolerance

def tolerance(n: int = 300) -> dict:
    imgs = ml.synth_images(range(10_000, 10_000 + n))
    e32, e8 = ml.ImageEncoder("fp32"), ml.ImageEncoder("int8")
    a, b = e32.encode(imgs), e8.encode(imgs)
    same = (a * b).sum(1)
    cross = a @ a.T
    off = cross[~np.eye(n, dtype=bool)]
    b2 = e8.encode(imgs[:20])
    a1 = ml.ImageEncoder("fp32", threads=1).encode(imgs[:20])
    return {"n": n, "same_fp32_int8": {"min": float(same.min()), "p001": float(np.percentile(same, 0.1)),
                                       "p50": float(np.median(same))},
            "cross_fp32": {"max": float(off.max()), "p99": float(np.percentile(off, 99)),
                           "p50": float(np.median(off))},
            "int8_rerun_bit_exact": bool(np.array_equal(b[:20], b2)),
            "fp32_threads_1_vs_all_bit_exact": bool(np.array_equal(a[:20], a1)),
            "fp32_threads_1_vs_all_max_abs": float(np.abs(a[:20] - a1).max()),
            "images": "synthetic (ml.synth_image), not photos"}


# ------------------------------------------------------------------ decision 1: delivery

def biggest_topic(M: int = 10_000):
    p = s16.BASE.with_(users=M)
    c = cm.build(p)
    hb, ht = po.holders(c, "P-sub")
    on = np.random.default_rng(19).random(M) < p.online
    big = np.bincount(ht).argmax()
    return p, np.unique(hb[(ht == big) & on[hb]])


def delivery(mode: str, up: float, members, p, n_honest: int, n_false: int = 0, seed: int = 21) -> dict:
    env = p.overhead_bytes + p.post_bytes + THUMB_B + (EMB_B if mode == "I3" else 0)
    d = ImageDelivery(members, p.users, mode, uplink_bps=up, hop_s=p.hop_s, enc_s=ml.chip_image_s(),
                      full_bytes=p.overhead_bytes + p.post_bytes + p.image_bytes, env_bytes=env, seed=seed)
    authors = d.rng.choice(members, size=n_honest + n_false)
    for i, a in enumerate(authors):   # 10 s apart, as CP16's propagation test
        d.eng.at(10.0 * i, d.publish, int(a), i >= n_honest)
    d.eng.run()
    return d.summary(members)


def decide_images(tol: dict) -> dict:
    p, members = biggest_topic()
    res = {"holders": int(len(members)), "image_enc_chip_ms": ml.chip_image_s() * 1e3, "runs": {}}
    for up in (5e6, 20e6, 100e6):
        for mode in ("I1", "I2", "I3"):
            r = delivery(mode, up, members, p, 300)
            res["runs"][f"{mode} {up / 1e6:g} Mbps"] = r
            log("deliv", mode, up, r["searchable_s"], r["encodes_per_post_per_box"])
    res["I3_attack_20Mbps"] = delivery("I3", 20e6, members, p, 20, 20, seed=23)
    log("I3 attack", res["I3_attack_20Mbps"].get("false_hidden_everywhere_s"))
    # R1
    verdict = {}
    for mode in ("I1", "I2", "I3"):
        r = res["runs"][f"{mode} 20 Mbps"]
        ok = r["searchable_s"]["p99"] <= 5.0
        why = [] if ok else ["freshness"]
        if mode == "I3":
            tau = tol["same_fp32_int8"]["p001"]
            a = tol["cross_fp32"]["max"] < tau          # another picture's embedding would be caught
            b = res["I3_attack_20Mbps"]["false_hidden_everywhere_s"]["p99"] <= 60
            cc = res["runs"]["I3 20 Mbps"]["p_unchecked"] <= 1e-6
            why += [k for k, v in (("R1-2a separation", a), ("R1-2b removal", b), ("R1-2c unchecked", cc)) if not v]
        verdict[mode] = {"pass": not why, "fails": why, "p99_box_encodes": r["encodes_per_post_per_box"]["p99_box"],
                         "tx_mean": r["tx_bytes_per_post_per_box"]["mean"]}
    ok = [m for m in verdict if verdict[m]["pass"]]
    pick = min(ok, key=lambda m: (round(verdict[m]["p99_box_encodes"], 3), verdict[m]["tx_mean"])) if ok else "none"
    res["R1"] = {"verdict": verdict, "pick": pick}
    return res


# ------------------------------------------------------------------ decision 2: vote proof

def decide_votes() -> dict:
    table = {
        "W1 RLN only": {"one_vote_per_post": False, "rate_limited": True, "unlinkable": True, "publics": 5},
        "W2 post nullifier only": {"one_vote_per_post": True, "rate_limited": False, "unlinkable": True, "publics": 4},
        "W3 RLN + post nullifier": {"one_vote_per_post": True, "rate_limited": True, "unlinkable": True, "publics": 6},
    }
    for v in table.values():   # cost relative to CP14's 5-input single proof: one window pass per extra input
        v["verify_cost_vs_cp14"] = 1 + (v["publics"] - 5) * W3_EXTRA
        v["proofs_per_s_single"] = 1 / (VERIFY_SINGLE_S * v["verify_cost_vs_cp14"])
    ok = [k for k, v in table.items() if v["one_vote_per_post"] and v["rate_limited"] and v["unlinkable"]]
    return {"table": table, "W4": "excluded: CP16 bundles held 1.00 vote; longer windows break Q2 (60 s)",
            "pick": min(ok, key=lambda k: table[k]["verify_cost_vs_cp14"]), "extra_mults_per_input": 32 * 11}


# ------------------------------------------------------------------ the day

def load_rows(n: int, rng) -> list[dict]:
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    pick = rng.choice(len(rows), size=n, replace=False)
    return [rows[i] for i in pick]


def day_run(cost: Cost, M: int = 10_000, n_sample: int = 20, seed: int = 17) -> dict:
    p = s16.BASE.with_(users=M)
    c = cm.build(p)
    hb, ht = po.holders(c, "P-sub")
    rng = np.random.default_rng(seed)
    on = rng.random(M) < p.online
    T = c.n_topics
    ROOM, MEMB = T, T + 1
    members = {t: np.unique(hb[(ht == t) & on[hb]]) for t in range(T)}
    online = np.nonzero(on)[0]
    room_users = np.sort(rng.choice(online, 8, replace=False))
    members[ROOM], members[MEMB] = room_users, online
    room_key = os.urandom(32)

    # sample boxes: the two room members first, then the busiest and the median box by expected load, then random
    rates = cm.topic_rates(c, "V1")
    tot_rate = rates["post"] + rates["comment"] + rates["vote"]
    exp_load = np.bincount(hb, weights=tot_rate[ht], minlength=M)
    order = online[np.argsort(exp_load[online])]
    pick = [int(room_users[0]), int(room_users[1]), int(order[-1]), int(order[len(order) // 2])]
    for b in rng.permutation(online):
        if len(pick) >= n_sample:
            break
        if int(b) not in pick:
            pick.append(int(b))
    W, S = World(M, seed=seed), Shared()
    sample = {b: Box(b, c.subs[b], {ROOM: room_key} if b in room_users else {}, S, {"text": 0, "image": 0})
              for b in pick}
    for b in sample.values():
        b.topics.add(MEMB)

    # content and encoders
    enc = ml.Encoder("int8", threads=0)
    ienc = ml.ImageEncoder("int8")
    DAY = cm.DAY
    n_bp = rng.poisson(rates["post"] * 30 * DAY)
    n_bc = rng.poisson(rates["comment"] * 30 * DAY)
    # today's specs
    specs = []
    ok = on[c.su]
    su, st = c.su[ok], c.st[ok]
    for hr in range(24):
        f = p.diurnal[hr]
        for kind in ("post", "comment", "vote"):
            lam = c.pair_rate(kind)[ok] * f * 3600
            n = rng.poisson(lam)
            for u, t, k in zip(su[n > 0], st[n > 0], n[n > 0]):
                for t0 in hr * 3600 + rng.random(k) * 3600:
                    specs.append({"t": float(t0), "kind": kind, "user": int(u), "topic": int(t)})
        n_s = rng.poisson(0.4 * p.dau_frac / DAY * f * 3600, len(online))
        n_f = rng.poisson(10 * p.dau_frac / DAY * f * 3600, len(online))
        for b, k in zip(online[n_s > 0], n_s[n_s > 0]):
            for t0 in hr * 3600 + rng.random(k) * 3600:
                specs.append({"t": float(t0), "kind": "search", "user": int(b)})
        for b, k in zip(online[n_f > 0], n_f[n_f > 0]):
            for t0 in hr * 3600 + rng.random(k) * 3600:
                specs.append({"t": float(t0), "kind": "feed", "user": int(b)})
    for t0 in np.sort(rng.random(50) * DAY):
        specs.append({"t": float(t0), "kind": "room", "user": int(rng.choice(room_users)), "topic": ROOM})
    for s in specs:
        if s["kind"] == "post":
            s["image"] = bool(rng.random() < p.image_frac)
    texts_needed = int(n_bp.sum() + n_bc.sum()) + sum(s["kind"] in ("post", "comment", "room") for s in specs) + 40
    rows = load_rows(texts_needed, rng)
    log("specs", len(specs), "backlog", int(n_bp.sum()), int(n_bc.sum()), "rows", len(rows))

    # attacks and deletes (specs with tags)
    text_specs = [s for s in specs if s["kind"] in ("post", "comment") and 3600 < s["t"] < DAY - 7200]
    for s in rng.choice(text_specs, 10, replace=False):
        s["delete_at"] = s["t"] + 60 + rng.random() * 3540
    vote_specs = [s for s in specs if s["kind"] == "vote" and s["t"] < DAY - 3600]
    for i, s in enumerate(rng.choice(vote_specs, 3, replace=False)):
        s["dup_id"] = i
        specs.append({"t": s["t"] + 120, "kind": "vote", "user": s["user"], "topic": s["topic"], "dup_of": i})
    spammer = int(rng.choice(members[0]))
    t_spam = 20 * 3600 + 300.0                         # inside one 600 s epoch at the peak hour
    for j in range(25):
        specs.append({"t": t_spam + 5 * j, "kind": "comment", "user": spammer, "topic": 0, "attack": "rate",
                      "reuse_k": j if j < 20 else j - 20})
    for j in range(5):
        t_bad = int(rng.integers(T))
        specs.append({"t": float(rng.random() * DAY), "kind": "comment", "user": int(rng.choice(members[t_bad])),
                      "topic": t_bad, "attack": "bad_proof"})
    specs.sort(key=lambda s: s["t"])

    # embed every text up front (same results a box would compute; computed once)
    row_i = iter(range(len(rows)))
    def text_for(kind):
        r = rows[next(row_i)]
        if kind == "post":
            return {"title": r["title"], "body": r["body"][:2000]}, r["title"] + "\n\n" + r["body"][:2000]
        b = r["body"][:180]
        return {"body": b}, b
    backlog_specs = []
    for t in range(T):
        for kind, n in (("post", n_bp[t]), ("comment", n_bc[t])):
            for _ in range(int(n)):
                backlog_specs.append({"t": float(-rng.random() * 30 * DAY), "kind": kind, "topic": t,
                                      "image": bool(kind == "post" and rng.random() < p.image_frac)})
    backlog_specs.sort(key=lambda s: s["t"])
    all_text = backlog_specs + [s for s in specs if s["kind"] in ("post", "comment", "room")]
    for s in all_text:
        s["body"], s["text"] = text_for("post" if s["kind"] == "post" else "comment")
    log("embedding", len(all_text), "texts")
    vec = []
    toks = []
    for i in range(0, len(all_text), 64):
        chunk = [s["text"] for s in all_text[i:i + 64]]
        feed = enc.tokenize(chunk)
        vec.append(enc.encode_tokens(feed))
        toks += [int(x) for x in feed["attention_mask"].sum(1)]
    vec = np.concatenate(vec)
    for s, v, k in zip(all_text, vec, toks):
        s["vec"], s["tokens"] = v, k
    img_specs = [s for s in all_text if s.get("image")]
    log("embedding", len(img_specs), "images")
    seeds = rng.integers(1 << 30, size=len(img_specs))
    ivec = ienc.encode(ml.synth_images(seeds)) if img_specs else []
    for s, sd, v in zip(img_specs, seeds, ivec):
        s["seed"], s["ivec"] = int(sd), v

    # backlog: authors are random holders of the topic; boxes already synced (no gossip)
    holders_of = {t: hb[ht == t] for t in range(T)}
    score0 = {}
    count_t = np.zeros(T)
    count_i = np.zeros(T)
    for s in backlog_specs:
        u = int(rng.choice(holders_of[s["topic"]]))
        size = p.overhead_bytes + (p.post_bytes if s["kind"] == "post" else p.comment_bytes)
        m = W.make(u, s["kind"], s["topic"], s["t"], s["body"], size, sub_key(s["topic"]),
                   image=s.get("seed"), tokens=s["tokens"])
        S.emb[m.mid] = s["vec"]
        if s.get("image"):
            S.img[m.mid] = s["ivec"]
            count_i[s["topic"]] += 1
        count_t[s["topic"]] += 1
        votes = rng.poisson(rng.lognormal(np.log(9) - 0.5 * 1.5 ** 2, 1.5))
        score0[m.mid] = int(votes - 2 * rng.binomial(votes, 0.1))
        for b in sample.values():
            if m.topic in b.topics:
                b.apply(m, m.t0, b.validate(m))
                b.items[m.mid]["score"] = score0[m.mid]
                if s.get("image"):
                    b.index("image", m.mid)
    W.over_limit = 0
    n_backlog = len(W.msgs)
    log("backlog made", n_backlog)

    watch = (pick[2], pick[3])
    acc = Accel(M, watch=watch, window=(20 * 3600, 21 * 3600))
    net = DayNet(members, M, W, S, sample, cost, acc, uplink_bps=p.uplink_bps, hop_s=p.hop_s, membership_topic=MEMB)
    net.index_n[:] = np.bincount(hb, weights=(count_t + count_i)[ht], minlength=M)

    topic_items = defaultdict(list)          # topic -> [(t, mid)] eligible vote / comment targets
    for m in W.msgs:
        if m.kind == "post" or m.kind == "comment":
            topic_items[m.topic].append((m.t0, m.mid))
    truth = defaultdict(int)
    dup_target, tags = {}, defaultdict(list)

    def target_in(topic, t, posts_only=False):
        cand = [mid for t0, mid in topic_items[topic][-300:] if t0 < t - 60
                and (not posts_only or W.msgs[mid].kind == "post")]
        return int(rng.choice(cand)) if cand else None

    def do(s):
        t, k, u = s["t"], s["kind"], s["user"]
        if k == "search":
            net.search(u, 12)
            return
        if k == "feed":
            net.feed(u)
            return
        topic = s["topic"]
        key = room_key if topic == ROOM else sub_key(topic)
        if k == "vote":
            tgt = dup_target[s["dup_of"]] if "dup_of" in s else target_in(topic, t)
            if tgt is None:
                return
            if "dup_id" in s:
                dup_target[s["dup_id"]] = tgt
            d = 1 if rng.random() < 0.9 else -1
            m = W.make(u, "vote", topic, t, {"target": tgt, "dir": d}, p.overhead_bytes + p.vote_bytes, key,
                       target=tgt, truth={"dir": d})
            if "dup_of" in s:
                tags["double_vote"].append(m.mid)
        else:
            body = dict(s["body"])
            if k == "comment":
                body["parent"] = target_in(topic, t, posts_only=True)
            img = s.get("image", False)
            size = p.overhead_bytes + (p.post_bytes + (THUMB_B if img else 0) if k == "post" else p.comment_bytes)
            m = W.make(u, k, topic, t, body, size, key, image=s.get("seed") if img else None, tokens=s["tokens"],
                       pull_bytes=p.image_bytes if img else 0.0, proof_ok=s.get("attack") != "bad_proof",
                       reuse_k=s.get("reuse_k"))
            S.emb[m.mid] = s["vec"]
            if img:
                S.img[m.mid] = s["ivec"]
                tags["image"].append(m.mid)
            if s.get("attack"):
                tags[s["attack"]].append(m.mid)
            if k != "room":
                topic_items[topic].append((t, m.mid))
            if "delete_at" in s:
                net.eng.at(s["delete_at"], delete, u, m.mid, topic)
        net.publish(m, u)
        if m.kind == "vote" and net.verdict[m.mid][0] == "ok":
            truth[m.truth["target"]] += m.truth["dir"]

    def delete(u, target, topic):
        m = W.make(u, "delete", topic, net.eng.t, {"target": target}, p.overhead_bytes + 40, sub_key(topic),
                   target=target)
        tags["delete"].append(m.mid)
        tags["deleted_target"].append(target)
        net.publish(m, u)

    for s in specs:
        net.eng.at(s["t"], do, s)
    log("running the day")
    t_wall = time.time()
    net.eng.run()
    wall = time.time() - t_wall
    log("day done", net.eng.events, "events", round(wall), "s")
    return evaluate(net, sample, W, S, members, tags, truth, score0, ienc, enc, rng, on, acc, p, wall,
                    n_backlog, watch, ROOM, room_users, spammer)


def evaluate(net, sample, W, S, members, tags, truth, score0, ienc, enc, rng, on, acc, p, wall, n_backlog,
             watch, ROOM, room_users, spammer) -> dict:
    res = {"events": net.eng.events, "wall_s": wall, "messages_today": len(W.msgs) - n_backlog,
           "backlog_items": n_backlog, "sample_boxes": list(sample), "watched": list(watch),
           "verdict_mismatch_sample_vs_ref": net.mismatch, "honest_over_limit": W.over_limit}
    kinds = defaultdict(int)
    for m in W.msgs[n_backlog:]:
        kinds[m.kind] += 1
    res["today_by_kind"] = dict(kinds)
    # completeness: valid messages reach every online holder of their topic
    miss = tot = 0
    for m in W.msgs[n_backlog:]:
        if net.verdict[m.mid][0] == "ok":
            mem = members[m.topic]
            tot += len(mem)
            miss += len(mem) - int(net.got[m.mid][mem].sum())
    res["completeness"] = 1 - miss / max(tot, 1)

    # F1 search
    hit1 = hit10 = nq = 0
    img_rank1 = img_n = img_top10 = 0
    for b in sample.values():
        own = [mid for mid, v in b.items.items() if v["kind"] == "post" and v["title"] and mid not in b.deleted]
        for mid in rng.choice(own, min(50, len(own)), replace=False):
            q = enc.encode([b.items[mid]["title"]])[0]
            r = b.search(q, "text", 10)
            nq += 1
            hit10 += mid in r
            hit1 += bool(r) and r[0] == mid
        imgs = [mid for mid in b.idx_ids["image"] if mid not in b.deleted]
        for mid in rng.choice(imgs, min(20, len(imgs)), replace=False) if imgs else []:
            x = ml.synth_image(W.msgs[mid].image).astype(np.float32)
            x = np.clip(x * 0.9 + rng.normal(0, 3, x.shape), 0, 255).astype(np.uint8)   # re-saved, a bit darker
            r = b.search(ienc.encode(x[None])[0], "image", 10)
            img_n += 1
            img_rank1 += bool(r) and r[0] == mid
            img_top10 += mid in r
    res["F1"] = {"queries": nq, "hit10": hit10 / max(nq, 1), "hit1": hit1 / max(nq, 1),
                 "image_queries": img_n, "image_rank1": img_rank1 / max(img_n, 1),
                 "image_top10": img_top10 / max(img_n, 1)}
    res["F1"]["pass"] = bool(res["F1"]["hit10"] >= 0.95 and res["F1"]["image_rank1"] == 1.0)

    # F2 forgetting
    viol = 0
    for b in sample.values():
        fd = set(b.feed(10_000))
        for tgt in tags["deleted_target"]:
            if tgt in b.items:
                if tgt not in b.deleted or tgt in fd:
                    viol += 1
                if b.items[tgt]["title"]:
                    if tgt in b.search(enc.encode([b.items[tgt]["title"]])[0], "text", 10):
                        viol += 1
    dd = []
    for mid in tags["delete"]:
        mem = members[W.msgs[mid].topic]
        dd.append(np.nanmax(net.arr[mid][mem]) - W.msgs[mid].t0)
    res["F2"] = {"deletes": len(tags["delete"]), "violations": viol, "spread_p99_s": float(np.percentile(dd, 99)),
                 "pass": viol == 0 and float(np.percentile(dd, 99)) <= 5.0}

    # F3 rooms
    room_msgs = [m for m in W.msgs[n_backlog:] if m.kind == "room"]
    mem_ok = all(any(x[1] == m.mid and x[2] == "ok" for x in sample[b].log) for b in sample if b in room_users
                 for m in room_msgs)
    non = [b for b in sample if b not in room_users]
    leaked = sum(1 for b in non for x in sample[b].log if W.msgs[x[1]].kind == "room")
    opened = sum(1 for b in non for m in room_msgs for t in sample[b].topics if S.open(m, sub_key(t)) is not None)
    res["F3"] = {"room_msgs": len(room_msgs), "members_in_sample": int(sum(b in room_users for b in sample)),
                 "members_decrypted_all": bool(mem_ok), "nonmember_received": leaked, "nonmember_opened": opened,
                 "pass": bool(mem_ok) and leaked == 0 and opened == 0}

    # F4 votes, slashing
    dv = tags["double_vote"]
    rate = tags["rate"]
    slash_msgs = [m for m in W.msgs[n_backlog:] if m.kind == "slash"]
    commit = W.ids[spammer].commit
    banned_all = all(commit in b.banned for b in sample.values() if on[b.bid])
    res["F4"] = {"double_votes": [net.verdict[m][0] for m in dv], "double_vote_reach": [net.n[m] for m in dv],
                 "rate_verdicts": [net.verdict[m][0] for m in rate], "slash_msgs": len(slash_msgs),
                 "spammer_banned_at_all_sample_boxes": bool(banned_all),
                 "after_slash_reach": [net.n[m] for m in rate if net.verdict[m][0] == "bad_proof"]}
    res["F4"]["pass"] = (all(v == "double_vote" for v in res["F4"]["double_votes"])
                         and "slash" in res["F4"]["rate_verdicts"] and banned_all)

    # F5 invalid proofs
    # a relay drops it: only the author's box and its mesh peers ever hold it (mesh links are symmetric, so a box
    # can have more than D peers)
    bp = tags["bad_proof"]
    beyond = []
    for mid in bp:
        m = W.msgs[mid]
        a = m.truth["user"]
        first = set(net.mesh[m.topic].get(a, np.array([], int)).tolist()) | {a}
        beyond.append(int(np.sum(net.got[mid])) - len(first & set(np.nonzero(net.got[mid])[0].tolist())))
    res["F5"] = {"n": len(bp), "reach": [net.n[m] for m in bp], "beyond_first_hop": beyond,
                 "author_degree": [len(net.mesh[W.msgs[m].topic].get(W.msgs[m].truth["user"], ())) for m in bp],
                 "pass": all(b == 0 for b in beyond)}

    # F6 tallies
    bad = checked = 0
    for b in sample.values():
        for mid, v in b.items.items():
            if v["kind"] in ("post", "comment"):
                checked += 1
                bad += v["score"] != score0.get(mid, 0) + truth.get(mid, 0)
    res["F6"] = {"items_checked": checked, "mismatched": bad, "pass": bad == 0}

    # searchable delay at sample boxes
    st = [t - W.msgs[k[1]].t0 for k, t in net.searchable.items() if len(k) == 2 and k[1] >= n_backlog]
    si = [t - W.msgs[k[1]].t0 for k, t in net.searchable.items() if len(k) == 3]
    res["searchable_s"] = {"text_p99": float(np.percentile(st, 99)) if st else None,
                           "image_p99": float(np.percentile(si, 99)) if si else None, "image_n": len(si)}
    res["search_latency_ms"] = {"p50": float(np.percentile(net.search_lat, 50) * 1e3),
                                "p99": float(np.percentile(net.search_lat, 99) * 1e3), "n": len(net.search_lat)}

    # per-box unit work, all online boxes
    onb = np.nonzero(on)[0]
    peak = int(np.argmax(p.diurnal))
    work = {}
    for u, B in acc.busy.items():
        x = B[onb, peak] / 3600
        work[u] = {"peak_hour_util_p50": float(np.percentile(x, 50)), "peak_hour_util_p99": float(np.percentile(x, 99)),
                   "peak_hour_util_max": float(x.max()), "day_s_p50": float(np.percentile(B[onb].sum(1), 50)),
                   "day_s_p99": float(np.percentile(B[onb].sum(1), 99)), "ops_day_p99": float(np.percentile(acc.ops[u][onb], 99))}
    enc_u = (acc.busy["enc_text"][onb, peak] + acc.busy["enc_image"][onb, peak]) / 3600
    res["work"] = work
    res["R3"] = {"encoder_peak_util_p99_box": float(np.percentile(enc_u, 99)),
                 "rule_lt_1pct": bool(np.percentile(enc_u, 99) < 0.01)}
    res["energy_mJ_day_p99_box"] = {
        "enc": float(np.percentile(ml.chip_mj(acc.busy["enc_text"][onb].sum(1) + acc.busy["enc_image"][onb].sum(1)), 99)),
        "verify": float(np.percentile(acc.ops["verify"][onb], 99) * 0.0685)}   # CP14 single, f = 1 row (추정)
    acc.trace.write(OUT / "trace_box_peak_hour.json")
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    enc = ml.Encoder("int8", threads=0)
    res["host_laptop_s"] = host_costs(enc)
    cost = cost_model(res["host_laptop_s"])
    res["cost_box_s"] = cost.__dict__
    res["chip"] = {"text_164tok_ms": ml.chip_text_s(164) * 1e3, "query_12tok_ms": ml.chip_text_s(12) * 1e3,
                   "image_ms": ml.chip_image_s() * 1e3, "image_mJ": ml.chip_mj(ml.chip_image_s()),
                   "macs_text_164": ml.text_macs(164), "macs_image": ml.image_macs()}
    log("costs", res["chip"])
    res["tolerance"] = tolerance()
    log("tolerance", res["tolerance"])
    res["images"] = decide_images(res["tolerance"])
    log("R1", res["images"]["R1"])
    res["votes"] = decide_votes()
    log("R2", res["votes"]["pick"])
    (OUT / "cp17.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if "--decide" in sys.argv:
        return
    if res["images"]["R1"]["pick"] != "I2":
        log("note: the day run models image pulls as I2; pick was", res["images"]["R1"]["pick"])
    res["day"] = day_run(cost)
    log("day", {k: v for k, v in res["day"].items() if k[0] == "F" or k in ("R3", "completeness")})
    (OUT / "cp17.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("wrote", OUT / "cp17.json")


if __name__ == "__main__":
    main()
