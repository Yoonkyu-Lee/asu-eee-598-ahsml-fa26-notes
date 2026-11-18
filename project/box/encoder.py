"""The box's encoders and the virtual chip (CP24): MiniLM, CLIP image and text towers on the CPU, one encoder queue by
priority (query, old text, new text, old pictures, new pictures; text up to 128 tokens), and for each job the time the
chip would take (chip/model/chip.py, 추정)."""
from __future__ import annotations

import time
import traceback

import numpy as np

from box_state import Node, log
from chip.model import chip as CHIPM   # the latest chip's numbers (CP31: one source)

CHIP = {"proof_s": CHIPM.PROOF_S, "image_s": CHIPM.IMAGE_S}     # chip/model/chip.py (추정)


def load(node: Node) -> None:
    from sim.ml import ClipText
    from sim import ml
    if node.cfg.get("text") == "fp32":                      # CP28: contract precision (CP3, CP27)
        node.text_enc = ml.Encoder("fp32", threads=1)
    else:
        try:
            node.text_enc = ml.Encoder("int8", threads=1)
        except Exception:                                  # e.g. an ONNX Runtime build without the INT8 kernels
            log("MiniLM INT8 failed, using FP32:", traceback.format_exc(limit=1).strip().splitlines()[-1])
            node.text_enc = ml.Encoder("fp32", threads=1)
    node.img_enc = ml.ImageEncoder("fp32", threads=1)    # FP32 = the FP16 tower's quality (CP20)
    node.clip_text = ClipText(threads=1)
    if node.cfg.get("text") == "fp32":       # CP28 chip: 16x16, 2 MB, FP16 text (chip/model/chip.py)
        node.chip_table = CHIPM.text_table("fp16_16x16")
    else:                                    # CP19 chip: 8x8, 1 MB, INT8 text
        node.chip_table = CHIPM.text_table("int8_8x8")


def enqueue(node: Node, cls: int, job: tuple) -> None:
    node.inc()
    node.encq.put((cls, next(node.seq), job))


def run(node: Node) -> None:
    """One encoder, lowest class first (CP22 T2/T3). The owner's queries do not wait here (class 0 = run now)."""
    while True:
        cls, _, job = node.encq.get()
        try:
            if job[0] == "text":
                _index_text(node, job[1], job[2])
            else:
                _index_image(node, job[1])
        except Exception:
            log("encoder job failed:", traceback.format_exc())
        finally:
            node.inc(-1)


def _index_text(node: Node, mid: int, feed: dict) -> None:
    t0 = time.perf_counter()
    e = node.text_enc.encode_tokens(feed)[0]
    ms = (time.perf_counter() - t0) * 1e3
    nt = int(feed["attention_mask"].sum())
    with node.lock:
        node.S.emb[mid] = e
        node.enc_tokens_max = max(node.enc_tokens_max, nt)
        node.chip_s += node.chip_table.text1(nt)[0]
        node.count["text"] += 1
        node.count["insert"] += 1
        if mid in node.box.items and mid not in node.box.deleted:
            node.box.index("text", mid)
            if node.st:
                node.st.ev("index", mid=str(mid), what="text", ms=ms, tokens=nt, replay=mid in node.replayed)


def _index_image(node: Node, mid: int) -> None:
    data = node.full.get(mid)
    if data is None:                                        # forgotten while it waited (CP25)
        return
    px = np.frombuffer(data, np.uint8).reshape(1, 224, 224, 3)
    t0 = time.perf_counter()
    e = node.img_enc.encode(px)[0]
    ms = (time.perf_counter() - t0) * 1e3
    with node.lock:
        node.S.img[mid] = e
        node.chip_s += CHIP["image_s"]
        node.count["image"] += 1
        node.count["insert"] += 1
        if mid in node.box.items and mid not in node.box.deleted:
            node.box.index("image", mid)
            if node.st:
                node.st.ev("index", mid=str(mid), what="image", ms=ms, replay=mid in node.replayed)
