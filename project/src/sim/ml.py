"""Encoders for the box (run for real on the laptop CPU) and their chip-time model.

Text: all-MiniLM-L6-v2 INT8 (node.Encoder, CP1-CP3). Image: CLIP ViT-B/32 image tower (ONNX, FP32, or the INT8
variant made locally with onnxruntime quantize_dynamic). Images are synthetic: no image dataset is downloaded.
Chip time: CP12's 32 x 32 array at 500 MHz takes 8.05 ms for a 164-token post; other inputs scale by MACs (추정).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from node import Encoder  # noqa: E402

CLIP_DIR = ROOT / "models" / "clip-vit-base-patch32" / "onnx"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)[:, None, None]
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)[:, None, None]

CP12_POST_TOKENS = 164      # CP4 sample mean, the post CP12 timed
CP12_POST_S = 8.05e-3       # CP12: 32 x 32, 500 MHz, B = 1
CP12_POST_MJ = 7.90


def text_macs(tokens: int, L: int = 6, d: int = 384, ff: int = 1536) -> float:
    """Weight GEMMs + attention score and context products, per encoder layer."""
    return L * (tokens * (4 * d * d + 2 * d * ff) + 2 * tokens * tokens * d)


def image_macs(L: int = 12, d: int = 768, ff: int = 3072, patches: int = 49, patch_px: int = 32 * 32 * 3,
               proj: int = 512) -> float:
    T = patches + 1  # + class token
    return patches * patch_px * d + L * (T * (4 * d * d + 2 * d * ff) + 2 * T * T * d) + d * proj


def chip_text_s(tokens: int) -> float:
    return CP12_POST_S * text_macs(tokens) / text_macs(CP12_POST_TOKENS)


def chip_image_s() -> float:
    return CP12_POST_S * image_macs() / text_macs(CP12_POST_TOKENS)


def chip_mj(seconds: float) -> float:
    return CP12_POST_MJ * seconds / CP12_POST_S


def quantize_clip() -> Path:
    out = CLIP_DIR / "vision_model_int8.onnx"
    if not out.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(CLIP_DIR / "vision_model.onnx"), str(out), weight_type=QuantType.QInt8,
                         per_channel=True)
    return out


class ImageEncoder:
    def __init__(self, variant: str = "fp32", threads: int = 0):
        path = CLIP_DIR / "vision_model.onnx" if variant == "fp32" else quantize_clip()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def encode(self, imgs: np.ndarray) -> np.ndarray:
        """imgs: (n, 224, 224, 3) uint8 -> unit-length (n, 512)."""
        x = (imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - MEAN) / STD
        out = []
        for i in range(0, len(x), 16):
            out.append(self.s.run(None, {"pixel_values": x[i:i + 16]})[0])
        e = np.concatenate(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def synth_image(seed: int, size: int = 224) -> np.ndarray:
    """A procedural picture: a two-colour gradient background, 2-6 filled shapes, and a stripe or noise texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    c0, c1 = rng.random(3), rng.random(3)
    a = rng.random() * 2 * np.pi
    g = (np.cos(a) * xx + np.sin(a) * yy)
    g = (g - g.min()) / (np.ptp(g) + 1e-9)
    img = c0 * (1 - g[..., None]) + c1 * g[..., None]
    for _ in range(rng.integers(2, 7)):
        col = rng.random(3)
        cx, cy, r = rng.random(), rng.random(), 0.05 + 0.25 * rng.random()
        if rng.random() < 0.5:
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        else:
            m = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r * (0.3 + rng.random()))
        img[m] = col
    if rng.random() < 0.5:
        f = 4 + 30 * rng.random()
        img *= (0.8 + 0.2 * np.sin(2 * np.pi * f * (xx if rng.random() < 0.5 else yy)))[..., None]
    else:
        img += 0.08 * rng.standard_normal(img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def synth_images(seeds) -> np.ndarray:
    return np.stack([synth_image(int(s)) for s in seeds])


__all__ = ["Encoder", "ImageEncoder", "synth_image", "synth_images", "chip_text_s", "chip_image_s", "chip_mj",
           "text_macs", "image_macs", "quantize_clip"]
