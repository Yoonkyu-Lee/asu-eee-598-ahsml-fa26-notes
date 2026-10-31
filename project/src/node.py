"""Reference pipeline of one server-blind node (CP1).

Ingest, per post:  verify signature -> decrypt -> tokenize -> encode (MiniLM) -> insert into index
Query:             tokenize -> encode -> exact top-k over the index

Everything runs on the CPU. The encoder is all-MiniLM-L6-v2 exported to ONNX by
sentence-transformers; `model_quint8_avx2.onnx` is its INT8 variant for CPUs without AVX-512.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
MAX_TOKENS = 256  # sentence-transformers default max_seq_length for this model
DIM = 384


# ---------------------------------------------------------------- crypto envelope

@dataclass
class Envelope:
    """What a storage node holds: ciphertext signed by the author. It cannot read the body."""
    author: Ed25519PublicKey
    nonce: bytes
    ciphertext: bytes
    signature: bytes  # over nonce || ciphertext


def seal(author_key: Ed25519PrivateKey, room_key: bytes, text: str) -> Envelope:
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(room_key).encrypt(nonce, text.encode("utf-8"), None)
    return Envelope(author_key.public_key(), nonce, ct, author_key.sign(nonce + ct))


def verify(env: Envelope) -> None:
    env.author.verify(env.signature, env.nonce + env.ciphertext)  # raises InvalidSignature


def decrypt(env: Envelope, room_key: bytes) -> str:
    return ChaCha20Poly1305(room_key).decrypt(env.nonce, env.ciphertext, None).decode("utf-8")


# ---------------------------------------------------------------- encoder

class Encoder:
    def __init__(self, variant: str = "int8", threads: int = 1):
        """variant: "int8" (HF dynamic INT8), "fp32", or a path to any ONNX export of the same model."""
        known = {"int8": "model_quint8_avx2.onnx", "fp32": "model.onnx"}
        path = MODEL_DIR / "onnx" / known[variant] if variant in known else Path(variant)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # 0 = let ORT use all cores
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        self.tok.enable_padding()  # pad to the longest text in this batch
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        return feed

    def encode_tokens(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        hidden = self.session.run(None, feed)[0]                 # (batch, seq, 384)
        mask = feed["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)  # mean pooling
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)       # unit length

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encode_tokens(self.tokenize(texts))


# ---------------------------------------------------------------- index

class ExactIndex:
    """Flat array of unit vectors; search = one matrix-vector product + partial sort."""

    def __init__(self, capacity: int):
        self.vecs = np.empty((capacity, DIM), dtype=np.float32)
        self.n = 0

    def insert(self, v: np.ndarray) -> None:
        self.vecs[self.n] = v
        self.n += 1

    def search(self, q: np.ndarray, k: int = 10) -> np.ndarray:
        scores = self.vecs[: self.n] @ q
        top = np.argpartition(scores, -k)[-k:]
        return top[np.argsort(scores[top])[::-1]]
