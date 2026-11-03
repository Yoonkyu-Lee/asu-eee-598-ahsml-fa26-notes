"""Download the demo box's models from Hugging Face into project/models/ (git-ignored), about 720 MB.

  python src/demo/fetch_models.py            # skips files already there with the right SHA-256
  python src/demo/fetch_models.py --check    # only check what is there

Files and hashes are the ones the project measured with (CP1, CP17, CP18). A hash mismatch is reported and the file
kept as *.mismatch, so the box never loads something nobody checked.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[2] / "models"
HF = "https://huggingface.co/{repo}/resolve/main/{path}"
FILES = [   # (repo, path in repo, local folder, sha256, bytes)
    ("sentence-transformers/all-MiniLM-L6-v2", "onnx/model_quint8_avx2.onnx", "all-MiniLM-L6-v2",
     "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21", 23046789),
    ("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx", "all-MiniLM-L6-v2",
     "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452", 90405214),
    ("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json", "all-MiniLM-L6-v2",
     "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037", 466247),
    ("Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx", "clip-vit-base-patch32",
     "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40", 351685709),
    ("Xenova/clip-vit-base-patch32", "onnx/text_model.onnx", "clip-vit-base-patch32",
     "3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc", 254058553),
    ("Xenova/clip-vit-base-patch32", "tokenizer.json", "clip-vit-base-patch32",
     "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd", 2224119),
]


def sha256(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    check = "--check" in sys.argv
    base = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--base=")), HF)   # tests: a local mirror
    bad = 0
    for repo, path, folder, digest, size in FILES:
        dst = MODELS / folder / path
        if dst.exists() and dst.stat().st_size == size and sha256(dst) == digest:
            print(f"ok       {folder}/{path}")
            continue
        if check:
            print(f"missing  {folder}/{path}")
            bad += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        url = base.format(repo=repo, path=path)
        print(f"get      {url} ({size / 1e6:.0f} MB)", flush=True)
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
            got = 0
            for b in iter(lambda: r.read(1 << 20), b""):
                f.write(b)
                got += len(b)
                print(f"\r         {got / 1e6:6.0f} / {size / 1e6:.0f} MB", end="", flush=True)
        print()
        if sha256(part) != digest:
            part.replace(dst.with_name(dst.name + ".mismatch"))
            print(f"MISMATCH {folder}/{path}: kept as .mismatch, not used")
            bad += 1
            continue
        part.replace(dst)
        print(f"ok       {folder}/{path}")
    print("all models in place" if not bad else f"{bad} file(s) not in place")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
