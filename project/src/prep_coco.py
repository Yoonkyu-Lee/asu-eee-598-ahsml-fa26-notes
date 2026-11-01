"""COCO val2017 -> CLIP input pixels (CP20). Needs Pillow, which the project .venv does not have, so run it with the
course venv in isolated mode (the images are downloaded data, read as data only):

  ../.venv/Scripts/python.exe -I src/prep_coco.py     -> data/coco/val2017_224.npy (uint8, n x 224 x 224 x 3),
                                                         data/coco/val2017_index.json (image ids, captions)
CLIP preprocessing: shortest side to 224 (bicubic), center crop 224 x 224, RGB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
COCO = ROOT / "data" / "coco"


def load(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def main() -> None:
    ann = json.loads((COCO / "captions_val2017.json").read_text(encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    caps: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].strip())
    ids = sorted(files)
    out = np.lib.format.open_memmap(COCO / "val2017_224.npy", mode="w+", dtype=np.uint8, shape=(len(ids), 224, 224, 3))
    for i, k in enumerate(ids):
        out[i] = load(COCO / "val2017" / files[k])
        if i % 500 == 0:
            print(i, flush=True)
    out.flush()
    (COCO / "val2017_index.json").write_text(json.dumps({"ids": ids, "captions": [caps[k] for k in ids]}),
                                            encoding="utf-8")
    print("done", len(ids))


if __name__ == "__main__":
    main()
