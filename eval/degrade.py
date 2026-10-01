"""Make clean renders look like scans — for building an evaluation set with
exact ground truth.

The eval plan (dev-communication/design.md §7) takes arXiv papers WITH their
LaTeX source, renders the compiled PDF, and degrades the page images with
this module. Because the source is known, every formula, table, and paragraph
has a ground truth, at any degradation level — something real scans can't
give. Degradations are seeded and parameterised by a single ``level``
(0 = clean, 1 = office scanner, 2 = old photocopy, 3 = bad phone photo):

    python -m eval.degrade in.png out.png --level 2 --seed 7
"""

from __future__ import annotations

import argparse
import io
import random

import numpy as np
from PIL import Image, ImageFilter

LEVELS = {
    #      skew deg, blur px, noise sd, jpeg q, contrast, binarize
    0: dict(skew=0.0, blur=0.0, noise=0.0, jpeg=None, contrast=1.0, binarize=False),
    1: dict(skew=0.6, blur=0.4, noise=4.0, jpeg=85, contrast=0.95, binarize=False),
    2: dict(skew=1.2, blur=0.8, noise=10.0, jpeg=60, contrast=0.85, binarize=True),
    3: dict(skew=2.5, blur=1.3, noise=16.0, jpeg=40, contrast=0.75, binarize=False),
}


def degrade(img: Image.Image, level: int = 1, seed: int = 0) -> Image.Image:
    p = LEVELS[level]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    img = img.convert("L")
    if p["skew"]:
        angle = rng.uniform(-p["skew"], p["skew"])
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if p["contrast"] != 1.0:
        lo = int(255 * (1 - p["contrast"]) / 2)
        img = img.point(lambda v: lo + v * (255 - 2 * lo) // 255)
    if p["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    if p["noise"]:
        arr = np.asarray(img, dtype=np.float32)
        arr += np_rng.normal(0.0, p["noise"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if p["binarize"]:
        thr = rng.randint(150, 190)
        img = img.point(lambda v: 255 if v > thr else 0)
    if p["jpeg"]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=p["jpeg"])
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img.convert("RGB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--level", type=int, default=1, choices=sorted(LEVELS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with Image.open(a.src) as im:
        degrade(im, a.level, a.seed).save(a.dst)


if __name__ == "__main__":
    main()
