from __future__ import annotations

import math

from PIL import Image

from ..backend import ChatClient
from ..schema import Block


class Reader:
    """Base class. Subclasses set ``name`` and implement ``read``."""

    name = "base"

    def __init__(self, client: ChatClient, max_tokens: int = 16384):
        self.client = client
        self.max_tokens = max_tokens

    @property
    def tag(self) -> str:
        return f"reader:{self.name}:{self.client.model}"

    def read(self, img: Image.Image, attempt: int = 0) -> list[Block]:
        """attempt > 0 is a retry after a degenerate (truncated) reading:
        adapters should perturb decoding (see RETRY_SAMPLING)."""
        raise NotImplementedError


# Decoding for retries. Greedy decoding is what makes a loop self-sustaining;
# a little temperature plus a mild repetition penalty usually breaks it.
RETRY_SAMPLING = {"temperature": 0.3, "repetition_penalty": 1.05}


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
                 ) -> tuple[int, int]:
    """Qwen2-VL-family resize: both sides multiples of ``factor``, area within
    [min_pixels, max_pixels], aspect ratio preserved as closely as possible.

    Resizing on our side to dimensions the processor will leave untouched
    means bboxes the model emits are in OUR image's pixel frame."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def normalise_bbox(box, w: int, h: int):
    """Pixel [x0,y0,x1,y1] in a w x h frame -> clipped page fractions."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    f = [min(max(x0 / w, 0.0), 1.0), min(max(y0 / h, 0.0), 1.0),
         min(max(x1 / w, 0.0), 1.0), min(max(y1 / h, 0.0), 1.0)]
    return f if f[2] > f[0] and f[3] > f[1] else None
