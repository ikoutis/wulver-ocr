from __future__ import annotations

import math
from collections import Counter

from PIL import Image

from ..backend import ChatClient
from ..schema import DROPPED_TYPES, Block


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
        adapters should perturb decoding (see RETRY_SAMPLING).

        A reply that was cut off (backend.TRUNCATION_MARKER, or output that
        ends inside an element) keeps its complete elements as they are and
        ends with the blocks of ``truncated_tail(...)``; the marker itself
        never goes into a block. A layout reader keeps an element the model
        repeated once, with meta["repeated"] = its number of copies
        (``drop_repeats``)."""
        raise NotImplementedError


# Decoding for retries. Greedy decoding is what makes a loop self-sustaining;
# a little temperature plus a mild repetition penalty usually breaks it.
RETRY_SAMPLING = {"temperature": 0.3, "repetition_penalty": 1.05}

TAIL_MIN_HEIGHT = 0.02      # page fraction: a tail box is never empty
COLUMN = 0.55               # a box ending before this is in the left half (a
                            # left column); one starting after 1 - COLUMN, the right
COLUMN_TEXT = 0.1           # how tall the boxes justified to one right edge in the
                            # left half must be together to be a column of text
BELOW_ROOM = 0.1            # page below a band of columns, past a bottom margin as
                            # deep as the top one, that can hold more content
_EPS = 0.01


def _tail(bbox, source: str) -> Block:
    return Block(type="text", content="", bbox=bbox, source=source,
                 meta={"truncated_tail": True})


def truncated_tail(kept: list[Block], source: str, cut=None) -> list[Block]:
    """The blocks standing for reader output lost at a cut: empty, marked
    meta["truncated_tail"] (validation flags them, so the page is re-read, and
    if it stays cut the reviewer transcribes each region from its crop).

    Their boxes are the regions the lost output most likely covers, in reading
    order. ``cut`` is the box of the element the model was writing when it was
    cut off, if the reply shows it; else the cut fell after the last kept
    element. From there the rest of that column is lost: the box runs between
    the kept columns beside it (on a two-column page cut in the right column,
    the left column is not re-read) down to where they end, and what lies
    below them, full width, is a second region (a full-width float or
    equation, or the next band of columns); when no more than a bottom margin
    is left below them (one about as deep as the page's top margin), or no
    kept column is beside, the box runs to the page bottom. A cut in a left
    column, below text of it and with nothing kept in the right half, also
    lost the whole right column: a second region. The evidence of a left
    column is two or more boxes in the band (the cut one included) justified
    to one right edge near the middle and as tall together as several lines
    of text: short lines and list items on a one-column page are no column,
    and splitting such a page there would cut every line below in two.
    Both regions then run to the page bottom,
    since nothing read says where the columns end. Running headers and
    footers are ignored (they sit at the page edges, not in the reading flow),
    and with nothing kept the region is the whole page. A reader without boxes
    (Markdown) cannot say where the cut fell: then the box is None."""
    body = [b.bbox for b in kept if b.bbox and b.type not in DROPPED_TYPES]
    if cut in [b.bbox for b in kept]:
        cut = None          # a copy of a kept element (a loop): nothing of it is lost
    if cut is None and kept and not any(b.bbox for b in kept):
        return [_tail(None, source)]
    boxes = body + ([cut] if cut else [])
    if not boxes:
        return [_tail([0.0, 0.0, 1.0, 1.0], source)]
    anchor = boxes[-1]
    y = min(anchor[1] if cut else anchor[3], 1.0 - TAIL_MIN_HEIGHT)
    # kept columns beside the anchor that reach below y
    beside = [b for b in body if b[3] > y and (b[2] <= anchor[0] or b[0] >= anchor[2])]
    x0 = max((b[2] for b in beside if b[2] <= anchor[0]), default=0.0)
    x1 = min((b[0] for b in beside if b[0] >= anchor[2]), default=1.0)
    end = max((b[3] for b in beside), default=1.0)      # where those columns end
    margin = min((b[1] for b in body), default=0.0)     # the page's top margin
    if 1.0 - end <= margin + BELOW_ROOM:                # only a bottom margin below
        regions = [[x0, y, x1, 1.0]]
    elif end - y >= TAIL_MIN_HEIGHT:
        regions = [[x0, y, x1, end], [0.0, end, 1.0, 1.0]]
    else:
        regions = [[0.0, y, 1.0, 1.0]]  # the cut fell where the columns end
    if anchor[2] <= COLUMN:
        # A left column? The columns start below the last full-width box above
        # the cut; there, boxes in the left half justified to one right edge
        # near the middle and tall together (a column of text, not short
        # lines), and none in the right half: the right column was never read.
        top = max((b[3] for b in body if b[0] < 1 - COLUMN and b[2] > COLUMN
                   and b[3] <= anchor[1] + _EPS), default=0.0)
        band = [b for b in boxes if b[1] >= top - _EPS]
        left = [b for b in band if b[2] <= COLUMN]
        # the right edge most of the band's text is justified to (a box that
        # overhangs it, an overfull line or a wide float, is no other column)
        column = max(([b for b in left if abs(b[2] - e) <= _EPS]
                      for e in {b[2] for b in left if b[2] >= 1 - COLUMN}),
                     key=lambda c: sum(b[3] - b[1] for b in c), default=[])
        edge = max((b[2] for b in left), default=0.0)
        if (len(column) >= 2
                and sum(b[3] - b[1] for b in column) >= COLUMN_TEXT
                and not any(b[0] >= 1 - COLUMN for b in band)):
            regions = [[x0, y, edge, 1.0], [edge, min(b[1] for b in left), 1.0, 1.0]]
    return [_tail(r, source) for r in regions]


def reading_loss(blocks: list[Block]) -> tuple:
    """How much of its page one reading lost, to choose among a page's read
    attempts (the smaller, the better; the blocks must carry their
    validation flags). First the page area lost: the regions of the tail
    blocks still empty ("truncated") and the boxes of blocks whose text loops
    ("repetition"), a block without a box counting as the whole page. So a
    cut weighs what it lost, not how many regions stand for it (a cut in a
    left column has two), and an attempt cut further down the page loses
    less. Then the number of elements the model repeated (meta["repeated"]:
    kept once, so nothing of them is lost, but the page is worth a retry).
    Then, between readings equal so far, the one that kept more text.
    Nothing is left to retry when the first two are 0."""
    lost, repeated, kept = 0.0, 0, 0
    for b in blocks:
        flags = set(b.flags)
        element_loop = (b.meta.get("repeated") or 0) >= 2
        # A "repetition" flag on a repeated element is that element's mark
        # when validation has no "repeated" flag of its own for it.
        text_loop = "repetition" in flags and ("repeated" in flags or not element_loop)
        if "truncated" in flags or text_loop:
            lost += (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) if b.bbox else 1.0
        else:
            kept += len(b.content or "")
        repeated += element_loop
    return round(lost, 6), repeated, -kept


REPEAT_TEXT = 5     # the same (label, text) this often on one page is a loop


def drop_repeats(items: list, box, text) -> list[list]:
    """Element-level loops (a model re-emitting one element until it runs
    out of tokens) -> [[item, copies], ...] in reading order. An item with
    the box of an earlier one, or with the (label, text) key of an earlier
    one when that key occurs REPEAT_TEXT times or more, is a copy: it is
    dropped and counted on the first. These are the two rules of dots.ocr's
    own OutputCleaner (remove_duplicate_category_text_pairs_and_bbox).
    ``box(item)`` and ``text(item)`` give the keys (hashable, or None)."""
    keys = [(box(it), text(it)) for it in items]
    common = Counter(t for _, t in keys if t)
    out: list[list] = []
    first: dict = {}
    for it, (b, t) in zip(items, keys):
        k = first.get(("box", b)) if b else None
        if k is None and t and common[t] >= REPEAT_TEXT:
            k = first.get(("text", t))
        if k is not None:
            out[k][1] += 1
            continue
        for key in (("box", b), ("text", t)):
            if key[1]:
                first.setdefault(key, len(out))
        out.append([it, 1])
    return out


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
