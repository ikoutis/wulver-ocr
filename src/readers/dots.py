"""dots.ocr-style layout readers: one request per page, JSON list of
{bbox, category, text} in reading order.

The prompt and label set are those of rednote-hilab/dots.ocr
(prompt_layout_all_en); formulas come back as LaTeX, tables as HTML, the rest
as Markdown, pictures with no text. Bboxes are pixels in the frame of the image
the model saw, so the page is resized here with the Qwen2-VL rule the model's
processor uses — the processor then leaves it unchanged and the boxes are in a
known frame.

A page whose JSON is cut off (a repetition loop that hit max_tokens is the
usual cause) is not lost: every complete element before the cut is kept as
it is, the element in progress is dropped, and a truncated_tail block
(base.py) stands for the rest of the page, so the page is re-read and, if
still cut, the reviewer transcribes that region from its crop. Malformed
JSON is read element by element, so one bad element costs only itself.
"""

from __future__ import annotations

import json
import re

from PIL import Image

from ..backend import TRUNCATION_MARKER, image_part, text_part
from ..schema import Block
from ..validate import strip_math_delims
from .base import RETRY_SAMPLING, Reader, normalise_bbox, smart_resize, truncated_tail

LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

CATEGORY_MAP = {
    "caption": "caption", "footnote": "footnote", "formula": "formula",
    "list-item": "list", "page-footer": "footer", "page-header": "header",
    "picture": "figure", "section-header": "heading", "table": "table",
    "text": "text", "title": "title",
}


# Where an element object starts. Resyncing on this, not on any '{', skips
# the braces of LaTeX inside text strings.
_ELEMENT = re.compile(r'\{\s*"(?:bbox|category|text)"\s*:')
_JSON_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)", re.S)


def _repair_escapes(s: str) -> str:
    """LaTeX written with single backslashes (\\alpha, \\in) is an invalid
    JSON escape: double those backslashes. Valid escapes are left alone."""
    return _JSON_ESCAPE.sub(lambda m: m.group(0) if len(m.group(1)) == 5
                            or m.group(1) in '"\\/bfnrt' else "\\" + m.group(0), s)


def _element_list(data):
    """The elements in decoded JSON: a list, one bare element, or a wrapper
    object ({"layout": [...]}, {"layout_dets": [...]}, ...); None if none."""
    if isinstance(data, dict):
        if "bbox" in data or "category" in data:
            return [data]
        lists = [v for v in data.values() if isinstance(v, list)]
        data = next((v for v in lists if any(isinstance(x, dict) for x in v)),
                    lists[0] if lists else None)
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict)]


def _salvage(fragment: str) -> dict:
    """What can be read of a malformed element (an unescaped quote in its
    text, say): its box and category. The text is left empty, so validation
    flags the block and the reviewer transcribes it from its crop."""
    el = {"text": ""}
    m = re.search(r'"bbox"\s*:\s*(\[[^\[\]]*\])', fragment)
    if m:
        try:
            el["bbox"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'"category"\s*:\s*"([^"\\]*)"', fragment)
    if m:
        el["category"] = m.group(1)
    return el


def parse_elements(reply: str) -> tuple[list[dict], bool]:
    """-> (elements, complete). ``complete`` is False when the reply ends
    inside an element (it was cut off) or holds no layout JSON at all.

    Raw control characters in strings and single-backslash LaTeX are
    tolerated. Past that, the reply is read element by element: an element
    that cannot be decoded is kept as its box and category (_salvage) and
    reading resumes at the next element, so one bad element never costs the
    ones after it; text after the array, or a trailing comma, costs nothing."""
    s = reply.replace(TRUNCATION_MARKER, "").strip()
    s = _repair_escapes(re.sub(r"^```(?:json)?\s*|\s*```$", "", s))
    try:
        els = _element_list(json.loads(s, strict=False))
        return (els, True) if els is not None else ([], False)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder(strict=False)
    starts = [m.start() for m in _ELEMENT.finditer(s)]
    out, complete, end = [], False, 0
    for k, i in enumerate(starts):
        if i < end:
            continue                    # inside an element already read
        try:
            obj, end = dec.raw_decode(s, i)
            out.append(obj)
            complete = True
        except json.JSONDecodeError:
            complete = False            # the last one is where the reply was cut
            if k + 1 < len(starts):
                out.append(_salvage(s[i:starts[k + 1]]))
    return out, complete


def _as_text(v) -> str:
    """An element's text; models occasionally emit a number, a list, or null."""
    if v is None or isinstance(v, str):
        return v or ""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)


class DotsReader(Reader):
    name = "dots"
    min_pixels = 3136
    max_pixels = 11289600

    def read(self, img: Image.Image, attempt: int = 0) -> list[Block]:
        h, w = smart_resize(img.height, img.width, 28, self.min_pixels, self.max_pixels)
        if (w, h) != img.size:
            img = img.resize((w, h), Image.BICUBIC)
        sampling = {"top_p": 0.9, **(RETRY_SAMPLING if attempt else {})}
        reply = self.client.chat(
            [image_part(img), text_part(f"<|img|><|imgpad|><|endofimg|>{LAYOUT_PROMPT}")],
            max_tokens=self.max_tokens, temperature=sampling.pop("temperature", 0.0),
            extra=sampling)
        elements, complete = parse_elements(reply)
        blocks = []
        for el in elements:
            cat = str(el.get("category", "")).strip().lower()
            btype = CATEGORY_MAP.get(cat, "other")
            text = _as_text(el.get("text"))
            if btype == "formula":
                text = strip_math_delims(text)
            meta = {"category": el.get("category")}
            if btype in ("heading", "title"):
                # Markdown hashes (assembly adds its own); a Section-header's
                # give its level
                hashes = re.match(r"^(#{1,6})\s+", text)
                if hashes:
                    if btype == "heading":
                        meta["level"] = max(1, len(hashes.group(1)) - 1)
                    text = text[hashes.end():]
            blocks.append(Block(type=btype, content=text.strip(),
                                bbox=normalise_bbox(el.get("bbox"), w, h),
                                source=self.tag, meta=meta))
        if reply.endswith(TRUNCATION_MARKER) or not complete:
            blocks.append(truncated_tail(blocks, self.tag))
        return blocks
