"""dots.ocr-style layout readers: one request per page, JSON list of
{bbox, category, text} in reading order.

The prompt and label set are those of rednote-hilab/dots.ocr
(prompt_layout_all_en); formulas come back as LaTeX, tables as HTML, the rest
as Markdown, pictures with no text. Bboxes are pixels in the frame of the image
the model saw, so the page is resized here with the Qwen2-VL rule the model's
processor uses — the processor then leaves it unchanged and the boxes are in a
known frame.

A page whose JSON is cut off (a repetition loop that hit max_tokens is the
usual cause) is not lost: every complete element before the cut is kept, and
the page is flagged so the reviewer re-reads the tail.
"""

from __future__ import annotations

import json
import re

from PIL import Image

from ..backend import TRUNCATION_MARKER, image_part, text_part
from ..schema import Block
from ..validate import strip_math_delims
from .base import RETRY_SAMPLING, Reader, normalise_bbox, smart_resize

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


def parse_elements(reply: str) -> tuple[list[dict], bool]:
    """-> (elements, complete). Recovers every complete object from a
    truncated or slightly malformed JSON array."""
    s = reply.replace(TRUNCATION_MARKER, "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            data = data.get("layout") or data.get("elements") or [data]
        return [d for d in data if isinstance(d, dict)], True
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    out, i = [], s.find("{")
    while 0 <= i < len(s):
        try:
            obj, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = s.find("{", end)
    return out, False


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
        truncated = reply.endswith(TRUNCATION_MARKER)
        elements, complete = parse_elements(reply)
        blocks = []
        for el in elements:
            cat = str(el.get("category", "")).strip().lower()
            btype = CATEGORY_MAP.get(cat, "other")
            text = el.get("text") or ""
            if btype == "formula":
                text = strip_math_delims(text)
            meta = {"category": el.get("category")}
            if btype == "heading":
                hashes = re.match(r"^(#{1,6})\s+", text)
                if hashes:
                    meta["level"] = max(1, len(hashes.group(1)) - 1)
                    text = text[hashes.end():]
            blocks.append(Block(type=btype, content=text.strip(),
                                bbox=normalise_bbox(el.get("bbox"), w, h),
                                source=self.tag, meta=meta))
        if (truncated or not complete) and blocks:
            # The element in progress at the cut is lost; mark the last kept one
            # so validation flags the page tail for review.
            blocks[-1].content += TRUNCATION_MARKER
        elif truncated or not complete:
            blocks = [Block(type="text", content=TRUNCATION_MARKER, source=self.tag)]
        return blocks
