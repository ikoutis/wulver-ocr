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
it is, the element in progress is dropped, and truncated_tail blocks
(base.py) stand for the rest of the page, from the box of the element in
progress on, so the page is re-read and, if still cut, the reviewer
transcribes those regions from their crops. An element the model re-emitted
in a loop is kept once (base.drop_repeats, meta["repeated"]). Malformed JSON
is read element by element, so one bad element costs only itself, and an
element whose single-backslash LaTeX had to be repaired is marked
meta["json_repaired"], so that it is reviewed.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from PIL import Image

from ..backend import TRUNCATION_MARKER, image_part, text_part
from ..schema import Block
from ..validate import strip_math_delims
from .base import (RETRY_SAMPLING, Reader, drop_repeats, normalise_bbox, smart_resize,
                   truncated_tail)

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
_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)
_JSON_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)(?=([A-Za-z]?))", re.S)
_CONTROL = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
# No line of a formula starts with these: after "\n" they are \nabla, \neq, …
_N_COMMAND = re.compile(r"(?:abla|eq|eg|otin|mid|leq|geq|exists|ewline)(?![A-Za-z])")
_REPAIRED = "\x00"      # marks a repaired "text" until parse_layout reads it


def _latex_escape(m: re.Match, formula: bool) -> bool:
    """An escape no JSON writer means: an invalid one (\\alpha, \\in), a
    backspace, form feed, CR or TAB before a letter (\\beta, \\frac, \\rho,
    \\theta), or in a formula a newline before 'abla', 'eq', … (\\nabla,
    \\neq). Each is LaTeX written with single backslashes."""
    e, letter = m.group(1), m.group(2)
    if len(e) > 1 or e in '"\\/':
        return False
    if e == "n":
        return formula and bool(_N_COMMAND.match(m.string, m.end()))
    return e not in "bfrt" or bool(letter)


def _single_backslash(body: str, formula: bool) -> str:
    """The text a JSON string body means when its LaTeX has single
    backslashes. Its escapes are LaTeX, except \\" \\/ \\uXXXX, the control
    escapes \\b \\f \\n \\r \\t when no letter follows, and \\n outside math
    (a newline: inside math it is \\nabla, \\neq, \\nu). A doubled backslash
    before a letter is an escaped command (\\\\frac), elsewhere it is a LaTeX
    row break (\\\\). ``formula``: the whole body is math."""
    out, math, i = [], formula, 0
    while i < len(body):
        c = body[i]
        if c == "$":
            j = i
            while j < len(body) and body[j] == "$":     # $$ is one delimiter
                j += 1
            out.append(body[i:j])
            math, i = formula or not math, j
            continue
        if c != "\\" or i + 1 == len(body):
            out.append(c)
            i += 1
            continue
        e, letter = body[i + 1], bool(re.match(r"[A-Za-z]", body[i + 2:i + 3]))
        if e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", body[i + 2:i + 6]):
            out.append(chr(int(body[i + 2:i + 6], 16)))
            i += 6
            continue
        if e == "\\":
            out.append("\\" if letter else "\\\\")
        elif e in '"/':
            out.append(e)
        elif e in _CONTROL and (not letter or (e == "n" and not math)):
            out.append(_CONTROL[e])
        else:
            out.append("\\" + e)
        i += 2
    return "".join(out)


def _repair_escapes(s: str) -> str:
    """Single-backslash LaTeX in a reply (\\alpha, \\frac): every string
    literal holding an escape no JSON writer means (_latex_escape) is read
    as such (_single_backslash) and re-encoded, and a repaired "text" is
    marked with _REPAIRED. Other literals, valid JSON included, are left
    alone."""
    out, pos, formula = [], 0, False
    for m in _STRING.finditer(s):
        key = re.search(r'"(\w+)"\s*:\s*$', s[max(0, m.start() - 40):m.start()])
        key = key.group(1) if key else None
        if key == "category":               # it comes before the text
            formula = m.group(1).strip().lower() == "formula"
        if not any(_latex_escape(e, formula) for e in _JSON_ESCAPE.finditer(m.group(1))):
            continue
        text = _single_backslash(m.group(1), formula)
        if key == "text":
            text = _REPAIRED + text
        out += [s[pos:m.start()], json.dumps(text, ensure_ascii=False)]
        pos = m.end()
    return "".join(out) + s[pos:]


def _unmark(elements: list[dict]) -> list[dict]:
    for el in elements:
        t = el.get("text")
        if isinstance(t, str) and t.startswith(_REPAIRED):
            el["text"], el["json_repaired"] = t[len(_REPAIRED):], True
    return elements


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


def parse_layout(reply: str) -> tuple[list[dict], bool, Optional[list]]:
    """-> (elements, complete, cut). ``complete`` is False when the reply
    ends inside an element (it was cut off) or holds no layout JSON at all;
    ``cut`` is then the box of that element, as written, if it shows one.

    Raw control characters in strings and single-backslash LaTeX are
    tolerated; an element whose text needed the LaTeX repair gets
    "json_repaired": True. Past that, the reply is read element by element:
    an element that cannot be decoded is kept as its box and category
    (_salvage) and reading resumes at the next element, so one bad element
    never costs the ones after it; text after the array, or a trailing comma,
    costs nothing."""
    s = reply.replace(TRUNCATION_MARKER, "").strip()
    s = _repair_escapes(re.sub(r"^```(?:json)?\s*|\s*```$", "", s))
    try:
        els = _element_list(json.loads(s, strict=False))
        return (_unmark(els), True, None) if els is not None else ([], False, None)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder(strict=False)
    starts = [m.start() for m in _ELEMENT.finditer(s)]
    out, complete, end, cut = [], False, 0, None
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
            else:
                cut = _salvage(s[i:]).get("bbox")
    return _unmark(out), complete, cut


def parse_elements(reply: str) -> tuple[list[dict], bool]:
    """-> (elements, complete): parse_layout without the cut."""
    return parse_layout(reply)[:2]


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
        elements, complete, cut = parse_layout(reply)
        blocks = []
        for el in elements:
            cat = str(el.get("category", "")).strip().lower()
            btype = CATEGORY_MAP.get(cat, "other")
            text = _as_text(el.get("text"))
            if btype == "formula":
                text = strip_math_delims(text)
            meta = {"category": el.get("category")}
            if el.get("json_repaired"):
                meta["json_repaired"] = True
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
        kept = drop_repeats(blocks, lambda b: b.bbox and tuple(b.bbox),
                            lambda b: b.content and (b.type, b.content))
        for b, copies in kept:
            if copies > 1:
                b.meta["repeated"] = copies
        blocks = [b for b, _ in kept]
        if reply.endswith(TRUNCATION_MARKER) or not complete:
            blocks.extend(truncated_tail(blocks, self.tag, normalise_bbox(cut, w, h)))
        return blocks
