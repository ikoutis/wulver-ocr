"""Chandra (datalab-to/chandra-ocr-2) reader: one request per page, HTML
layout blocks back.

The model answers with top-level <div data-label=… data-bbox="x0 y0 x1 y1">
blocks, boxes normalised to 0–1000, math as <math> (display="block" for
display math), tables as HTML, and images described in an <img alt=…> (charts
as data, diagrams as Mermaid). The prompt, the image scaling rule, and the
decoding settings below are copied from datalab-to/chandra (Apache-2.0 code;
chandra/prompts.py, chandra/model/util.py, chandra/model/vllm.py), since
the model was trained on exactly that prompt. The model WEIGHTS are under a
modified OpenRAIL-M licence: free for research and personal use — see
design.md §4 before any other use.
"""

from __future__ import annotations

import re

from PIL import Image

from ..backend import TRUNCATION_MARKER, image_part, text_part
from ..schema import Block
from . import htmlmd
from .base import Reader

ALLOWED_TAGS = ["math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td",
                "p", "th", "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li",
                "input", "a", "span", "img", "hr", "tbody", "small", "caption", "strong",
                "thead", "big", "code", "chem"]
ALLOWED_ATTRIBUTES = ["class", "colspan", "rowspan", "display", "checked", "type",
                      "border", "value", "style", "href", "alt", "align", "data-bbox",
                      "data-label"]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{PROMPT_ENDING}
""".strip()

LABEL_MAP = {
    "caption": "caption", "footnote": "footnote", "equation-block": "formula",
    "list-group": "list", "page-header": "header", "page-footer": "footer",
    "image": "figure", "figure": "figure", "diagram": "figure",
    "section-header": "heading", "table": "table", "text": "text",
    "complex-block": "text", "code-block": "code", "form": "text",
    "table-of-contents": "text", "chemical-block": "other",
    "bibliography": "reference",
}
_EQNUM = re.compile(r"^\(?\s*([0-9]+[a-z]?|[A-Z]?\.?[0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)?$")


def scale_to_fit(img: Image.Image, max_size=(3072, 2048), min_size=(1792, 28),
                 grid: int = 28) -> Image.Image:
    """Chandra's resize: area within bounds, both sides multiples of ``grid``,
    aspect ratio preserved as well as the grid allows."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    ar, px = w / h, w * h
    max_px, min_px = max_size[0] * max_size[1], min_size[0] * min_size[1]
    scale = (max_px / px) ** 0.5 if px > max_px else (
        (min_px / px) ** 0.5 if px < min_px else 1.0)
    wb, hb = max(1, round(w * scale / grid)), max(1, round(h * scale / grid))
    while wb * hb * grid * grid > max_px and (wb > 1 or hb > 1):
        if wb == 1:
            hb -= 1
        elif hb == 1:
            wb -= 1
        elif abs((wb - 1) / hb - ar) < abs(wb / (hb - 1) - ar):
            wb -= 1
        else:
            hb -= 1
    size = (wb * grid, hb * grid)
    return img if size == (w, h) else img.resize(size, Image.LANCZOS)


def parse_bbox(s: str):
    try:
        x0, y0, x1, y1 = (float(v) for v in s.replace(",", " ").split())
    except (AttributeError, ValueError):
        return None
    f = [min(max(v / 1000.0, 0.0), 1.0) for v in (x0, y0, x1, y1)]
    return f if f[2] > f[0] and f[3] > f[1] else None


def _formula_blocks(div: htmlmd.Node, bbox, tag) -> list[Block]:
    maths = div.find_all("math")
    if not maths:
        return [Block(type="text", content=htmlmd.to_markdown(div), bbox=bbox,
                      source=tag, meta={"category": "Equation-Block"})]
    rest = div.text()
    for m in maths:
        rest = rest.replace(m.text(), " ", 1)
    number = _EQNUM.match(rest.strip())
    out = []
    for m in maths:
        out.append(Block(type="formula", content=m.text().strip(), bbox=bbox,
                         source=tag, meta={"category": "Equation-Block"}))
    if number and out and r"\tag" not in out[-1].content:
        out[-1].content += rf" \tag{{{number.group(1)}}}"
    return out


def html_to_blocks(html_text: str, tag: str) -> list[Block]:
    root = htmlmd.parse(html_text)
    blocks: list[Block] = []
    for div in root.children:
        if not isinstance(div, htmlmd.Node) or div.tag != "div":
            continue
        label = div.attrs.get("data-label", "Text")
        if label == "Blank-Page":
            continue
        btype = LABEL_MAP.get(label.lower(), "other")
        bbox = parse_bbox(div.attrs.get("data-bbox", ""))
        meta = {"category": label}
        if btype == "formula":
            blocks.extend(_formula_blocks(div, bbox, tag))
            continue
        if btype == "figure":
            imgs = div.find_all("img")
            alt = " ".join(i.attrs.get("alt", "") for i in imgs).strip()
            for i in imgs:
                i.attrs.pop("alt", None)
            body = htmlmd.to_markdown(div)
            meta["reader_description"] = "\n\n".join(x for x in (alt, body) if x)
            blocks.append(Block(type="figure", content="", bbox=bbox, source=tag, meta=meta))
            continue
        if btype == "table":
            tables = div.find_all("table")
            content = "\n".join(htmlmd.table_html(t) for t in tables) if tables \
                else htmlmd.to_markdown(div)
        elif btype == "heading":
            hs = [c for c in div.children if isinstance(c, htmlmd.Node)
                  and re.fullmatch(r"h[1-6]", c.tag)]
            if hs:
                meta["level"] = int(hs[0].tag[1])
                content = htmlmd._squash(htmlmd.inline(hs[0]))
            else:
                content = htmlmd._squash(htmlmd.inline(div))
        elif btype == "code":
            pre = div.find_all("pre")
            content = (pre[0].text() if pre else div.text()).strip("\n")
        else:
            content = htmlmd.to_markdown(div)
        blocks.append(Block(type=btype, content=content, bbox=bbox, source=tag, meta=meta))
    return blocks


class ChandraReader(Reader):
    name = "chandra"

    def __init__(self, client, max_tokens: int = 12384):
        super().__init__(client, max_tokens)

    def read(self, img: Image.Image, attempt: int = 0) -> list[Block]:
        img = scale_to_fit(img)
        # Chandra's own decoding: near-greedy, and on a retry after a loop,
        # temperature +0.2 per attempt with top_p 0.95.
        temperature, top_p = (0.0, 0.1) if attempt == 0 else (min(0.2 * attempt, 0.8), 0.95)
        reply = self.client.chat([image_part(img), text_part(OCR_LAYOUT_PROMPT)],
                                 max_tokens=self.max_tokens, temperature=temperature,
                                 extra={"top_p": top_p})
        truncated = reply.endswith(TRUNCATION_MARKER)
        blocks = html_to_blocks(reply.replace(TRUNCATION_MARKER, ""), self.tag)
        if truncated:
            if blocks:
                blocks[-1].content += TRUNCATION_MARKER
            else:
                blocks = [Block(type="text", content=TRUNCATION_MARKER, source=self.tag)]
        return blocks
