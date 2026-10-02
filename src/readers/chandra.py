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

A reply cut off at max_tokens (a repetition loop is the usual cause) keeps
its complete layout blocks as they are; the block in progress at the cut is
dropped, and a truncated_tail block (base.py) stands for the rest of the
page, so the page is re-read and, if still cut, the reviewer transcribes that
region from its crop.
"""

from __future__ import annotations

import re

from PIL import Image

from ..backend import TRUNCATION_MARKER, image_part, text_part
from ..schema import Block
from . import htmlmd
from .base import Reader, truncated_tail

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
# An equation number written outside the math, with the punctuation around
# it: (3) (2.1) (3a) (A.3) (2.1-a) (3') (ii) (b) (*) (†), [5], or a bare 2.1.
_LABEL = r"(?:[A-Z]?\.?\d+(?:[.\-]\d+)*(?:[.\-]?[a-z])?|[ivx]+|[IVX]+|[a-zA-Z]|[*†‡§]{1,3})['′]*"
_EQNUM = re.compile(rf"\s*[,.;:]?\s*(?:\(\s*({_LABEL})\s*\)|\[\s*({_LABEL})\s*\]"
                    rf"|(\d+(?:\.\d+)*)(?!\w|\.\d))\s*[,.;:]?\s*")


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


def _wrapped(div: htmlmd.Node, children: list) -> htmlmd.Node:
    node = htmlmd.Node(div.tag, dict(div.attrs))
    node.children = list(children)
    return node


def _has_content(pieces: list) -> bool:
    return any(not isinstance(p, str) or p.strip() for p in pieces)


def _split(node: htmlmd.Node, pred) -> list:
    """node's content in document order with the elements matching ``pred``
    lifted out: a list of the matching Nodes and, between them, lists of the
    other pieces. An element holding a match is opened up (its own
    formatting is lost, its content is not)."""
    segs: list = []

    def walk(n):
        for c in n.children:
            if isinstance(c, htmlmd.Node) and pred(c):
                segs.append(c)
            elif isinstance(c, htmlmd.Node) and any(pred(d) for d in c.iter()):
                walk(c)
            else:
                if not segs or isinstance(segs[-1], htmlmd.Node):
                    segs.append([])
                segs[-1].append(c)
    walk(node)
    return segs


def _drop_prefix(pieces: list, n: int):
    """pieces without their first n characters of text; None if the cut
    would fall inside an element."""
    out = list(pieces)
    while n > 0 and out:
        p = out[0]
        t = p if isinstance(p, str) else p.text()
        if len(t) <= n:
            out.pop(0)
            n -= len(t)
        elif isinstance(p, str):
            out[0], n = p[n:], 0
        else:
            return None
    return out


def _leading_number(pieces: list):
    """-> (equation number, the pieces after it) if their text starts with
    one (after punctuation, as in '…</math>, (3).'), else (None, pieces). An
    unbracketed number counts only on its own ('… 2 times' is prose)."""
    m = _EQNUM.match("".join(p if isinstance(p, str) else p.text() for p in pieces))
    if m:
        rest = _drop_prefix(pieces, m.end())
        if rest is not None and (m.group(3) is None or not _has_content(rest)):
            return next(g for g in m.groups() if g), rest
    return None, pieces


def _add_tag(b: Block, number: str) -> None:
    if r"\tag" not in b.content:    # else the number duplicates the \tag
        b.content += rf" \tag{{{number}}}"


def _formula_blocks(div: htmlmd.Node, bbox, tag, meta) -> list[Block]:
    """An Equation-Block: each display formula becomes a formula block, an
    equation number written after it (outside the math) becomes its \\tag,
    and any other text stays, in order, as text blocks with the same box.
    Inline <math> belongs to that text, unless the block has no display math
    at all (then every <math> is an equation)."""
    maths = div.find_all("math")
    if not maths:
        return [Block(type="text", content=htmlmd.to_markdown(div), bbox=bbox,
                      source=tag, meta=meta)]
    display = any(htmlmd.is_display(m) for m in maths)
    out: list[Block] = []
    number = None                       # a number written before its equation
    for seg in _split(div, htmlmd.is_display if display else lambda n: n.tag == "math"):
        if isinstance(seg, htmlmd.Node):
            out.append(Block(type="formula", content=seg.text().strip(), bbox=bbox,
                             source=tag, meta=dict(meta)))
            if number:
                _add_tag(out[-1], number)
                number = None
            continue
        found, rest = _leading_number(seg)
        if found and out and out[-1].type == "formula":
            _add_tag(out[-1], found)
            seg = rest
        elif found and not _has_content(rest):
            number = found
            continue
        text = htmlmd.to_markdown(_wrapped(div, seg))
        if text.strip(" \n,.;:"):       # more than the sentence's punctuation
            out.append(Block(type="text", content=text, bbox=bbox, source=tag,
                             meta=dict(meta)))
    return out


def _is_layout(c) -> bool:
    return isinstance(c, htmlmd.Node) and c.tag == "div" and (
        "data-label" in c.attrs or "data-bbox" in c.attrs)


def _div_blocks(div: htmlmd.Node, tag: str, inherited=None) -> list[Block]:
    """One layout div -> blocks. Layout divs nested in it (a Complex-Block
    holding Text, Equation-Block and Image divs) become blocks of their own,
    so their formulas are reviewed and their images cropped; the parent's
    loose content stays a block of the parent's label, in reading order."""
    label = div.attrs.get("data-label") or "Text"
    if label == "Blank-Page":
        return []
    bbox = parse_bbox(div.attrs.get("data-bbox", "")) or inherited
    if not any(_is_layout(c) for c in div.children):
        return _leaf_blocks(div, label, bbox, tag)
    out: list[Block] = []
    loose: list = []
    for c in div.children + [None]:
        if c is None or _is_layout(c):
            if _has_content(loose):
                out.extend(_leaf_blocks(_wrapped(div, loose), label, bbox, tag))
            loose = []
            if c is not None:
                out.extend(_div_blocks(c, tag, bbox))
        else:
            loose.append(c)
    return out


def _leaf_blocks(div: htmlmd.Node, label: str, bbox, tag: str) -> list[Block]:
    btype = LABEL_MAP.get(label.lower(), "other")
    meta = {"category": label}

    def block(btype, content, **more):
        return Block(type=btype, content=content, bbox=bbox, source=tag,
                     meta={**meta, **more})

    if btype == "formula":
        return _formula_blocks(div, bbox, tag, meta)
    if btype == "figure":
        alt = " ".join(i.attrs.get("alt", "") for i in div.find_all("img")).strip()
        body = htmlmd.to_markdown(div)
        return [block("figure", "", reader_description="\n\n".join(
            x for x in (htmlmd.escape_html(alt), body) if x))]
    inner = {"table": "table", "code": "pre"}.get(btype)
    if inner and div.find_all(inner):
        # Each table / <pre> is a block of its own; the text around it (a
        # table's title and notes, an algorithm's header) is kept too.
        out = []
        for seg in _split(div, lambda n: n.tag == inner):
            if isinstance(seg, htmlmd.Node):
                out.append(block(btype, htmlmd.table_html(seg) if inner == "table"
                                 else seg.text().strip("\n")))
            elif _has_content(seg):
                text = htmlmd.to_markdown(_wrapped(div, seg))
                if text:
                    out.append(block("caption" if inner == "table" else "text", text))
        return out
    if btype == "code":
        # Pseudocode as paragraphs and <br> lines with <math>: Markdown text
        # (in a code fence the $…$ and the line breaks would show literally).
        return [block("text", htmlmd.to_markdown(div))]
    if btype == "heading":
        h = next((n for n in div.iter() if re.fullmatch(r"h[1-6]", n.tag)), None)
        content = htmlmd.one_line(div)      # every <hN> in it; a <br> is a space
        if h is not None and h.tag == "h1":
            return [block("title", content)]
        # <h2> is a top-level section, '##' (as dots' levels and design.md)
        return [block("heading", content, **({"level": int(h.tag[1]) - 1} if h else {}))]
    content = htmlmd.to_markdown(div)
    if btype == "caption":      # one line: a <br> is a space
        content = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
    return [block(btype, content)]


def _layout_root(root: htmlmd.Node) -> htmlmd.Node:
    """The node whose children are the layout divs: the root, or an
    <html>/<body> wrapper around them."""
    for wrapper in ("html", "body"):
        nodes = [c for c in root.children if isinstance(c, htmlmd.Node)]
        inner = next((c for c in nodes if c.tag == wrapper), None)
        if inner is not None and not any(c.tag == "div" for c in nodes):
            root = inner
    return root


def html_to_blocks(html_text: str, tag: str, truncated: bool = False) -> list[Block]:
    """Chandra's HTML -> blocks. Content outside every layout div is kept, as
    a text block without a box. ``truncated``: the reply was cut off; the
    element in progress at the cut is dropped and a truncated_tail block
    stands for everything lost. A reply that yields no block at all (and no
    Blank-Page) is treated the same way."""
    root = _layout_root(htmlmd.parse(html_text))
    items = list(root.children)
    last = next((k for k in reversed(range(len(items))) if _has_content([items[k]])), None)
    if truncated and last is not None and (
            isinstance(items[last], str) or not items[last].complete):
        del items[last:]                # the element in progress at the cut
    blocks: list[Block] = []
    loose: list = []
    for c in items + [None]:
        if c is None or (isinstance(c, htmlmd.Node) and c.tag == "div"):
            text = htmlmd.to_markdown(_wrapped(root, loose)) if _has_content(loose) else ""
            if text:
                blocks.append(Block(type="text", content=text, source=tag,
                                    meta={"category": None}))
            loose = []
            if c is not None:
                blocks.extend(_div_blocks(c, tag))
        else:
            loose.append(c)
    blank = any(isinstance(c, htmlmd.Node) and c.attrs.get("data-label") == "Blank-Page"
                for c in items)
    if truncated or not (blocks or blank):
        blocks.append(truncated_tail(blocks, tag))
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
        return html_to_blocks(reply.replace(TRUNCATION_MARKER, ""), self.tag,
                              truncated=reply.endswith(TRUNCATION_MARKER))
