"""Whole-page Markdown readers (no layout boxes).

For models whose native output is one Markdown document per page — olmOCR
(the ``olmocr`` reader, with olmOCR's own prompt), or any general VLM prompted
to transcribe a page (``markdown``). The Markdown is split back into blocks
(the '#' title, headings, display math, tables, paragraphs) so that the
validators and the per-block review still apply; without boxes the reviewer
sees the whole page for each block, and figures cannot be cropped.

A reply cut off at max_tokens keeps its complete blocks; the line and the
block in progress at the cut are dropped, and a truncated_tail block (base.py)
stands for the rest. Without boxes its region is unknown (bbox None).
"""

from __future__ import annotations

import re

from PIL import Image

from ..backend import TRUNCATION_MARKER, image_part, text_part
from ..schema import Block
from ..validate import strip_math_delims
from .base import RETRY_SAMPLING, Reader, truncated_tail
from .htmlmd import escape_html

PAGE_PROMPT = """Transcribe this page of a research paper into Markdown, reading it in natural order.
- Inline math as $...$ and display math as $$...$$ on their own lines, in LaTeX; keep equation numbers as \\tag{...}.
- Tables as HTML <table> (Markdown if they have no merged cells).
- Headings as #, ##, ###. Omit running headers, footers, and page numbers.
- For each figure, write a single line: ![figure](figure)
- Transcribe exactly what is visible. Do not summarise, translate, or complete text."""

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FIGURE = re.compile(r"^!\[([^\]]*)\]\([^)]*\)\s*$")
# One fence around the whole reply, as chat models like to add.
_OUTER_FENCE = re.compile(r"^\s*```(?:markdown|md)?[ \t]*\n", re.I)


def strip_front_matter(md: str) -> tuple[str, dict]:
    """olmOCR-style YAML front matter -> (body, dict of simple key: value)."""
    m = re.match(r"^\s*---\s*\n(.*?)\n---\s*\n?", md, re.S)
    if not m:
        return md, {}
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return md[m.end():], meta


def strip_outer_fence(md: str) -> str:
    """'```markdown\\n…\\n```' -> '…' (also a bare ``` or ```md fence that
    opens the reply; the closing fence may be missing, as after a cut). A
    reply that starts with a real code block (```python) is left alone."""
    m = _OUTER_FENCE.match(md)
    return re.sub(r"\n?```\s*$", "", md[m.end():]) if m else md


def _display_opener(s: str) -> bool:
    """$$, or \\[ when it opens display math: alone, closed at the end of the
    line, or continued on later lines. '\\[1\\] D. Spielman …' is a
    Markdown-escaped citation, not math."""
    if s.startswith("$$"):
        return True
    return s.startswith(r"\[") and (r"\]" not in s or s.endswith(r"\]"))


def split_markdown(md: str, truncated: bool = False) -> list[Block]:
    """Markdown page -> blocks. Display math and HTML tables may span blank
    lines, so they are collected up to their closing delimiter.

    ``truncated``: the text was cut off, so what was in progress at the cut —
    the last line, an unclosed formula or table, a paragraph not yet ended by
    a blank line — is incomplete and dropped."""
    lines = md.replace("\r\n", "\n").split("\n")
    if truncated:
        lines.pop()     # the line in progress ('' if the cut fell at a line end)
    blocks: list[Block] = []
    para: list[str] = []

    def flush():
        if para:
            text = "\n".join(para).strip()
            if text:
                kind = "table" if text.lstrip().startswith("|") else (
                    "list" if re.match(r"^\s*([-*+]|\d+[.)])\s", text) else "text")
                blocks.append(Block(type=kind, content=text))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if _display_opener(s):
            flush()
            opener = "$$" if s.startswith("$$") else r"\["
            closer = "$$" if opener == "$$" else r"\]"
            buf = [s]
            closed = closer in s[len(opener):]
            if not closed:
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if closer in lines[i]:
                        closed = True
                        break
                    i += 1
            if closed or not truncated:
                blocks.append(Block(type="formula",
                                    content=strip_math_delims("\n".join(buf).strip())))
        elif s.lower().startswith("<table"):
            flush()
            buf = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
            if "</table>" in lines[i].lower() or not truncated:
                blocks.append(Block(type="table", content="\n".join(buf).strip()))
        elif _HEADING.match(s):
            flush()
            hashes, title = _HEADING.match(s).groups()
            # '#' is the title, '##' a top-level section (level 1), as dots
            blocks.append(Block(type="title", content=title.strip()) if len(hashes) == 1
                          else Block(type="heading", content=title.strip(),
                                     meta={"level": len(hashes) - 1}))
        elif _FIGURE.match(s):
            flush()
            alt = _FIGURE.match(s).group(1).strip()
            meta = {"reader_description": escape_html(alt)} \
                if alt.lower() not in ("", "figure", "image") else {}
            blocks.append(Block(type="figure", content="", meta=meta))
        elif not s:
            flush()
        else:
            para.append(line)
        i += 1
    if truncated:
        para.clear()    # not ended by a blank line: in progress at the cut
    flush()
    return blocks


class PageMarkdownReader(Reader):
    """Any VLM served behind an OpenAI-compatible endpoint, one Markdown page
    per request, with the generic PAGE_PROMPT."""

    name = "markdown"
    target_long_side = 1600

    def __init__(self, client, max_tokens: int = 8192, prompt: str = PAGE_PROMPT):
        super().__init__(client, max_tokens)
        self.prompt = prompt

    def prepare(self, img: Image.Image) -> Image.Image:
        scale = self.target_long_side / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)),
                             Image.LANCZOS)
        return img

    def request(self, img: Image.Image, attempt: int) -> str:
        sampling = dict(RETRY_SAMPLING) if attempt else {}
        return self.client.chat([image_part(img), text_part(self.prompt)],
                                max_tokens=self.max_tokens,
                                temperature=sampling.pop("temperature", 0.0),
                                extra=sampling or None)

    def parse(self, reply: str) -> list[Block]:
        truncated = reply.endswith(TRUNCATION_MARKER)
        body, _ = strip_front_matter(strip_outer_fence(reply.replace(TRUNCATION_MARKER, "")))
        blocks = split_markdown(body, truncated)
        if truncated:
            blocks.extend(truncated_tail(blocks, self.tag))
        for b in blocks:
            b.source = self.tag
        return blocks

    def read(self, img: Image.Image, attempt: int = 0) -> list[Block]:
        return self.parse(self.request(self.prepare(img), attempt))


# olmOCR-2's prompt, verbatim from allenai/olmocr (Apache-2.0),
# olmocr/prompts/prompts.py: build_no_anchoring_v4_yaml_prompt(), the prompt
# olmocr/pipeline.py sends to olmOCR-2.
OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)
# olmocr/pipeline.py: TEMPERATURE_BY_ATTEMPT (a retry samples hotter)
OLMOCR_TEMPERATURES = (0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0)
# Inline math \( … \), within one paragraph.
_INLINE_PARENS = re.compile(r"\\\(\s*(\S(?:(?!\\\(|\n\s*\n).)*?)\s*\\\)", re.S)
# Display math \[ … \] (olmOCR was trained on text whose $$ … $$ became this,
# wherever it stood). The '\[' of a LaTeX row break with spacing (\\[4pt])
# opens nothing.
_DISPLAY_BRACKETS = re.compile(r"(?<!\\)\\\[((?:(?!(?<!\\)\\\[).)*?)\\\]", re.S)
# What only math holds: a command, a script, a relation, a group. A
# Markdown-escaped citation (\[1, 2\], \[Spi04\], \[ABC+20\], \[3, §2\],
# \[Spi04, Thm. 2(b)\]) or interval (\[−1, 1\]) holds none of these.
_MATHY = re.compile(r"\\[A-Za-z]|[\^_=<>{}|]|\s[-+*/\u2212]\s|\)\s*\(")
_TABLE = re.compile(r"(<table\b.*?</table>)", re.S | re.I)
_TURN = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
         270: Image.Transpose.ROTATE_270}


def _lift_display(md: str) -> str:
    """Display math \\[ … \\] sharing a line with text -> $$ lines of its own,
    so that split_markdown makes it a formula block (outside tables). Only
    brackets around math: a citation in Markdown-escaped brackets stays
    text."""
    def lift(part: str) -> str:
        def sub(m: re.Match) -> str:
            before = part[part.rfind("\n", 0, m.start()) + 1:m.start()]
            end = part.find("\n", m.end())
            after = part[m.end():end if end >= 0 else len(part)]
            if not (before.strip() or after.strip()) or not _MATHY.search(m.group(1)):
                return m.group(0)       # on lines of its own already, or not math
            return f"\n$$\n{m.group(1).strip()}\n$$\n"
        return _DISPLAY_BRACKETS.sub(sub, part)
    return "".join(p if k % 2 else lift(p) for k, p in enumerate(_TABLE.split(md)))


def _rotation_asked(reply: str) -> int:
    """The turn olmOCR's front matter asks for (is_rotation_valid: false,
    rotation_correction: 90/180/270), else 0."""
    _, meta = strip_front_matter(strip_outer_fence(reply))
    turn = meta.get("rotation_correction", "")
    if meta.get("is_rotation_valid", "").lower() != "false" or not turn.isdigit():
        return 0
    return int(turn) if int(turn) in _TURN else 0


class OlmOCRReader(PageMarkdownReader):
    """olmOCR-2 (allenai/olmOCR-2-7B-1025), run as olmOCR's own pipeline runs
    it (olmocr/pipeline.py): the v4 no-anchoring YAML prompt, the text part
    before the image, the page rendered at 1288 px on its longest side, at
    most 8000 tokens, temperature by attempt, and a page the model says is
    rotated read again turned as it asks (up to max_turns times; the blocks
    of a turned page carry meta["rotation"]). Its reply is Markdown under
    YAML front matter (stripped), inline math as \\(…\\) and display math as
    \\[…\\] (converted to $…$ and $$…$$ here)."""

    name = "olmocr"
    target_long_side = 1288
    max_turns = 3

    def __init__(self, client, max_tokens: int = 8000):
        super().__init__(client, max_tokens, prompt=OLMOCR_PROMPT)

    def read(self, img: Image.Image, attempt: int = 0) -> list[Block]:
        img, rotation = self.prepare(img), 0
        for k in range(self.max_turns + 1):
            # each turn is a further attempt, as in olmOCR's pipeline
            reply = self.request(img.transpose(_TURN[rotation]) if rotation else img,
                                 attempt + k)
            turn = _rotation_asked(reply)
            if not turn or k == self.max_turns:
                break
            rotation = (rotation + turn) % 360
        blocks = self.parse(reply)
        for b in blocks if rotation else ():
            if not b.meta.get("truncated_tail"):
                b.meta["rotation"] = rotation
        return blocks

    def prepare(self, img: Image.Image) -> Image.Image:
        scale = self.target_long_side / max(img.size)     # up or down, as olmOCR renders
        if scale != 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)),
                             Image.LANCZOS)
        return img

    def request(self, img: Image.Image, attempt: int) -> str:
        return self.client.chat(
            [text_part(self.prompt), image_part(img)], max_tokens=self.max_tokens,
            temperature=OLMOCR_TEMPERATURES[min(attempt, len(OLMOCR_TEMPERATURES) - 1)])

    def parse(self, reply: str) -> list[Block]:
        reply = _INLINE_PARENS.sub(lambda m: "$" + " ".join(m.group(1).split()) + "$", reply)
        return super().parse(_lift_display(reply))
