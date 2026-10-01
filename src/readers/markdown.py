"""Whole-page Markdown readers (no layout boxes).

For models whose native output is one Markdown document per page — olmOCR
style, or any general VLM prompted to transcribe a page. The Markdown is split
back into blocks (headings, display math, tables, paragraphs) so that the
validators and the per-block review still apply; without boxes the reviewer
sees the whole page for each block, and figures cannot be cropped.
"""

from __future__ import annotations

import re

from PIL import Image

from ..backend import image_part, text_part
from ..schema import Block
from ..validate import strip_math_delims
from .base import RETRY_SAMPLING, Reader

PAGE_PROMPT = """Transcribe this page of a research paper into Markdown, reading it in natural order.
- Inline math as $...$ and display math as $$...$$ on their own lines, in LaTeX; keep equation numbers as \\tag{...}.
- Tables as HTML <table> (Markdown if they have no merged cells).
- Headings as #, ##, ###. Omit running headers, footers, and page numbers.
- For each figure, write a single line: ![figure](figure)
- Transcribe exactly what is visible. Do not summarise, translate, or complete text."""

_DISPLAY = re.compile(r"^\s*(\$\$|\\\[)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FIGURE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")


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


def split_markdown(md: str) -> list[Block]:
    """Markdown page -> blocks. Display math and HTML tables may span blank
    lines, so they are collected up to their closing delimiter."""
    lines = md.replace("\r\n", "\n").split("\n")
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
        if _DISPLAY.match(line):
            flush()
            opener = "$$" if s.startswith("$$") else r"\["
            closer = "$$" if opener == "$$" else r"\]"
            buf = [s]
            rest = s[len(opener):]
            if closer not in rest:
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if closer in lines[i]:
                        break
                    i += 1
            blocks.append(Block(type="formula",
                                content=strip_math_delims("\n".join(buf).strip())))
        elif s.lower().startswith("<table"):
            flush()
            buf = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
            blocks.append(Block(type="table", content="\n".join(buf).strip()))
        elif _HEADING.match(s):
            flush()
            hashes, title = _HEADING.match(s).groups()
            blocks.append(Block(type="heading", content=title.strip(),
                                meta={"level": max(1, len(hashes) - 1)}))
        elif _FIGURE.match(s):
            flush()
            blocks.append(Block(type="figure", content=""))
        elif not s:
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


class PageMarkdownReader(Reader):
    """Any VLM served behind an OpenAI-compatible endpoint, one Markdown page
    per request. With ``--reader-prompt olmocr`` the olmOCR prompt and front
    matter are used instead of the generic prompt."""

    name = "markdown"
    target_long_side = 1600

    def __init__(self, client, max_tokens: int = 8192, prompt: str = PAGE_PROMPT):
        super().__init__(client, max_tokens)
        self.prompt = prompt

    def read(self, img: Image.Image, attempt: int = 0) -> list[Block]:
        scale = self.target_long_side / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)),
                             Image.LANCZOS)
        sampling = dict(RETRY_SAMPLING) if attempt else {}
        reply = self.client.chat([image_part(img), text_part(self.prompt)],
                                 max_tokens=self.max_tokens,
                                 temperature=sampling.pop("temperature", 0.0),
                                 extra=sampling or None)
        body, _ = strip_front_matter(reply)
        blocks = split_markdown(body)
        for b in blocks:
            b.source = self.tag
        return blocks
