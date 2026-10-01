"""Stage 3: pages of blocks -> one Markdown document.

Deterministic and CPU-only. Conventions of the output:
  * display math as  $$ ... $$  on its own lines, inline math as $...$
    (renders on GitHub, Obsidian, Pandoc, Jupyter, VS Code, MkDocs+arithmatex)
  * tables: GFM when the reader gave GFM, HTML otherwise (merged cells survive)
  * figures: the crop is linked, the generated description follows in a
    collapsible <details> so the Markdown stays readable; a graph drawing
    carries two marked versions — "Graph — Markdown (simple)" and
    "Graph — TikZ"
  * running headers/footers/page numbers are dropped (kept in the page JSON)
  * <!-- page N --> comments mark page boundaries for traceability
  * a paragraph split across a page break is re-joined (de-hyphenated when the
    break falls inside a word)
"""

from __future__ import annotations

import re

from .backend import TRUNCATION_MARKER
from .schema import DROPPED_TYPES, Block, Page
from .validate import strip_math_delims

_TERMINAL = re.compile(r"[.!?:;)\]\"'”’]\s*$|\$\$\s*$")


def _clean(s: str) -> str:
    return (s or "").replace(TRUNCATION_MARKER, "").strip()


def render_block(b: Block) -> str:
    c = _clean(b.content)
    t = b.type
    if t == "title":
        return f"# {c}"
    if t == "heading":
        level = int(b.meta.get("level", 1))
        return f"{'#' * min(6, level + 1)} {c}"
    if t == "formula":
        body = strip_math_delims(c)
        return f"$$\n{body}\n$$" if body else ""
    if t == "figure":
        parts = []
        img = b.meta.get("image")
        if img:
            parts.append(f"![{b.meta.get('alt', 'figure')}]({img})")
        desc = b.meta.get("description") or b.meta.get("reader_description")
        if desc:
            summary = "Figure description (generated)"
            if b.meta.get("kind") == "graph" and "Graph — TikZ" in desc:
                summary += " — graph as Markdown and as TikZ"
            parts.append(f"<details><summary>{summary}</summary>\n\n"
                         f"{desc.strip()}\n\n</details>")
        if c:                 # some readers transcribe text inside figures
            parts.append(c)
        return "\n\n".join(parts)
    if t == "caption":
        return f"*{c}*" if c and not c.startswith("*") and "\n" not in c else c
    if t == "code":
        return c if c.startswith("```") else f"```\n{c}\n```"
    if t == "footnote":
        return f"<sub>{c}</sub>" if c else ""
    return c


def _joinable(prev: Block | None, nxt: Block | None) -> bool:
    if prev is None or nxt is None or prev.type != "text" or nxt.type != "text":
        return False
    a, b = _clean(prev.content), _clean(nxt.content)
    return bool(a and b and not _TERMINAL.search(a) and b[0].islower())


def _join(a: str, b: str) -> str:
    a, b = _clean(a), _clean(b)
    if re.search(r"[A-Za-z]-$", a):           # word broken across the page
        return a[:-1] + b
    return f"{a} {b}"


def assemble(pages: list[Page], page_markers: bool = True) -> str:
    """Concatenate pages in order into Markdown."""
    chunks: list[str] = []
    carry: Block | None = None   # last body text block of the previous page
    for page in sorted(pages, key=lambda p: p.index):
        blocks = [b for b in page.blocks if b.type not in DROPPED_TYPES]
        if page_markers:
            chunks.append(f"<!-- page {page.index + 1} -->")
        if carry is not None and blocks and _joinable(carry, blocks[0]):
            # Re-join the paragraph split by the page break: rewrite the
            # previous page's last text chunk, then skip this page's first.
            for k in range(len(chunks) - 1, -1, -1):
                if chunks[k] == render_block(carry):
                    carry = Block(type="text",
                                  content=_join(carry.content, blocks[0].content))
                    chunks[k] = render_block(carry)
                    break
            blocks = blocks[1:]
        last_text = carry if not blocks else None
        for b in blocks:
            r = render_block(b)
            if r:
                chunks.append(r)
            last_text = b if b.type == "text" else (
                last_text if b.type in ("footnote", "figure", "caption") else None)
        carry = last_text
    md = "\n\n".join(chunks).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", md)
