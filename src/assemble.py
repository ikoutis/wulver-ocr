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
  * <!-- page N --> comments mark page boundaries for traceability; a page the
    reader could not read is always marked <!-- page N: OCR failed, see
    report.json -->, and so is text lost where the reader's output was cut off
  * a paragraph, list item or reference split across a page break is
    re-joined, also when floats (figures, captions, tables, footnotes) sit
    between its halves. A word broken at the break loses its hyphen unless
    the document spells it with one elsewhere ("well-known") or, lacking
    evidence either way, it starts with a prefix that keeps one ("so-called")
"""

from __future__ import annotations

import re
from collections import Counter

from .backend import TRUNCATION_MARKER
from .schema import DROPPED_TYPES, Block, Page
from .validate import strip_math_delims

_TERMINAL = re.compile(r"[.!?:;)\]\"'”’]\s*$|\$\$\s*$")
_JOINABLE = frozenset({"text", "list", "reference"})
# Blocks set apart from the text flow (LaTeX floats them to the top or bottom
# of a page): a paragraph split by a page break may have these between its halves.
_FLOATS = frozenset({"figure", "caption", "table", "footnote"})
# A word broken after one of these keeps its hyphen when the document gives
# no evidence either way ("well-known", "so-called"; but "sparsi-fication").
_HYPHEN_PREFIXES = frozenset({"well", "so", "self", "non", "ill", "half", "first", "quasi"})
_LETTERS = r"[^\W\d_]+"
_WORD = re.compile(rf"{_LETTERS}(?:-{_LETTERS})*")


def _clean(s: str) -> str:
    return (s or "").replace(TRUNCATION_MARKER.strip(), "").strip()


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
        desc = _clean(b.meta.get("description") or b.meta.get("reader_description"))
        if desc:
            summary = "Figure description (generated)"
            if b.meta.get("kind") == "graph" and "Graph — TikZ" in desc:
                summary += (" — graph as TikZ (Markdown version unavailable)"
                            if "not available" in desc else " — graph as Markdown and as TikZ")
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
    if (prev is None or nxt is None or prev.type != nxt.type
            or prev.type not in _JOINABLE):
        return False
    a, b = _clean(prev.content), _clean(nxt.content)
    return bool(a and b and not _TERMINAL.search(a) and b[0].islower())


def _word_counts(pages: list[Page]) -> Counter:
    """How often each word, and each hyphenated pair of words, occurs."""
    counts: Counter = Counter()
    for p in pages:
        for b in p.blocks:
            for w in _WORD.findall(_clean(b.content).lower()):
                parts = w.split("-")
                counts.update(parts)
                counts.update(f"{x}-{y}" for x, y in zip(parts, parts[1:]))
    return counts


def _keep_hyphen(head: str, tail: str, counts: Counter) -> bool:
    """Is "head-" / "tail" at a page break the compound "head-tail" rather
    than one word hyphenated by the typesetter? The spelling the document
    uses elsewhere decides; without evidence, the prefix list does."""
    head, tail = head.lower(), tail.lower()
    hyphenated, solid = counts[f"{head}-{tail}"], counts[head + tail]
    if hyphenated != solid:
        return hyphenated > solid
    return head in _HYPHEN_PREFIXES


def _join(a: str, b: str, counts: Counter | None = None) -> str:
    a, b = _clean(a), _clean(b)
    if re.search(r"\S-$", a) and b[:1].isalpha():      # broken at a hyphen
        head = re.search(rf"({_LETTERS})-$", a)
        if head is None or _keep_hyphen(head.group(1), re.match(_LETTERS, b).group(0),
                                        counts or Counter()):
            return a + b                                # "well-known", "$k$-connected"
        return a[:-1] + b                               # "sparsification"
    return f"{a} {b}"


def _failed(page: Page) -> bool:
    return bool(page.meta.get("failed")) or any("page_failed" in b.flags for b in page.blocks)


def assemble(pages: list[Page], page_markers: bool = True) -> str:
    """Concatenate pages in order into Markdown."""
    counts = _word_counts(pages)
    chunks: list[str] = []
    # The last body block before the page break (text, list item or
    # reference) and its index in chunks: the next page may continue it.
    carry: tuple[Block, int] | None = None
    for page in sorted(pages, key=lambda p: p.index):
        n = page.index + 1
        blocks = [b for b in page.blocks if b.type not in DROPPED_TYPES]
        if _failed(page):
            chunks.append(f"<!-- page {n}: OCR failed, see report.json -->")
            carry = None                # its text is missing: nothing joins across it
        elif page_markers:
            chunks.append(f"<!-- page {n} -->")
        last = carry
        if carry is not None:
            # The continuation is the first block that is not a float (a
            # figure or table at the top of the page comes before it).
            j = next((k for k, b in enumerate(blocks) if b.type not in _FLOATS), None)
            if j is not None and _joinable(carry[0], blocks[j]):
                prev, k = carry
                joined = Block(type=prev.type,
                               content=_join(prev.content, blocks[j].content, counts))
                chunks[k] = render_block(joined)    # the floats follow it
                last = (joined, k)
                blocks = blocks[:j] + blocks[j + 1:]
        for b in blocks:
            r = render_block(b)
            if not r and b.meta.get("truncated_tail"):
                r = f"<!-- page {n}: the reader's output was cut off here, see report.json -->"
            if r:
                chunks.append(r)
            if b.type not in _FLOATS:   # floats do not interrupt the text
                last = (b, len(chunks) - 1) if b.type in _JOINABLE and _clean(b.content) \
                    else None
        carry = last
    md = "\n\n".join(chunks).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", md)
