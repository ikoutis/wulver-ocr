"""Deterministic, CPU-only checks on block content.

These are the hinge of the two-model design: the reader's output is trusted
unless a validator flags it (or the review policy asks for a block type to be
reviewed regardless — formulas by default), and an edit proposed by the
reviewer is accepted only if it does not leave the block with flags the reader's
version did not have. Every check returns a list of short flag strings; empty
means clean.

Flags:
  empty            no content where content is expected
  truncated        the model hit max_tokens (backend appended the marker)
  repetition       degenerate loop (the classic VLM-OCR failure mode)
  latex_braces     unbalanced { } in math
  latex_env        \\begin/\\end mismatch
  latex_leftright  \\left/\\right count mismatch
  latex_delims     stray $ / \\( \\) / \\[ \\] inside a formula body
  latex_katex      KaTeX cannot parse it (only when node + katex are
                   available; see katex_check.py)
  inline_math      odd number of unescaped $ in a text block
  table_shape      rows with inconsistent effective column counts
  table_parse      table content is neither parseable HTML nor GFM
"""

from __future__ import annotations

import re
import zlib
from html.parser import HTMLParser

from .backend import TRUNCATION_MARKER
from .katex_check import katex_error
from .schema import Block

# ----------------------------------------------------------------- repetition


def has_repetition(text: str, min_len: int = 200, max_period: int = 60,
                   min_repeats: int = 6) -> bool:
    """True if text ends in (or contains) a short unit repeated many times,
    or is suspiciously compressible for its length.

    Two complementary tests: (1) a token-level tail scan for a period p
    repeated >= min_repeats times — catches "\\cdot \\cdot \\cdot ..." and
    "the the the ..." loops; (2) zlib ratio < 0.12 on long texts — catches
    loops whose unit is longer than max_period tokens.
    """
    if len(text) < min_len:
        return False
    toks = text.split()
    n = len(toks)
    for p in range(1, min(max_period, n // min_repeats) + 1):
        unit = toks[n - p:]
        reps = 1
        i = n - 2 * p
        while i >= 0 and toks[i:i + p] == unit:
            reps += 1
            i -= p
        if reps >= min_repeats:
            return True
    raw = text.encode("utf-8")
    return len(raw) > 2000 and len(zlib.compress(raw)) / len(raw) < 0.12


# ---------------------------------------------------------------------- latex

_ESCAPED_BRACE = re.compile(r"\\[{}]")
_BEGIN_END = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")
_LEFT = re.compile(r"\\left(?![a-zA-Z])")
_RIGHT = re.compile(r"\\right(?![a-zA-Z])")


def _braces_balanced(s: str) -> bool:
    depth = 0
    for ch in _ESCAPED_BRACE.sub("", s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _envs_balanced(s: str) -> bool:
    stack = []
    for kind, name in _BEGIN_END.findall(s):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def check_latex(body: str, display: bool = True) -> list[str]:
    """Structural checks on a formula body (no surrounding delimiters), then,
    if the structure is sound, a KaTeX parse."""
    flags = []
    if not _braces_balanced(body):
        flags.append("latex_braces")
    if not _envs_balanced(body):
        flags.append("latex_env")
    if len(_LEFT.findall(body)) != len(_RIGHT.findall(body)):
        flags.append("latex_leftright")
    stripped = body.replace(r"\$", "")
    if "$" in stripped or re.search(r"\\[()\[\]]", stripped.replace(r"\\", "")):
        flags.append("latex_delims")
    if not flags and body.strip() and katex_error(body, display) is not None:
        flags.append("latex_katex")
    return flags


def strip_math_delims(s: str) -> str:
    """'$$ x $$' / '\\[ x \\]' / '$x$' -> 'x' (readers disagree on this)."""
    s = s.strip()
    for a, b in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            return s[len(a):len(s) - len(b)].strip()
    return s


_INLINE_MATH = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", re.S)


def check_inline_math(text: str) -> list[str]:
    unescaped = re.sub(r"\\\$", "", text).replace("$$", "")
    if unescaped.count("$") % 2:
        return ["inline_math"]
    flags = []
    for m in _INLINE_MATH.finditer(text):
        flags.extend(check_latex(m.group(1), display=False))
    return sorted(set(flags))


# --------------------------------------------------------------------- tables


class _TableShape(HTMLParser):
    """Effective column count per row, honouring colspan AND rowspan."""

    def __init__(self):
        super().__init__()
        self.rows: list[int] = []
        self._carry: dict[int, int] = {}   # column -> rows still covered by a rowspan
        self._col = 0
        self._in_row = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._in_row, self._col = True, 0
        elif tag in ("td", "th") and self._in_row:
            while self._carry.get(self._col, 0) > 0:
                self._col += 1
            a = dict(attrs)
            span = _int(a.get("colspan"), 1)
            rspan = _int(a.get("rowspan"), 1)
            if rspan > 1:
                for c in range(self._col, self._col + span):
                    self._carry[c] = rspan          # includes this row
            self._col += span

    def handle_endtag(self, tag):
        if tag == "tr" and self._in_row:
            while self._carry.get(self._col, 0) > 0:
                self._col += 1
            self.rows.append(self._col)
            self._carry = {c: k - 1 for c, k in self._carry.items() if k > 1}
            self._in_row = False


def _int(v, default):
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return default


def check_table(content: str) -> list[str]:
    c = content.strip()
    if "<table" in c.lower():
        p = _TableShape()
        try:
            p.feed(c)
        except Exception:   # noqa: BLE001
            return ["table_parse"]
        if not p.rows:
            return ["table_parse"]
        return ["table_shape"] if len(set(p.rows)) > 1 else []
    lines = [ln for ln in c.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return ["table_parse"]
    counts = {len(re.split(r"(?<!\\)\|", ln.strip().strip("|"))) for ln in lines}
    return ["table_shape"] if len(counts) > 1 else []


# ---------------------------------------------------------------------- block


def validate_block(b: Block) -> list[str]:
    """All flags for a block, sorted. Figures need no content."""
    c = b.content or ""
    flags: list[str] = []
    if c.endswith(TRUNCATION_MARKER) or "<<TRUNCATED>>" in c:
        flags.append("truncated")
    c = c.replace(TRUNCATION_MARKER, "")
    if b.type == "figure":
        return sorted(set(flags))
    if not c.strip() and b.type not in ("header", "footer", "page_number"):
        flags.append("empty")
    if has_repetition(c):
        flags.append("repetition")
    if b.type == "formula":
        flags.extend(check_latex(c))
    elif b.type == "table":
        flags.extend(check_table(c))
    elif b.type in ("text", "list", "caption", "footnote", "heading", "title",
                    "reference"):
        flags.extend(check_inline_math(c))
    return sorted(set(flags))
