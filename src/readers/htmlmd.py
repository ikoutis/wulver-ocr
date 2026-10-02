"""Small, dependency-free HTML -> Markdown conversion for OCR-model HTML.

Some readers (Chandra) emit a restricted HTML vocabulary with math in
<math>…</math> (display="block" for display math). Math is what this project
cares about most, so the conversion is done here, where the delimiters are
under our control, rather than by a general converter:

  <math>x</math>                 -> $x$
  <math display="block">x</math> -> $$\\nx\\n$$  (the Chandra adapter lifts an
                                    Equation-Block's display math out into
                                    formula blocks of their own)
  <table>…</table>               -> HTML kept (merged cells survive), with any
                                    math inside cells turned into $…$
  <img>                          -> nothing (a figure block reads its alt)
  everything else                -> Markdown (paragraphs, headings, lists,
                                    emphasis, code, links)

Text is escaped so that it renders as written (Chandra's own converter also
escapes * and _): $ * _ ` get a backslash, a '<' that could open a tag and an
'&' that could start an entity are HTML-escaped, and a line that would begin
a heading, list, or quote gets a backslash at its start.

A <br> in a paragraph or list item is a hard line break (a backslash at the
end of the line), as in Chandra's own Markdown; in a heading it is a space.
An ordered list keeps its start number, and a lettered or roman one
(type="a", "i", …) writes its labels, "(a)", "(ii)", into its items.

Parsing uses html.parser plus HTML's optional end tags for <p>, <li>, and
table cells and rows; it tolerates the unclosed tags models produce, and caps
nesting depth so that a looping reply cannot make the tree deep enough to
overflow the recursion.
"""

from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser
from typing import Iterator, Optional, Union

VOID = {"br", "img", "hr", "input", "meta", "link"}
KEEP_TABLE_ATTRS = {"colspan", "rowspan"}
BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table",
              "pre", "hr", "caption"}
# A start tag that ends an open <p> (HTML's implied </p>), and the elements
# whose content a <p> outside them cannot reach.
_CLOSES_P = (BLOCK_TAGS - {"caption"}) | {"li", "blockquote"}
_P_SCOPE = {"div", "li", "td", "th", "caption", "table"}
_LI_SCOPE = {"ul", "ol"} | (_P_SCOPE - {"li"})
# A new cell ends the open cell of its row, a new row the open row of its
# table (HTML's implied </td> </th> </tr>).
_ROW_SCOPE = {"table", "thead", "tbody", "tfoot"}
_CELL_SCOPE = _ROW_SCOPE | {"tr"}
# Deeper than this is a repetition loop, not a layout: further start tags are
# kept but not nested, so every traversal below stays shallow.
MAX_DEPTH = 100


class Node:
    __slots__ = ("tag", "attrs", "children", "complete")

    def __init__(self, tag: str, attrs: Optional[dict] = None):
        self.tag = tag
        self.attrs = attrs or {}
        self.children: list[Union["Node", str]] = []
        self.complete = True        # False: the input ended inside this element

    def iter(self) -> Iterator["Node"]:
        """Descendant elements in document order (not self)."""
        stack = [c for c in reversed(self.children) if isinstance(c, Node)]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(c for c in reversed(n.children) if isinstance(c, Node))

    def text(self) -> str:
        """Raw text content (entities already decoded)."""
        out, stack = [], list(reversed(self.children))
        while stack:
            c = stack.pop()
            if isinstance(c, str):
                out.append(c)
            else:
                stack.extend(reversed(c.children))
        return "".join(out)

    def find_all(self, tag: str) -> list["Node"]:
        return [n for n in self.iter() if n.tag == tag]


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def _close_implied(self, tag: str, scope: set) -> None:
        """Pop back to and including the innermost open ``tag``, unless an
        element in ``scope`` is open inside it."""
        for i in range(len(self.stack) - 1, 0, -1):
            t = self.stack[i].tag
            if t == tag:
                del self.stack[i:]
                return
            if t in scope:
                return

    def _open(self, tag, attrs, push: bool):
        if tag in _CLOSES_P:
            self._close_implied("p", _P_SCOPE)
        if tag == "li":     # <li>a<li>b: the second item ends the first
            self._close_implied("li", _LI_SCOPE)
        if tag in ("td", "th", "tr"):
            self._close_implied("td", _CELL_SCOPE)
            self._close_implied("th", _CELL_SCOPE)
        if tag == "tr":
            self._close_implied("tr", _ROW_SCOPE)
        node = Node(tag, {k: (v if v is not None else "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if push and tag not in VOID and len(self.stack) < MAX_DEPTH:
            self.stack.append(node)

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs, push=True)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, push=False)

    def handle_endtag(self, tag):
        # close up to the matching open tag; ignore stray closers
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


# A math span, not crossing into another math or layout element.
_MATH_SPAN = re.compile(r"(<math\b[^>]*>)((?:(?!</?math\b|</?div\b).)*?)(</math>)",
                        re.S | re.I)


def parse(html_text: str) -> Node:
    # A raw '<' in LaTeX (0<x<1, i<j) would open a phantom tag that swallows
    # the </math> and the text after it: escape it before parsing.
    html_text = _MATH_SPAN.sub(lambda m: m[1] + m[2].replace("<", "&lt;") + m[3],
                               html_text)
    b = _TreeBuilder()
    b.feed(html_text)
    b.close()
    for n in b.stack[1:]:
        n.complete = False
    return b.root


def is_display(node: Node) -> bool:
    return node.tag == "math" and node.attrs.get("display", "").lower() == "block"


# ----------------------------------------------------------------- escaping

_HTML_SPECIAL = re.compile(r"<(?=[A-Za-z/!?])|&(?=#?\w+;)")
_TEXT_SPECIAL = re.compile(r"(?<!\\)[$*_`]|<(?=[A-Za-z/!?])|&(?=#?\w+;)")
# What makes a line a heading, list item, quote, or rule.
_BLOCK_START = re.compile(
    r"^(?:#{1,6}(?=[ \t]|$)|[-+](?=[ \t]|$)|>|-+[ \t]*$|\d{1,9}(?=[.)](?:[ \t]|$)))")


def _escape_char(m: re.Match) -> str:
    c = m.group(0)
    return "&lt;" if c == "<" else "&amp;" if c == "&" else "\\" + c


def _escape_block_start(m: re.Match) -> str:
    s = m.group(0)      # for '2016. ' the escape goes before the '.'
    return s + "\\" if s[0].isdigit() else "\\" + s


def escape_html(s: str) -> str:
    """Make free text safe inside Markdown without changing its own Markdown:
    a '<' that could open a tag and an '&' that could start an entity."""
    return _HTML_SPECIAL.sub(_escape_char, s)


def escape_text(s: str) -> str:
    """Text that must render as written: $ * _ ` and HTML specials."""
    return _TEXT_SPECIAL.sub(_escape_char, s)


def _escape_block_starts(s: str) -> str:
    """Escape what would start a Markdown block at the start of each line
    ('# ', '- ', '+ ', '>', '2016. '), outside $$ display math."""
    lines, in_math = s.split("\n"), False
    for k, line in enumerate(lines):
        if line.strip().strip("*~") == "$$":    # '**$$' if the math was in <b>
            in_math = not in_math
        elif not in_math:
            lines[k] = _BLOCK_START.sub(_escape_block_start, line)
    return "\n".join(lines)


# ------------------------------------------------------------------ inline

# inline() marks a <br> with _BR; a paragraph turns it into HARD_BREAK (a
# backslash at the end of the line), a heading into a space.
_BR = "\ue000"
HARD_BREAK = "\\\n"
_SPACE = " \t\n\r\f\v" + _BR


def _wrap(inner: str, delim: str) -> str:
    """Emphasis delimiters go inside the element's surrounding whitespace:
    '<b>Proof. </b>By' is '**Proof.** By', not '**Proof.**By'."""
    core = inner.strip(_SPACE)
    if not core:
        return inner
    lead = inner[:len(inner) - len(inner.lstrip(_SPACE))]
    trail = inner[len(inner.rstrip(_SPACE)):]
    return f"{lead}{delim}{core}{delim}{trail}"


def inline(node: Union[Node, str], pre: bool = False) -> str:
    if isinstance(node, str):
        return node if pre else escape_text(re.sub(r"\s+", " ", node))
    t = node.tag
    if t == "math":
        if is_display(node):
            return f"\n$$\n{node.text().strip()}\n$$\n"
        tex = " ".join(node.text().split())     # a newline inside $…$ could start a list
        return f"${tex}$" if tex else ""
    if t == "br":
        return _BR
    if t == "img":      # outside a figure block an <img> is a hallucination
        return ""       # (Chandra's own parser deletes these)
    if t == "input":
        return "[x] " if "checked" in node.attrs else "[ ] "
    if t in ("pre", "code"):
        return f"`{node.text()}`"
    inner = "".join(inline(c, pre) for c in node.children)
    if t in ("b", "strong"):
        return _wrap(inner, "**")
    if t in ("i", "em"):
        return _wrap(inner, "*")
    if t == "del":
        return _wrap(inner, "~~")
    if t in ("sup", "sub"):
        return f"<{t}>{inner}</{t}>"
    if t == "a" and node.attrs.get("href"):
        return f"[{inner}]({node.attrs['href']})"
    if t in BLOCK_TAGS or t == "li":    # siblings must not run together
        return f" {inner} "
    return inner


def one_line(node: Node) -> str:
    """Inline Markdown on one line (headings): a <br> becomes a space."""
    return _squash(inline(node).replace(_BR, " ").replace("\n", " "))


# ------------------------------------------------------------------- block


def table_html(node: Node) -> str:
    """Re-serialise a table: structural tags + colspan/rowspan only, math as $..$."""
    def ser(n: Union[Node, str]) -> str:
        if isinstance(n, str):
            return _html.escape(re.sub(r"\s+", " ", n), quote=False)
        if n.tag == "math":
            return _html.escape(f"${n.text().strip()}$", quote=False)
        parts, after_block = [], False
        for c in n.children:
            s = ser(c)
            if not s.strip():
                parts.append(s)
                continue
            block = isinstance(c, Node) and c.tag in ("p", "div")
            if (block or after_block) and "".join(parts).strip():
                parts.append("<br>")    # <p>first</p><p>second</p> in a cell
            parts.append(s)
            after_block = block
        inner = "".join(parts)
        if n.tag in ("table", "thead", "tbody", "tr", "td", "th", "caption"):
            attrs = "".join(f' {k}="{v}"' for k, v in n.attrs.items()
                            if k in KEEP_TABLE_ATTRS)
            return f"<{n.tag}{attrs}>{inner}</{n.tag}>"
        if n.tag == "br":
            return "<br>"
        if n.tag in ("sup", "sub", "b", "i", "strong", "em"):
            return f"<{n.tag}>{inner}</{n.tag}>"
        return inner
    return ser(node)


_LETTERED = ("a", "A", "i", "I")
_ROMAN = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
          (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))


def _count(v, default: int) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return default


def _ol_label(k: int, kind: str) -> str:
    """Item k's label in an <ol type=kind>: a, b, …, aa / i, ii, … / A / I."""
    label = ""
    if kind in "iI":
        for value, numeral in _ROMAN:
            n, k = divmod(k, value)
            label += numeral * n
    else:
        while k > 0:
            k, r = divmod(k - 1, 26)
            label = chr(ord("a") + r) + label
    return label.upper() if kind in "AI" else label


def _list(node: Node, depth: int = 0, br: str = HARD_BREAK) -> str:
    """An ordered list is numbered from its start (or an item's value).
    Markdown has no lettered lists: the items of one get a bullet and their
    label, '- (a) …', unless they already start with it."""
    kind = node.attrs.get("type", "") if node.tag == "ol" else None
    lines, k = [], _count(node.attrs.get("start"), 1) - 1
    for c in node.children:
        if isinstance(c, Node) and c.tag == "li":
            k = _count(c.attrs.get("value"), k + 1)
            text_parts, nested = [], []
            for cc in c.children:
                if isinstance(cc, Node) and cc.tag in ("ul", "ol"):
                    nested.append(_list(cc, depth + 1, br))
                else:
                    text_parts.append(inline(cc))
            text = _paragraph("".join(text_parts), br)
            if kind in _LETTERED and k > 0:
                label = _ol_label(k, kind)
                if not re.match(rf"\(?{re.escape(label)}[.)]", text, re.I):
                    text = f"({label}) {text}"
            marker = f"{k}." if kind is not None and kind not in _LETTERED else "-"
            lines.append("  " * depth + f"{marker} " + text)
            lines.extend(nested)
    return "\n".join(lines)


def _squash(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def _paragraph(s: str, br: str = HARD_BREAK) -> str:
    """Inline Markdown as one escaped paragraph. A run of <br>s becomes
    ``br`` (default: a hard line break), except where a line already ends:
    there, and at the ends, a backslash would render literally."""
    s = re.sub(rf"[ \t]*{_BR}[ \t{_BR}]*", _BR, _squash(s))
    s = re.sub(rf"{_BR}(?=\n|$)|(?:^|(?<=\n)){_BR}", "", s)
    return _escape_block_starts(s.replace(_BR, br))


def to_markdown(node: Node, br: str = HARD_BREAK) -> str:
    """Block-level conversion of a node's children to Markdown. ``br``: what
    a <br> becomes inside a paragraph (default: a hard line break)."""
    out: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            s = _paragraph("".join(buf), br)
            if s:
                out.append(s)
            buf.clear()

    for c in node.children:
        if isinstance(c, Node) and (c.tag in BLOCK_TAGS or is_display(c)):
            flush()
            t = c.tag
            if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
                out.append("#" * int(t[1]) + " " + one_line(c))
            elif t in ("ul", "ol"):
                out.append(_list(c, 0, br))
            elif t == "table":
                out.append(table_html(c))
            elif t == "pre":
                out.append(f"```\n{c.text().strip(chr(10))}\n```")
            elif t == "hr":
                out.append("---")
            elif is_display(c):
                out.append(f"$$\n{c.text().strip()}\n$$")
            else:   # p, div, caption
                out.append(to_markdown(c, br) if any(
                    isinstance(x, Node) and (x.tag in BLOCK_TAGS or is_display(x))
                    for x in c.children) else _paragraph(inline(c), br))
        else:
            buf.append(inline(c))
    flush()
    return "\n\n".join(s for s in out if s)
