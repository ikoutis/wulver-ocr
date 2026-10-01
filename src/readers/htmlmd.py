"""Small, dependency-free HTML -> Markdown conversion for OCR-model HTML.

Some readers (Chandra) emit a restricted HTML vocabulary with math in
<math>…</math> (display="block" for display math). Math is what this project
cares about most, so the conversion is done here, where the delimiters are
under our control, rather than by a general converter:

  <math>x</math>                 -> $x$
  <math display="block">x</math> -> $$\\nx\\n$$  (or, via ``math_blocks``, a
                                    separate formula block)
  <table>…</table>               -> HTML kept (merged cells survive), with any
                                    math inside cells turned into $…$
  everything else                -> Markdown (paragraphs, headings, lists,
                                    emphasis, code, links); literal $ escaped

Parsing uses html.parser, which tolerates the unclosed tags models produce.
"""

from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser
from typing import Optional, Union

VOID = {"br", "img", "hr", "input", "meta", "link"}
KEEP_TABLE_ATTRS = {"colspan", "rowspan"}


class Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs: Optional[dict] = None):
        self.tag = tag
        self.attrs = attrs or {}
        self.children: list[Union["Node", str]] = []

    def text(self) -> str:
        """Raw text content (entities already decoded)."""
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)

    def find_all(self, tag: str) -> list["Node"]:
        out = []
        for c in self.children:
            if isinstance(c, Node):
                if c.tag == tag:
                    out.append(c)
                out.extend(c.find_all(tag))
        return out


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: (v if v is not None else "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, {k: v or "" for k, v in attrs}))

    def handle_endtag(self, tag):
        # close up to the matching open tag; ignore stray closers
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse(html_text: str) -> Node:
    b = _TreeBuilder()
    b.feed(html_text)
    b.close()
    return b.root


def is_display(node: Node) -> bool:
    return node.tag == "math" and node.attrs.get("display", "").lower() == "block"


# ------------------------------------------------------------------ inline


def _escape_text(s: str) -> str:
    return re.sub(r"(?<!\\)\$", r"\\$", s)


def inline(node: Union[Node, str], pre: bool = False) -> str:
    if isinstance(node, str):
        return node if pre else _escape_text(re.sub(r"\s+", " ", node))
    t = node.tag
    if t == "math":
        tex = node.text().strip()
        return f"\n$$\n{tex}\n$$\n" if is_display(node) else f"${tex}$"
    if t == "br":
        return "\n"
    if t == "img":
        return node.attrs.get("alt", "")
    if t == "input":
        return "[x] " if "checked" in node.attrs else "[ ] "
    if t in ("pre", "code"):
        return f"`{node.text()}`"
    inner = "".join(inline(c, pre) for c in node.children)
    if t in ("b", "strong"):
        return f"**{inner.strip()}**" if inner.strip() else inner
    if t in ("i", "em"):
        return f"*{inner.strip()}*" if inner.strip() else inner
    if t == "del":
        return f"~~{inner}~~"
    if t in ("sup", "sub"):
        return f"<{t}>{inner}</{t}>"
    if t == "a" and node.attrs.get("href"):
        return f"[{inner}]({node.attrs['href']})"
    return inner


# ------------------------------------------------------------------- block


def table_html(node: Node) -> str:
    """Re-serialise a table: structural tags + colspan/rowspan only, math as $..$."""
    def ser(n: Union[Node, str]) -> str:
        if isinstance(n, str):
            return _html.escape(re.sub(r"\s+", " ", n), quote=False)
        if n.tag == "math":
            return _html.escape(f"${n.text().strip()}$", quote=False)
        inner = "".join(ser(c) for c in n.children)
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


def _list(node: Node, depth: int = 0) -> str:
    lines, k = [], 0
    for c in node.children:
        if isinstance(c, Node) and c.tag == "li":
            k += 1
            marker = f"{k}." if node.tag == "ol" else "-"
            text_parts, nested = [], []
            for cc in c.children:
                if isinstance(cc, Node) and cc.tag in ("ul", "ol"):
                    nested.append(_list(cc, depth + 1))
                else:
                    text_parts.append(inline(cc))
            lines.append("  " * depth + f"{marker} " + _squash("".join(text_parts)))
            lines.extend(nested)
    return "\n".join(lines)


def _squash(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table",
              "pre", "hr", "caption"}


def to_markdown(node: Node) -> str:
    """Block-level conversion of a node's children to Markdown."""
    out: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            s = _squash("".join(buf))
            if s:
                out.append(s)
            buf.clear()

    for c in node.children:
        if isinstance(c, Node) and (c.tag in BLOCK_TAGS or is_display(c)):
            flush()
            t = c.tag
            if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
                out.append("#" * int(t[1]) + " " + _squash(inline(c)))
            elif t in ("ul", "ol"):
                out.append(_list(c))
            elif t == "table":
                out.append(table_html(c))
            elif t == "pre":
                out.append(f"```\n{c.text().strip(chr(10))}\n```")
            elif t == "hr":
                out.append("---")
            elif is_display(c):
                out.append(f"$$\n{c.text().strip()}\n$$")
            else:   # p, div, caption
                out.append(to_markdown(c) if any(
                    isinstance(x, Node) and (x.tag in BLOCK_TAGS or is_display(x))
                    for x in c.children) else _squash(inline(c)))
        else:
            buf.append(inline(c))
    flush()
    return "\n\n".join(s for s in out if s)
