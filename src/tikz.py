"""TikZ for figures: graph drawings as a canonical tikzpicture, parsed back.

Graph drawings (vertices and edges) are transcribed as TikZ, so they can be
pasted into LaTeX (and, later, into the compilable .tex export), and so the
picture can be re-rendered close to the original layout. The reviewer is asked
for one canonical subset:

    \\begin{tikzpicture}
      \\node[circle, draw, inner sep=1.5pt] (a) at (0,2) {$a$};
      \\draw (a) -- (b);                           undirected edge
      \\draw[->] (b) -- (c);                       directed edge
      \\draw (a) -- node[midway, auto] {$3$} (c);  edge with a weight/label
      \\draw (b) to[bend left] (c);                curved edge
      \\draw (c) to[loop above] (c);               self-loop
    \\end{tikzpicture}

``parse_graph`` reads that subset (and the common variants: chained paths
``(a) -- (b) -- (c)``, ``edge`` operations, labels after the target,
``<-``/``<->`` arrows, comments) back into vertices and edges. That gives:
  * checks — braces balance, every edge's endpoints are declared vertices,
    the picture has vertices, no constructs we cannot verify (\\foreach);
  * data — the vertex/edge lists go into the page JSON (meta["graph"]), so
    graphs are queryable and can be scored against ground truth (design §7).
"""

from __future__ import annotations

import re
from typing import Optional

_ENV = re.compile(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", re.S)
_ARROW_FWD = re.compile(r"(?<![<])->|-\s*(latex|stealth|Stealth|Latex|>|\{)")


def strip_comments(code: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", code)


def braces_balanced(s: str) -> bool:
    depth = 0
    for ch in re.sub(r"\\[{}]", "", s):
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0


def normalise(code: str, env: str = "tikzpicture") -> str:
    """Strip code fences; wrap a bare body in its environment."""
    code = re.sub(r"^```[a-zA-Z-]*\n|\n?```\s*$", "", code.strip()).strip()
    if code and f"\\begin{{{env}}}" not in code:
        code = f"\\begin{{{env}}}\n{code}\n\\end{{{env}}}"
    return code


# ------------------------------------------------------------------ scanning


class _Scan:
    """A tiny cursor over one TikZ statement."""

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def peek(self, lit: str) -> bool:
        self.ws()
        return self.s.startswith(lit, self.i)

    def take(self, lit: str) -> bool:
        if self.peek(lit):
            self.i += len(lit)
            return True
        return False

    def group(self, open_: str, close: str) -> Optional[str]:
        """Balanced group starting at the cursor, e.g. [..] or {..}."""
        self.ws()
        if not self.s.startswith(open_, self.i):
            return None
        depth, j = 0, self.i
        while j < len(self.s):
            c = self.s[j]
            if c == "\\" and j + 1 < len(self.s):
                j += 2
                continue
            if c == open_:
                depth += 1
            elif c == close:
                depth -= 1
                if depth == 0:
                    out = self.s[self.i + 1:j]
                    self.i = j + 1
                    return out
            j += 1
        raise ValueError(f"unclosed {open_}")

    def name(self) -> Optional[str]:
        """A node reference '(id)' (anchors like (a.north) map to 'a')."""
        self.ws()
        m = re.match(r"\(\s*([A-Za-z0-9_\-]+)(?:\.[A-Za-z0-9 ]+)?\s*\)", self.s[self.i:])
        if not m:
            return None
        self.i += m.end()
        return m.group(1)

    def done(self) -> bool:
        self.ws()
        return self.i >= len(self.s)


def _statements(body: str) -> list[str]:
    """Split on ';' outside braces/brackets."""
    out, depth, cur = [], 0, []
    for ch in body:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == ";" and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _direction(opts: str) -> str:
    o = opts.replace(" ", "")
    if "<->" in o:
        return "both"
    if "<-" in o and not _ARROW_FWD.search(o.replace("<-", "")):
        return "back"
    if _ARROW_FWD.search(o):
        return "forward"
    return "none"


def _float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[a-z]+$", "", s.strip()))
    except ValueError:
        return None


def _parse_node(sc: _Scan) -> dict:
    sc.group("[", "]")
    nid = sc.name()
    if nid is None:
        raise ValueError("node without a (name)")
    x = y = None
    if sc.take("at"):
        sc.ws()
        coord = sc.group("(", ")") or ""
        parts = coord.split(",")
        if len(parts) == 2:
            x, y = _float(parts[0]), _float(parts[1])
    label = sc.group("{", "}")
    return {"id": nid, "label": (label or "").strip(), "x": x, "y": y}


def _parse_path(sc: _Scan, kind: str) -> list[dict]:
    """Edges of one \\draw / \\path statement."""
    path_opts = sc.group("[", "]") or ""
    edges: list[dict] = []
    current = sc.name()
    if current is None:
        raise ValueError("path does not start at a (node)")
    pending = None                    # (connector kind, opts, label)
    while not sc.done():
        if sc.take("--"):
            pending = ("--", "", None)
        elif sc.take("edge"):
            pending = ("edge", sc.group("[", "]") or "", None)
        elif sc.take("to"):
            pending = ("to", sc.group("[", "]") or "", None)
        elif sc.take("node"):
            sc.group("[", "]")
            label = (sc.group("{", "}") or "").strip()
            if pending is not None:
                pending = (pending[0], pending[1], label)
            elif edges:
                edges[-1]["label"] = label
        elif sc.peek("("):
            target = sc.name()
            if target is None:
                raise ValueError("bad coordinate in path (use named nodes)")
            if pending is None:
                current = target                 # a move, not an edge
                continue
            conn, copts, label = pending
            drawn = kind == "draw" or conn == "edge"
            if drawn:
                direction = _direction(copts) if _direction(copts) != "none" \
                    else _direction(path_opts)
                u, v = (target, current) if direction == "back" else (current, target)
                edges.append({"u": u, "v": v,
                              "directed": direction in ("forward", "back", "both"),
                              "both": direction == "both", "label": label,
                              "curved": "bend" in copts or "loop" in copts})
            if conn != "edge":                   # edge ops keep the current point
                current = target
            pending = None
        else:
            raise ValueError(f"cannot parse path near {sc.s[sc.i:sc.i + 25]!r}")
    return edges


def parse_graph(code: str) -> tuple[Optional[dict], list[str], list[str]]:
    """-> (graph or None, flags, human-readable problems)."""
    code = strip_comments(normalise(code))
    flags, problems = [], []
    if not braces_balanced(code):
        return None, ["tikz_parse"], ["unbalanced braces"]
    m = _ENV.search(code)
    if not m:
        return None, ["tikz_parse"], ["no \\begin{tikzpicture} ... \\end{tikzpicture}"]
    nodes, edges = [], []
    for st in _statements(m.group(1)):
        sc = _Scan(st)
        try:
            if sc.take("\\node"):
                nodes.append(_parse_node(sc))
            elif sc.take("\\draw"):
                edges.extend(_parse_path(sc, "draw"))
            elif sc.take("\\path"):
                edges.extend(_parse_path(sc, "path"))
            elif sc.peek("\\foreach"):
                flags.append("tikz_unsupported")
                problems.append("\\foreach is not allowed: write every vertex and edge out")
        except ValueError as e:
            flags.append("tikz_parse")
            problems.append(f"{e} in: {st[:60]}")
    ids = [n["id"] for n in nodes]
    if not nodes:
        flags.append("tikz_empty")
        problems.append("no \\node vertices declared")
    if len(set(ids)) != len(ids):
        flags.append("tikz_parse")
        problems.append("duplicate vertex names")
    missing = sorted({e[k] for e in edges for k in ("u", "v")} - set(ids))
    if missing:
        flags.append("tikz_undeclared")
        problems.append(f"edges use undeclared vertices: {', '.join(missing)}")
    return {"nodes": nodes, "edges": edges}, sorted(set(flags)), problems


def check_tikzcd(code: str) -> tuple[list[str], list[str]]:
    code = strip_comments(normalise(code, "tikzcd"))
    if not braces_balanced(code):
        return ["tikz_parse"], ["unbalanced braces"]
    if not re.search(r"\\begin\{tikzcd\}.*\\end\{tikzcd\}", code, re.S):
        return ["tikz_parse"], ["no \\begin{tikzcd} ... \\end{tikzcd}"]
    return [], []


def edge_set(graph: dict, by: str = "label") -> set:
    """Comparable edge set: endpoints by vertex label (or id), undirected
    edges as frozensets. Used to score against ground truth (design §7)."""
    name = {n["id"]: (n["label"] or n["id"]) if by == "label" else n["id"]
            for n in graph["nodes"]}
    out = set()
    for e in graph["edges"]:
        u, v = name.get(e["u"], e["u"]), name.get(e["v"], e["v"])
        out.add((u, v) if e["directed"] and not e["both"] else frozenset((u, v)))
    return out


def graph_markdown(graph: dict) -> str:
    """The simple-Markdown rendering of a parsed graph: vertex list + one
    bullet per edge (— undirected, → directed, ↔ both ways; edge labels in
    parentheses). Derived from the TikZ, so the two versions always agree."""
    def name(n):
        return n["label"] or n["id"]
    names = {n["id"]: name(n) for n in graph["nodes"]}
    edges = graph["edges"]
    n_dir = sum(e["directed"] for e in edges)
    kind = ("directed" if n_dir == len(edges) else
            "undirected" if n_dir == 0 else "mixed") if edges else "no edges"
    lines = [f"{len(graph['nodes'])} vertices, {len(edges)} edges ({kind})", "",
             "- Vertices: " + ", ".join(names.values())]
    if edges:
        lines.append("- Edges:")
        for e in edges:
            arrow = "↔" if e["both"] else ("→" if e["directed"] else "—")
            label = f" ({e['label']})" if e.get("label") else ""
            lines.append(f"  - {names.get(e['u'], e['u'])} {arrow} "
                         f"{names.get(e['v'], e['v'])}{label}")
    return "\n".join(lines)
