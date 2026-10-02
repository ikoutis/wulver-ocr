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

``parse_graph`` reads that subset back into vertices and edges, along with
the common variants models write instead: chained paths ``(a) -- (b) -- (c)``,
``edge`` operations (and the automata self-loop ``edge[loop above] ()``),
labels after the target, node options in any order, a vertex label given as
``label=above:$v_1$``, any arrow spec (``<-``, ``latex-latex``,
``-{Stealth[length=2mm]}``), arrows set for the whole picture or a scope or
through a style (``name/.style={..}`` in the picture's or a scope's options,
``\\tikzset``, ``\\tikzstyle``, ``every edge``), and comments.
That gives:
  * checks — braces balance, one picture (a figure with several panels is
    drawn as one, with scopes), every edge's endpoints are declared
    vertices, the picture has vertices, every statement that draws
    something was understood, no constructs we cannot verify (\\foreach);
  * data — the vertex/edge lists go into the page JSON (meta["graph"]), so
    graphs are queryable and can be scored against ground truth (design §7).
"""

from __future__ import annotations

import re
from typing import Optional

_ENV = re.compile(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", re.S)
_DRAWING = re.compile(r"\\(?:node|draw|path|fill|filldraw|coordinate|graph|foreach|matrix)\b")
_STYLE_KEY = re.compile(r"\s*(.+?)\s*/\.(append )?style\s*")
# The position part of label=<pos>:<text>: above, below left, north east, 45, ...
_LABEL_POS = re.compile(r"\s*(?:[-+]?\d+(?:\.\d+)?|(?:above|below|left|right|north|south|east"
                        r"|west|center)(?:\s+(?:left|right|east|west))?)\s*")
# One side of an arrow spec that draws an arrowhead: < or >, or a tip that
# points (latex, Stealth, to, Triangle, angle 90, Kite, Straight Barb, ...),
# possibly braced with options ({Stealth[length=2mm]}). Other tips (bars,
# hooks, circles) and empty sides draw no head.
_HEAD = re.compile(
    r"[{|\s]*(?:(?:open|straight|arc|classical\s+tikz|computer\s+modern)\s+)?"
    r"(?:<|>|latex|stealth|to\b|triangle|angle|kite|barb|implies|imply|rightarrow)", re.I)


def strip_comments(code: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", code)


def braces_balanced(s: str) -> bool:
    depth = 0
    for ch in re.sub(r"\\[\\{}]", "", s):     # "\\{" is a line break + group
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
    """Split on ';' outside braces/brackets. Brackets count only outside
    braces, so a label such as {$[0,1)$} cannot swallow later statements."""
    out, braces, brackets, start, i = [], 0, 0, 0, 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":                  # an escaped character (\{, \;) is text
            i += 2
            continue
        if ch in "{}":
            braces += 1 if ch == "{" else -1
        elif braces == 0 and ch in "[]":
            brackets += 1 if ch == "[" else -1
        elif ch == ";" and braces == 0 and brackets == 0:
            out.append(body[start:i].strip())
            start = i + 1
        i += 1
    if body[start:].strip():
        out.append(body[start:].strip())
    return out


def _split_top(s: str, sep: str, maxsplit: int = -1) -> list[str]:
    """Split at ``sep`` outside {..} and [..]."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        depth += (ch in "{[") - (ch in "}]")
        if ch == sep and depth == 0 and len(out) != maxsplit:
            out.append(s[start:i])
            start = i + 1
    return out + [s[start:]]


def _unbrace(s: str) -> str:
    """'{-Latex}' -> '-Latex': one group around a whole option value."""
    sc = _Scan(s)
    inner = sc.group("{", "}")
    return inner.strip() if inner is not None and sc.done() else s.strip()


def _add_styles(opts: str, styles: dict) -> None:
    """Record the styles an option list defines: name/.style={..} and
    name/.append style={..}."""
    for item in _split_top(opts, ","):
        kv = _split_top(item, "=", 1)
        m = _STYLE_KEY.fullmatch(kv[0]) if len(kv) == 2 else None
        if m:
            before = styles.get(m.group(1), "") + "," if m.group(2) else ""
            styles[m.group(1)] = before + _unbrace(kv[1])


def _expand(opts: str, styles: dict, depth: int = 0) -> str:
    """The option list with each style it uses replaced by the style's
    options (#1 by the value of name=value), as TikZ applies them."""
    if depth > 8:
        raise ValueError("a style that uses itself")
    out = []
    for item in _split_top(opts, ","):
        key, *value = _split_top(item, "=", 1)
        if key.strip() in styles:
            arg = _unbrace(value[0]) if value else ""
            item = _expand(styles[key.strip()].replace("#1", arg), styles, depth + 1)
        out.append(item)
    return ",".join(out)


def _direction(opts: str) -> Optional[str]:
    """The arrows an option list sets: 'forward', 'back', 'both', 'none' (an
    explicit '-', or tips without a head), or None when it sets no arrows.
    The arrow spec is the option with a '-' outside groups and no '=' (so
    out=-30 and >=stealth are not specs), or the value of arrows=; the two
    sides of that '-' are the start and end tips. As in TikZ, the last
    spec wins."""
    found = None
    for item in _split_top(opts, ","):
        kv = _split_top(item, "=", 1)
        if len(kv) == 2:
            if kv[0].strip() != "arrows":
                continue
            item = _unbrace(kv[1])      # arrows={-Latex}
        sides = _split_top(item, "-", 1)
        if len(sides) == 2:
            start, end = (bool(_HEAD.match(x)) for x in sides)
            found = ("both" if start else "forward") if end else ("back" if start else "none")
    return found


def _arrows(*opts: str) -> Optional[str]:
    """The direction set by the first of these option lists that sets one."""
    return next((d for d in map(_direction, opts) if d), None)


def _draws(opts: str) -> Optional[bool]:
    """True for a draw (or draw=<colour>) option, False for draw=none, else
    None. As in TikZ, the last one wins."""
    found = None
    for item in _split_top(opts, ","):
        key, *value = [x.strip() for x in _split_top(item, "=", 1)]
        if key == "draw":
            found = value != ["none"]
    return found


def _float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[a-z]+$", "", s.strip()))
    except ValueError:
        return None


def _label_option(opts: list[str]) -> str:
    """The text of a label=<pos>:<text> node option, or ''."""
    for o in opts:
        for item in _split_top(o, ","):
            key, *value = _split_top(item, "=", 1)
            if key.strip() == "label" and value:
                v = re.sub(r"^\[[^\]]*\]", "", _unbrace(value[0]))     # {[red]above:$x$}
                pos, *text = _split_top(v, ":", 1)
                return _unbrace(text[0] if text and _LABEL_POS.fullmatch(pos) else v)
    return ""


def _parse_node(sc: _Scan) -> dict:
    """[opts] (name) at (x,y) {label}: options, name, and position in any
    order, as TikZ allows (\\node (b) [right=of a] {..}, \\node at (3,0)
    (c) {..}); the {label} must come last. An empty {label} takes the text
    of a label= option (\\node[label=above:$v_1$] (v1) at (0,0) {};)."""
    nid = x = y = None
    opts: list[str] = []
    while True:
        o = sc.group("[", "]")
        if o is not None:
            opts.append(o)
            continue
        if sc.take("at"):
            parts = (sc.group("(", ")") or "").split(",")
            if len(parts) == 2:
                x, y = _float(parts[0]), _float(parts[1])
        elif nid is None and sc.peek("("):
            nid = sc.name()
            if nid is None:
                break
        else:
            break
    if nid is None:
        raise ValueError("node without a (name)")
    label = sc.group("{", "}")
    if label is None:
        raise ValueError("node without a {label}")
    if not sc.done():
        raise ValueError(f"unexpected text after the node label: {sc.s[sc.i:sc.i + 25]!r}")
    return {"id": nid, "label": label.strip() or _label_option(opts), "x": x, "y": y}


def _parse_path(sc: _Scan, kind: str, inherited: list[str], styles: dict) -> list[dict]:
    """Edges of one \\draw / \\path statement. ``inherited``: the option
    lists of the enclosing scopes and the picture, innermost first (their
    arrows apply when the path sets none). ``styles``: the styles defined so
    far, expanded wherever an option list uses one."""
    path_opts = _expand(sc.group("[", "]") or "", styles)
    edges: list[dict] = []
    current = sc.name()
    if current is None:
        raise ValueError("path does not start at a (node)")
    pending = None                    # (connector kind, opts, label)
    while not sc.done():
        if sc.take("--"):
            pending = ("--", "", None)
        elif sc.take("edge"):           # TikZ applies "every edge" first
            pending = ("edge", _expand("every edge," + (sc.group("[", "]") or ""), styles),
                       None)
        elif sc.take("to"):
            pending = ("to", _expand(sc.group("[", "]") or "", styles), None)
        elif sc.take("node"):
            sc.group("[", "]")
            label = (sc.group("{", "}") or "").strip()
            if pending is not None:
                pending = (pending[0], pending[1], label)
            elif edges:
                edges[-1]["label"] = label
        elif sc.peek("("):
            loop = re.match(r"\(\s*\)", sc.s[sc.i:])
            if loop:                             # '()' is the current node:
                sc.i += loop.end()               # edge[loop above] ()
                target = current
            else:
                target = sc.name()
            if target is None:
                raise ValueError("bad coordinate in path (use named nodes)")
            if pending is None:
                current = target                 # a move, not an edge
                continue
            conn, copts, label = pending
            drawn = next((d for d in (_draws(copts), _draws(path_opts)) if d is not None),
                         kind == "draw" or conn == "edge")
            own = _arrows(copts, path_opts)
            if not drawn and own not in (None, "none"):
                # \path[->] (a) -- (b) draws nothing in TikZ: not an edge, but
                # surely meant as one.
                raise ValueError("arrows on a path that is not drawn (use \\draw)")
            if drawn:
                direction = own or _arrows(*(_expand(o, styles) for o in inherited)) or "none"
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


def _settings(sc: _Scan, scopes: list[str], styles: dict) -> None:
    """Consume what may come before a statement's command: \\tikzset{..} and
    \\tikzstyle{..}=[..] (often written without a ';'), whose style
    definitions go into ``styles``, and \\begin{scope}[..] / \\end{scope},
    whose options ``scopes`` tracks (outermost first)."""
    while True:
        if sc.take("\\tikzset"):
            _add_styles(sc.group("{", "}") or "", styles)
        elif sc.take("\\tikzstyle"):
            name = (sc.group("{", "}") or "").strip()
            append = sc.take("+")
            sc.take("=")
            opts = sc.group("[", "]") or ""
            if name:
                styles[name] = styles.get(name, "") + "," + opts if append else opts
        elif sc.take("\\begin{scope}"):
            scopes.append(sc.group("[", "]") or "")
            _add_styles(scopes[-1], styles)
        elif sc.take("\\end{scope}"):
            if len(scopes) > 1:          # scopes[0] is the picture's
                scopes.pop()
        else:
            return


def parse_graph(code: str) -> tuple[Optional[dict], list[str], list[str]]:
    """-> (graph or None, flags, human-readable problems)."""
    code = strip_comments(normalise(code))
    flags, problems = [], []
    if not braces_balanced(code):
        return None, ["tikz_parse"], ["unbalanced braces"]
    pictures = _ENV.findall(code)
    if not pictures:
        return None, ["tikz_parse"], ["no \\begin{tikzpicture} ... \\end{tikzpicture}"]
    if len(pictures) > 1:
        return None, ["tikz_parse"], ["more than one tikzpicture: draw all panels in one "
                                      "picture, each in its own \\begin{scope}[xshift=...]"]
    outside = _ENV.sub(" ", code)
    if _DRAWING.search(outside):
        return None, ["tikz_parse"], ["drawing commands outside the tikzpicture"]
    styles: dict = {}
    body = _Scan(pictures[0])
    try:
        for m in re.finditer(r"\\tikz(?:set|style)\b", outside):   # e.g. before the picture
            sc = _Scan(outside)
            sc.i = m.start()
            _settings(sc, [], styles)
        # \begin{tikzpicture}[->, >=stealth, arc/.style={..}]: for every path
        scopes = [body.group("[", "]") or ""]
        _add_styles(scopes[0], styles)
    except ValueError as e:
        return None, ["tikz_parse"], [f"{e} in the picture's options or styles"]
    nodes, edges = [], []
    for st in _statements(body.s[body.i:]):
        sc = _Scan(st)
        try:
            _settings(sc, scopes, styles)
            # Resume at the command: skip a font switch or \def before it,
            # but not \fill, \coordinate, \graph, ... — they draw something
            # we cannot read, and the graph would be silently incomplete.
            cmd = re.search(r"\\(?:node|draw|path|foreach)\b", st[sc.i:])
            if re.search(r"\(|--|->|<-", st[sc.i:sc.i + cmd.start()] if cmd else st[sc.i:]):
                raise ValueError("not a \\node or \\draw statement")
            if cmd is None:
                continue
            sc.i += cmd.start()
            if sc.take("\\node"):
                nodes.append(_parse_node(sc))
            elif sc.take("\\draw"):
                edges.extend(_parse_path(sc, "draw", scopes[::-1], styles))
            elif sc.take("\\path"):
                edges.extend(_parse_path(sc, "path", scopes[::-1], styles))
            else:
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
