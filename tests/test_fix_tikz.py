"""Regression tests for graph TikZ in the forms models actually write, and for
what a figure description publishes when its checks fail or its answer was
cut off (findings gate-1..5, gate-15, gate-16, docs-6)."""

import pytest
from PIL import Image

from conftest import FakeServer
from src.assemble import render_block
from src.figures import check_structure, describe_figures, format_description, parse_description
from src.schema import Block, Page
from src.tikz import parse_graph

AB = r"\node (a) at (0,0) {$a$}; \node (b) at (1,0) {$b$}; "
FWD, BACK, BOTH, NONE = ("a", "b", True, False), ("b", "a", True, False), \
    ("a", "b", True, True), ("a", "b", False, False)


def parsed(code):
    g, flags, _ = parse_graph(code)
    return ([n["id"] for n in g["nodes"]],
            [(e["u"], e["v"], e["directed"], e["both"]) for e in g["edges"]], flags)


def graph_reply(tikz, kind="KIND: graph"):
    return f"{kind}\n<description>d</description>\n<structure>\n{tikz}\n</structure>"


FLOW = r"""\begin{tikzpicture}[->, >=stealth, scale=1.2]
  \node[circle, draw] (s) at (0,1) {$s$};
  \node[circle, draw] (u) at (1,2) {$u$};
  \node[circle, draw] (v) at (1,0) {$v$};
  \node[circle, draw] (t) at (2,1) {$t$};
  \draw (s) -- (u); \draw (s) -- (v); \draw (u) -- (v);
  \draw (u) -- (t); \draw (v) -- (t);
\end{tikzpicture}"""

# The TikZ manual's automaton: picture-level arrows, a \tikzstyle with no ';'
# before the first node, node options after the name, edge operations, and
# the self-loop idiom edge[loop above] ().
AUTOMATON = r"""\begin{tikzpicture}[->,>=stealth',shorten >=1pt,auto,node distance=2.8cm]
  \tikzstyle{every state}=[fill=red,draw=none,text=white]
  \node[initial,state] (A)                    {$q_a$};
  \node[state]         (B) [above right of=A] {$q_b$};
  \node[state]         (D) [below right of=A] {$q_d$};
  \node[state]         (C) [below right of=B] {$q_c$};
  \node[state]         (E) [below of=D]       {$q_e$};
  \path (A) edge              node {0,1,L} (B)
            edge              node {1,1,R} (C)
        (B) edge [loop above] node {1,1,L} ()
            edge              node {1,1,R} (C)
        (C) edge              node {0,1,L} (D)
            edge [bend left]  node {1,0,R} (E)
        (D) edge [loop below] node {1,1,R} ()
            edge              node {0,1,R} (A)
        (E) edge [bend left]  node {1,0,R} (A);
\end{tikzpicture}"""


class TestPictureAndScopeOptions:          # gate-1
    def test_picture_options_keep_first_node_and_set_arrows(self):
        ids, es, flags = parsed(FLOW)
        assert flags == [] and ids == ["s", "u", "v", "t"]
        assert len(es) == 5 and all(d and not both for _, _, d, both in es)
        ids, _, flags = parsed(r"\begin{tikzpicture}[scale=0.8]" + AB + r"\draw (a) -- (b);"
                               r"\end{tikzpicture}")
        assert flags == [] and ids == ["a", "b"]

    def test_explicit_dash_overrides_picture_arrows(self):
        _, es, _ = parsed(r"\begin{tikzpicture}[->]" + AB + r"\draw[-] (a) -- (b);"
                          r"\end{tikzpicture}")
        assert es == [NONE]

    def test_tikzset_without_semicolon(self):
        ids, es, flags = parsed(r"\tikzset{vertex/.style={circle, draw}}"
                                r"\node[vertex] (a) at (0,0) {$a$}; \node (b) at (1,0) {$b$};"
                                r"\draw (a) -- (b);")
        assert flags == [] and ids == ["a", "b"] and es == [NONE]

    def test_scope_options_apply_inside_only(self):
        ids, es, flags = parsed(r"\node (a) at (0,0) {a}; \begin{scope}[->]"
                                r"\node (b) at (1,0) {b}; \draw (a) -- (b); \end{scope}"
                                r"\node (c) at (2,0) {c}; \draw (a) -- (c);")
        assert flags == [] and ids == ["a", "b", "c"]
        assert es == [FWD, ("a", "c", False, False)]

    def test_unrecognised_drawing_statement_is_flagged(self):
        _, _, flags = parsed(AB + r"\draw (a) -- (b); \fill (a) circle (2pt);")
        assert flags == ["tikz_parse"]
        _, _, flags = parsed(AB + r"\graph {a -> b};")
        assert flags == ["tikz_parse"]
        _, es, flags = parsed(AB + r"\small \draw (a) -- (b);")      # draws nothing itself
        assert flags == [] and es == [NONE]

    def test_textbook_automaton(self):
        g, flags, _ = parse_graph(AUTOMATON)
        assert flags == [] and len(g["nodes"]) == 5
        es = [(e["u"], e["v"], e["directed"], e["label"]) for e in g["edges"]]
        assert len(es) == 9 and all(d for _, _, d, _ in es)
        assert ("B", "B", True, "1,1,L") in es and ("D", "D", True, "1,1,R") in es
        assert ("E", "A", True, "1,0,R") in es


@pytest.mark.parametrize("spec, edge", [                                # gate-3
    ("->", FWD), ("<-", BACK), ("<->", BOTH), ("-", NONE),
    ("-Triangle", FWD), ("-to", FWD), ("-angle 90", FWD), ("-triangle 45", FWD),
    ("-Straight Barb", FWD), ("-Kite", FWD), ("-{Stealth[length=3mm]}", FWD),
    ("latex-latex", BOTH), ("Stealth-Stealth", BOTH), ("{Stealth}-{Stealth}", BOTH),
    ("latex-", BACK), ("Stealth-", BACK), ("stealth-", BACK),
    ("|->", FWD), ("arrows={-Latex}", FWD),
    ("thick, out=-30, in=210", NONE), (">=stealth, shorten >=1pt", NONE),
])
def test_arrow_specs(spec, edge):
    _, es, flags = parsed(AB + rf"\draw[{spec}] (a) -- (b);")
    assert flags == [] and es == [edge]


class TestNodeForms:                       # gate-4
    def test_options_and_position_in_any_order(self):
        g, flags, _ = parse_graph(r"\node (a) at (0,0) [circle, draw] {$u$};"
                                  r"\node (b) [circle, draw, right=of a] {$w$};"
                                  r"\node at (3,0) (c) {$c$}; \draw (a) -- (b) -- (c);")
        assert flags == []
        assert g["nodes"] == [{"id": "a", "label": "$u$", "x": 0.0, "y": 0.0},
                              {"id": "b", "label": "$w$", "x": None, "y": None},
                              {"id": "c", "label": "$c$", "x": 3.0, "y": 0.0}]

    def test_lost_label_is_never_silent(self):
        assert parse_graph(r"\node (a) at (0,0); \node (b) at (1,0) {b};")[1] == ["tikz_parse"]
        _, flags, problems = parse_graph(AB + r"\node (c) at (2,0) {c} edge (a);")
        assert flags == ["tikz_parse"] and "after the node label" in problems[0]


class TestEdgeIdioms:                      # gate-5
    def test_empty_target_is_a_self_loop(self):
        _, es, flags = parsed(AB + r"\path[->] (a) edge node {0} (b)"
                                   r" edge [loop above] node {1} ();")
        assert flags == [] and es == [FWD, ("a", "a", True, False)]

    def test_path_with_draw_option_draws(self):
        assert parsed(AB + r"\path[draw] (a) -- (b);")[1] == [NONE]
        assert parsed(AB + r"\path[draw=blue, ->] (a) to (b);")[1] == [FWD]
        assert parsed(AB + r"\draw[draw=none] (a) -- (b);")[1] == []

    def test_arrows_on_an_undrawn_path_are_flagged(self):
        _, es, flags = parsed(AB + r"\path[->] (a) -- (b);")
        assert es == [] and flags == ["tikz_parse"]

    def test_brackets_inside_a_label(self):
        _, es, flags = parsed(AB + r"\node (c) at (2,0) {c};"
                                   r"\draw (a) -- node {$[0,1)$} (b); \draw (b) -- (c);")
        assert flags == [] and es == [NONE, ("b", "c", False, False)]


@pytest.mark.parametrize("label, kind", [                               # gate-16
    ("**KIND:** graph", "graph"), ("KIND: **graph**", "graph"), ("KIND: `graph`", "graph"),
    ("KIND: <graph>", "graph"), ("Kind: Graph.", "graph"), ("KIND: graph drawing", "graph"),
    ("KIND: commutative diagram", "commutative_diagram"), ("KIND: spaceship", "other"),
])
def test_decorated_kind_label(label, kind):
    assert parse_description(f"{label}\n<description>x</description>")["kind"] == kind


def test_bold_kind_still_gives_both_graph_versions():                   # gate-16
    info, flags, problems = check_structure(parse_description(graph_reply(FLOW, "**KIND:** graph")))
    out = format_description(info, problems)
    assert flags == [] and "**Graph — Markdown (simple):** 4 vertices, 5 edges (directed)" in out
    assert "**Graph — TikZ:**" in out


@pytest.fixture
def figure_page(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "figures").mkdir()
    Image.new("RGB", (400, 400), "white").save(tmp_path / "pages" / "p0001.png")
    Image.new("RGB", (100, 100), "white").save(tmp_path / "figures" / "f.png")
    fig = Block(type="figure", bbox=[0.1, 0.1, 0.5, 0.5], meta={"image": "figures/f.png"})
    return Page(doc_id="d", index=0, image=str(tmp_path / "pages" / "p0001.png"),
                width=400, height=400, blocks=[fig])


def describe(page, responder):
    srv = FakeServer(responder)
    describe_figures(srv.client(), [page], "reviewer:fake")
    return page.blocks[0], len(srv.requests)


class TestFailedChecksPublishNoPartialGraph:     # gate-2, docs-6
    BAD = graph_reply(r"""\begin{tikzpicture}
  \node (a) at (0,0) {$a$};
  \node (b) at (1,0) {$b$};
  \draw (a) -- (b);
  \draw (b) -- (c);
  \draw (a) -- (3,1);
\end{tikzpicture}""")

    def test_still_failing_tikz_has_no_markdown_version(self, figure_page):
        fig, requests = describe(figure_page, lambda p, n: self.BAD)
        assert requests == 2 and fig.flags == ["tikz_parse", "tikz_undeclared"]
        desc = fig.meta["description"]
        assert "**Graph — Markdown (simple):** *(not available: the TikZ below failed " \
               "the checks: " in desc
        assert "undeclared vertices: c" in desc and "vertices, " not in desc
        assert "**Graph — TikZ:**\n\n```latex\n\\begin{tikzpicture}" in desc
        # the partial parse is kept, but never as the graph to score
        assert "graph" not in fig.meta and len(fig.meta["graph_partial"]["nodes"]) == 2
        assert "graph as TikZ (Markdown version unavailable)" in render_block(fig)

    def test_passing_tikz_gives_the_same_graph_twice(self, figure_page):
        fig, requests = describe(figure_page, lambda p, n: graph_reply(FLOW))
        assert requests == 1 and fig.flags == []
        assert "**Graph — Markdown (simple):** 4 vertices, 5 edges (directed)" \
            in fig.meta["description"]
        assert "  - $s$ → $u$" in fig.meta["description"]
        assert len(fig.meta["graph"]["nodes"]) == 4 and "graph_partial" not in fig.meta

    def test_format_description_needs_a_clean_check(self):
        info, _, problems = check_structure(parse_description(graph_reply(
            r"\node (a) at (0,0) {a}; \draw (a) -- (z);")))
        assert problems and info["graph"]["nodes"]
        assert "not available" in format_description(info, problems)
        assert "not available" not in format_description(dict(info, graph={
            "nodes": info["graph"]["nodes"], "edges": []}))


class TestCutOffAnswers:                    # gate-15
    MERMAID = ("KIND: diagram\n<description>An encoder.</description>\n<structure>\n"
               "```mermaid\nflowchart LR\n  A[Input] --> B[Encoder]\n  B --> C[Lat", "length")
    CUT_GRAPH = ("KIND: graph\n<description>A path.</description>\n<structure>\n"
                 "\\begin{tikzpicture}\n  \\node (a) at (0,0) {$a$};\n  \\draw (a) -- ", "length")

    def test_cut_off_mermaid_is_flagged_and_left_out(self, figure_page):
        fig, requests = describe(figure_page, lambda p, n: self.MERMAID)
        out = render_block(fig)
        assert requests == 1 and fig.flags == ["description_truncated"]
        assert "TRUNCATED" not in out and "```mermaid" not in out and "C[Lat" not in out
        assert "An encoder." in out and "cut off at the length limit" in out

    def test_cut_off_graph_is_sent_back(self, figure_page):
        good = graph_reply(FLOW)
        fig, requests = describe(figure_page,
                                 lambda p, n: good if "could not be checked" in p else self.CUT_GRAPH)
        assert requests == 2 and fig.flags == [] and len(fig.meta["graph"]["nodes"]) == 4

    def test_still_cut_off_graph(self, figure_page):
        prompts = []
        fig, requests = describe(figure_page,
                                 lambda p, n: prompts.append(p) or self.CUT_GRAPH)
        assert requests == 2 and "cut off at the length limit" in prompts[1]
        assert "description_truncated" in fig.flags and "graph" not in fig.meta
        assert "TRUNCATED" not in fig.meta["description"]
        assert "Graph — TikZ" not in fig.meta["description"]
