from PIL import Image

from conftest import FakeServer
from src.figures import check_structure, describe_figures, format_description, parse_description
from src.schema import Block, Page
from src.tikz import edge_set, graph_markdown, normalise, parse_graph

TRIANGLE = r"""\begin{tikzpicture}
  % a weighted triangle and a pendant vertex
  \node[circle, draw, inner sep=1.5pt] (a) at (0,2) {$a$};
  \node[circle, draw, inner sep=1.5pt] (b) at (2cm,2) {$b_{1}$};
  \node[circle, draw, inner sep=1.5pt] (c) at (1,0) {$c$};
  \node[circle, draw, inner sep=1.5pt] (v1) at (3,0) {};
  \draw (a) -- (b) -- (c) -- node[midway, auto] {$3$} (a);
  \draw[->] (b) to[bend left] (v1);
  \draw[<-] (c) -- (v1);
  \draw (v1) to[loop below] (v1);
\end{tikzpicture}"""


class TestParseGraph:
    def test_nodes_positions_labels(self):
        g, flags, _ = parse_graph(TRIANGLE)
        assert flags == []
        assert [n["id"] for n in g["nodes"]] == ["a", "b", "c", "v1"]
        assert g["nodes"][1] == {"id": "b", "label": "$b_{1}$", "x": 2.0, "y": 2.0}
        assert g["nodes"][3]["label"] == ""

    def test_edges_chain_label_direction(self):
        g, _, _ = parse_graph(TRIANGLE)
        es = [(e["u"], e["v"], e["directed"], e["label"]) for e in g["edges"]]
        assert es == [("a", "b", False, None), ("b", "c", False, None),
                      ("c", "a", False, "$3$"), ("b", "v1", True, None),
                      ("v1", "c", True, None),             # <- reverses the edge
                      ("v1", "v1", False, None)]           # loop
        assert g["edges"][3]["curved"]

    def test_edge_operation_keeps_current_point(self):
        g, flags, _ = parse_graph(r"""\node (a) at (0,0) {a}; \node (b) at (1,0) {b};
            \node (c) at (2,0) {c}; \path[->] (a) edge (b) edge node {x} (c);""")
        assert flags == []
        assert [(e["u"], e["v"], e["directed"], e["label"]) for e in g["edges"]] == \
            [("a", "b", True, None), ("a", "c", True, "x")]

    def test_label_after_target_and_both_ways(self):
        g, _, _ = parse_graph(r"""\node (a) at (0,0) {}; \node (b) at (1,0) {};
            \draw[<->] (a) -- (b) node[midway] {$w$};""")
        e = g["edges"][0]
        assert e["both"] and e["directed"] and e["label"] == "$w$"

    def test_undeclared_vertex_flagged(self):
        _, flags, problems = parse_graph(r"\node (a) at (0,0) {}; \draw (a) -- (z);")
        assert flags == ["tikz_undeclared"] and "z" in problems[0]

    def test_foreach_and_coordinates_flagged(self):
        _, flags, _ = parse_graph(r"\node (a) at (0,0) {}; \foreach \i in {1,2} {\draw (a) -- (\i);}")
        assert "tikz_unsupported" in flags
        _, flags, _ = parse_graph(r"\node (a) at (0,0) {}; \draw (0,0) -- (1,1);")
        assert "tikz_parse" in flags

    def test_unbalanced_and_empty(self):
        assert parse_graph(r"\node (a) at (0,0) {$a$;")[1] == ["tikz_parse"]
        assert "tikz_empty" in parse_graph(r"\draw (a) -- (b);")[1]

    def test_normalise_wraps_and_unfences(self):
        code = normalise("```latex\n\\node (a) at (0,0) {};\n```")
        assert code.startswith("\\begin{tikzpicture}") and "```" not in code

    def test_edge_set_by_label(self):
        g, _, _ = parse_graph(TRIANGLE)
        s = edge_set(g)
        assert frozenset(("$a$", "$b_{1}$")) in s and ("$b_{1}$", "v1") in s


class TestMarkdownVersion:
    def test_graph_markdown(self):
        g, _, _ = parse_graph(TRIANGLE)
        md = graph_markdown(g)
        assert md.startswith("4 vertices, 6 edges (mixed)")
        assert "- Vertices: $a$, $b_{1}$, $c$, v1" in md
        assert "  - $c$ — $a$ ($3$)" in md and "  - $b_{1}$ → v1" in md

    def test_both_versions_marked(self):
        reply = f"KIND: graph\n<description>A triangle.</description>\n<structure>\n{TRIANGLE}\n</structure>"
        info, flags, _ = check_structure(parse_description(reply))
        out = format_description(info)
        assert flags == []
        assert "**Graph — Markdown (simple):** 4 vertices" in out
        assert "**Graph — TikZ:**\n\n```latex\n\\begin{tikzpicture}" in out
        assert out.index("Markdown (simple)") < out.index("Graph — TikZ")

    def test_unparseable_tikz_still_marked(self):
        reply = "KIND: graph\n<description>x</description>\n<structure>\\draw (a) -- (b);</structure>"
        info, flags, _ = check_structure(parse_description(reply))
        out = format_description(info)
        assert "tikz_empty" in flags and "not available" in out and "Graph — TikZ" in out


class TestRepairLoop:
    def _page(self, tmp_path):
        (tmp_path / "pages").mkdir()
        (tmp_path / "figures").mkdir()
        Image.new("RGB", (400, 400), "white").save(tmp_path / "pages" / "p0001.png")
        Image.new("RGB", (100, 100), "white").save(tmp_path / "figures" / "f.png")
        fig = Block(type="figure", bbox=[0.1, 0.1, 0.5, 0.5], meta={"image": "figures/f.png"})
        return Page(doc_id="d", index=0, image=str(tmp_path / "pages" / "p0001.png"),
                    width=400, height=400, blocks=[fig, Block(type="caption", content="Fig 1")])

    def test_bad_tikz_is_sent_back_once(self, tmp_path):
        bad = "KIND: graph\n<description>d</description>\n<structure>\\node (a) at (0,0) {}; \\draw (a) -- (q);</structure>"
        good = f"KIND: graph\n<description>d</description>\n<structure>{TRIANGLE}</structure>"
        srv = FakeServer(lambda p, n: good if "could not be checked" in p else bad)
        page = self._page(tmp_path)
        describe_figures(srv.client(), [page], "reviewer:fake")
        fig = page.blocks[0]
        assert len(srv.requests) == 2 and fig.meta["describe_attempts"] == 2
        assert "undeclared vertices: q" in srv.requests[1]["messages"][0]["content"][1]["text"]
        assert fig.flags == [] and len(fig.meta["graph"]["nodes"]) == 4
        assert '"Fig 1"' in srv.requests[0]["messages"][0]["content"][1]["text"]

    def test_still_bad_after_repair_is_flagged(self, tmp_path):
        bad = "KIND: graph\n<description>d</description>\n<structure>\\node (a) at (0,0) {}; \\draw (a) -- (q);</structure>"
        srv = FakeServer(lambda p, n: bad)
        page = self._page(tmp_path)
        describe_figures(srv.client(), [page], "reviewer:fake")
        assert page.blocks[0].flags == ["tikz_undeclared"] and len(srv.requests) == 2

    def test_non_graph_not_repaired(self, tmp_path):
        srv = FakeServer(lambda p, n: "KIND: plot\n<description>a plot</description>\n<structure></structure>")
        page = self._page(tmp_path)
        describe_figures(srv.client(), [page], "reviewer:fake")
        assert len(srv.requests) == 1 and page.blocks[0].meta["kind"] == "plot"
