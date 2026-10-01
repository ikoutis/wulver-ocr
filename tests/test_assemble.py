from src.assemble import assemble, render_block
from src.schema import Block, Page


def page(i, blocks):
    return Page(doc_id="d", index=i, image="", width=1, height=1, blocks=blocks)


def B(t, c="", **meta):
    return Block(type=t, content=c, meta=meta)


class TestRender:
    def test_types(self):
        assert render_block(B("title", "T")) == "# T"
        assert render_block(B("heading", "Intro", level=1)) == "## Intro"
        assert render_block(B("heading", "Sub", level=2)) == "### Sub"
        assert render_block(B("formula", "$$x$$")) == "$$\nx\n$$"
        assert render_block(B("caption", "Figure 1: a")) == "*Figure 1: a*"
        assert render_block(B("code", "for i")) == "```\nfor i\n```"

    def test_figure(self):
        b = B("figure", image="figures/p0001_b03.png", alt="Figure (page 1)",
              description="**Kind:** graph.")
        r = render_block(b)
        assert r.startswith("![Figure (page 1)](figures/p0001_b03.png)")
        assert "<details>" in r and "**Kind:** graph." in r


class TestAssemble:
    def test_drops_running_heads_and_marks_pages(self):
        md = assemble([page(0, [B("header", "J. Foo"), B("text", "Body."),
                                B("page_number", "1")])])
        assert "J. Foo" not in md and "<!-- page 1 -->" in md and "Body." in md

    def test_joins_paragraph_across_pages(self):
        md = assemble([page(0, [B("text", "The Laplacian of a"), B("footer", "x")]),
                       page(1, [B("header", "y"), B("text", "graph is PSD.")])],
                      page_markers=False)
        assert "The Laplacian of a graph is PSD." in md

    def test_dehyphenates(self):
        md = assemble([page(0, [B("text", "we use sparsi-")]),
                       page(1, [B("text", "fication here.")])], page_markers=False)
        assert "we use sparsification here." in md

    def test_join_skips_trailing_footnote(self):
        md = assemble([page(0, [B("text", "results of the"), B("footnote", "1 note")]),
                       page(1, [B("text", "previous section.")])], page_markers=False)
        assert "results of the previous section." in md
        assert md.index("previous section") < md.index("1 note")

    def test_no_join_after_sentence_end(self):
        md = assemble([page(0, [B("text", "Done.")]),
                       page(1, [B("text", "next starts lower.")])], page_markers=False)
        assert "Done.\n\nnext starts lower." in md

    def test_no_join_into_capital(self):
        md = assemble([page(0, [B("text", "see Section")]),
                       page(1, [B("text", "Proof. trivial")])], page_markers=False)
        assert "see Section\n\nProof. trivial" in md
