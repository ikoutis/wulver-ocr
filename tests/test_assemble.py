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


class TestJoinsAcrossFloats:
    """Finding core-12: floats between the halves, lists, references, 3 pages."""

    def test_top_of_page_figure_before_continuation(self):
        md = assemble([page(0, [B("text", "The spectral sparsifier of the")]),
                       page(1, [B("header", "J. Foo"), B("figure", image="f.png"),
                                B("caption", "Figure 2: A sparsifier."),
                                B("text", "graph $G$ is computed in two phases.")])],
                      page_markers=False)
        assert md == ("The spectral sparsifier of the graph $G$ is computed in two phases."
                      "\n\n![figure](f.png)\n\n*Figure 2: A sparsifier.*\n")

    def test_reference_and_list_item_split_across_pages(self):
        md = assemble([page(0, [B("reference", "[12] D. Spielman and S.-H. Teng. Spectral "
                                               "sparsification of")]),
                       page(1, [B("reference", "graphs. SIAM J. Comput., 2011."),
                                B("reference", "[13] Next entry.")])], page_markers=False)
        assert "sparsification of graphs. SIAM J." in md and "\n\n[13] Next entry." in md
        md = assemble([page(0, [B("list", "- an item that")]),
                       page(1, [B("list", "continues here."), B("list", "- next")])],
                      page_markers=False)
        assert md == "- an item that continues here.\n\n- next\n"

    def test_no_join_across_types(self):
        md = assemble([page(0, [B("list", "- an item that")]),
                       page(1, [B("text", "starts lower.")])], page_markers=False)
        assert md == "- an item that\n\nstarts lower.\n"

    def test_three_pages_with_footnote_in_the_middle(self):
        md = assemble([page(0, [B("text", "A proof that")]),
                       page(1, [B("text", "spans a full page and"), B("footnote", "1 A note.")]),
                       page(2, [B("text", "ends here.")])], page_markers=False)
        assert md == "A proof that spans a full page and ends here.\n\n<sub>1 A note.</sub>\n"

    def test_float_page_between_the_halves(self):
        md = assemble([page(0, [B("text", "as the table on the next")]),
                       page(1, [B("table", "<table><tr><td>1</td></tr></table>")]),
                       page(2, [B("text", "page shows.")])])
        assert "as the table on the next page shows.\n\n<!-- page 2 -->" in md

    def test_heading_breaks_the_carry(self):
        md = assemble([page(0, [B("text", "the end of the")]),
                       page(1, [B("heading", "Results", level=1), B("text", "rest.")])],
                      page_markers=False)
        assert md == "the end of the\n\n## Results\n\nrest.\n"


class TestHyphenAtPageBreak:
    """Finding core-13: compounds broken at their own hyphen keep it."""

    def join(self, a, b, *elsewhere):
        md = assemble([page(0, [B("text", a)]), page(1, [B("text", b)]),
                       page(2, [B("text", e) for e in elsewhere])], page_markers=False)
        return md.split("\n\n")[0].strip()

    def test_prefix_list_without_evidence(self):
        assert self.join("is a well-", "known result.") == "is a well-known result."
        assert self.join("the so-", "called Cheeger constant.") \
            == "the so-called Cheeger constant."
        assert self.join("we use sparsi-", "fication here.") == "we use sparsification here."

    def test_document_evidence_decides(self):
        assert self.join("a positive semi-", "definite matrix.",
                         "Every semidefinite program.") == "a positive semidefinite matrix."
        assert self.join("a positive semi-", "definite matrix.",
                         "A semi-definite program.") == "a positive semi-definite matrix."
        assert self.join("the so-", "lution is", "Our solution.") == "the solution is"

    def test_math_before_the_hyphen(self):
        assert self.join("every $k$-", "connected graph.") == "every $k$-connected graph."


class TestFailedAndCutOffPages:
    def failed_page(self, i):
        return Page(doc_id="d", index=i, image="", width=1, height=1, meta={"failed": True},
                    blocks=[Block(type="other", flags=["page_failed"],
                                  meta={"error": "HTTP 400"})])

    def test_failed_page_is_marked_and_breaks_joins(self):
        pages = [page(0, [B("text", "the proof of")]), self.failed_page(1),
                 page(2, [B("text", "lemma two follows.")])]
        for markers in (True, False):
            md = assemble(pages, page_markers=markers)
            assert ("the proof of\n\n<!-- page 2: OCR failed, see report.json -->\n\n"
                    in md) and "<!-- page 2 -->" not in md
            assert md.endswith("\n\nlemma two follows.\n")

    def test_truncated_tail_is_marked_and_breaks_joins(self):
        tail = Block(type="text", content="", bbox=[0, 0.4, 1, 1],
                     flags=["empty", "truncated"], meta={"truncated_tail": True})
        md = assemble([page(0, [B("text", "the first half of"), tail]),
                       page(1, [B("text", "something else.")])], page_markers=False)
        assert md == ("the first half of\n\n<!-- page 1: the reader's output was cut off "
                      "here, see report.json -->\n\nsomething else.\n")

    def test_truncation_marker_never_reaches_markdown(self):
        md = assemble([page(0, [B("text", "cut\n<<TRUNCATED>>"), B("text", "<<TRUNCATED>>"),
                                B("figure", image="f.png", description="A plot\n<<TRUNCATED>>")])])
        assert "TRUNCATED" not in md and "cut" in md and "A plot" in md
