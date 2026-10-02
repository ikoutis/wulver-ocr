"""Regression tests for the reader fixes (finding ids in each test's name or
docstring): Chandra's HTML conversion, dots' JSON recovery, the Markdown and
olmOCR readers, and the truncated-tail contract shared by all of them."""

import json
import os

import pytest
from PIL import Image

from conftest import FakeServer
from src import run_ocr
from src.assemble import assemble, render_block
from src.backend import TRUNCATION_MARKER
from src.readers import READERS, htmlmd, markdown
from src.readers.base import truncated_tail
from src.readers.chandra import ChandraReader, html_to_blocks
from src.readers.dots import DotsReader, parse_elements
from src.readers.markdown import (OLMOCR_PROMPT, OlmOCRReader, PageMarkdownReader,
                                  split_markdown, strip_outer_fence)
from src.schema import Block, Page

MARKER = TRUNCATION_MARKER.strip()      # "<<TRUNCATED>>", in any form


def chandra(html, **kw):
    return [(b.type, b.content) for b in html_to_blocks(html, "t", **kw)]


def md(html):
    return htmlmd.to_markdown(htmlmd.parse(html))


EQ = '<div data-label="Equation-Block" data-bbox="100 100 900 200">%s</div>'


class TestEquationNumbers:
    """readers-1: numbers and prose outside the <math> of an Equation-Block."""

    def test_each_number_tags_its_own_equation(self):
        assert chandra(EQ % ('<math display="block">a = b</math> (2.1)<br>'
                             '<math display="block">c = d</math> (2.2)')) == [
            ("formula", r"a = b \tag{2.1}"), ("formula", r"c = d \tag{2.2}")]

    @pytest.mark.parametrize("after, tag", [
        (", (3)", "3"), (" (3).", "3"), (" (3')", "3'"), (" (*)", "*"), (" (ii)", "ii"),
        (" [5]", "5"), (" (2.1-a)", "2.1-a"), (" (A.3)", "A.3"), (" 2.1.", "2.1"),
        ("<p>(4b)</p>", "4b")])
    def test_number_forms(self, after, tag):
        assert chandra(EQ % f'<math display="block">x = 1</math>{after}') == [
            ("formula", rf"x = 1 \tag{{{tag}}}")]

    def test_prose_between_equations_is_kept_and_inline_math_stays_inline(self):
        assert chandra(EQ % ('<math display="block">f(x)=0</math> for all <math>x \\in A</math>,'
                             ' and <math display="block">g(x)=1</math>')) == [
            ("formula", "f(x)=0"), ("text", r"for all $x \in A$, and"), ("formula", "g(x)=1")]

    def test_number_then_prose(self):
        assert chandra(EQ % '<math display="block">a = b</math> (6) where <math>b</math> is') == [
            ("formula", r"a = b \tag{6}"), ("text", "where $b$ is")]

    def test_number_on_the_left_and_duplicate_of_a_tag(self):
        assert chandra(EQ % '(4) <math display="block">a = b</math>') == [
            ("formula", r"a = b \tag{4}")]
        assert chandra(EQ % r'<math display="block">a = b \tag{2}</math> (2)') == [
            ("formula", r"a = b \tag{2}")]

    def test_bare_number_followed_by_prose_is_prose(self):
        assert chandra(EQ % '<math display="block">a = b</math> 2 times') == [
            ("formula", "a = b"), ("text", "2 times")]

    def test_block_without_display_math(self):
        assert chandra(EQ % '<math>a = b</math> (5)') == [("formula", r"a = b \tag{5}")]


class TestInlineConversion:
    def test_emphasis_whitespace_moves_outside_the_delimiters(self):
        """readers-2"""
        assert md('<p><b>Proof. </b>By induction.</p>') == "**Proof.** By induction."
        assert md('<p><b>Theorem 1.2. </b><i>Let </i><math>G</math><i> be a graph on </i>'
                  '<math>n</math><i> vertices.</i></p>') == \
            "**Theorem 1.2.** *Let* $G$ *be a graph on* $n$ *vertices.*"

    def test_raw_less_than_inside_math(self):
        """readers-3"""
        assert md("<p>Assume <math>0<x<1</math> and <math>a<b</math> holds.</p>") == \
            "Assume $0<x<1$ and $a<b$ holds."
        assert md('<math display="block">|x|<\\epsilon \\implies f(x)<g(x)</math>') == \
            "$$\n|x|<\\epsilon \\implies f(x)<g(x)\n$$"

    def test_inline_math_on_one_line(self):
        assert md("<p>so <math>a\n- b</math> holds</p>") == "so $a - b$ holds"

    @pytest.mark.parametrize("html, out", [
        ("<p>Alice Smith<sup>*</sup>, Bob Jones<sup>*</sup></p>",
         r"Alice Smith<sup>\*</sup>, Bob Jones<sup>\*</sup>"),
        ("<p>__init__ and `x`</p>", r"\_\_init\_\_ and \`x\`"),
        ("<p>&lt;S&gt; and &lt;img src=x onerror=alert(1)&gt;</p>",
         "&lt;S> and &lt;img src=x onerror=alert(1)>"),
        ("<p>AT&amp;T, &amp;amp;, 1 &lt; 2</p>", "AT&T, &amp;amp;, 1 < 2"),
        ("<p># of vertices</p>", r"\# of vertices"),
        ("<p>2016. The year</p>", r"2016\. The year"),
        ("<p>- a</p><p>+ b</p><p>&gt; c</p><p>---</p>", "\\- a\n\n\\+ b\n\n\\> c\n\n\\---"),
        ("<p>line<br>- not a list</p>", "line\\\n\\- not a list"),    # v2-readers-10
        ("<p>-1 and +2 and x-y</p>", "-1 and +2 and x-y"),
        ("<ul><li># one</li></ul>", r"- \# one"),
    ])
    def test_text_is_escaped(self, html, out):
        """readers-4"""
        assert md(html) == out

    def test_display_math_lines_are_not_escaped(self):
        assert md('<p>so <math display="block">a\n- b</math> holds</p>') == \
            "so\n\n$$\na\n- b\n$$\n\nholds"
        assert md('<p>so <b><math display="block">a\n- b</math></b></p>') == \
            "so\n**$$\na\n- b\n$$**"

    def test_figure_alt_is_html_escaped(self):
        """readers-4"""
        (fig,) = html_to_blocks('<div data-label="Image" data-bbox="1 1 900 900">'
                                '<img alt="See <img src=x onerror=alert(1)> here"/></div>', "t")
        assert fig.meta["reader_description"] == "See &lt;img src=x onerror=alert(1)> here"

    def test_img_outside_figures_is_dropped(self):
        """readers-5"""
        assert chandra('<div data-label="Text" data-bbox="1 1 900 100"><p>The results are '
                       'summarised below.<img alt="A bar chart"/></p></div>') == [
            ("text", "The results are summarised below.")]
        assert chandra('<div data-label="Caption" data-bbox="1 1 900 100">Figure 3: '
                       'Accuracy.<img alt="Bar chart"/></div>') == [
            ("caption", "Figure 3: Accuracy.")]

    def test_block_children_are_separated(self):
        """readers-8"""
        assert md("<ol><li><p>First para.</p><p>Second para.</p></li></ol>") == \
            "1. First para. Second para."
        assert chandra('<div data-label="Section-Header" data-bbox="1 1 900 100">'
                       '<p>3</p><p>Main Results</p></div>') == [("heading", "3 Main Results")]
        table = htmlmd.parse("<table><tr><td><p>first</p><p>second</p></td>"
                             "<td>x<br>y</td><td>a <b>b</b></td></tr></table>").children[0]
        assert htmlmd.table_html(table) == \
            "<table><tr><td>first<br>second</td><td>x<br>y</td><td>a <b>b</b></td></tr></table>"

    def test_unclosed_list_items(self):
        """readers-10"""
        assert md("<ul><li>first item<li>second item<li>third item</ul>") == \
            "- first item\n- second item\n- third item"
        assert md("<ol><li>[1] A. Author, Paper one.<li>[2] B. Author, Paper two.</ol>") == \
            "1. [1] A. Author, Paper one.\n2. [2] B. Author, Paper two."
        assert md("<ul><li>a<ul><li>a1<li>a2</ul><li>b</ul>") == "- a\n  - a1\n  - a2\n- b"

    def test_unclosed_paragraphs(self):
        assert md("<p>one <b>two<p>three") == "one **two**\n\nthree"

    def test_looping_unclosed_tags_do_not_overflow(self):
        """readers-22"""
        for unit in ("<p>the the ", "<b>x ", "<li>x ", "<span>x "):
            html = '<div data-label="Text" data-bbox="1 1 900 900">' + unit * 3000
            (b,) = html_to_blocks(html + "</div>", "t")
            assert b.type == "text" and b.content
            assert html_to_blocks(html, "t", truncated=True)[-1].meta == {"truncated_tail": True}


class TestChandraBlocks:
    def test_nested_layout_divs_become_blocks(self):
        """readers-6"""
        blocks = html_to_blocks(
            '<div data-label="Complex-Block" data-bbox="80 100 920 400">'
            '<div data-label="Text" data-bbox="80 100 920 150"><p><b>Theorem 3.</b> For every '
            'graph <math>G</math>,</p></div>'
            '<div data-label="Equation-Block" data-bbox="200 160 800 220"><math display="block">'
            '\\lambda_2(L_G) \\le \\frac{n}{n-1}</math> (3.4)</div>'
            '<div data-label="Image" data-bbox="300 230 700 400"><img alt="The graph G"/></div>'
            '<p>Loose text.</p><div data-label="Caption"><p>Figure 1.</p></div></div>', "t")
        assert [(b.type, b.content) for b in blocks] == [
            ("text", "**Theorem 3.** For every graph $G$,"),
            ("formula", r"\lambda_2(L_G) \le \frac{n}{n-1} \tag{3.4}"),
            ("figure", ""), ("text", "Loose text."), ("caption", "Figure 1.")]
        assert blocks[1].bbox == [0.2, 0.16, 0.8, 0.22]
        assert blocks[2].meta["reader_description"] == "The graph G"
        assert blocks[3].meta["category"] == "Complex-Block"
        assert blocks[4].bbox == [0.08, 0.1, 0.92, 0.4]          # inherits the parent's box

    def test_code_block(self):
        """readers-7"""
        algo = ('<div data-label="Code-Block" data-bbox="1 1 900 900"><p><b>Algorithm 1</b> Greedy'
                '</p><p>1: <b>for</b> <math>i = 1</math> to <math>n</math> <b>do</b><br>'
                '2: <math>M \\gets M \\cup \\{e_i\\}</math><br>3: <b>end for</b></p></div>')
        assert chandra(algo) == [("text", "**Algorithm 1** Greedy\n\n1: **for** $i = 1$ to $n$ "
                                          "**do**\\\n2: $M \\gets M \\cup \\{e_i\\}$\\\n"
                                          "3: **end for**")]     # hard breaks: v2-readers-10
        assert chandra('<div data-label="Code-Block" data-bbox="1 1 900 900"><p>Listing 1</p>'
                       '<pre>def f(x):\n    return x</pre><pre>print(f(1))</pre></div>') == [
            ("text", "Listing 1"), ("code", "def f(x):\n    return x"), ("code", "print(f(1))")]

    def test_section_header(self):
        """readers-9, readers-20"""
        def header(inner):
            return html_to_blocks('<div data-label="Section-Header" data-bbox="1 1 900 100">'
                                  f'{inner}</div>', "t")[0]
        for inner in ("<h1>Spectral Sparsification</h1><h2>of Graphs</h2>",
                      "<h1>Spectral Sparsification<br>of Graphs</h1>"):
            b = header(inner)
            assert (b.type, b.content) == ("title", "Spectral Sparsification of Graphs")
            assert render_block(b) == "# Spectral Sparsification of Graphs"
        assert render_block(header("<h2>1 Introduction</h2>")) == "## 1 Introduction"
        assert render_block(header("<h3>1.1 Background</h3>")) == "### 1.1 Background"
        assert render_block(header("<p>Untagged</p>")) == "## Untagged"

    def test_caption_br_is_a_space(self):
        """readers-9"""
        (b,) = html_to_blocks('<div data-label="Caption" data-bbox="1 1 900 100">'
                              '<p>Figure 1: A<br>graph.</p></div>', "t")
        assert render_block(b) == "*Figure 1: A graph.*"

    def test_content_outside_the_expected_container_is_kept(self):
        """readers-17"""
        assert chandra('<div data-label="Table" data-bbox="1 1 900 500"><p>Table 2: Running '
                       'times</p><table><tr><td>a</td></tr></table><p>* measured on 8 cores.</p>'
                       '</div>') == [
            ("caption", "Table 2: Running times"), ("table", "<table><tr><td>a</td></tr></table>"),
            ("caption", r"\* measured on 8 cores.")]
        assert chandra("<p>Whole page without <b>layout</b> <i>divs</i>.</p>") == [
            ("text", "Whole page without **layout** *divs*.")]
        assert chandra('<div data-label="Text" data-bbox="1 1 900 100"><p>One.</p></div>'
                       "Two outside.") == [("text", "One."), ("text", "Two outside.")]
        assert chandra("<html><body><div data-label='Text' data-bbox='1 1 900 100'>"
                       "<p>x</p></div></body></html>") == [("text", "x")]

    def test_empty_reply_is_degenerate_but_blank_page_is_not(self):
        """readers-17"""
        (b,) = html_to_blocks("", "t")
        assert b.meta == {"truncated_tail": True} and b.bbox == [0.0, 0.0, 1.0, 1.0]
        assert html_to_blocks('<div data-label="Blank-Page" data-bbox="0 0 1000 1000"></div>',
                              "t") == []


class TestDotsJson:
    E = [{"bbox": [10, 10 + 50 * k, 500, 50 + 50 * k], "category": "Text", "text": f"Para {k}."}
         for k in range(6)]

    def texts(self, reply):
        els, complete = parse_elements(reply)
        return [e.get("text") for e in els], complete

    def test_bad_element_costs_only_itself(self):
        """readers-11"""
        s = json.dumps(self.E)
        # an unescaped quote: the element is kept as its box, the rest intact
        els, complete = parse_elements(s.replace("Para 2.", 'the so-called "spectral" method'))
        assert complete and [e["text"] for e in els] == [
            "Para 0.", "Para 1.", "", "Para 3.", "Para 4.", "Para 5."]
        assert els[2] == {"text": "", "bbox": [10, 110, 500, 150], "category": "Text"}
        # a raw newline, single-backslash LaTeX: read as meant
        assert self.texts(s.replace("Para 1.", "line\nbreak"))[0][1] == "line\nbreak"
        assert self.texts(s.replace("Para 3.", r"\alpha \in A"))[0][3] == r"\alpha \in A"
        assert self.texts(s.replace("Para 3.", r"\\frac{a}{b}"))[0][3] == r"\frac{a}{b}"

    def test_text_after_the_array_is_not_truncation(self):
        """readers-11"""
        s = json.dumps(self.E)
        assert self.texts(s + "\nI hope this helps.") == ([f"Para {k}." for k in range(6)], True)
        assert self.texts(s[:-1] + ",]")[1]

    def test_resync_ignores_latex_braces(self):
        """readers-11"""
        s = json.dumps([{"bbox": [1, 2, 3, 4], "category": "Formula", "text": "x^{2"},
                        {"bbox": [1, 5, 3, 9], "category": "Text", "text": "ok {a} {b}"}])
        assert self.texts(s.replace('"x^{2"', '"x^{2" junk')) == (["", "ok {a} {b}"], True)

    def test_wrappers(self):
        """readers-18"""
        for key in ("layout", "elements", "layout_dets", "result", "cells"):
            assert self.texts(json.dumps({key: self.E}))[0] == [f"Para {k}." for k in range(6)]
        assert parse_elements(json.dumps({"layout": []})) == ([], True)
        els, complete = parse_elements(json.dumps({"layout": self.E})[:-30])
        assert not complete and len(els) == 5

    @pytest.mark.parametrize("reply", ["null", "42", '"hello"', "{}", "", "I cannot read this."])
    def test_not_a_layout(self, reply):
        """readers-13"""
        assert parse_elements(reply) == ([], False)

    def test_odd_values_and_title_hashes(self):
        """readers-13, readers-14"""
        srv = FakeServer(lambda p, n: json.dumps([
            {"bbox": [1, 2, 300, 40], "category": "Title", "text": "# Spectral Sparsifiers"},
            {"bbox": [1, 50, 300, 90], "category": "Section-header", "text": "## 1 Intro"},
            {"bbox": [1, 900, 30, 940], "category": "Page-footer", "text": 12},
            {"bbox": [1, 100, 300, 140], "category": "Text", "text": ["a", "b"]},
            {"bbox": [1, 150, 300, 190], "category": "Text", "text": None},
            {"bbox": None, "category": None}]))
        blocks = DotsReader(srv.client()).read(Image.new("RGB", (896, 1092)))
        assert [(b.type, b.content) for b in blocks] == [
            ("title", "Spectral Sparsifiers"), ("heading", "1 Intro"), ("footer", "12"),
            ("text", '["a", "b"]'), ("text", ""), ("other", "")]
        assert render_block(blocks[0]) == "# Spectral Sparsifiers"
        assert "level" not in blocks[0].meta and render_block(blocks[1]) == "## 1 Intro"

    def test_null_reply_is_a_degenerate_page(self):
        """readers-13"""
        srv = FakeServer(lambda p, n: "null")
        (b,) = DotsReader(srv.client()).read(Image.new("RGB", (896, 1092)))
        assert b.meta == {"truncated_tail": True} and b.bbox == [0.0, 0.0, 1.0, 1.0]


# --------------------------------------------------------------- truncation


def no_marker(blocks):
    return not any(MARKER in b.content for b in blocks)


class TestTruncatedTail:
    """readers-12, readers-16, gate-8, docs-5: the truncated-tail contract."""

    def test_helper(self):
        kept = [Block(type="header", content="h", bbox=[0, 0.9, 1, 0.95]),
                Block(type="text", content="a", bbox=[0.1, 0.1, 0.5, 0.3]),
                Block(type="text", content="b", bbox=[0.5, 0.1, 0.9, 0.4]),
                Block(type="text", content="c")]
        (tail,) = truncated_tail(kept, "src")
        assert (tail.type, tail.content, tail.source) == ("text", "", "src")
        assert tail.bbox == [0.0, 0.4, 1.0, 1.0] and tail.meta == {"truncated_tail": True}
        assert [t.bbox for t in truncated_tail([], "s")] == [[0.0, 0.0, 1.0, 1.0]]
        assert [t.bbox for t in truncated_tail([Block(type="text", content="x")], "s")] == [None]
        assert [t.bbox for t in truncated_tail(
            [Block(type="text", content="x", bbox=[0, 0.5, 1, 1.0])], "s")] == [[0.0, 0.98, 1.0, 1.0]]

    def test_dots_looping_element(self):
        els = [{"bbox": [50, 50, 800, 150], "category": "Section-header", "text": "## 1 Intro"},
               {"bbox": [50, 170, 800, 300], "category": "Formula", "text": "E = mc^{2}"}]
        reply = json.dumps(els)[:-1] + ', {"bbox": [50, 310, 800, 700], "category": "Text", ' \
                                       '"text": "the the the the'
        srv = FakeServer(lambda p, n: (reply, "length"))
        blocks = DotsReader(srv.client()).read(Image.new("RGB", (896, 1092)))
        assert [(b.type, b.content) for b in blocks] == [
            ("heading", "1 Intro"), ("formula", "E = mc^{2}"), ("text", "")]
        assert blocks[-1].meta == {"truncated_tail": True}
        # the tail starts at the looping element's own box (y 310-700 of 1092)
        assert blocks[-1].bbox == [0.0, 310 / 1092, 1.0, 1.0]

    def test_dots_cut_between_elements(self):
        s = json.dumps(TestDotsJson.E)
        srv = FakeServer(lambda p, n: (s[:-1] + ", ", "length"))
        blocks = DotsReader(srv.client()).read(Image.new("RGB", (896, 1092)))
        assert len(blocks) == 7 and blocks[-1].meta == {"truncated_tail": True} and no_marker(blocks)

    def test_chandra_cut_inside_a_div(self):
        srv = FakeServer(lambda p, n: (
            '<div data-label="Text" data-bbox="100 100 900 300"><p>Done.</p></div>\n'
            '<div data-label="Text" data-bbox="100 310 900 700"><p>the the the', "length"))
        blocks = ChandraReader(srv.client()).read(Image.new("RGB", (1700, 2200)))
        assert [(b.type, b.content) for b in blocks] == [("text", "Done."), ("text", "")]
        assert blocks[-1].bbox == [0.0, 0.31, 1.0, 1.0] and blocks[-1].meta["truncated_tail"]

    def test_chandra_cut_between_divs(self):
        srv = FakeServer(lambda p, n: (
            '<div data-label="Text" data-bbox="100 100 900 300"><p>Done.</p></div>\n', "length"))
        blocks = ChandraReader(srv.client()).read(Image.new("RGB", (1700, 2200)))
        assert [(b.type, b.content) for b in blocks] == [("text", "Done."), ("text", "")]

    @pytest.mark.parametrize("reply, kept", [
        ("# Title\n\nSome text.\n\n", [("title", "Title"), ("text", "Some text.")]),
        ("Para one.\n\n$$\nx = 1\n$$\n", [("text", "Para one."), ("formula", "x = 1")]),
        ("Para one.\n\n<table><tr><td>a</td></tr></table>\n",
         [("text", "Para one."), ("table", "<table><tr><td>a</td></tr></table>")]),
        ("Para one.\n\nPara two\nis cut", [("text", "Para one.")]),
        ("Para one.\n\n$$\nx = ", [("text", "Para one.")]),
        ("Para one.\n\n<table><tr><td>a</td></tr>\n", [("text", "Para one.")]),
    ])
    def test_markdown(self, reply, kept):
        srv = FakeServer(lambda p, n: (reply, "length"))
        blocks = PageMarkdownReader(srv.client()).read(Image.new("RGB", (1700, 2200)))
        assert [(b.type, b.content) for b in blocks[:-1]] == kept
        assert blocks[-1].meta == {"truncated_tail": True} and blocks[-1].content == ""
        assert blocks[-1].bbox is None and no_marker(blocks)

    def test_marker_never_reaches_the_markdown(self, tmp_path):
        html = '<div data-label="Text" data-bbox="100 100 900 300"><p>Done.</p></div>'
        pages = []
        for k, (reader, reply) in enumerate([
                (ChandraReader, html + "<div data-label='Text'><p>cut"),
                (DotsReader, json.dumps(TestDotsJson.E)[:-40]),
                (PageMarkdownReader, "# Title\n\nSome text.\n\n")]):
            srv = FakeServer(lambda p, n, r=reply: (r, "length"))
            blocks = reader(srv.client()).read(Image.new("RGB", (896, 1092)))
            pages.append(Page(doc_id="d", index=k, image="", width=896, height=1092,
                              blocks=blocks))
        out = assemble(pages)
        assert MARKER not in out and "Done." in out and "Some text." in out


def test_read_stage_saves_tail_block(tmp_path, pdf_path, monkeypatch):
    """readers-12, docs-5: through run_ocr, a cut-off page is saved (and
    assembles) with its complete blocks and the tail block."""
    els = [{"bbox": [100, 100, 1500, 300], "category": "Text", "text": "Kept."}]
    reply = json.dumps(els)[:-1] + ', {"bbox": [100, 400, 1500, 900], "category": "Text", "text": "x x'
    srv = FakeServer(lambda p, n: (reply, "length"))
    monkeypatch.setattr(run_ocr, "ChatClient", lambda url, model=None, timeout=0, **kw: srv.client())
    work, out = str(tmp_path / "w"), str(tmp_path / "o")
    for stage in ("read", "assemble"):
        assert run_ocr.main([stage, "--inputs", pdf_path, "--work", work, "--out", out,
                             "--workers", "1"]) == 0
    (doc,) = os.listdir(work)
    page = json.load(open(os.path.join(work, doc, "read", "p0001.json")))
    assert [b["content"] for b in page["blocks"]] == ["Kept.", ""]
    assert page["blocks"][-1]["meta"] == {"truncated_tail": True}
    text = open(os.path.join(out, doc, doc + ".md")).read()
    assert "Kept." in text and MARKER not in text


# ----------------------------------------------------------- Markdown readers


class TestMarkdownReader:
    def test_outer_fence_is_stripped(self):
        """readers-15"""
        reply = ("```markdown\n## 1 Introduction\n\nLet $G$ be a graph.\n\n$$\nL = D - A \\tag{1}"
                 "\n$$\n\nMore text.\n```")
        assert [(b.type, b.content) for b in split_markdown(strip_outer_fence(reply))] == [
            ("heading", "1 Introduction"), ("text", "Let $G$ be a graph."),
            ("formula", r"L = D - A \tag{1}"), ("text", "More text.")]
        assert strip_outer_fence("```\nText.\n```\n") == "Text."
        assert strip_outer_fence("```md\nText.") == "Text."            # cut: no closing fence
        code = "```python\nx = 1\n```\n\nText."
        assert strip_outer_fence(code) == code

    def test_fenced_reply_through_the_reader(self):
        """readers-15"""
        srv = FakeServer(lambda p, n: "```markdown\n# T\n\nBody.\n```")
        blocks = PageMarkdownReader(srv.client()).read(Image.new("RGB", (800, 1000)))
        assert [(b.type, b.content) for b in blocks] == [("title", "T"), ("text", "Body.")]

    def test_escaped_citations_are_not_display_math(self):
        """readers-19"""
        page = ("## References\n\n\\[1\\] D. Spielman and S. Teng.\n\n\\[Spi04, 2\\] X.\n\n"
                "\\[\nx = 1\n\\]\n\n\\[ y = 2 \\]\n\n\\[ z\n= 3 \\]\n")
        assert [(b.type, b.content) for b in split_markdown(page)] == [
            ("heading", "References"), ("text", r"\[1\] D. Spielman and S. Teng."),
            ("text", r"\[Spi04, 2\] X."), ("formula", "x = 1"), ("formula", "y = 2"),
            ("formula", "z\n= 3")]

    def test_figure_alt_becomes_reader_description(self):
        blocks = split_markdown("![figure](figure)\n\n![A <b>path</b> graph](p.png)\n")
        assert blocks[0].meta == {}
        assert blocks[1].meta == {"reader_description": "A &lt;b>path&lt;/b> graph"}


OLMOCR_REPLY = r"""---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
# A Title

Let \( G=(V,E) \) be a graph and \(x\) a vector.

\[
L = D - A
\]

![A path graph on four vertices](page_100_200_300_400.png)
"""


class TestOlmOCR:
    """readers-21, docs-12: a real olmOCR-2 adapter instead of a docstring."""

    def test_registered_and_documented(self):
        assert READERS["olmocr"] is OlmOCRReader
        assert issubclass(OlmOCRReader, PageMarkdownReader)
        assert "--reader-prompt" not in (markdown.__doc__ + PageMarkdownReader.__doc__)
        assert "math_blocks" not in htmlmd.__doc__
        args = run_ocr.build_parser().parse_args(["read", "--reader", "olmocr"])
        assert args.reader == "olmocr"

    def test_prompt_is_olmocrs_v4(self):
        # olmocr/prompts/prompts.py: build_no_anchoring_v4_yaml_prompt()
        assert OLMOCR_PROMPT.startswith("Attached is one page of a document that you must "
                                        "process. Just return the plain text representation")
        assert "Convert equations to LateX and tables to HTML.\n" in OLMOCR_PROMPT
        assert OLMOCR_PROMPT.endswith("primary_language, is_rotation_valid, rotation_correction,"
                                      " is_table, and is_diagram parameters.")

    def test_request_and_parse(self):
        srv = FakeServer(lambda p, n: OLMOCR_REPLY)
        r = OlmOCRReader(srv.client())
        blocks = r.read(Image.new("RGB", (850, 1100)))
        assert [(b.type, b.content) for b in blocks] == [
            ("title", "A Title"), ("text", "Let $G=(V,E)$ be a graph and $x$ a vector."),
            ("formula", "L = D - A"), ("figure", "")]
        assert blocks[3].meta["reader_description"] == "A path graph on four vertices"
        body = srv.requests[0]
        assert body["max_tokens"] == 8000 and body["temperature"] == 0.1
        content = body["messages"][0]["content"]
        assert [p["type"] for p in content] == ["text", "image_url"]
        assert content[0]["text"] == OLMOCR_PROMPT
        r.read(Image.new("RGB", (850, 1100)), attempt=2)
        assert srv.requests[1]["temperature"] == 0.2

    def test_page_rendered_at_1288_on_the_long_side(self):
        r = OlmOCRReader(FakeServer(lambda p, n: "").client())
        assert r.prepare(Image.new("RGB", (1700, 2200))).size == (995, 1288)
        assert r.prepare(Image.new("RGB", (850, 1100))).size == (995, 1288)
