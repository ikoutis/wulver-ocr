"""Regression tests for the second round of reader fixes (finding ids in each
class's or test's docstring): where the regions lost at a cut lie, dots'
single-backslash LaTeX, element-level loops, left-numbered equations, and the
Markdown made of <br>, lists, tables, and olmOCR's math and rotation."""

import base64
import io
import json
import os

import pytest
from PIL import Image, ImageDraw

from conftest import FakeServer
from src import run_ocr
from src.assemble import render_block
from src.readers import htmlmd
from src.readers.base import drop_repeats, truncated_tail
from src.readers.chandra import html_to_blocks
from src.readers.dots import DotsReader, parse_layout
from src.readers.markdown import OlmOCRReader, split_markdown
from src.schema import Block
from src.validate import check_table


def chandra(html, **kw):
    return [(b.type, b.content) for b in html_to_blocks(html, "t", **kw)]


def md(html):
    return htmlmd.to_markdown(htmlmd.parse(html))


def tails(blocks):
    return [b for b in blocks if b.meta.get("truncated_tail")]


def inside(inner, outer):
    return all(o <= i + 1e-9 for o, i in zip(outer[:2], inner[:2])) and \
        all(i <= o + 1e-9 for i, o in zip(inner[2:], outer[2:]))


def dots_read(reply, finish="length", size=(1008, 1008)):
    srv = FakeServer(lambda p, n: (reply, finish))
    return DotsReader(srv.client()).read(Image.new("RGB", size))


def div(box, label, inner):
    return f'<div data-bbox="{box}" data-label="{label}">{inner}</div>'


# A two-column page (1008 px square: no resize for dots): title, left column
# with a footnote at its bottom, right column. Read in reading order.
TWO_COL = [
    {"bbox": [60, 40, 950, 90], "category": "Title", "text": "# A Paper"},
    {"bbox": [60, 110, 490, 400], "category": "Text", "text": "Left para 1."},
    {"bbox": [60, 410, 490, 700], "category": "Text", "text": "Left para 2."},
    {"bbox": [60, 710, 490, 900], "category": "Text", "text": "Left para 3."},
    {"bbox": [60, 910, 490, 960], "category": "Footnote", "text": "1 Supported by NSF."},
    {"bbox": [520, 110, 950, 300], "category": "Text", "text": "Right para 1."},
    {"bbox": [520, 310, 950, 960], "category": "Text", "text": "Right para 2 the the the"},
]
TWO_COL_HTML = (div("60 40 950 90", "Section-Header", "<h1>A Paper</h1>")
                + div("60 110 490 400", "Text", "<p>Left para 1.</p>")
                + div("60 410 490 900", "Text", "<p>Left para 2.</p>")
                + div("60 910 490 960", "Footnote", "<p>1 Supported by NSF.</p>")
                + div("520 110 950 300", "Text", "<p>Right para 1.</p>"))
CUT_RIGHT = '<div data-bbox="520 310 950 960" data-label="Text"><p>Right para 2 the the the'


class TestTailRegions:
    """v2-readers-1, v2-core-5, v2-gate-2, v2-e2e-1, reopen-readers-12,
    reopen-docs-5: the tail regions cover what the cut lost, on two-column
    pages too, and start from the box of the element cut off."""

    def test_dots_cut_in_the_right_column(self):
        s = json.dumps(TWO_COL)
        (tail,) = tails(dots_read(s[:s.index("Right para 2") + 10]))
        assert tail.meta == {"truncated_tail": True} and tail.content == ""
        # the rest of the right column, beside the kept left column
        assert tail.bbox == [490 / 1008, 310 / 1008, 1.0, 1.0]
        assert inside([v / 1008 for v in TWO_COL[6]["bbox"]], tail.bbox)

    def test_chandra_cut_in_the_right_column(self):
        blocks = html_to_blocks(TWO_COL_HTML + CUT_RIGHT, "t", truncated=True)
        assert [b.content for b in blocks[:-1]] == [
            "A Paper", "Left para 1.", "Left para 2.", "1 Supported by NSF.", "Right para 1."]
        (tail,) = tails(blocks)
        assert tail.bbox == [0.49, 0.31, 1.0, 1.0]

    def test_cut_in_the_left_column_loses_the_right_one_too(self):
        html = (div("60 40 950 90", "Section-Header", "<h1>A Paper</h1>")
                + div("60 110 490 400", "Text", "<p>Left para 1.</p>")
                + '<div data-bbox="60 410 490 960" data-label="Text"><p>Left para 2 the the')
        left, right = tails(html_to_blocks(html, "t", truncated=True))
        assert left.bbox == [0.0, 0.41, 0.49, 1.0]         # the rest of the left column
        assert right.bbox == [0.49, 0.11, 1.0, 1.0]        # the whole, unread right column
        assert inside([0.52, 0.11, 0.95, 0.96], right.bbox)

    def test_cut_between_elements_in_the_right_column(self):
        # no element in progress: the region starts below the last one read
        (tail,) = tails(html_to_blocks(TWO_COL_HTML, "t", truncated=True))
        assert tail.bbox == [0.49, 0.3, 1.0, 1.0]

    def test_one_column_page(self):
        kept = [Block(type="text", content="a", bbox=[0.1, 0.1, 0.9, 0.3]),
                Block(type="heading", content="Proof.", bbox=[0.1, 0.31, 0.2, 0.34])]
        # a centred formula cut off: the lines below are full width
        assert [t.bbox for t in truncated_tail(kept, "s", cut=[0.3, 0.35, 0.7, 0.4])] == [
            [0.0, 0.35, 1.0, 1.0]]
        # a narrow element alone is no evidence of a second column
        assert [t.bbox for t in truncated_tail(kept[:1], "s", cut=[0.1, 0.31, 0.3, 0.34])] == [
            [0.0, 0.31, 1.0, 1.0]]

    def test_no_second_region_without_an_unread_column(self):
        # short left-aligned items on a one-column page are not a column
        items = [Block(type="text", content="a", bbox=[0.1, 0.1, 0.9, 0.3]),
                 Block(type="list", content="- one", bbox=[0.1, 0.31, 0.3, 0.33]),
                 Block(type="list", content="- two", bbox=[0.1, 0.34, 0.35, 0.36])]
        assert [t.bbox for t in truncated_tail(items, "s", cut=[0.1, 0.37, 0.4, 0.39])] == [
            [0.0, 0.37, 1.0, 1.0]]
        # a cut in the page footer, both columns read
        page = [Block(type="text", content="x", bbox=[v / 1008 for v in e["bbox"]])
                for e in TWO_COL]
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.45, 0.97, 0.55, 0.99])] == [
            [0.0, 0.97, 1.0, 1.0]]

    def test_cut_inside_a_copy_of_a_kept_element(self):
        kept = [Block(type="header", content="J. ACM", bbox=[0.1, 0.02, 0.9, 0.04]),
                Block(type="text", content="a", bbox=[0.1, 0.1, 0.9, 0.3])]
        for copy in ([0.1, 0.1, 0.9, 0.3], [0.1, 0.02, 0.9, 0.04]):     # a looping header too
            assert [t.bbox for t in truncated_tail(kept, "s", cut=copy)] == [[0.0, 0.3, 1.0, 1.0]]

    def test_boxless_and_empty(self):
        assert [t.bbox for t in truncated_tail([Block(type="text", content="x")], "s")] == [None]
        assert [t.bbox for t in truncated_tail([], "s")] == [[0.0, 0.0, 1.0, 1.0]]
        assert [t.bbox for t in truncated_tail([], "s", cut=[0.1, 0.5, 0.9, 0.7])] == [
            [0.0, 0.5, 1.0, 1.0]]

    def test_read_stage_saves_the_column_region(self, tmp_path, pdf_path, monkeypatch):
        """reopen-docs-5: cut on every attempt, through run_ocr read"""
        srv = FakeServer(lambda p, n: (TWO_COL_HTML + CUT_RIGHT, "length"))
        monkeypatch.setattr(run_ocr, "ChatClient",
                            lambda url, model=None, timeout=0, **kw: srv.client())
        work = str(tmp_path / "w")
        assert run_ocr.main(["read", "--inputs", pdf_path, "--work", work, "--reader",
                             "chandra", "--workers", "1"]) == 0
        (doc,) = os.listdir(work)
        page = json.load(open(os.path.join(work, doc, "read", "p0001.json")))
        assert len(srv.requests) == 4                  # two pages, two attempts each
        tail = page["blocks"][-1]
        assert tail["meta"] == {"truncated_tail": True} and tail["bbox"] == [0.49, 0.31, 1.0, 1.0]
        assert "truncated" in tail["flags"]


class TestSingleBackslashLatex:
    """v2-readers-2, v2-e2e-3: LaTeX with single backslashes in dots' JSON."""

    @staticmethod
    def texts(reply):
        els, _, _ = parse_layout(reply)
        return [(e.get("text"), bool(e.get("json_repaired"))) for e in els]

    @pytest.mark.parametrize("tex", [
        r"\frac{a}{b} \leq \alpha",                         # \f
        r"x^T L_H x \tag{1} \times \theta \to \text{ok}",    # \t
        r"\beta \bar{x} \binom{n}{k}",                       # \b
        r"\rho_i \left( x \right) \rangle",                  # \r
        r"\nabla f \neq \nu",                                # \n, in math
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",     # a row break stays one
    ])
    def test_formula(self, tex):
        reply = '[{"bbox": [1, 1, 500, 50], "category": "Formula", "text": "%s"}]' % tex
        assert self.texts(reply) == [(tex, True)]

    def test_text(self):
        reply = (r'[{"bbox": [1, 1, 500, 50], "category": "Text", "text": "Let $\theta \in '
                 r'\Theta$, $\nu \geq 0$, $\rho < \alpha$, $x \to \infty$.\nNext, \$5."}]')
        assert self.texts(reply) == [(
            "Let $\\theta \\in \\Theta$, $\\nu \\geq 0$, $\\rho < \\alpha$, "
            "$x \\to \\infty$.\nNext, \\$5.", True)]     # \n outside math is a newline

    def test_only_valid_looking_escapes(self):
        # \frac alone is valid JSON (a form feed and 'rac'), still LaTeX
        reply = r'[{"bbox": [1, 1, 500, 50], "category": "Text", "text": "only $\frac{a}{b}$"}]'
        assert self.texts(reply) == [("only $\\frac{a}{b}$", True)]

    def test_valid_json_is_untouched(self):
        els = [{"bbox": [1, 1, 500, 50], "category": "Text",
                "text": "$\\frac{a}{b}$ and\nnext\t(tab) \u00e9 \"q\" C:\\temp"},
               {"bbox": [1, 60, 500, 90], "category": "Formula",
                "text": "\\begin{aligned}\na &= b \\\\\nc &= \\nabla d\n\\end{aligned}"}]
        assert self.texts(json.dumps(els)) == [(e["text"], False) for e in els]

    def test_only_the_repaired_element_is_marked_and_survives_a_cut(self):
        reply = (r'[{"bbox": [1, 1, 500, 50], "category": "Text", "text": "$\alpha$ ok"}, '
                 r'{"bbox": [1, 60, 500, 90], "category": "Text", "text": "plain"}, '
                 r'{"bbox": [1, 100, 500, 400], "category": "Text", "text": "cut \beta the')
        els, complete, cut = parse_layout(reply)
        assert [(e["text"], bool(e.get("json_repaired"))) for e in els] == [
            ("$\\alpha$ ok", True), ("plain", False)]
        assert not complete and cut == [1, 100, 500, 400]

    def test_reader_puts_the_mark_in_meta(self):
        blocks = dots_read(r'[{"bbox": [1, 1, 500, 50], "category": "Formula", '
                           r'"text": "$$\frac{a}{b}$$"}, '
                           r'{"bbox": [1, 60, 500, 90], "category": "Text", "text": "x"}]', "stop")
        assert [(b.content, b.meta) for b in blocks] == [
            ("\\frac{a}{b}", {"category": "Formula", "json_repaired": True}),
            ("x", {"category": "Text"})]


class TestRepeatedElements:
    """v2-readers-4: an element the model re-emitted is kept once, marked."""

    A = {"bbox": [60, 60, 950, 120], "category": "Text", "text": "We now prove it."}
    C = {"bbox": [60, 130, 950, 260], "category": "Text", "text": "By Lemma 3, the gap is large."}
    D = {"bbox": [60, 270, 950, 330], "category": "Text", "text": "This completes the proof."}

    def test_loop_the_model_left(self):
        blocks = dots_read(json.dumps([self.A] + [self.C] * 4 + [self.D]), "stop")
        assert [b.content for b in blocks] == [self.A["text"], self.C["text"], self.D["text"]]
        assert [b.meta.get("repeated") for b in blocks] == [None, 4, None]

    def test_loop_to_max_tokens(self):
        s = json.dumps([self.A] + [self.C] * 40)
        blocks = dots_read(s[:-50])
        assert [b.content for b in blocks] == [self.A["text"], self.C["text"], ""]
        assert blocks[1].meta["repeated"] == 39
        # the cut fell in one more copy: what is lost lies below the element
        assert blocks[2].bbox == [0.0, 260 / 1008, 1.0, 1.0]

    def test_same_text_in_new_boxes(self):
        moving = [{**self.C, "bbox": [60, 130 + 10 * k, 950, 260 + 10 * k]} for k in range(5)]
        blocks = dots_read(json.dumps([self.A] + moving), "stop")
        assert [b.meta.get("repeated") for b in blocks] == [None, 5]
        # fewer than five copies in distinct boxes: kept (dots' own threshold)
        assert len(dots_read(json.dumps([self.A] + moving[:4]), "stop")) == 5

    def test_chandra(self):
        c = div("60 130 950 260", "Text", "<p>By Lemma 3, the gap is large.</p>")
        blocks = html_to_blocks(div("60 60 950 120", "Text", "<p>We now prove it.</p>")
                                + c * 30 + c[:60], "t", truncated=True)
        assert [(b.content, b.meta.get("repeated")) for b in blocks] == [
            ("We now prove it.", None), ("By Lemma 3, the gap is large.", 30), ("", None)]
        assert blocks[-1].bbox == [0.0, 0.26, 1.0, 1.0]

    def test_drop_repeats(self):
        items = [("b1", "x"), ("b2", "y"), ("b1", "z"), ("b3", "y"), (None, "y")]
        kept = drop_repeats(items, lambda it: it[0], lambda it: it[1])
        assert kept == [[("b1", "x"), 2], [("b2", "y"), 1], [("b3", "y"), 1], [(None, "y"), 1]]


EQ = '<div data-label="Equation-Block" data-bbox="100 100 900 200">%s</div>'


class TestLeftNumberedEquations:
    """v2-readers-3, reopen-readers-1"""

    def test_each_number_tags_the_next_equation(self):
        assert chandra(EQ % ('<p>(2.1) <math display="block">A</math></p>'
                             '<p>(2.2) <math display="block">B</math></p>'
                             '<p>(2.3) <math display="block">C</math></p>')) == [
            ("formula", r"A \tag{2.1}"), ("formula", r"B \tag{2.2}"), ("formula", r"C \tag{2.3}")]
        assert chandra(EQ % ('(1.1)<br><math display="block">a=b</math><br>(1.2)<br>'
                             '<math display="block">c=d</math>')) == [
            ("formula", r"a=b \tag{1.1}"), ("formula", r"c=d \tag{1.2}")]

    def test_a_number_after_the_last_equation_is_not_lost(self):
        assert chandra(EQ % ('<p>(1) <math display="block">A</math></p>'
                             '<p><math display="block">B</math> (2)</p>')) == [
            ("formula", r"A \tag{1}"), ("formula", r"B \tag{2}")]

    def test_right_numbers_are_unchanged(self):
        assert chandra(EQ % ('<math display="block">A</math> (2.1)<br>'
                             '<math display="block">B</math> (2.2)')) == [
            ("formula", r"A \tag{2.1}"), ("formula", r"B \tag{2.2}")]

    def test_roman_section_label(self):
        assert chandra(EQ % '<math display="block">a=b</math> (II.3)') == [
            ("formula", r"a=b \tag{II.3}")]


class TestOlmOCRMath:
    """v2-readers-5: olmOCR's \\[ … \\] mid-line and \\( … \\) over lines."""

    @staticmethod
    def parse(reply):
        reader = OlmOCRReader(FakeServer(lambda p, n: "").client())
        return [(b.type, b.content) for b in reader.parse(reply)]

    def test_display_math_inside_a_paragraph(self):
        assert self.parse(r"We obtain \[ \sum_i x_i = 1 \] for all feasible \(x\).") == [
            ("text", "We obtain"), ("formula", r"\sum_i x_i = 1"),
            ("text", "for all feasible $x$.")]
        assert self.parse(r"\[ E = mc^2 \] where \(m\) is the mass.") == [
            ("formula", "E = mc^2"), ("text", "where $m$ is the mass.")]

    def test_inline_math_over_a_line_break(self):
        assert self.parse("Let \\(f(x) =\nx^2\\) be given.") == [
            ("text", "Let $f(x) = x^2$ be given.")]
        # not across paragraphs
        assert self.parse("a \\( b\n\nc \\) d") == [("text", "a \\( b"), ("text", "c \\) d")]

    def test_citations_and_tables_are_left_alone(self):
        assert self.parse(r"As in \[12\] and \[Spi04, 2\], see \(x\).") == [
            ("text", r"As in \[12\] and \[Spi04, 2\], see $x$.")]
        table = r"<table><tr><td>\[ a+b \] c</td></tr></table>"
        assert self.parse(table) == [("table", table)]


class TestTablesWithoutEndTags:
    """v2-readers-6: implied </td> </th> </tr>, as HTML has them."""

    def test_long_table(self):
        rows = "".join("<tr>" + "".join(f"<td>r{i}c{j}" for j in range(6)) for i in range(20))
        closed = "".join("<tr>" + "".join(f"<td>r{i}c{j}</td>" for j in range(6)) + "</tr>"
                         for i in range(20))
        (b,) = html_to_blocks(div("1 1 900 900", "Table", f"<table>{rows}</table>"), "t")
        (ref,) = html_to_blocks(div("1 1 900 900", "Table", f"<table>{closed}</table>"), "t")
        assert b.content == ref.content and check_table(b.content) == []
        assert b.content.count("<tr>") == 20 and "r19c5</td></tr></table>" in b.content

    def test_cells_and_nested_tables(self):
        t = htmlmd.parse("<table><tr><th>h<td>a<table><tr><td>x<td>y</table><td>b"
                         "<tr><td>c</table>").children[0]
        assert htmlmd.table_html(t) == (
            "<table><tr><th>h</th><td>a<table><tr><td>x</td><td>y</td></tr></table></td>"
            "<td>b</td></tr><tr><td>c</td></tr></table>")


class TestOrderedLists:
    """v2-readers-7: start, value, and lettered or roman types."""

    @pytest.mark.parametrize("html, out", [
        ('<ol start="4"><li>fourth</li><li>fifth</li></ol>', "4. fourth\n5. fifth"),
        ('<ol><li value="4">fourth</li><li>fifth</li></ol>', "4. fourth\n5. fifth"),
        ('<ol start="x"><li>one</li></ol>', "1. one"),
        ('<ol type="a"><li>connected;</li><li>regular.</li></ol>',
         "- (a) connected;\n- (b) regular."),
        ('<ol type="i"><li>a</li><li>b</li><li>c</li><li>d</li></ol>',
         "- (i) a\n- (ii) b\n- (iii) c\n- (iv) d"),
        ('<ol type="I" start="3"><li>a</li></ol>', "- (III) a"),
        ('<ol type="A"><li>a</li></ol>', "- (A) a"),
        ('<ol type="a"><li>(a) connected;</li><li>b) regular.</li></ol>',
         "- (a) connected;\n- b) regular."),
        ('<ol type="1"><li>one</li></ol>', "1. one"),
        ("<ul><li>x</li></ul>", "- x"),
    ])
    def test_markers(self, html, out):
        assert md(html) == out


class TestMarkdownTitle:
    """v2-readers-8: '#' is the title, as design.md §6 promises."""

    def test_levels(self):
        blocks = split_markdown("# Spectral Sparsification\n\n## 1 Introduction\n\n"
                                "### 1.1 Background\n\nSome text.\n")
        assert [render_block(b) for b in blocks] == [
            "# Spectral Sparsification", "## 1 Introduction", "### 1.1 Background", "Some text."]
        assert blocks[0].type == "title" and blocks[1].meta == {"level": 1}


FRONT = ("---\nprimary_language: en\nis_rotation_valid: {valid}\nrotation_correction: {turn}\n"
         "is_table: False\nis_diagram: False\n---\n")


def request_image(body) -> Image.Image:
    url = next(p for p in body["messages"][0]["content"] if p["type"] == "image_url")
    return Image.open(io.BytesIO(base64.b64decode(url["image_url"]["url"].split(",", 1)[1])))


class TestOlmOCRRotation:
    """v2-readers-9: a page olmOCR says is rotated is read again, turned."""

    def page(self):
        img = Image.new("RGB", (850, 1100), "white")
        ImageDraw.Draw(img).rectangle((50, 50, 300, 120), fill="black")
        return img

    def test_turned_as_asked(self):
        replies = iter([FRONT.format(valid="False", turn=90) + "eht",
                        FRONT.format(valid="True", turn=0) + "Some text.\n"])
        srv = FakeServer(lambda p, n: next(replies))
        reader = OlmOCRReader(srv.client())
        blocks = reader.read(self.page())
        assert [(b.type, b.content, b.meta) for b in blocks] == [
            ("text", "Some text.", {"rotation": 90})]
        first, second = (request_image(r) for r in srv.requests)
        turned = reader.prepare(self.page()).transpose(Image.Transpose.ROTATE_90)
        assert first.size == (995, 1288) and second.size == (1288, 995)
        assert second.convert("RGB").tobytes() == turned.tobytes()
        assert [r["temperature"] for r in srv.requests] == [0.1, 0.1]

    def test_upright_page_is_read_once(self):
        srv = FakeServer(lambda p, n: FRONT.format(valid="True", turn=0) + "Text.\n")
        blocks = OlmOCRReader(srv.client()).read(self.page())
        assert len(srv.requests) == 1 and blocks[0].meta == {}

    def test_turns_are_bounded(self):
        srv = FakeServer(lambda p, n: FRONT.format(valid="False", turn=90) + "Text.\n")
        reader = OlmOCRReader(srv.client())
        blocks = reader.read(self.page())
        assert len(srv.requests) == reader.max_turns + 1
        assert blocks[0].meta == {"rotation": 270}


class TestLineBreaks:
    """v2-readers-10: <br> is a hard line break in paragraphs and lists."""

    @pytest.mark.parametrize("html, out", [
        ("<p>Alice Smith<br>Department of CS<br>NJIT</p>",
         "Alice Smith\\\nDepartment of CS\\\nNJIT"),
        ("<ul><li>one<br>two</li></ul>", "- one\\\ntwo"),
        ("<p><br>a <br> <br>b<br></p>", "a\\\nb"),          # none at the ends, one per run
        ("<p><b>do<br></b>next</p>", "**do**\\\nnext"),
        ("<h2>A<br>title</h2>", "## A title"),
    ])
    def test_breaks(self, html, out):
        assert md(html) == out

    def test_caption_and_heading_blocks_stay_on_one_line(self):
        (cap,) = html_to_blocks(div("1 1 900 100", "Caption", "<p>Figure 1: A<br>graph.</p>"), "t")
        (head,) = html_to_blocks(div("1 1 900 100", "Section-Header", "<h2>2 Main<br>Results</h2>"),
                                 "t")
        assert (cap.content, head.content) == ("Figure 1: A graph.", "2 Main Results")
