"""Regression tests for the third round of reader fixes (finding ids in each
class's docstring): where the regions lost at a cut lie on one-column pages
and below a band of columns, how much a reading lost, olmOCR's \\[ … \\], and
dots' LaTeX escapes."""

import json
import random
import re

import pytest
from PIL import Image

from conftest import FakeServer
from src.readers.base import reading_loss, truncated_tail
from src.readers.chandra import html_to_blocks
from src.readers.dots import DotsReader, parse_layout
from src.readers.markdown import OlmOCRReader
from src.schema import Block
from src.validate import validate_block

W = 1008        # a square page this size is not resized for dots


def px(box):
    return [round(v * W) for v in box]


def dots_reply(elements, cut_at=None):
    """dots' JSON for (box, category, text) elements, cut off 5 characters
    into the text of element ``cut_at`` (None: complete)."""
    s = json.dumps([{"bbox": px(b), "category": c, "text": t} for b, c, t in elements])
    if cut_at is None:
        return s, "stop"
    pos = 0
    for k in range(cut_at + 1):
        pos = s.index('{"bbox"', pos + (1 if k else 0))
    return s[:s.index('"text": "', pos) + len('"text": "') + 5], "length"


def dots_read(elements, cut_at=None):
    reply = dots_reply(elements, cut_at)
    blocks = DotsReader(FakeServer(lambda p, n: reply).client()).read(Image.new("RGB", (W, W)))
    for b in blocks:
        b.flags = validate_block(b)
    return blocks


def div(box, label, inner):
    return f'<div data-bbox="{box}" data-label="{label}">{inner}</div>'


def tail_boxes(blocks):
    return [[round(v, 3) for v in b.bbox] if b.bbox else None
            for b in blocks if b.meta.get("truncated_tail")]


def inside(inner, outer):
    return all(o <= i + 1e-9 for o, i in zip(outer[:2], inner[:2])) and \
        all(i <= o + 1e-9 for i, o in zip(inner[2:], outer[2:]))


def kept(*boxes):
    return [Block(type="text", content="x", bbox=list(b)) for b in boxes]


# A one-column page: paragraph, then the conditions of a theorem as short
# list items (the cut falls in item (c)), then a full-width paragraph.
ONE_COL_ITEMS = [([0.1, 0.05, 0.9, 0.08], "Title", "A Paper"),
                 ([0.1, 0.1, 0.9, 0.45], "Text", "Para one."),
                 ([0.12, 0.46, 0.50, 0.48], "List-item", "(a) $G$ is connected and regular;"),
                 ([0.12, 0.49, 0.38, 0.51], "List-item", "(b) $d \\geq 3$;"),
                 ([0.12, 0.52, 0.47, 0.54], "List-item", "(c) $\\lambda_2 > 0$ holds here."),
                 ([0.1, 0.56, 0.9, 0.9], "Text", "Then the theorem holds for every n.")]

# A two-column page: title, abstract, a left column of three paragraphs,
# the right column.
TWO_COL = [([0.06, 0.04, 0.94, 0.09], "Title", "Spectral Sparsification"),
           ([0.06, 0.10, 0.94, 0.25], "Text", "Abstract. We show that every graph ..."),
           ([0.06, 0.27, 0.486, 0.50], "Text", "1 Introduction. Left para 1."),
           ([0.06, 0.51, 0.486, 0.70], "Text", "Left para 2."),
           ([0.06, 0.71, 0.486, 0.95], "Text", "Left para 3."),
           ([0.516, 0.27, 0.94, 0.95], "Text", "Right column.")]


class TestOneColumnPages:
    """v3-readers-1: short left-aligned elements on a one-column page are no
    left column; the rest of the page is one full-width region."""

    def test_dots_short_list_items(self):
        blocks = dots_read(ONE_COL_ITEMS, cut_at=4)
        assert tail_boxes(blocks) == [[0.0, 0.52, 1.0, 1.0]]
        for box, _, _ in ONE_COL_ITEMS[4:]:
            assert inside(box, blocks[-1].bbox)

    def test_chandra_header_line_and_list_group(self):
        html = (div("100 50 900 300", "Text", "<p>A full-width paragraph ...</p>")
                + div("100 320 330 340", "Section-Header", "<h3>2.1 Notation</h3>")
                + div("100 350 500 370", "Text", "<p>We use the following notation.</p>")
                + '<div data-bbox="120 380 540 470" data-label="List-Group"><ul>'
                  '<li><math>G=(V,E)</math> a graph;</li><li><math>L_G</math> L L L L')
        assert tail_boxes(html_to_blocks(html, "t", truncated=True)) == [[0.0, 0.38, 1.0, 1.0]]

    def test_one_tall_left_box_is_no_column(self):
        # a kept list group reaching the middle, then a cut in a short header
        page = kept([0.1, 0.1, 0.9, 0.3], [0.12, 0.31, 0.54, 0.5])
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.1, 0.52, 0.3, 0.54])] == [
            [0.0, 0.52, 1.0, 1.0]]

    def test_short_lines_on_one_edge_are_no_column(self):
        # two one-line items that happen to end together: not as tall as text
        page = kept([0.1, 0.1, 0.9, 0.3], [0.1, 0.31, 0.5, 0.33], [0.1, 0.34, 0.5, 0.36])
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.1, 0.37, 0.3, 0.39])] == [
            [0.0, 0.37, 1.0, 1.0]]

    def test_two_column_page_still_gets_the_right_column(self):
        blocks = dots_read(TWO_COL, cut_at=4)
        assert tail_boxes(blocks) == [[0.0, 0.71, 0.486, 1.0], [0.486, 0.27, 1.0, 1.0]]
        # a cut in its first paragraph, below a short heading of the column
        page = kept([0.06, 0.04, 0.94, 0.25], [0.06, 0.27, 0.2, 0.29])
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.06, 0.3, 0.486, 0.6])] == [
            [0.0, 0.3, 1.0, 1.0]]       # nothing of the column read: both below the cut
        page = kept([0.06, 0.04, 0.94, 0.25], [0.06, 0.27, 0.486, 0.33])
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.06, 0.34, 0.486, 0.6])] == [
            [0.0, 0.34, 0.486, 1.0], [0.486, 0.27, 1.0, 1.0]]


# A Physical Review page: two columns, a widetext equation across both, two
# columns again. The reader is cut in the upper right column.
PRL = [([0.06, 0.06, 0.486, 0.48], "Text", "Left column, upper part."),
       ([0.516, 0.06, 0.94, 0.25], "Text", "Right column, upper part 1."),
       ([0.516, 0.26, 0.94, 0.48], "Text", "Right column, upper part 2."),
       ([0.10, 0.50, 0.90, 0.58], "Formula", "H = \\sum_{i<j} J_{ij} \\sigma_i \\sigma_j"),
       ([0.06, 0.60, 0.486, 0.95], "Text", "Left column, lower part."),
       ([0.516, 0.60, 0.94, 0.95], "Text", "Right column, lower part.")]


class TestBelowTheColumns:
    """v3-readers-3: the region between the kept columns ends where they end;
    what lies below them is a full-width region of its own."""

    def test_widetext_band(self):
        blocks = dots_read(PRL, cut_at=2)
        assert tail_boxes(blocks) == [[0.486, 0.26, 1.0, 0.48], [0.0, 0.48, 1.0, 1.0]]
        right, below = (b.bbox for b in blocks if b.meta.get("truncated_tail"))
        assert inside(PRL[2][0], right)
        for box, _, _ in PRL[3:]:       # the equation whole, the lower columns
            assert inside(box, below)

    def test_full_width_table_at_the_bottom(self):
        page = [([0.06, 0.04, 0.94, 0.09], "Title", "A Paper"),
                ([0.06, 0.11, 0.486, 0.60], "Text", "Left column."),
                ([0.516, 0.11, 0.94, 0.30], "Text", "Right para 1."),
                ([0.516, 0.31, 0.94, 0.60], "Text", "Right para 2."),
                ([0.06, 0.63, 0.94, 0.92], "Table", "<table><tr><td>a</td></tr></table>")]
        html = "".join(div(" ".join(str(round(v * 1000)) for v in b),
                           {"Title": "Section-Header"}.get(c, c), f"<p>{t}</p>")
                       for b, c, t in page[:3])
        html += '<div data-bbox="516 310 940 600" data-label="Text"><p>Right para 2 the'
        expected = [[0.486, 0.31, 1.0, 0.6], [0.0, 0.6, 1.0, 1.0]]
        assert tail_boxes(dots_read(page, cut_at=3)) == expected
        assert tail_boxes(html_to_blocks(html, "t", truncated=True)) == expected

    def test_only_a_bottom_margin_below(self):
        # 1-inch margins on letter paper: the left column ends at 0.905, the
        # page number below it is no reason for a region of its own
        page = kept([0.09, 0.09, 0.91, 0.14], [0.09, 0.15, 0.49, 0.905],
                    [0.51, 0.15, 0.91, 0.5])
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.51, 0.51, 0.91, 0.9])] == [
            [0.49, 0.51, 1.0, 1.0]]

    def test_cut_where_the_columns_end(self):
        # the left column ends just below the cut: one full-width region
        page = kept([0.06, 0.04, 0.94, 0.09], [0.06, 0.11, 0.486, 0.505],
                    [0.516, 0.11, 0.94, 0.49])
        assert [t.bbox for t in truncated_tail(page, "s", cut=[0.516, 0.5, 0.94, 0.52])] == [
            [0.0, 0.5, 1.0, 1.0]]

    def test_left_cut_regions_run_to_the_bottom(self):
        # cut in the upper left column: nothing read says where the unread
        # columns end, so both regions run to the page bottom
        page = [([0.06, 0.06, 0.486, 0.2], "Text", "Left column, upper part 1."),
                ([0.06, 0.21, 0.486, 0.48], "Text", "Left column, upper part 2.")] + PRL[1:]
        assert tail_boxes(dots_read(page, cut_at=1)) == [
            [0.0, 0.21, 0.486, 1.0], [0.486, 0.06, 1.0, 1.0]]


class TestReadingLoss:
    """v3-readers-2: attempts are compared by what they lost, not by how
    many blocks stand for it."""

    def test_a_left_column_cut_loses_less_than_a_cut_in_the_abstract(self):
        late = dots_read(TWO_COL, cut_at=4)         # two tail regions
        early = dots_read(TWO_COL, cut_at=1)        # one
        assert len([b for b in late if "truncated" in b.flags]) == 2
        assert reading_loss(late) < reading_loss(early)
        assert reading_loss(late)[0] == pytest.approx(0.29 * 0.486 + 0.73 * 0.514, abs=1e-3)

    def test_a_repeated_element_counts_once_and_loses_nothing(self):
        eq = div("100 400 900 520", "Equation-Block",
                 '<p><math display="block">\\lambda_2 \\le 2d</math> (3.1)</p><p>where</p>'
                 '<p><math display="block">d = \\max_v \\deg v</math> (3.2)</p>')
        page = (div("100 50 900 90", "Section-Header", "<h1>Spectral Sparsification</h1>")
                + div("100 100 900 390", "Text", "<p>Intro.</p>") + eq + eq
                + div("100 530 900 900", "Text", "<p>Rest of the page.</p>"))
        blocks = html_to_blocks(page, "t")
        assert [(b.type, b.meta.get("repeated")) for b in blocks] == [
            ("title", None), ("text", None), ("formula", 2), ("text", None),
            ("formula", None), ("text", None)]
        for b in blocks:
            b.flags = validate_block(b)
        cut = html_to_blocks(page[:page.index("<p>Intro") + 10], "t", truncated=True)
        for b in cut:
            b.flags = validate_block(b)
        assert reading_loss(blocks)[:2] == (0.0, 1)         # complete, worth a retry
        assert reading_loss(blocks) < reading_loss(cut)

    def test_flags(self):
        def block(flags, bbox=None, content="text", **meta):
            return Block(type="text", content=content, bbox=bbox, flags=flags, meta=meta)
        # a repeated element whose text also loops lost its box
        both = block(["repeated", "repetition"], [0.0, 0.0, 0.5, 0.2], repeated=3)
        assert reading_loss([both]) == (0.1, 1, 0)
        # validation that marks the element with "repetition" only
        assert reading_loss([block(["repetition"], repeated=3)]) == (0.0, 1, -4)
        assert reading_loss([block(["repetition"])]) == (1.0, 0, 0)
        assert reading_loss([block([])]) == (0.0, 0, -4)

    def test_boxless_readings_prefer_the_one_that_read_further(self):
        tail = Block(type="text", content="", flags=["empty", "truncated"],
                     meta={"truncated_tail": True})
        short = [Block(type="text", content="One."), tail]
        longer = [Block(type="text", content="One. Two. Three."), tail]
        assert reading_loss(longer) < reading_loss(short)
        assert reading_loss(longer)[0] == reading_loss(short)[0] == 1.0


class TestOlmocrDisplayBrackets:
    """v3-readers-4, v3-readers-5: a row break's '\\[' opens no display math,
    and only brackets around math are lifted out of a line."""

    @staticmethod
    def parse(reply):
        reader = OlmOCRReader(FakeServer(lambda p, n: "").client())
        return [(b.type, b.content) for b in reader.parse(reply)]

    def test_row_break_with_spacing(self):
        reply = ("Hence\n\n\\[\n\\begin{cases} 1 & x > 0 \\\\[2pt] 0 & \\text{otherwise} "
                 "\\end{cases}\n\\]\n\n## 3 Main result\n\nTheorem 3. Every graph has a "
                 "sparsifier.\n\nProof. Combine:\n\n\\[\nL_H \\preceq (1+\\epsilon) L_G\n\\]\n\n"
                 "which finishes the proof.")
        assert self.parse(reply) == [
            ("text", "Hence"),
            ("formula", "\\begin{cases} 1 & x > 0 \\\\[2pt] 0 & \\text{otherwise} \\end{cases}"),
            ("heading", "3 Main result"),
            ("text", "Theorem 3. Every graph has a sparsifier."),
            ("text", "Proof. Combine:"),
            ("formula", "L_H \\preceq (1+\\epsilon) L_G"),
            ("text", "which finishes the proof.")]
        reply = ("\\[\n\\begin{aligned}\nE &= \\sum_i x_i^2 \\\\[4pt]\n&\\le n\n\\end{aligned}\n"
                 "\\]\nfor every feasible \\(x\\).")
        assert self.parse(reply) == [
            ("formula", "\\begin{aligned}\nE &= \\sum_i x_i^2 \\\\[4pt]\n&\\le n\n\\end{aligned}"),
            ("text", "for every feasible $x$.")]

    def test_math_in_a_line_is_still_lifted(self):
        assert self.parse(r"We obtain \[ \sum_i x_i = 1 \] for all \(x\).") == [
            ("text", "We obtain"), ("formula", r"\sum_i x_i = 1"), ("text", "for all $x$.")]
        assert self.parse(r"so \[ a^2 + b^2 \] holds") == [
            ("text", "so"), ("formula", "a^2 + b^2"), ("text", "holds")]

    @pytest.mark.parametrize("line", [
        r"As shown in \[ABC+20\], the bound holds.",
        r"By \[3, §2\] we have the claim.",
        r"See \[Spi04, Thm. 2(b)\] for details.",
        r"The interval \[−1, 1\] is compact.",
        r"\[ABC+20\] A. Andoni, B. Brown. Sparsifiers. In STOC, 2020.",
    ])
    def test_citations_stay_text(self, line):
        assert self.parse(line) == [("text", line)]


class TestDotsEscapes:
    """v3-readers-6, reopen-v2-readers-2: LaTeX escapes in dots' JSON."""

    @staticmethod
    def texts(reply):
        els, _, _ = parse_layout(reply)
        return [(e.get("text"), bool(e.get("json_repaired"))) for e in els]

    @staticmethod
    def element(category, raw_text):
        return '[{"bbox": [1, 1, 500, 50], "category": "%s", "text": "%s"}]' % (
            category, raw_text)

    def test_one_slip_in_doubled_latex(self):
        raw = r"Let $S = \\{x \in A : \\|x\\| \\leq 1\\}$ and $\\,\\alpha$."
        assert self.texts(self.element("Text", raw)) == [
            (r"Let $S = \{x \in A : \|x\| \leq 1\}$ and $\,\alpha$.", True)]
        raw = r"\\left\\{ x \in \\mathbb{R} \\;:\\; \\|x\\| \\le 1 \\right\\}"
        assert self.texts(self.element("Formula", raw)) == [
            (r"\left\{ x \in \mathbb{R} \;:\; \|x\| \le 1 \right\}", True)]
        # a slip that is valid JSON (\t, \r) in otherwise doubled math
        raw = r"$\\alpha + \theta \right)$ and \\(x\\)"
        assert self.texts(self.element("Text", raw)) == [
            (r"$\alpha + \theta \right)$ and \(x\)", True)]

    def test_doubled_row_breaks_and_escaped_dollars(self):
        raw = r"\\begin{pmatrix} a \\\\ b \end{pmatrix}"
        assert self.texts(self.element("Formula", raw)) == [
            (r"\begin{pmatrix} a \\ b \end{pmatrix}", True)]
        # a doubled row break or \( is evidence of doubling too
        assert self.texts(self.element("Formula", r"a \\\\ \theta")) == [
            (r"a \\ \theta", True)]
        assert self.texts(self.element("Text", r"\\(x\\) and $\theta$")) == [
            (r"\(x\) and $\theta$", True)]
        # a single-backslash row break stays one
        assert self.texts(self.element("Formula", r"a \\ \theta \\[2pt] b")) == [
            (r"a \\ \theta \\[2pt] b", True)]
        # \\$ is an escaped dollar: the TAB after it is not in math
        raw = r"costs \\$5;\tEach \\alpha, \in"
        assert self.texts(self.element("Text", raw)) == [
            ("costs \\$5;\tEach \\alpha, \\in", True)]

    @pytest.mark.parametrize("text", [
        "Step\tSet $\\{x \\in A\\}$, then $\\|x\\|_2 \\, \\leq 1$",
        "Case\tTwo: $$\\begin{cases} 1 & x>0 \\\\ 0 & \\text{else}\\end{cases}$$",
        "Step\tSet x to\tzero.",
        "$\\nabla f$\nnext line $\\theta$",
        "$a$\nunder $b$, price $5 per unit,\ne.g. more\nunder",
        "$$\nu = 1\n$$",
    ])
    def test_valid_json_is_untouched(self, text):
        reply = json.dumps([{"bbox": [1, 1, 500, 50], "category": "Text", "text": text}])
        assert self.texts(reply) == [(text, False)]

    def test_valid_json_round_trips(self):
        """Valid JSON changes only where a TAB before a letter in math is read
        as \\t (\\theta), or a newline in math before 'abla', 'mid', … as \\n
        (\\nabla, \\nmid): nothing else in such a literal changes."""
        rng = random.Random(7)
        toks = ["\\frac{a}{b}", "\\alpha", "$", "$$", " ", "x", "The", "\n", "\\\\", "\\{",
                "\\}", '"', "/", "\t", "é", "\\u00e9", "C:\\Users", "\\nabla", "{", "}",
                "\\(", "\\)", "\\[", "\\]", "a & b", "u", "e.g.", "mid", "1"]
        changed = 0
        for _ in range(3000):
            texts = ["".join(rng.choice(toks) for _ in range(rng.randint(1, 10)))
                     for _ in range(3)]
            cats = [rng.choice(["Text", "Formula", "Table"]) for _ in texts]
            reply = json.dumps([{"bbox": [1, 1 + i, 9, 9 + i], "category": c, "text": t}
                                for i, (c, t) in enumerate(zip(cats, texts))])
            for (got, repaired), t in zip(self.texts(reply), texts):
                if got != t:
                    changed += 1
                    assert repaired
                    allowed = "".join({"\t": r"(?:\t|\\t)", "\n": r"(?:\n|\\n)"}.get(
                        ch, re.escape(ch)) for ch in t)
                    assert re.fullmatch(allowed, got, re.S), (t, got)
        assert 0 < changed < 300

    def test_n_commands_in_text_math(self):
        raw = r"At a minimum $\nabla f(x^*) = 0$, and the walk is $\nu$-mixing."
        assert self.texts(self.element("Text", raw)) == [
            (r"At a minimum $\nabla f(x^*) = 0$, and the walk is $\nu$-mixing.", True)]
        for raw, text in [(r"$\nu$-mixing", r"$\nu$-mixing"), (r"if $a \ne b$", r"if $a \ne b$"),
                          (r"$\not\in$ here", r"$\not\in$ here")]:
            assert self.texts(self.element("Text", raw)) == [(text, True)]
        # outside math \n is a newline
        assert self.texts(self.element("Text", r"$x$\nunder")) == [("$x$\nunder", False)]
