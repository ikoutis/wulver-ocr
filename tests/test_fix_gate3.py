"""Regression tests for the third review round of the gate, validators and
TikZ: v3-readers-7 / v3-gate-1 (a reader's repeat marker is the flag
"repeated", not the degenerate "repetition"), v3-gate-2 (whitespace that
renders), v3-gate-3 (mid-arrow decorations), v3-gate-4 (escape check on HTML
tables), v3-gate-5 (placeholder notes), v3-gate-6 (multi-panel repair) and
v3-gate-7 (the reviewer sees a turned page turned)."""

import base64
import io
import json

import pytest
from PIL import Image, ImageDraw

from conftest import FakeServer
from src.assemble import assemble
from src.figures import DESCRIBE_PROMPT, describe_figures
from src.readers.base import smart_resize
from src.readers.dots import DotsReader
from src.readers.htmlmd import parse, table_html
from src.readers.markdown import OlmOCRReader
from src.review import (DEGENERATE, ReviewPolicy, _TURN, dropped_escapes, gate, review_block,
                        same_text)
from src.run_ocr import make_report
from src.schema import Block, Page
from src.tikz import graph_markdown, parse_graph
from src.validate import validate_block

IMG = Image.new("RGB", (1000, 1400), "white")


def block(type_, content="", bbox=(0.1, 0.1, 0.9, 0.2), **meta):
    b = Block(type=type_, content=content, bbox=list(bbox) if bbox else None,
              source="reader:x", meta=meta)
    b.flags = validate_block(b)
    return b


def tail():
    return block("text", bbox=(0.0, 0.9, 1.0, 1.0), truncated_tail=True)


def review(b, answer, policy=None, img=IMG):
    srv = FakeServer(lambda p, n: answer)
    review_block(srv.client(), img, b, policy or ReviewPolicy(), "reviewer:fake")
    return b, srv


def reply(verdict, body, tag="text"):
    return f"VERDICT: {verdict}\n<{tag}>\n{body}\n</{tag}>"


def sent_image(srv) -> Image.Image:
    for part in srv.requests[-1]["messages"][-1]["content"]:
        if part["type"] == "image_url":
            return Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",")[1])))
    raise AssertionError("no image in the request")


# ------------------------------------------------- v3-readers-7 / v3-gate-1


class TestRepeatedIsNotDegenerate:
    ESCAPED = (r"We store results in a List&lt;String> and run the A\* search, where "
               r"a\*b and c\*d are products.")

    def test_flag(self):
        assert block("text", "Fine text.", repeated=3).flags == ["repeated"]
        assert block("formula", r"x^{2}", repeated=2).flags == ["repeated"]
        assert "repeated" not in DEGENERATE and "repetition" in DEGENERATE
        # a loop inside one block's text is still "repetition"
        assert "repetition" in block("formula", r"\cdot " * 300).flags
        assert "repeated" not in block("formula", r"\cdot " * 300).flags

    def test_still_reviewed(self):
        assert ReviewPolicy().wants(block("text", "Fine text.", repeated=3))

    def test_escape_check_applies(self):
        b = block("text", self.ESCAPED, repeated=3)
        proposal = self.ESCAPED.replace("&lt;", "<").replace(r"\*", "*")
        ok, why = gate(b, proposal, ReviewPolicy())
        assert not ok and why.startswith("drops Markdown escapes")
        b, _ = review(b, reply("fixed", proposal))
        assert b.meta["reviewed"] == "rejected" and b.content == self.ESCAPED
        assert b.flags == ["repeated"]

    def test_change_limit_applies(self):
        b = block("formula", r"\|x_{t+1} - x^\star\| \le \rho \, \|x_t - x^\star\|", repeated=4)
        ok, why = gate(b, r"\int_0^1 f(t)\,dt = \sum_{k \ge 0} c_k", ReviewPolicy())
        assert not ok and why.startswith("change") and "limit 0.60" in why
        ok, why = gate(b, r"\|x_{t+1} - x^\star\| \le \rho \, \|x_{t} - x^\star\|",
                       ReviewPolicy())
        assert ok and "degenerate" not in why

    def test_short_correction_is_accepted(self):       # v3-gate-5 (b)
        b, _ = review(block("text", "(3)", repeated=3), reply("fixed", "(8)"),
                      ReviewPolicy(review_types={"formula", "text"}))
        assert b.meta["reviewed"] == "edited" and b.content == "(8)" and b.flags == []

    def test_prompt_explains_the_flag(self):
        b, srv = review(block("text", "A paragraph.", repeated=3), reply("correct", ""))
        prompt = srv.requests[-1]["messages"][-1]["content"][1]["text"]
        assert "Automatic checks flagged: repeated (the OCR wrote this element more than once" \
            in prompt

    def test_dots_loop_end_to_end(self):
        img = Image.new("RGB", (1700, 2200), "white")
        h, w = smart_resize(img.height, img.width, 28, 3136, 11289600)

        def bx(x0, y0, x1, y1):
            return [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)]
        f = {"bbox": bx(.1, .30, .9, .36), "category": "Formula",
             "text": r"\|x_{t+1} - x^\star\| \le \rho \, \|x_t - x^\star\|"}
        head = {"bbox": bx(.1, .05, .9, .25), "category": "Text", "text": "By the lemma,"}
        srv = FakeServer(lambda p, n: json.dumps([head, f, f, f, f]))
        blocks = DotsReader(srv.client()).read(img)
        for b in blocks:
            b.flags = validate_block(b)
        formula = blocks[1]
        assert formula.meta["repeated"] == 4 and formula.flags == ["repeated"]
        b, _ = review(formula, reply("fixed", r"\int_0^1 f(t)\,dt = \sum_{k \ge 0} c_k", "latex"))
        assert b.meta["reviewed"] == "rejected" and b.source.startswith("reader:")

    def test_accepted_edit_clears_the_flag(self):
        b, _ = review(block("text", "A paragraph with a typo.", repeated=2),
                      reply("fixed", "A paragraph with no typo."))
        assert b.meta["reviewed"] == "edited" and b.flags == []


# ---------------------------------------------------------------- v3-gate-2


class TestWhitespaceThatRenders:
    @pytest.mark.parametrize("draft,fix", [
        (r"x \in A\text{and}y \in B", r"x \in A \text{ and } y \in B"),
        (r"f(x) = 0 \quad\text{for all}x > 0", r"f(x) = 0 \quad\text{for all }x > 0"),
        (r"u_t = \Delta u\text{in}\Omega", r"u_t = \Delta u \text{ in } \Omega"),
        (r"a \mbox{if}b", r"a \mbox{ if } b"),
    ])
    def test_text_group_spaces_are_an_edit(self, draft, fix):
        assert not same_text(draft, fix, "formula")
        b, _ = review(block("formula", draft), reply("fixed", fix, "latex"))
        assert b.meta["reviewed"] == "edited" and b.content == fix

    def test_math_spaces_still_do_not_matter(self):
        assert same_text(r"\text{and}", r"\text {and}", "formula")
        assert same_text(r"\text{a  b}", r"\text{a b}", "formula")
        assert same_text(r"x\tag{3}", r"x \tag{3}", "formula")
        assert same_text(r"\textstyle{a + b}", r"\textstyle{a+b}", "formula")
        b, _ = review(block("formula", r"a+b \text{ if } c"),
                      reply("fixed", r"a + b \text{ if } c", "latex"))
        assert b.meta["reviewed"] == "agreed" and b.source == "reader:x"

    def test_list_nesting_is_an_edit(self):
        flat, nested = "- Step one\n- substep a\n- Step two", "- Step one\n  - substep a\n- Step two"
        assert not same_text(flat, nested, "text")
        assert not same_text("1. a\n2. b", "1. a\n   2. b", "text")
        assert not same_text("- a\n\nb", "- a\n\n  b", "text")      # inside the item or not
        b, _ = review(block("list", flat), reply("fixed", nested))
        assert b.meta["reviewed"] == "edited" and b.content == nested

    def test_continuation_indentation_does_not_matter(self):
        assert same_text("Let  $x$ be\n given.", "Let $x$ be\ngiven. ", "text")
        assert same_text("- a long item\n  continued", "- a long item\ncontinued", "text")
        assert same_text("para\n\n   \n\nnext", "para\n\nnext", "text")
        assert same_text("<table>\n  <tr><td>1</td></tr>\n</table>",
                         "<table>\n<tr><td>1</td></tr>\n</table>", "table")


# ---------------------------------------------------------------- v3-gate-3

AB = r"\node (a) at (0,0) {$a$}; \node (b) at (1,0) {$b$}; "
MID = r"decoration={markings, mark=at position .5 with {\arrow{>}}}, postaction={decorate}"


def edges(code):
    g, flags, problems = parse_graph(code)
    assert flags == [], problems
    return [(e["u"], e["v"], e["directed"], e["both"]) for e in g["edges"]]


FWD, BACK, NONE, BOTH = (("a", "b", True, False), ("b", "a", True, False),
                         ("a", "b", False, False), ("a", "b", True, True))


class TestMidArrows:
    @pytest.mark.parametrize("code,want", [
        (r"\tikzset{->-/.style={" + MID + r"}} \begin{tikzpicture}" + AB
         + r"\draw[->-] (a) -- (b); \end{tikzpicture}", FWD),
        (r"\begin{tikzpicture}[->-/.style={" + MID + r"}]" + AB
         + r"\draw[->-] (a) -- (b); \end{tikzpicture}", FWD),
        (r"\begin{tikzpicture}[mid arrow/.style={postaction={decorate,decoration={markings,"
         r"mark=at position .5 with {\arrow[scale=1.5]{Stealth}}}}}]" + AB
         + r"\draw[mid arrow] (a) -- (b); \end{tikzpicture}", FWD),
        (r"\tikzset{->-/.style={decoration={markings, mark=at position #1 with {\arrow{>}}},"
         r" postaction={decorate}}} \begin{tikzpicture}" + AB
         + r"\draw[->-=.6] (a) -- (b); \end{tikzpicture}", FWD),
        (r"\begin{tikzpicture}" + AB + r"\draw[" + MID.replace(r"\arrow{>}", r"\arrow{<}")
         + r"] (a) -- (b); \end{tikzpicture}", BACK),
        (r"\begin{tikzpicture}" + AB + r"\draw[" + MID.replace(r"\arrow", r"\arrowreversed")
         + r"] (a) -- (b); \end{tikzpicture}", BACK),
        (r"\begin{tikzpicture}" + AB + r"\draw[" + MID.replace(r"{>}", r"{Stealth[reversed]}")
         + r"] (a) -- (b); \end{tikzpicture}", BACK),
        (r"\begin{tikzpicture}[->-/.style={" + MID + r"}, every edge/.style={draw, ->-}]" + AB
         + r"\path (a) edge (b); \end{tikzpicture}", FWD),
        # a style defined but not used, a decoration never applied, a tip with no head
        (r"\begin{tikzpicture}[->-/.style={" + MID + r"}]" + AB
         + r"\draw (a) -- (b); \end{tikzpicture}", NONE),
        (r"\begin{tikzpicture}" + AB + r"\draw[decoration={markings, mark=at position .5 with"
         r" {\arrow{>}}}] (a) -- (b); \end{tikzpicture}", NONE),
        (r"\begin{tikzpicture}" + AB + r"\draw[" + MID.replace(r"\arrow{>}", r"\arrow{|}")
         + r"] (a) -- (b); \end{tikzpicture}", NONE),
        (r"\begin{tikzpicture}" + AB + r"\draw[->, " + MID.replace(r"\arrow{>}", r"\arrow{<}")
         + r"] (a) -- (b); \end{tikzpicture}", BOTH),
    ])
    def test_direction(self, code, want):
        assert edges(code) == [want]

    def test_markdown_version_is_directed(self):
        g, flags, _ = parse_graph(r"\tikzset{->-/.style={" + MID + r"}} \begin{tikzpicture}"
                                  + AB + r"\node (c) at (2,0) {$c$};"
                                  r"\draw[->-] (a) -- (b); \draw[->-] (b) -- (c);"
                                  r"\end{tikzpicture}")
        assert flags == [] and "3 vertices, 2 edges (directed)" in graph_markdown(g)


# ---------------------------------------------------------------- v3-gate-4


class TestTableEscapes:
    DRAFT = table_html(parse(
        "<table><tr><th>Method</th><th>n</th><th>Time</th></tr>"
        "<tr><td>Ours</td><td>&lt; 100</td><td>1.2 s</td></tr>"
        "<tr><td>Baseline</td><td>2.5 s</td></tr></table>").children[0])     # a cell missing
    POLICY = ReviewPolicy(review_types={"formula", "table"})

    @pytest.mark.parametrize("cell", ["< 100", "$< 100$", "&lt; 100"])
    def test_restored_cell_is_accepted(self, cell):
        b = block("table", self.DRAFT, bbox=(0.1, 0.1, 0.9, 0.4))
        assert b.flags == ["table_shape"] and "&lt; 100" in self.DRAFT
        fix = (self.DRAFT.replace("<td>Baseline</td>", "<td>Baseline</td><td>500</td>")
               .replace("&lt; 100", cell))
        b, _ = review(b, reply("fixed", fix, "table_out"), self.POLICY)
        assert b.meta["reviewed"] == "edited" and b.flags == []

    def test_real_drop_is_still_caught(self):
        draft = ("<table><tr><td>List&lt;String&gt;</td><td>1</td></tr>"
                 "<tr><td>x</td></tr></table>")
        b = block("table", draft)
        # the tag-like '<' comes back even as a spurious cell goes
        proposal = "<table><tr><td>List<String></td><td>1</td></tr></table>"
        ok, why = gate(b, proposal, self.POLICY)
        assert not ok and why == "drops Markdown escapes ['&lt;']"
        assert dropped_escapes(draft, draft.replace("<tr><td>x</td></tr>", "<thead></thead>"),
                               "table") == []

    def test_text_blocks_still_count_tags(self):
        assert dropped_escapes("the &lt;b&gt; tag", "the <b> tag") == ["&lt;"]
        assert dropped_escapes("the &lt;b&gt; tag", "the <b> tag", "text") == ["&lt;"]


# ---------------------------------------------------------------- v3-gate-5


class TestPlaceholderNotes:
    @pytest.mark.parametrize("text", [
        "(If $S$ is the empty set, there is nothing to prove.)",
        "(The case of the empty graph is trivial.)",
        "(Here $G$ is nonempty.)",
        "(Nothing to prove.)",
        "(Only if $G$ is connected.)",
        "(Page numbers refer to the journal version.)",
        "[Empty set notation follows Bourbaki.]",
        "(See the transcript of the talk.)",
    ])
    def test_remark_is_a_transcription(self, text):
        b, _ = review(tail(), reply("fixed", text))
        assert b.meta["reviewed"] == "edited" and b.content == text and b.flags == []
        assert b.meta["tail_recovered"] is True
        page = Page(doc_id="d", index=0, image="x", width=1, height=1, reader="r",
                    stage="review", blocks=[Block(type="text", content="Kept text."), b])
        assert make_report({"doc_id": "d", "source": "s", "n_pages": 1},
                           [page])["truncated_pages"] == []
        assert assemble([page]).strip().endswith(text)

    @pytest.mark.parametrize("note", [
        "(nothing to transcribe: only the page number 7)", "(blank)", "[illegible]",
        "(page number only)", "(No text in this region.)", "(The image is blank.)",
        "(the transcription, or nothing)", "(Nothing visible.)", "(nothing else)",
        "(empty region)", "(page number 7)", "(Page number: 12)", "(only the page number 7)",
        "(No transcription needed)", "(The region contains only the page number.)",
        "(nothing here but the page number)", "(empty)",
    ])
    def test_note_is_not_a_transcription(self, note):
        b, _ = review(tail(), reply("fixed", note))
        assert b.meta["reviewed"] == "rejected" and b.content == ""
        assert b.history[-1]["decision"] == "rejected: not a transcription"
        assert b.flags == ["empty", "truncated"]


# ---------------------------------------------------------------- v3-gate-6


def panel(names, edges_, xshift=None):
    nodes = "\n".join(rf"  \node[circle, draw] ({v}) at ({k},0) {{${lab}$}};"
                      for k, (v, lab) in enumerate(names))
    draws = "\n".join(rf"  \draw ({u}) -- ({v});" for u, v in edges_)
    body = nodes + "\n" + draws
    return (rf"\begin{{scope}}[xshift={xshift}]" + "\n" + body + "\n\\end{scope}"
            if xshift else body)


G = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")]


@pytest.fixture
def figure_page(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "figures").mkdir()
    Image.new("RGB", (400, 400), "white").save(tmp_path / "pages" / "p0001.png")
    Image.new("RGB", (100, 100), "white").save(tmp_path / "figures" / "f.png")
    fig = Block(type="figure", bbox=[0.1, 0.1, 0.5, 0.5], meta={"image": "figures/f.png"})
    return Page(doc_id="d", index=0, image=str(tmp_path / "pages" / "p0001.png"),
                width=400, height=400, blocks=[fig])


def graph_reply(code):
    return ("KIND: graph\n<description>\n(a) A path $G$; (b) its complement.\n</description>\n"
            f"<structure>\n{code}\n</structure>")


class TestPanels:
    TWO = ("\\begin{tikzpicture}\n" + panel(G, [("1", "2"), ("2", "3")])
           + "\n\\end{tikzpicture}\n\\quad\n\\begin{tikzpicture}\n"
           + panel(G, [("1", "3"), ("2", "4")]) + "\n\\end{tikzpicture}")
    SAME_NAMES = ("\\begin{tikzpicture}\n" + panel(G, [("1", "2")], "0cm") + "\n"
                  + panel(G, [("1", "3")], "4cm") + "\n\\end{tikzpicture}")
    UNIQUE = ("\\begin{tikzpicture}\n"
              + panel([("a" + v, lab) for v, lab in G], [("a1", "a2")], "0cm") + "\n"
              + panel([("b" + v, lab) for v, lab in G], [("b1", "b3")], "4cm")
              + "\n\\end{tikzpicture}")

    def test_problems_ask_for_unique_names(self):
        _, flags, problems = parse_graph(self.TWO)
        assert flags == ["tikz_parse"] and "more than one tikzpicture" in problems[0]
        assert "vertex names unique across panels" in problems[0]
        _, flags, problems = parse_graph(self.SAME_NAMES)
        assert flags == ["tikz_parse"]
        assert "duplicate vertex names: 1, 2, 3, 4." in problems[0]
        assert "unique across panels" in problems[0]

    def test_prompt_asks_for_unique_names(self):
        assert "vertex names unique across panels" in DESCRIBE_PROMPT

    def test_repair_round_with_unique_names(self, figure_page):
        prompts = []
        answers = [graph_reply(self.TWO), graph_reply(self.UNIQUE)]
        srv = FakeServer(lambda p, n: prompts.append(p) or answers[len(prompts) - 1])
        describe_figures(srv.client(), [figure_page], "reviewer:fake")
        fig = figure_page.blocks[0]
        assert "vertex names unique across panels" in prompts[1]
        assert fig.flags == [] and len(fig.meta["graph"]["nodes"]) == 8
        assert [n["label"] for n in fig.meta["graph"]["nodes"]].count("$1$") == 2
        assert "8 vertices, 2 edges (undirected)" in fig.meta["description"]


# ---------------------------------------------------------------- v3-gate-7


def portrait_page():
    page = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(page).rectangle((0, 0, 99, 49), fill="black")      # marks the top-left
    return page


class TestTurnedPage:
    def test_same_turns_as_the_reader(self):
        from src.readers import markdown
        assert _TURN == markdown._TURN

    @pytest.mark.parametrize("rotation,size,dark", [
        (None, (850, 1100), (10, 10)), (90, (1100, 850), (10, 840)),
        (180, (850, 1100), (840, 1090)), (270, (1100, 850), (1090, 10)),
    ])
    def test_reviewer_sees_the_turned_page(self, rotation, size, dark):
        meta = {"rotation": rotation} if rotation is not None else {}
        b = block("formula", r"\sum_{i=1}^n x_i", bbox=None, **meta)
        b, srv = review(b, reply("correct", "", "latex"), img=portrait_page())
        img = sent_image(srv)
        assert img.size == size and img.convert("L").getpixel(dark) < 50

    def test_olmocr_page_end_to_end(self):
        replies = iter([
            "---\nprimary_language: en\nis_rotation_valid: false\nrotation_correction: 90\n"
            "is_table: true\nis_diagram: false\n---\n",
            "---\nprimary_language: en\nis_rotation_valid: true\nrotation_correction: 0\n"
            "is_table: true\nis_diagram: false\n---\nTable 3: results.\n\n$$\n\\sum_{i=1}^n x_i"
            "\n$$\n"])
        reader_srv = FakeServer(lambda p, n: next(replies))
        page = portrait_page()
        blocks = OlmOCRReader(reader_srv.client()).read(page)
        reader_size = sent_image(reader_srv).size
        formula = next(b for b in blocks if b.type == "formula")
        assert formula.meta["rotation"] == 90 and formula.bbox is None
        formula.flags = validate_block(formula)
        _, srv = review(formula, reply("correct", "", "latex"), img=page)
        seen = sent_image(srv).size
        assert seen == (1100, 850)
        assert seen[0] / seen[1] == pytest.approx(reader_size[0] / reader_size[1], rel=0.01)
