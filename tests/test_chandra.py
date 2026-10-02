import pytest
from PIL import Image

from conftest import FakeServer
from src.backend import TRUNCATION_MARKER, strip_thinking
from src.katex_check import katex_available, katex_error
from src.readers import htmlmd
from src.readers.chandra import ChandraReader, html_to_blocks, parse_bbox, scale_to_fit
from src.schema import Block
from src.validate import validate_block

PAGE_HTML = r"""<div data-bbox="40 30 960 60" data-label="Page-Header"><p>J. ACM, Vol. 12</p></div>
<div data-bbox="100 80 900 120" data-label="Section-Header"><h2>2 Spectral sparsifiers</h2></div>
<div data-bbox="100 130 900 260" data-label="Text"><p>Let <math>G=(V,E,w)</math> be a graph with Laplacian <math>L_G</math>. It costs $5 &amp; more.</p></div>
<div data-bbox="200 270 800 320" data-label="Equation-Block"><math display="block">x^T L_G x = \sum_{(u,v)\in E} w_{uv}(x_u - x_v)^2</math> (3)</div>
<div data-bbox="100 330 900 600" data-label="Image"><img alt="A path graph on four vertices."/><p>graph LR; a---b; b---c</p></div>
<div data-bbox="100 610 900 640" data-label="Caption"><p>Figure 2: The path <math>P_4</math>.</p></div>
<div data-bbox="100 650 900 800" data-label="Table"><table><tr><th rowspan="2">n</th><th colspan="2">time</th></tr><tr><td><math>\epsilon=0.1</math></td><td>a &lt; b</td></tr></table></div>
<div data-bbox="100 810 900 880" data-label="List-Group"><ol><li>first <b>bold</b></li><li>second<ul><li>nested</li></ul></li></ol></div>
<div data-bbox="100 900 900 950" data-label="Footnote"><p><sup>1</sup>Supported by NSF.</p></div>
<div data-bbox="0 0 1000 1000" data-label="Blank-Page"></div>"""


class TestHtmlToBlocks:
    def setup_method(self):
        self.blocks = html_to_blocks(PAGE_HTML, "reader:chandra:x")

    def test_types_in_order(self):
        assert [b.type for b in self.blocks] == [
            "header", "heading", "text", "formula", "figure", "caption", "table",
            "list", "footnote"]

    def test_inline_math_and_escaped_dollar(self):
        t = self.blocks[2].content
        assert t == r"Let $G=(V,E,w)$ be a graph with Laplacian $L_G$. It costs \$5 & more."
        assert validate_block(Block(type="text", content=t)) == []

    def test_equation_number_becomes_tag(self):
        assert self.blocks[3].content.endswith(r"(x_u - x_v)^2 \tag{3}")

    def test_heading_level(self):
        # <h2> is a top-level section: level 1, rendered '##' (as dots)
        assert self.blocks[1].meta["level"] == 1 and self.blocks[1].content == "2 Spectral sparsifiers"

    def test_figure_keeps_reader_description(self):
        f = self.blocks[4]
        assert f.content == "" and "path graph" in f.meta["reader_description"]
        assert "graph LR" in f.meta["reader_description"]

    def test_table_keeps_spans_and_math(self):
        t = self.blocks[6].content
        assert 'rowspan="2"' in t and 'colspan="2"' in t and "$\\epsilon=0.1$" in t
        assert "a &lt; b" in t and validate_block(Block(type="table", content=t)) == []

    def test_nested_list(self):
        assert self.blocks[7].content == "1. first **bold**\n2. second\n  - nested"

    def test_bboxes(self):
        assert self.blocks[0].bbox == [0.04, 0.03, 0.96, 0.06]
        assert parse_bbox("bad") is None and parse_bbox("10 10 5 20") is None


class TestHtmlMd:
    def test_display_math_inside_paragraph(self):
        md = htmlmd.to_markdown(htmlmd.parse(
            '<p>so that <math display="block">a=b</math> holds.</p>'))
        assert "$$\na=b\n$$" in md and md.startswith("so that")

    def test_unclosed_tags_tolerated(self):
        md = htmlmd.to_markdown(htmlmd.parse("<p>one <b>two<p>three"))
        assert "one" in md and "three" in md

    def test_pre_block(self):
        md = htmlmd.to_markdown(htmlmd.parse("<pre>for i in range(n):\n    x</pre>"))
        assert md == "```\nfor i in range(n):\n    x\n```"


class TestChandraReader:
    def test_scale_to_fit(self):
        for size in [(1700, 2200), (850, 1100), (6000, 8000)]:
            out = scale_to_fit(Image.new("RGB", size))
            w, h = out.size
            assert w % 28 == 0 and h % 28 == 0 and w * h <= 3072 * 2048
            assert abs(w / h - size[0] / size[1]) < 0.03

    def test_read_and_decoding(self):
        srv = FakeServer(lambda p, n: PAGE_HTML)
        r = ChandraReader(srv.client())
        blocks = r.read(Image.new("RGB", (1700, 2200), "white"))
        assert len(blocks) == 9
        body = srv.requests[0]
        assert body["max_tokens"] == 12384 and body["temperature"] == 0.0
        assert body["top_p"] == 0.1
        assert "Bboxes are normalized 0-1000" in body["messages"][0]["content"][1]["text"]
        r.read(Image.new("RGB", (1700, 2200)), attempt=2)
        assert srv.requests[1]["temperature"] == pytest.approx(0.4)
        assert srv.requests[1]["top_p"] == 0.95

    def test_truncation(self):
        # cut inside the Equation-Block: the three complete divs are kept as
        # they are, the cut one is dropped, and a tail block covers the rest
        srv = FakeServer(lambda p, n: (PAGE_HTML[:400], "length"))
        blocks = ChandraReader(srv.client()).read(Image.new("RGB", (1700, 2200)))
        assert [b.type for b in blocks] == ["header", "heading", "text", "text"]
        assert not any(TRUNCATION_MARKER.strip() in b.content for b in blocks)
        tail = blocks[-1]
        assert tail.meta == {"truncated_tail": True} and tail.content == ""
        assert tail.bbox == [0.0, 0.26, 1.0, 1.0]


class TestBackendExtras:
    def test_strip_thinking(self):
        assert strip_thinking("<think>hmm\n</think>\nVERDICT: correct") == "VERDICT: correct"
        assert strip_thinking("<think>ran out of budget") == ""
        assert strip_thinking("plain") == "plain"

    def test_default_extra_merged(self):
        srv = FakeServer(lambda p, n: "ok")
        c = srv.client()
        c.default_extra = {"chat_template_kwargs": {"enable_thinking": False}}
        c.chat("hi", extra={"top_p": 0.5})
        assert srv.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert srv.requests[0]["top_p"] == 0.5


@pytest.mark.skipif(not katex_available(), reason="node + katex not installed "
                    "(set WOCR_KATEX_DIR to a dir with node_modules/katex)")
class TestKatex:
    def test_parses_valid(self):
        assert katex_error(r"\frac{a}{b} + \sum_{i=1}^n x_i \tag{2}") is None
        assert katex_error(r"\begin{aligned} a &= b \\ c &= d \end{aligned}") is None

    def test_rejects_unknown_command(self):
        assert katex_error(r"\fraq{a}{b}") is not None

    def test_flag_in_validator(self):
        assert "latex_katex" in validate_block(Block(type="formula", content=r"\fraq{a}{b}"))
        assert "latex_katex" in validate_block(Block(type="text", content=r"see $\fraq{a}{b}$"))

    def test_worker_restarted_after_it_dies(self, capsys):
        from src import katex_check
        old = katex_check._SERVER._proc
        old.kill()
        old.wait()
        assert katex_error(r"\fraq{a}{b}") is not None     # asked again, of a new worker
        assert katex_error(r"\frac{a}{b}") is None
        assert katex_check._SERVER._proc is not old
        assert "restarting" in capsys.readouterr().err

    def test_display_math_in_a_paragraph(self):
        text = ("Let $x$ satisfy\n\n$$\n\\sum x_i = 1 \\tag{3}\n$$\n\n"
                "where Smith & Jones's $x_i \\ge 0$ for all $i$.")
        assert validate_block(Block(type="text", content=text)) == []

    def test_gate(self):
        from src.review import ReviewPolicy, gate
        b = Block(type="formula", content=r"\mbox{for all } x: \frac{a}{b \leq 1")
        b.flags = validate_block(b)
        assert b.flags == ["latex_braces"]          # KaTeX knows no \mbox either
        assert gate(b, r"\mbox{for all } x: \frac{a}{b} \leq 1", ReviewPolicy())[0]
        b = Block(type="formula", content=r"\begin{aligned} a &= b \\ &= c \end{aligned}")
        b.flags = validate_block(b)
        assert gate(b, r"\begin{aligned} a &= b \\{}&= c \end{aligned}", ReviewPolicy())[0]
