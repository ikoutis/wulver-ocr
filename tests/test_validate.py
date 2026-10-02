import random

import pytest

import src.validate as validate
from src.backend import TRUNCATION_MARKER
from src.katex_check import UNCHECKED
from src.schema import Block
from src.validate import (check_inline_math, check_latex, check_table,
                          has_repetition, strip_math_delims, table_repetition,
                          validate_block)


@pytest.fixture
def katex_calls(monkeypatch):
    """A stand-in KaTeX that parses everything and records what it was asked."""
    calls = []
    monkeypatch.setattr(validate, "katex_error",
                        lambda tex, display=True: calls.append((tex.strip(), display)))
    return calls


class TestLatex:
    def test_clean_formulas_pass(self):
        for body in [r"\frac{a}{b} + \sqrt{x_{i}^{2}}",
                     r"\left( \sum_{i=1}^{n} x_i \right)^2 \tag{3}",
                     r"\begin{aligned} a &= b \\ c &= d \end{aligned}",
                     r"\{ x : x \in S \}",
                     r"\begin{pmatrix} 1 & 0 \\[2pt] 0 & 1 \end{pmatrix}",
                     r"L = D - A, \quad \lambda_2(L) > 0"]:
            assert check_latex(body) == [], body

    def test_unbalanced_braces(self):
        assert "latex_braces" in check_latex(r"\frac{a}{b")
        assert "latex_braces" in check_latex(r"x}^{2")

    def test_line_break_before_a_group_is_balanced(self):
        # "\\{}" is a line break, then an empty group: not an escaped brace
        assert check_latex(r"\begin{aligned} f(x) &= a + b \\{}&\quad + c \end{aligned}") == []
        assert check_latex(r"\sum_{\substack{i < j \\}} x_{ij}") == []
        assert check_latex(r"a \\\{ b \\\}") == []          # line break, then \{ ... \}
        assert "latex_braces" in check_latex(r"a \\{ b")

    def test_env_mismatch(self):
        assert "latex_env" in check_latex(r"\begin{aligned} a \end{array}")
        assert "latex_env" in check_latex(r"\begin{cases} a")

    def test_left_right(self):
        assert "latex_leftright" in check_latex(r"\left( x")

    def test_stray_delimiters(self):
        assert "latex_delims" in check_latex(r"$x$ + y")
        assert "latex_delims" in check_latex(r"\[ x \]")
        assert check_latex(r"\$5") == []

    def test_code_fence_is_a_stray_delimiter(self):
        # KaTeX parses backticks, so only this check keeps a fence out of $$ ... $$
        assert "latex_delims" in check_latex("```latex\nx^{2}\n```")

    def test_katex_unchecked_is_flagged(self, monkeypatch):
        monkeypatch.setattr(validate, "katex_error", lambda tex, display=True: UNCHECKED)
        assert check_latex(r"\frac{a}{b}") == ["latex_unchecked"]
        assert check_latex(r"\frac{a}{b") == ["latex_braces"]     # structure still checked
        monkeypatch.setattr(validate, "katex_error", lambda tex, display=True: "bad")
        assert check_latex(r"\frac{a}{b}") == ["latex_katex"]

    def test_strip_delims(self):
        assert strip_math_delims("$$ x+1 $$") == "x+1"
        assert strip_math_delims(r"\[x\]") == "x"
        assert strip_math_delims("$x$") == "x"
        assert strip_math_delims("x") == "x"


class TestInlineMath:
    def test_balanced(self):
        assert check_inline_math(r"Let $G=(V,E)$ and $\lambda_2$ be given.") == []

    def test_odd_dollars(self):
        assert check_inline_math(r"Let $G=(V,E) be") == ["inline_math"]

    def test_escaped_dollar_ok(self):
        assert check_inline_math(r"costs \$5 and $x$") == []

    def test_inner_latex_checked(self):
        assert "latex_braces" in check_inline_math(r"see $\frac{a}{b$ here")

    def test_display_math_in_a_paragraph(self, katex_calls):
        # Chandra nests <math display="block"> in a <p>: the inline math after
        # the display block must still be paired, and checked
        text = "Let $x$ satisfy\n\n$$\nx = 1\n$$\n\nwhere $\\frac{a}{b$ holds."
        assert check_inline_math(text) == ["latex_braces"]
        katex_calls.clear()
        text = ("Let $x$ satisfy\n\n$$\n\\sum x_i = 1 \\tag{3}\n$$\n\n"
                "where Smith & Jones's $x_i \\ge 0$ for all $i$.")
        assert check_inline_math(text) == []
        assert katex_calls == [(r"\sum x_i = 1 \tag{3}", True), ("x", False),
                               (r"x_i \ge 0", False), ("i", False)]

    def test_broken_display_math_in_a_paragraph(self):
        assert check_inline_math("so $$ \\frac{a}{b $$ holds") == ["latex_braces"]


def _html_table(rows):
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


class TestRepetition:
    def test_loop_detected(self):
        assert has_repetition("Some text. " + r"\cdot " * 300)
        assert has_repetition("intro " + "the graph Laplacian " * 100)

    def test_loop_inside_a_block_detected(self):
        text = ("We now bound the second term. By Lemma 2 and the triangle inequality "
                "we obtain " + "\\cdot " * 150 + " which completes the proof of Lemma 3.")
        assert len(text) < 2000 and has_repetition(text)

    def test_legitimate_runs_not_flagged(self):
        toc = "\n".join(f"{k} Section number {k} " + ". " * 40 + str(3 * k) for k in range(1, 8))
        assert not has_repetition(toc)                 # dot leaders
        zero = (r"Z = \begin{bmatrix} " + r" \\ ".join([" & ".join(["0"] * 24)] * 3)
                + r" \end{bmatrix}")
        assert not has_repetition(zero)                # a written-out zero matrix

    def test_normal_text_not_flagged(self):
        text = ("Spectral sparsification approximates a graph by a sparse one "
                "whose Laplacian quadratic form is within a factor of the original. ") * 3
        assert not has_repetition(text)

    def test_short_text_never_flagged(self):
        assert not has_repetition("a a a a a a a a")

    def test_tables_with_repetitive_cells_not_flagged(self):
        rnd = random.Random(0)
        tables = [
            _html_table([["Method"] + [f"F{k}" for k in range(7)]]
                        + [[f"M{i}"] + [rnd.choice("✓✗–") for _ in range(7)] for i in range(30)]),
            _html_table([[rnd.choice("0001") for _ in range(10)] for _ in range(20)]),
            _html_table([[rnd.randint(0, 9) for _ in range(8)] for _ in range(60)]),
            "| Method | A | B | C | D | E | F |\n|---|---|---|---|---|---|---|\n"
            "| Ours | 1 | 2 | 3 | 4 | 5 | 6 |\n| Random | 0 | 0 | 0 | 0 | 0 | 0 |",
        ]
        for t in tables:
            assert validate_block(Block(type="table", content=t)) == [], t[:80]

    def test_table_loops_detected(self):
        assert table_repetition(_html_table([["n", "time"]] + [["8", "0.1"]] * 12))
        assert table_repetition(_html_table([["a", "1"], ["b", "2"]] * 10))
        assert table_repetition("| a | b |\n|---|---|\n" + "| 1 | 2 |\n" * 9)
        assert table_repetition(_html_table([["a", "the " * 600], ["b", "c"]]))
        b = Block(type="table", content=_html_table([["8", "0.1"]] * 12))
        assert "repetition" in validate_block(b)


class TestTables:
    def test_html_consistent(self):
        t = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
        assert check_table(t) == []

    def test_html_colspan_rowspan(self):
        t = ("<table><tr><td rowspan=2>a</td><td colspan=2>b</td></tr>"
             "<tr><td>c</td><td>d</td></tr></table>")
        assert check_table(t) == []

    def test_html_ragged(self):
        t = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
        assert check_table(t) == ["table_shape"]

    def test_gfm(self):
        assert check_table("| a | b |\n|---|---|\n| 1 | 2 |") == []
        assert check_table("| a | b |\n|---|---|\n| 1 |") == ["table_shape"]

    def test_unparseable(self):
        assert check_table("just text") == ["table_parse"]


class TestValidateBlock:
    def test_truncation_flag(self):
        b = Block(type="text", content="some words" + TRUNCATION_MARKER)
        assert "truncated" in validate_block(b)

    def test_truncated_tail_block(self):
        # the block a reader adds for the region its cut-off output lost
        b = Block(type="text", content="", bbox=[0, 0.6, 1, 1], meta={"truncated_tail": True})
        assert validate_block(b) == ["empty", "truncated"]
        b.content = "The rest of the page, transcribed by the reviewer."
        assert validate_block(b) == []

    def test_empty(self):
        assert validate_block(Block(type="formula", content="")) == ["empty"]

    def test_figure_needs_no_content(self):
        assert validate_block(Block(type="figure")) == []
