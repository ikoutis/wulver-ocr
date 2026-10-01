from src.backend import TRUNCATION_MARKER
from src.schema import Block
from src.validate import (check_inline_math, check_latex, check_table,
                          has_repetition, strip_math_delims, validate_block)


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

    def test_env_mismatch(self):
        assert "latex_env" in check_latex(r"\begin{aligned} a \end{array}")
        assert "latex_env" in check_latex(r"\begin{cases} a")

    def test_left_right(self):
        assert "latex_leftright" in check_latex(r"\left( x")

    def test_stray_delimiters(self):
        assert "latex_delims" in check_latex(r"$x$ + y")
        assert "latex_delims" in check_latex(r"\[ x \]")
        assert check_latex(r"\$5") == []

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


class TestRepetition:
    def test_loop_detected(self):
        assert has_repetition("Some text. " + r"\cdot " * 300)
        assert has_repetition("intro " + "the graph Laplacian " * 100)

    def test_normal_text_not_flagged(self):
        text = ("Spectral sparsification approximates a graph by a sparse one "
                "whose Laplacian quadratic form is within a factor of the original. ") * 3
        assert not has_repetition(text)

    def test_short_text_never_flagged(self):
        assert not has_repetition("a a a a a a a a")


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

    def test_empty(self):
        assert validate_block(Block(type="formula", content="")) == ["empty"]

    def test_figure_needs_no_content(self):
        assert validate_block(Block(type="figure")) == []
