"""Regression tests for the second review round of the gate, validators,
figures and TikZ (findings v2-gate-1, 3..9, v2-e2e-4, v2-e2e-5), and for the
reader markers validate_block turns into flags (json_repaired, repeated) and
the tail_recovered marker review sets."""

import pytest
from PIL import Image

import src.validate as validate
from conftest import FakeServer
from src.assemble import render_block
from src.figures import check_structure, describe_figures, format_description, parse_description
from src.katex_check import UNCHECKED, katex_available
from src.review import ReviewPolicy, dropped_escapes, gate, review_block, same_text
from src.schema import Block, Page
from src.tikz import parse_graph
from src.validate import check_inline_math, validate_block

IMG = Image.new("RGB", (1000, 1400), "white")


def block(type_, content="", bbox=(0.1, 0.1, 0.9, 0.2), **meta):
    b = Block(type=type_, content=content, bbox=list(bbox) if bbox else None,
              source="reader:x", meta=meta)
    b.flags = validate_block(b)
    return b


def tail():
    return block("text", bbox=(0.0, 0.9, 1.0, 1.0), truncated_tail=True)


def review(b, answer, policy=None):
    prompts = []
    srv = FakeServer(lambda p, n: prompts.append(p) or answer)
    review_block(srv.client(), IMG, b, policy or ReviewPolicy(), "reviewer:fake")
    return b, prompts


def reply(verdict, body, tag="text"):
    return f"VERDICT: {verdict}\n<{tag}>\n{body}\n</{tag}>"


class TestTailTranscription:                    # v2-gate-1, contract 3
    @pytest.mark.parametrize("body", ["...", "7", "N/A", "A last line.",
                                      "(nothing to transcribe: only the page number 7)"])
    def test_correct_means_nothing_to_transcribe(self, body):
        b, _ = review(tail(), reply("correct", body))
        assert b.meta["reviewed"] == "agreed" and b.content == ""
        assert b.flags == ["empty", "truncated"] and "tail_recovered" not in b.meta
        assert b.history[-1]["decision"] == "kept (reviewer: nothing to transcribe)"
        assert b.history[-1]["proposal"] == body          # kept for the audit

    @pytest.mark.parametrize("body", ["...", "7", "- 7 -", "N/A", "None.", "Nothing to transcribe.",
                                      "(nothing to transcribe: only the page number 7)",
                                      "[blank]", "(the transcription, or nothing)"])
    def test_placeholder_is_not_a_transcription(self, body):
        b, _ = review(tail(), reply("fixed", body))
        assert b.meta["reviewed"] == "rejected" and b.content == ""
        assert b.history[-1]["decision"] == "rejected: not a transcription"
        assert b.flags == ["empty", "truncated"] and "tail_recovered" not in b.meta

    def test_template_shows_no_placeholder(self):
        _, prompts = review(tail(), reply("correct", ""))
        assert "<text>\n(the transcription, or nothing)\n</text>" in prompts[0]
        assert "\n...\n" not in prompts[0]
        assert "Answer VERDICT: fixed with the transcription." in prompts[0]

    def test_accepted_transcription_is_marked_recovered(self):
        text = "The proof of Lemma 7 ends here, since $x_{7} = 0$."
        b, _ = review(tail(), reply("fixed", text))
        assert b.meta["reviewed"] == "edited" and b.content == text and b.flags == []
        assert b.meta["tail_recovered"] is True and b.meta["truncated_tail"] is True

    def test_real_text_with_brackets_or_numbers_passes(self):
        for text in ("(i) the set is empty (see Lemma 2)", "12 3 4", "Nothing is known here."):
            assert gate(tail(), text, ReviewPolicy())[0], text

    def test_only_tails_are_marked(self):
        b, _ = review(block("formula", r"\cdot " * 300, bbox=(0.1, 0.1, 0.9, 0.2)),
                      reply("fixed", r"x \cdot y", "latex"))
        assert b.meta["reviewed"] == "edited" and "tail_recovered" not in b.meta


class TestNoOpAnswers:                          # v2-e2e-5
    F = r"c_n = \frac{a_n}{2} \tag{3.1}"

    @pytest.mark.parametrize("body", [f"$${F}$$", r"c_n=\frac{a_n}{2}\tag{3.1}",
                                      r"c_n  =  \frac{a_n}{2}   \tag{3.1}"])
    def test_same_formula_is_agreement(self, body):
        b, _ = review(block("formula", self.F), reply("fixed", body, "latex"))
        assert b.meta["reviewed"] == "agreed" and b.content == self.F
        assert b.source == "reader:x" and b.history[-1]["verdict"] == "fixed"
        assert b.history[-1]["decision"] == "kept (reviewer agreed)"

    def test_no_proposal_is_still_no_proposal(self):
        b, _ = review(block("formula", self.F), "I cannot tell.")
        assert b.meta["reviewed"] == "no-proposal"
        b, _ = review(block("formula", self.F), "VERDICT: correct")
        assert b.meta["reviewed"] == "agreed"

    def test_whitespace_that_matters(self):
        assert not same_text(r"\alpha b", r"\alphab", "formula")
        assert not same_text(r"a\ b", r"a\b", "formula")
        assert same_text(r"\frac{a}{b} + c", r"\frac{a}{b}+c", "formula")
        assert same_text("Let  $x$ be\n given.", "Let $x$ be\ngiven. ", "text")
        assert not same_text("Two items: - a - b", "Two items:\n- a\n- b", "text")

    def test_line_structure_fix_is_an_edit(self):
        b, _ = review(block("list", "- first $x_1$ - second $x_2$"),
                      reply("fixed", "- first $x_1$\n- second $x_2$"))
        assert b.meta["reviewed"] == "edited" and b.content == "- first $x_1$\n- second $x_2$"


class TestEscapes:                              # v2-gate-5
    DRAFT = (r"We store results in a List&lt;String> and run the A\* search, where "
             r"a\*b and c\*d are products, with cost $x^{2$ per step.")

    def test_prompt_asks_to_keep_escapes(self):
        _, prompts = review(block("text", self.DRAFT), reply("correct", ""))
        assert r"Keep the draft's Markdown escapes (\*, \_, \$, \#)" in prompts[0]

    def test_dropping_escapes_is_rejected(self):
        b = block("text", self.DRAFT)
        assert b.flags == ["latex_braces"]
        fixed = self.DRAFT.replace("x^{2$", "x^{2}$")
        dropped = fixed.replace("&lt;", "<").replace(r"\*", "*")
        b, _ = review(b, reply("fixed", dropped))
        assert b.meta["reviewed"] == "rejected" and b.content == self.DRAFT
        assert b.history[-1]["decision"] == r"rejected: drops Markdown escapes ['\\*', '&lt;']"
        b, _ = review(block("text", self.DRAFT), reply("fixed", fixed))
        assert b.meta["reviewed"] == "edited" and b.content == fixed

    @pytest.mark.parametrize("draft, proposal, dropped", [
        (r"a\*b and **c**", "a*b and **c**", [r"\*"]),
        (r"snake\_case", "snake_case", [r"\_"]),
        ("in List&lt;T> we", "in List<T> we", ["&lt;"]),
        ("the &amp;copy; sign", "the &copy; sign", ["&amp;"]),
        ("Year\n2016\\. was good", "Year\n2016. was good", [r"\."]),
        ("x\n\\# 1 is it", "x\n# 1 is it", [r"\#"]),
        (r"where x\_1 is", "where $x_1$ is", []),               # into math: no drop
        (r"a\*b", "a times b", []),                             # reworded, not bare
        ("Smith &amp; Jones", "Smith & Jones", []),            # renders the same
    ])
    def test_dropped_escapes(self, draft, proposal, dropped):
        assert dropped_escapes(draft, proposal) == dropped

    def test_math_and_formulas_are_not_escapes(self):
        # \_ and \# inside math are LaTeX, and a formula has no Markdown escapes
        b = block("text", r"Let $a\_b + \#S$ be given, with $x^{2$.")
        assert gate(b, r"Let $a_b + |S|$ be given, with $x^{2}$.", ReviewPolicy())[0]
        b = block("formula", r"a\_b + \frac{1}{2")
        assert gate(b, r"a_b + \frac{1}{2}", ReviewPolicy())[0]

    def test_degenerate_draft_is_a_reread(self):
        b = block("text", r"a\*b " * 120)
        assert "repetition" in b.flags
        assert gate(b, "a*b, once.", ReviewPolicy())[0]


class TestPolicy:
    def test_empty_boxless_block_is_not_reviewed(self):   # v2-gate-6
        p = ReviewPolicy()
        formula = block("formula", "", bbox=None)              # split_markdown of "$$\n$$"
        assert formula.flags == ["empty"] and not p.wants(formula)
        assert not p.wants(block("text", "", bbox=None, truncated_tail=True))
        assert p.wants(block("formula", "", bbox=(0.1, 0.4, 0.9, 0.5)))
        assert p.wants(block("formula", "x^{2", bbox=None))   # a draft to proofread

    def test_unchecked_text_is_rechecked(self, monkeypatch):   # v2-gate-7
        monkeypatch.setattr(validate, "katex_error", lambda tex, display=True: UNCHECKED)
        bad = block("text", r"The indicator $\mathbbm{1}\{x>0\}$ is used.")
        good = block("caption", r"Figure 1: the map $x \mapsto x^2$.")
        assert bad.flags == good.flags == ["latex_unchecked"]
        p = ReviewPolicy()
        assert not p.wants(bad)                         # KaTeX still down: no reason
        monkeypatch.setattr(validate, "katex_error", lambda tex, display=True:
                            "Undefined control sequence" if r"\mathbbm" in tex else None)
        assert p.wants(bad) and bad.flags == ["latex_katex"]
        assert not p.wants(good) and good.flags == []

    @pytest.mark.skipif(not katex_available(), reason="node + katex not installed")
    def test_unchecked_text_is_rechecked_with_katex(self, monkeypatch):
        monkeypatch.setattr(validate, "katex_error", lambda tex, display=True: UNCHECKED)
        b = block("text", r"The indicator $\mathbbm{1}\{x>0\}$ is used.")
        assert b.flags == ["latex_unchecked"]
        monkeypatch.undo()                              # the review process's KaTeX works
        assert ReviewPolicy().wants(b) and b.flags == ["latex_katex"]


class TestUnclosedDisplayMath:                  # v2-gate-9
    def test_lone_dollars(self):
        assert check_inline_math("Let $$x = 1 and then $y$ holds.") == ["inline_math"]
        assert check_inline_math("By the lemma,\n\n$$\n\\|x\\| \\le \\rho\n\nwhere $\\rho$.") \
            == ["inline_math"]
        assert check_inline_math("$$x$$ and then $$y") == ["inline_math"]

    def test_paired_and_adjacent_math_pass(self):
        assert check_inline_math("Let $a$$b$ hold.") == []
        assert check_inline_math("Let $$x = 1$$ hold.") == []
        assert check_inline_math("Let\n\n$$\nx\n$$\n\nhold.") == []
        assert check_inline_math("It costs \\$$5$ here.") == []

    def test_unclosed_transcription_is_rejected(self):
        text = ("By the previous lemma,\n\n$$\n\\|x_{t+1} - x^*\\| \\le \\rho \\|x_t - x^*\\|"
                "\n\nwhere $\\rho = 1 - \\mu/L$, and hence the claim.")
        b, _ = review(tail(), reply("fixed", text))
        assert b.meta["reviewed"] == "rejected" and "inline_math" in b.history[-1]["decision"]
        b, _ = review(tail(), reply("fixed", text.replace("\n\nwhere", "\n$$\n\nwhere")))
        assert b.meta["reviewed"] == "edited" and b.flags == []


class TestReaderMarkers:                        # contract 2 (readers2 -> gate2)
    def test_markers_become_flags(self):
        assert block("text", "Fine text.", json_repaired=True).flags == ["json_repaired"]
        assert block("text", "Fine text.", repeated=3).flags == ["repeated"]       # [gate3]
        assert block("text", "Fine text.", repeated=1).flags == []
        assert block("figure", "", repeated=2).flags == ["repeated"]

    def test_json_repaired_is_reviewed_under_the_flagged_bound(self):
        b = block("formula", r"\frac{a}{b} + \theta", json_repaired=True)
        assert ReviewPolicy().wants(b) and b.flags == ["json_repaired"]
        ok, why = gate(b, r"\int_0^1 f(t)\, dt = \sum_k c_k", ReviewPolicy())
        assert not ok and why.startswith("change")
        b, _ = review(b, reply("fixed", r"\frac{a}{b} + \Theta", "latex"))
        assert b.meta["reviewed"] == "edited" and b.flags == []     # the reviewer's text now

    def test_repeated_is_reviewed(self):     # no longer degenerate: see test_fix_gate3
        b, _ = review(block("text", "A paragraph.", repeated=12),
                      reply("fixed", "A paragraph, read again."))
        assert b.meta["reviewed"] == "edited" and b.flags == []
        b, _ = review(block("text", "A paragraph.", repeated=12), reply("correct", ""))
        assert b.meta["reviewed"] == "agreed" and b.flags == ["repeated"]


AB = r"\node (a) at (0,0) {$a$}; \node (b) at (1,0) {$b$}; "


def edges(code):
    g, flags, problems = parse_graph(code)
    assert flags == [], problems
    return [(e["u"], e["v"], e["directed"], e["both"]) for e in g["edges"]]


FWD, NONE = ("a", "b", True, False), ("a", "b", False, False)


class TestTikzStyles:                           # v2-gate-4
    @pytest.mark.parametrize("code", [
        r"\begin{tikzpicture}[vertex/.style={circle,draw}, arc/.style={->,>=stealth}]"
        + AB + r"\draw[arc] (a) -- (b); \end{tikzpicture}",
        r"\begin{tikzpicture} \tikzset{directed/.style={-latex}}" + AB
        + r"\draw[directed] (a) -- (b); \end{tikzpicture}",
        r"\tikzset{directed/.style={-latex}} \begin{tikzpicture}" + AB
        + r"\draw[directed] (a) -- (b); \end{tikzpicture}",
        r"\begin{tikzpicture}[every edge/.style={draw,-Stealth}]" + AB
        + r"\path (a) edge (b); \end{tikzpicture}",
        r"\begin{tikzpicture} \tikzset{every edge/.append style={->}}" + AB
        + r"\path (a) edge (b); \end{tikzpicture}",
        r"\begin{tikzpicture} \tikzstyle{arrow}=[->]" + AB
        + r"\draw[arrow] (a) -- (b); \end{tikzpicture}",
        r"\begin{tikzpicture}[arc/.style={#1,->}]" + AB
        + r"\draw[arc=thick] (a) -- (b); \end{tikzpicture}",
        r"\begin{tikzpicture}[a1/.style={->}, a2/.style={a1, thick}]" + AB
        + r"\draw[a2] (a) -- (b); \end{tikzpicture}",
        r"\begin{tikzpicture}[arc/.style={->}]" + AB
        + r"\begin{scope}[arc] \draw (a) -- (b); \end{scope} \end{tikzpicture}",
    ])
    def test_arrows_set_through_a_style(self, code):
        assert edges(code) == [FWD]

    def test_later_options_win(self):
        assert edges(r"\begin{tikzpicture}[arc/.style={->}]" + AB
                     + r"\draw[arc, -] (a) -- (b); \end{tikzpicture}") == [NONE]
        assert edges(r"\begin{tikzpicture}[every edge/.style={draw,->}]" + AB
                     + r"\path (a) edge[-] (b); \end{tikzpicture}") == [NONE]
        assert edges(AB + r"\draw[->, -] (a) -- (b);") == [NONE]
        assert edges(r"\begin{tikzpicture}[every edge/.style={draw}]" + AB
                     + r"\path (a) edge[draw=none] (b); \end{tikzpicture}") == []

    def test_style_that_uses_itself_is_flagged(self):
        g, flags, problems = parse_graph(r"\begin{tikzpicture}[arc/.style={arc}]" + AB
                                         + r"\draw[arc] (a) -- (b); \end{tikzpicture}")
        assert flags == ["tikz_parse"] and "uses itself" in problems[0]

    @pytest.mark.parametrize("opt, label", [
        ("label=above:$v_1$", "$v_1$"), ("label={below left:{$v_1$}}", "$v_1$"),
        ("label={[red]90:$v_1$}", "$v_1$"), ("label=$f:A$", "$f:A$"), ("label distance=2pt", ""),
    ])
    def test_label_option(self, opt, label):
        g, flags, _ = parse_graph(rf"\node[circle, fill, {opt}] (v1) at (0,0) {{}};"
                                  r"\node (v2) at (1,0) {$w$}; \draw (v1) -- (v2);")
        assert flags == [] and [n["label"] for n in g["nodes"]] == [label, "$w$"]


TWO = r"""\begin{tikzpicture}
  \node (a) at (0,0) {$a$}; \node (b) at (2,0) {$b$}; \draw (a) -- (b);
\end{tikzpicture}
\quad
\begin{tikzpicture}
  \node (c) at (0,0) {$c$}; \node (d) at (2,0) {$d$}; \node (e) at (1,1) {$e$};
  \draw (c) -- (d); \draw (d) -- (e); \draw (e) -- (c);
\end{tikzpicture}"""
ONE = r"""\begin{tikzpicture}
  \begin{scope}
    \node (a) at (0,0) {$a$}; \node (b) at (2,0) {$b$}; \draw (a) -- (b);
  \end{scope}
  \begin{scope}[xshift=4cm]
    \node (c) at (0,0) {$c$}; \node (d) at (2,0) {$d$}; \node (e) at (1,1) {$e$};
    \draw (c) -- (d); \draw (d) -- (e); \draw (e) -- (c);
  \end{scope}
\end{tikzpicture}"""


def graph_reply(structure, kind="graph", desc="(a) an edge; (b) a triangle."):
    return f"KIND: {kind}\n<description>\n{desc}\n</description>\n{structure}"


@pytest.fixture
def figure_page(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "figures").mkdir()
    Image.new("RGB", (400, 400), "white").save(tmp_path / "pages" / "p0001.png")
    Image.new("RGB", (100, 100), "white").save(tmp_path / "figures" / "f.png")
    fig = Block(type="figure", bbox=[0.1, 0.1, 0.5, 0.5], meta={"image": "figures/f.png"})
    return Page(doc_id="d", index=0, image=str(tmp_path / "pages" / "p0001.png"),
                width=400, height=400, blocks=[fig])


def describe(page, *answers):
    prompts = []
    srv = FakeServer(lambda p, n: prompts.append(p) or answers[min(len(prompts), len(answers)) - 1])
    describe_figures(srv.client(), [page], "reviewer:fake")
    return page.blocks[0], prompts


class TestOnePicture:                           # v2-gate-3
    def test_several_pictures_are_flagged(self):
        g, flags, problems = parse_graph(TWO)
        assert g is None and flags == ["tikz_parse"]
        assert "more than one tikzpicture" in problems[0]

    def test_drawing_outside_the_picture_is_flagged(self):
        _, flags, problems = parse_graph(r"\begin{tikzpicture}" + AB + r"\end{tikzpicture}"
                                         r" \draw (a) -- (b);")
        assert flags == ["tikz_parse"] and "outside" in problems[0]

    def test_multi_panel_figure_is_sent_back_and_redrawn(self, figure_page):
        fig, prompts = describe(figure_page, graph_reply(f"<structure>\n{TWO}\n</structure>"),
                                graph_reply(f"<structure>\n{ONE}\n</structure>"))
        assert len(prompts) == 2 and "more than one tikzpicture" in prompts[1]
        assert fig.flags == [] and len(fig.meta["graph"]["nodes"]) == 5
        assert "5 vertices, 4 edges (undirected)" in fig.meta["description"]

    def test_still_several_pictures_has_no_markdown_version(self, figure_page):
        fig, prompts = describe(figure_page, graph_reply(f"<structure>\n{TWO}\n</structure>"))
        assert len(prompts) == 2 and fig.flags == ["tikz_parse"]
        assert "graph" not in fig.meta and "not available" in fig.meta["description"]
        assert "vertices, " not in fig.meta["description"]


class TestMissingStructure:                     # v2-gate-8, v2-e2e-4
    def test_bare_fence_is_the_structure(self):
        fenced = f"```latex\n{ONE}\n```"
        for answer in (graph_reply(fenced), graph_reply("**Structure:**\n" + fenced)):
            info, flags, problems = check_structure(parse_description(answer))
            assert flags == [] and info["summary"] == "(a) an edge; (b) a triangle."
            assert len(info["graph"]["nodes"]) == 5
            assert "**Graph — TikZ:**" in format_description(info, problems)

    def test_fence_without_closing_description(self):
        info = parse_description(f"KIND: graph\n<description>\nAn edge.\n```latex\n{ONE}\n```")
        assert info["summary"] == "An edge." and info["structure"] == ONE

    def test_fence_inside_a_closed_description_stays_there(self):
        info = parse_description("KIND: plot\n<description>\nSee:\n```\nx\n```\n</description>")
        assert info["structure"] == "" and "```" in info["summary"]

    @pytest.mark.parametrize("structure", ["<structure>\n</structure>", ""])
    def test_graph_without_tikz_is_sent_back_and_flagged(self, figure_page, structure):
        fig, prompts = describe(figure_page, graph_reply(structure, desc="A dense graph."))
        assert len(prompts) == 2
        assert "no TikZ was given" in prompts[1] and "Previous TikZ:\n(none)" in prompts[1]
        assert fig.flags == ["tikz_missing"] and "graph" not in fig.meta
        assert "**Graph — Markdown (simple):** *(not available: no TikZ was given)*" \
            in fig.meta["description"]
        assert "Figure description (generated)</summary>" in render_block(fig)

    def test_repair_round_recovers(self, figure_page):
        fig, prompts = describe(figure_page, graph_reply(""),
                                graph_reply(f"<structure>\n{ONE}\n</structure>"))
        assert len(prompts) == 2 and fig.flags == [] and len(fig.meta["graph"]["nodes"]) == 5

    def test_commutative_diagram_without_structure(self):
        _, flags, problems = check_structure(parse_description(
            graph_reply("", kind="commutative_diagram")))
        assert flags == ["tikz_missing"] and "tikz-cd" in problems[0]

    def test_other_kinds_need_no_structure(self):
        assert check_structure(parse_description(graph_reply("", kind="plot")))[1] == []
        assert parse_description(graph_reply("```\nnoise\n```", kind="photo"))["structure"] == ""
        info = parse_description("KIND: graph\n<description>A path.</description>\n" + "x" * 5
                                 + "<<TRUNCATED>>")
        assert check_structure(info)[1] == ["description_truncated"]
