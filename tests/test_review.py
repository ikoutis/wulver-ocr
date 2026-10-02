import pytest
from PIL import Image

import src.validate as validate
from conftest import FakeServer
from src.assemble import render_block
from src.backend import TRUNCATION_MARKER
from src.katex_check import UNCHECKED
from src.review import ReviewPolicy, change_fraction, crop, gate, parse_reply, review_block
from src.schema import Block
from src.validate import validate_block

IMG = Image.new("RGB", (1000, 1400), "white")


def formula(content, bbox=(0.2, 0.3, 0.8, 0.35)):
    b = Block(type="formula", content=content, bbox=list(bbox), source="reader:x")
    b.flags = validate_block(b)
    return b


def reply(verdict, body, tag="latex"):
    return f"VERDICT: {verdict}\n<{tag}>\n{body}\n</{tag}>"


@pytest.fixture
def fake_katex(monkeypatch):
    """A stand-in KaTeX that knows no \\mathbbm (nor does the real one)."""
    monkeypatch.setattr(validate, "katex_error", lambda tex, display=True:
                        "Undefined control sequence: \\mathbbm" if r"\mathbbm" in tex else None)


class TestParse:
    def test_basic(self):
        assert parse_reply(reply("fixed", r"x_{i}^{2}"), "formula") == ("fixed", "x_{i}^{2}")

    def test_strips_delims_and_tolerates_missing_close(self):
        v, c = parse_reply("VERDICT: correct\n<latex>\n$$ a+b $$", "formula")
        assert (v, c) == ("correct", "a+b")

    def test_no_tag(self):
        assert parse_reply("I think it is fine", "formula") == ("fixed", None)

    def test_verdict_in_markdown_emphasis(self):
        assert parse_reply("**VERDICT:** unreadable\n<latex>\n</latex>", "formula") \
            == ("unreadable", "")
        assert parse_reply("VERDICT: **correct**\n<latex>\nx\n</latex>", "formula")[0] == "correct"
        assert parse_reply("**Verdict**: `unreadable`", "formula")[0] == "unreadable"

    def test_fence_inside_the_tags_is_dropped(self):
        assert parse_reply(reply("fixed", "```latex\nx_{i}^{2}\n```"), "formula") \
            == ("fixed", "x_{i}^{2}")
        assert parse_reply(reply("fixed", "```markdown\nLet $x$ be.\n```", "text"), "text") \
            == ("fixed", "Let $x$ be.")
        table = "<table><tr><td>1</td></tr></table>"
        assert parse_reply(reply("fixed", f"```html\n{table}\n```", "table_out"), "table") \
            == ("fixed", table)

    def test_fenced_answer_without_tags(self):
        r = "VERDICT: fixed\nThe draft has y_1:\n```\ny_1\n```\nIt should be:\n```latex\ny_{i}\n```"
        assert parse_reply(r, "formula") == ("fixed", "y_{i}")

    def test_fenced_draft_keeps_its_fence(self):
        code = "```\nfor i in range(n):\n    x\n```"
        assert parse_reply(reply("fixed", code, "text"), "text", draft=code) == ("fixed", code)


class TestGate:
    P = ReviewPolicy()

    def test_small_fix_accepted(self):
        ok, _ = gate(formula(r"\lambda_2(L) \leq \frac{d}{n}"),
                     r"\lambda_2(L) \geq \frac{d}{n}", self.P)
        assert ok

    def test_rewrite_rejected(self):
        ok, why = gate(formula(r"\lambda_2(L) \leq \frac{d}{n}"),
                       r"\mu \in \mathbb{R}^{k \times k}, \quad \|M\|_F = 1", self.P)
        assert not ok and "change" in why

    def test_new_flags_rejected(self):
        ok, why = gate(formula(r"\frac{a}{b}"), r"\frac{a}{b", self.P)
        assert not ok and "latex_braces" in why

    def test_fixing_flags_accepted(self):
        b = formula(r"\frac{a}{b")
        assert b.flags == ["latex_braces"]
        ok, _ = gate(b, r"\frac{a}{b}", self.P)
        assert ok

    def test_degenerate_draft_accepts_reread(self):
        b = formula(r"\cdot " * 300)
        assert "repetition" in b.flags
        ok, _ = gate(b, r"x \cdot y", self.P)
        assert ok

    def test_truncation_marker_never_accepted(self):
        b = formula(r"E = mc^{2}" + TRUNCATION_MARKER)
        assert b.flags == ["truncated"]                 # degenerate: no change bound
        for proposal in (r"E = mc^{2} <<TRUNCATED>>", r"E = mc^{2}" + TRUNCATION_MARKER):
            assert gate(b, proposal, self.P) == (False, "truncated proposal")

    def test_structural_fix_with_a_katex_unknown_macro(self, fake_katex):
        # the draft's KaTeX status was never computed: fixing its brace does
        # not "introduce" the KaTeX error it had all along
        b = formula(r"\mathbbm{1}\{x>0\} \frac{a}{b")
        assert b.flags == ["latex_braces"]
        assert gate(b, r"\mathbbm{1}\{x>0\} \frac{a}{b}", self.P)[0]
        ok, why = gate(formula(r"1\{x>0\} + \frac{a}{b}"),
                       r"\mathbbm{1}\{x>0\} + \frac{a}{b}", self.P)
        assert not ok and "latex_katex" in why

    def test_unchecked_proposal_rejected(self, monkeypatch):
        b = formula(r"\frac{a}{b} + 1")
        assert b.flags == []
        monkeypatch.setattr(validate, "katex_error", lambda tex, display=True: UNCHECKED)
        ok, why = gate(b, r"\frac{a}{b} + \mathbbm{1}", self.P)
        assert not ok and "latex_unchecked" in why

    def test_line_break_group_fix_accepted(self):
        b = formula(r"\begin{aligned} a &= b \\ &= c \end{aligned}")
        assert gate(b, r"\begin{aligned} a &= b \\{}&= c \end{aligned}", self.P)[0]

    def test_table_with_equal_cells_keeps_the_bound(self):
        t = Block(type="table", content="| Method | A | B | C | D | E | F |\n"
                  "|---|---|---|---|---|---|---|\n| Ours | 1 | 2 | 3 | 4 | 5 | 6 |\n"
                  "| Random | 0 | 0 | 0 | 0 | 0 | 0 |")
        t.flags = validate_block(t)
        assert t.flags == []
        ok, why = gate(t, "| a | b |\n|---|---|\n| 1 | 2 |", self.P)
        assert not ok and "change" in why

    def test_change_fraction(self):
        assert change_fraction("a b  c", "a b c") == 0.0
        assert change_fraction("abc", "xyz") == 1.0


class TestCrop:
    def test_crop_pads_and_clips(self):
        c = crop(IMG, [0.0, 0.0, 0.5, 0.5], 0.01)
        assert c.size[0] > 500 - 1 and c.size[0] < 540

    def test_no_bbox_full_page(self):
        assert crop(IMG, None, 0.01).size == IMG.size


LOSS = (r"\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_{i}"
        r" + (1 - y_{i}) \log (1 - \hat{y}_{i}) \right] + \lambda \|\theta\|_{2}^{2}")


class TestReviewBlock:
    def run(self, block, answer):
        srv = FakeServer(lambda p, n: answer)
        return review_block(srv.client(), IMG, block, ReviewPolicy(), "reviewer:fake"), srv

    def test_agreed_keeps_content(self):
        b, srv = self.run(formula("a+b"), reply("correct", "a+b"))
        assert b.content == "a+b" and b.meta["reviewed"] == "agreed"
        assert b.source == "reader:x"
        msg = srv.requests[0]["messages"][0]["content"]
        assert msg[0]["type"] == "image_url" and "<draft>\na+b\n</draft>" in msg[1]["text"]

    def test_accepted_edit_keeps_history(self):
        b, _ = self.run(formula(r"x^{2} + y_{1}"), reply("fixed", r"x^{2} + y_{i}"))
        assert b.content == r"x^{2} + y_{i}" and b.source == "reviewer:fake"
        assert b.history[-1]["previous"] == r"x^{2} + y_{1}"
        assert b.meta["reviewed"] == "edited"

    def test_rejected_edit_recorded(self):
        b, _ = self.run(formula(r"x^{2} + y_{1}"),
                        reply("fixed", r"\int_0^\infty e^{-t^2} dt = \sqrt{\pi}/2"))
        assert b.content == r"x^{2} + y_{1}" and b.meta["reviewed"] == "rejected"
        assert b.history[-1]["proposal"].startswith(r"\int")

    def test_unreadable(self):
        b, _ = self.run(formula("a+b"), reply("unreadable", ""))
        assert "unreadable" in b.flags and b.content == "a+b"

    def test_unreadable_whatever_comes_with_it(self):
        for answer in (reply("unreadable", r"x^{2} + y_{1}"),     # the draft echoed
                       "VERDICT: unreadable\nThe crop is too blurry to read."):
            b, _ = self.run(formula(r"x^{2} + y_{1}"), answer)
            assert b.flags == ["unreadable"] and b.meta["reviewed"] == "unreadable"
            assert b.content == r"x^{2} + y_{1}"

    def test_fenced_answer_lands_without_the_fence(self):
        b, _ = self.run(formula(LOSS), reply("fixed", "```latex\n"
                                             + LOSS.replace("y_i", "y_{i}") + "\n```"))
        assert b.meta["reviewed"] == "edited" and b.content == LOSS.replace("y_i", "y_{i}")
        assert "```" not in render_block(b)

    def test_marker_never_shown_or_accepted(self):
        b = formula(r"E = mc^{2}" + TRUNCATION_MARKER)
        b, srv = self.run(b, reply("fixed", r"E = mc^{2} <<TRUNCATED>>"))
        prompt = srv.requests[0]["messages"][0]["content"][1]["text"]
        assert "<draft>\nE = mc^{2}\n</draft>" in prompt and "TRUNCATED>>" not in prompt
        assert b.meta["reviewed"] == "rejected" and "truncated" in b.history[-1]["decision"]

    def test_reviewer_reply_cut_off_is_rejected(self):
        srv = FakeServer(lambda p, n: ("VERDICT: fixed\n<latex>\nx \\cdot", "length"))
        b = review_block(srv.client(), IMG, formula(r"\cdot " * 300), ReviewPolicy(), "r:f")
        assert b.meta["reviewed"] == "rejected"
        assert b.history[-1]["decision"] == "rejected: truncated proposal"

    def test_empty_tail_block_is_transcribed(self):
        tail = Block(type="text", content="", bbox=[0.0, 0.62, 1.0, 1.0], source="reader:x",
                     meta={"truncated_tail": True})
        tail.flags = validate_block(tail)
        assert ReviewPolicy().wants(tail)
        b, srv = self.run(tail, reply("fixed", "The rest of the page, with $x_{i}$.", "text"))
        prompt = srv.requests[0]["messages"][0]["content"][1]["text"]
        assert "Transcribe" in prompt and "<draft>" not in prompt and "<text>" in prompt
        assert b.content == "The rest of the page, with $x_{i}$."
        assert b.meta["reviewed"] == "edited" and b.flags == []

    def test_empty_tail_block_answers(self):
        def tail():
            b = Block(type="text", content="", bbox=[0.0, 0.9, 1.0, 1.0],
                      meta={"truncated_tail": True})
            b.flags = validate_block(b)
            return b
        b, _ = self.run(tail(), reply("correct", "", "text"))        # nothing there
        assert b.meta["reviewed"] == "agreed" and b.flags == ["empty", "truncated"]
        # "correct" means nothing to transcribe, whatever came with it
        b, _ = self.run(tail(), reply("correct", "A last line.", "text"))
        assert b.meta["reviewed"] == "agreed" and b.content == ""
        assert b.flags == ["empty", "truncated"]
        b, _ = self.run(tail(), reply("fixed", "A last line.", "text"))
        assert b.meta["reviewed"] == "edited" and b.content == "A last line."
        assert b.meta["tail_recovered"]
        b, _ = self.run(tail(), reply("unreadable", "", "text"))
        assert b.flags == ["empty", "truncated", "unreadable"]

    def test_unchecked_draft_is_rechecked(self, fake_katex):
        b = Block(type="formula", content=r"\mathbbm{1} + x", bbox=[0.1, 0.1, 0.9, 0.2],
                  flags=["latex_unchecked"])
        b, srv = self.run(b, reply("fixed", r"\mathbb{1} + x"))
        prompt = srv.requests[0]["messages"][0]["content"][1]["text"]
        assert "flagged: latex_katex." in prompt
        assert b.meta["reviewed"] == "edited" and b.flags == []

    def test_table_prompt(self):
        t = Block(type="table", content="| a | b |\n|---|---|\n| 1 |", bbox=[0, 0, 1, 1])
        t.flags = validate_block(t)
        b, _ = self.run(t, reply("fixed", "| a | b |\n|---|---|\n| 1 | 2 |", "table_out"))
        assert b.flags == [] and b.meta["reviewed"] == "edited"

    def test_policy_selection(self):
        p = ReviewPolicy()
        assert p.wants(formula("a"))
        assert not p.wants(Block(type="text", content="fine"))
        assert p.wants(Block(type="text", content="x", flags=["inline_math"]))
        assert not p.wants(Block(type="figure"))
        assert not ReviewPolicy(flagged=False).wants(
            Block(type="text", content="x", flags=["inline_math"]))
        # KaTeX down at read time is no reason to review a block on its own
        assert not p.wants(Block(type="text", content="$x$", flags=["latex_unchecked"]))
        assert p.wants(Block(type="formula", content="x", flags=["latex_unchecked"]))
        # a failed page's placeholder is read again later, not transcribed here
        assert not p.wants(Block(type="other", content="", flags=["page_failed"]))
