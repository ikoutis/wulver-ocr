from PIL import Image

from conftest import FakeServer
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


class TestParse:
    def test_basic(self):
        assert parse_reply(reply("fixed", r"x_{i}^{2}"), "formula") == ("fixed", "x_{i}^{2}")

    def test_strips_delims_and_tolerates_missing_close(self):
        v, c = parse_reply("VERDICT: correct\n<latex>\n$$ a+b $$", "formula")
        assert (v, c) == ("correct", "a+b")

    def test_no_tag(self):
        assert parse_reply("I think it is fine", "formula") == ("fixed", None)


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

    def test_change_fraction(self):
        assert change_fraction("a b  c", "a b c") == 0.0
        assert change_fraction("abc", "xyz") == 1.0


class TestCrop:
    def test_crop_pads_and_clips(self):
        c = crop(IMG, [0.0, 0.0, 0.5, 0.5], 0.01)
        assert c.size[0] > 500 - 1 and c.size[0] < 540

    def test_no_bbox_full_page(self):
        assert crop(IMG, None, 0.01).size == IMG.size


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
