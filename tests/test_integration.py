"""Seams between the fix groups: behaviour that spans reader, review, tikz,
assembly, and the CLI (see dev-communication/log.md [O-004])."""

import json
import os

from src import run_ocr
from src.readers import READERS
from src.review import ReviewPolicy
from src.schema import Block, Page
from src.tikz import braces_balanced


def test_boxless_truncated_tail_not_reviewed_but_boxed_one_is():
    p = ReviewPolicy()
    tail = Block(type="text", content="", flags=["empty", "truncated"],
                 meta={"truncated_tail": True})
    assert not p.wants(tail)                      # markdown/olmocr: region unknown
    tail.bbox = [0, 0.6, 1, 1]
    assert p.wants(tail)                          # layout readers: crop that region


def test_tikz_brace_check_line_break_before_group():
    assert braces_balanced(r"\node (a) at (0,0) {$\begin{matrix} a \\{} b \end{matrix}$};")
    assert not braces_balanced(r"\node (a) at (0,0) {$a$;")


def test_report_lists_truncated_and_failed_pages():
    tail = Block(type="text", content="", bbox=[0, 0.5, 1, 1], flags=["empty", "truncated"],
                 meta={"truncated_tail": True})
    pages = [Page(doc_id="d", index=0, image="", width=1, height=1,
                  blocks=[Block(type="text", content="x"), tail]),
             Page(doc_id="d", index=1, image="", width=1, height=1, meta={"failed": True},
                  blocks=[Block(type="other", flags=["page_failed"])])]
    rep = run_ocr.make_report({"doc_id": "d", "source": "s", "n_pages": 2}, pages)
    assert rep["truncated_pages"] == [1] and rep["failed_pages"] == [2]


def test_stale_failed_marker_removed_after_successful_ingest(tmp_path, pdf_path):
    from src.ingest import make_doc_id, sha256_file
    doc_id = make_doc_id(pdf_path, sha256_file(pdf_path))
    marker = tmp_path / "o" / doc_id / "FAILED.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"stage": "ingest", "error": "ENOSPC"}))
    a = run_ocr.build_parser().parse_args(
        ["ingest", "--inputs", pdf_path, "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert run_ocr.select_docs(a) == [os.path.join(str(tmp_path / "w"), doc_id)]
    assert not marker.exists()


def test_every_profile_names_a_registered_adapter():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
    import stage_models
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
    for f in sorted(os.listdir(root)):
        if f.endswith(".sh"):
            prof = stage_models.read_profile(f[:-3])
            assert prof["READER_ADAPTER"] in READERS, f


def test_gate_keeps_reader_hard_line_breaks():
    # htmlmd turns <br> into a Markdown hard break; a reviewer that "cleans"
    # the backslash away would silently turn it into a soft break
    from src.readers.htmlmd import HARD_BREAK
    from src.review import dropped_escapes
    draft = f"Department of Mathematics{HARD_BREAK}NJIT, Newark"
    assert dropped_escapes(draft, "Department of Mathematics\nNJIT, Newark") == [HARD_BREAK]
    assert dropped_escapes(draft, draft) == []


def test_read_attempts_scored_by_reading_loss(tmp_path):
    # one scoring for read retries (readers.base.reading_loss): a reading
    # whose only defect is a collapsed element loop still earns one re-read
    from src.readers.base import reading_loss
    rep = Block(type="text", content="Same paragraph.", bbox=[0, 0, 1, 0.2], meta={"repeated": 3})
    from src.validate import validate_block
    rep.flags = validate_block(rep)
    assert "repeated" in rep.flags and "repetition" not in rep.flags
    assert reading_loss([rep])[:2] == (0, 1)        # not clean: worth a retry


def test_repeated_block_gets_the_clean_change_limit():
    from src.review import gate
    b = Block(type="formula", content=r"\|x_{t+1}\| \le \rho \|x_t\|", meta={"repeated": 4})
    from src.validate import validate_block
    b.flags = validate_block(b)
    ok, why = gate(b, r"\int_0^1 f(t)\,dt", ReviewPolicy())
    assert not ok and "limit 0.35" in why
