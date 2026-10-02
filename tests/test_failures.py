"""Failure semantics: server down, rejected requests, stops mid-review,
unreadable pages, unreadable inputs (see src/run_ocr.py's module docstring)."""

import json
import os
import stat

import httpx
import pytest

from conftest import FakeServer
from src import run_ocr
from src.backend import RequestRejected, ServerError, strip_thinking
from src.schema import atomic_write_text

READER_JSON = json.dumps([
    {"bbox": [100, 100, 1500, 300], "category": "Text", "text": "Some text."},
    {"bbox": [300, 350, 1300, 450], "category": "Formula", "text": r"L = D - A_{1}"},
    {"bbox": [200, 1200, 1000, 1800], "category": "Picture"},
])
FIXED = "VERDICT: fixed\n<latex>\nL = D - A\n</latex>"
PLOT = "KIND: plot\n<description>A plot.</description>\n<structure></structure>"


def editor_ok(prompt, n):
    return PLOT if "KIND:" in prompt else FIXED


def down(prompt, n):
    raise httpx.ConnectError("connection refused")


@pytest.fixture
def servers(monkeypatch):
    state = {"reader": FakeServer(lambda p, n: READER_JSON, model="r"),
             "editor": FakeServer(editor_ok, model="e")}

    def factory(url, model=None, timeout=0, **kw):
        return state["reader" if url.endswith("8001") else "editor"].client()

    monkeypatch.setattr(run_ocr, "ChatClient", factory)
    yield state
    run_ocr.STOP.clear()


def run(*argv):
    return run_ocr.main(list(argv))


def args(pdf, tmp_path, *more):
    return ["--inputs", pdf, "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o"),
            "--workers", "1", *more]


def review_dir(tmp_path):
    (doc,) = os.listdir(tmp_path / "w")
    return tmp_path / "w" / doc / "review"


class TestReviewServerDown:
    def test_pages_not_saved_and_exit_3_then_resume(self, tmp_path, pdf_path, servers):
        assert run("read", *args(pdf_path, tmp_path)) == 0
        servers["editor"].responder = down
        assert run("review", *args(pdf_path, tmp_path)) == run_ocr.EXIT_SERVER
        assert not review_dir(tmp_path).exists()           # nothing saved as reviewed
        a = run_ocr.build_parser().parse_args(["todo", "--stage", "review", *args(pdf_path, tmp_path)])
        assert run_ocr.stage_todo(a) == 2
        servers["editor"].responder = editor_ok             # server back
        assert run("review", *args(pdf_path, tmp_path)) == 0
        assert sorted(os.listdir(review_dir(tmp_path))) == ["p0001.json", "p0002.json"]
        page = json.load(open(review_dir(tmp_path) / "p0001.json"))
        assert all("error" not in str(h) for b in page["blocks"] for h in b["history"])

    def test_circuit_breaker_fails_fast(self):
        calls = []

        def counting_down(p, n):
            calls.append(1)
            raise httpx.ConnectError("refused")
        srv = FakeServer(counting_down)
        c = srv.client(breaker=3)
        for _ in range(6):
            with pytest.raises(ServerError):
                c.chat("x")
        assert len(calls) == 3 + 0      # /models is not counted; after 3, no more requests


class TestDeterministicErrors:
    def test_rejected_review_is_final_and_reported(self, tmp_path, pdf_path, servers):
        servers["editor"].responder = lambda p, n: httpx.Response(
            400, json={"message": "maximum context length is 32768 tokens"}) \
            if "display equation" in p else PLOT
        assert run("all", *args(pdf_path, tmp_path)) == 0
        page = json.load(open(review_dir(tmp_path) / "p0001.json"))
        f = [b for b in page["blocks"] if b["type"] == "formula"][0]
        assert f["meta"]["reviewed"] == "error"
        assert "maximum context length" in f["history"][-1]["decision"]   # body kept
        (doc,) = os.listdir(tmp_path / "o")
        rep = json.load(open(tmp_path / "o" / doc / "report.json"))
        assert rep["review_errors"] == 2 and rep["reviewed_pages"] == 2

    def test_4xx_raises_with_body(self):
        srv = FakeServer(lambda p, n: httpx.Response(400, text='{"message": "too many images"}'))
        with pytest.raises(RequestRejected, match="too many images"):
            srv.client().chat("x")


class TestStopMidReview:
    def test_finished_pages_saved_rest_requeued(self, tmp_path, pdf_path, servers):
        assert run("read", *args(pdf_path, tmp_path)) == 0
        seen = []

        def stop_after_two(p, n):
            seen.append(1)
            if len(seen) == 2:          # page 1's figure: page 1 is now complete
                run_ocr.STOP.set()
            return editor_ok(p, n)
        servers["editor"].responder = stop_after_two
        assert run("review", *args(pdf_path, tmp_path)) == run_ocr.EXIT_REQUEUE
        assert os.listdir(review_dir(tmp_path)) == ["p0001.json"]
        assert len(seen) == 2           # no request started after the stop
        page = json.load(open(review_dir(tmp_path) / "p0001.json"))
        assert "Stopped" not in json.dumps(page)


class TestReadFailures:
    def test_deterministic_read_failure_becomes_placeholder(self, tmp_path, pdf_path, servers):
        calls = []

        def reject(p, n):
            calls.append(1)
            return httpx.Response(400, text="image too large")
        servers["reader"].responder = reject
        assert run("all", *args(pdf_path, tmp_path)) == 0
        assert len(calls) == 4          # 2 pages x (first try + one retry)
        (doc,) = os.listdir(tmp_path / "o")
        rep = json.load(open(tmp_path / "o" / doc / "report.json"))
        assert rep["failed_pages"] == [1, 2] and rep["open_flags"] == {"page_failed": 2}
        # --retry-failed reads them again once the problem is fixed
        servers["reader"].responder = lambda p, n: READER_JSON
        assert run("read", *args(pdf_path, tmp_path, "--retry-failed")) == 0
        page = json.load(open(tmp_path / "w" / doc / "read" / "p0001.json"))
        assert not page["meta"].get("failed") and len(page["blocks"]) == 3

    def test_server_down_during_read(self, tmp_path, pdf_path, servers):
        servers["reader"].responder = down
        assert run("read", *args(pdf_path, tmp_path)) == run_ocr.EXIT_SERVER
        (doc,) = os.listdir(tmp_path / "w")
        assert not (tmp_path / "w" / doc / "read").exists()

    def test_stop_preferred_over_server_error(self, tmp_path, pdf_path, servers):
        def killed(p, n):
            run_ocr.STOP.set()          # preemption: SIGTERM to us and the server
            raise httpx.ConnectError("refused")
        servers["reader"].responder = killed
        assert run("read", *args(pdf_path, tmp_path)) == run_ocr.EXIT_REQUEUE


class TestIngestFailure:
    def test_unreadable_input_marked_failed(self, tmp_path, pdf_path, servers):
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"%PDF-1.4 not really a pdf")
        assert run("ingest", "--inputs", str(bad), pdf_path, "--work", str(tmp_path / "w"),
                   "--out", str(tmp_path / "o")) == 0
        marks = [d for d in os.listdir(tmp_path / "o")
                 if os.path.exists(tmp_path / "o" / d / "FAILED.json")]
        assert len(marks) == 1 and marks[0].startswith("broken-")
        info = json.load(open(tmp_path / "o" / marks[0] / "FAILED.json"))
        assert info["stage"] == "ingest" and info["source"].endswith("broken.pdf")


class TestSmallThings:
    def test_strip_thinking_bare_close(self):
        assert strip_thinking("reasoning...</think>\nVERDICT: correct") == "VERDICT: correct"

    def test_atomic_write_permissions(self, tmp_path):
        p = tmp_path / "x.json"
        atomic_write_text(str(p), "{}")
        mode = stat.S_IMODE(os.stat(p).st_mode)
        assert mode & 0o044 and p.read_text() == "{}"
        assert [f for f in os.listdir(tmp_path)] == ["x.json"]    # no temp left behind

    def test_duplicate_inputs_selected_once(self, tmp_path, pdf_path):
        a = run_ocr.build_parser().parse_args(
            ["ingest", "--inputs", pdf_path, pdf_path, "--work", str(tmp_path / "w")])
        assert len(run_ocr.select_docs(a)) == 1
