"""End-to-end: ingest -> read -> review -> assemble against fake servers,
plus resume and signal-stop behaviour."""

import json
import os

import pytest

from conftest import FakeServer
from src import run_ocr
from src.figures import parse_description

READER_JSON = json.dumps([
    {"bbox": [0, 0, 900, 40], "category": "Page-header", "text": "Running head"},
    {"bbox": [100, 100, 1500, 300], "category": "Text",
     "text": "Let $G$ be a graph whose Laplacian is"},
    {"bbox": [300, 350, 1300, 450], "category": "Formula", "text": r"L = D - A_{1}"},
    {"bbox": [200, 1200, 1000, 1800], "category": "Picture"},
    {"bbox": [200, 1810, 1000, 1850], "category": "Caption", "text": "Figure 1: K_3."},
])

DESCRIPTION = """KIND: graph
<description>
A triangle on three labelled vertices.
</description>
<structure>
a -- b
b -- c
a -- c
</structure>"""


def editor_answer(prompt, n_images):
    if "KIND:" in prompt:
        return DESCRIPTION
    if "display equation" in prompt:
        return "VERDICT: fixed\n<latex>\nL = D - A\n</latex>"
    return "VERDICT: correct\n<text>\nunchanged\n</text>"


@pytest.fixture
def servers(monkeypatch):
    reader = FakeServer(lambda p, n: READER_JSON, model="dots-fake")
    editor = FakeServer(editor_answer, model="vlm-fake")

    def factory(url, model=None, timeout=0, **kw):
        return (reader if url.endswith("8001") else editor).client()

    monkeypatch.setattr(run_ocr, "ChatClient", factory)
    return reader, editor


def run(*argv):
    return run_ocr.main(list(argv))


def test_end_to_end(tmp_path, pdf_path, servers):
    reader, editor = servers
    work, out = str(tmp_path / "work"), str(tmp_path / "out")
    assert run("all", "--inputs", pdf_path, "--work", work, "--out", out,
               "--workers", "2") == 0
    (doc_id,) = os.listdir(out)
    assert doc_id.startswith("paper_one-")
    md = open(os.path.join(out, doc_id, doc_id + ".md")).read()

    assert "Running head" not in md
    assert "$$\nL = D - A\n$$" in md                       # reviewer's fix accepted
    assert "![Figure (page 1)](figures/p0001_b03.png)" in md
    assert "```text\na -- b" in md                          # graph -> edge list
    assert os.path.exists(os.path.join(out, doc_id, "figures", "p0001_b03.png"))
    # the paragraph ending page 1 without punctuation is NOT joined to page 2,
    # because page 2 starts with a header (dropped) and then the same text
    # with a capital letter
    assert md.count("<!-- page") == 2

    rep = json.load(open(os.path.join(out, doc_id, "report.json")))
    assert rep["n_pages"] == 2 and rep["review"]["edited"] == 2
    page = json.load(open(os.path.join(work, doc_id, "review", "p0001.json")))
    f = [b for b in page["blocks"] if b["type"] == "formula"][0]
    assert f["history"][0]["previous"] == r"L = D - A_{1}"

    # 2 pages read, 2 formulas reviewed + 2 figures described
    assert len(reader.requests) == 2 and len(editor.requests) == 4


def test_resume_does_no_new_work(tmp_path, pdf_path, servers):
    reader, editor = servers
    work = str(tmp_path / "work")
    args = ["--inputs", pdf_path, "--work", work, "--workers", "1"]
    assert run("all", *args) == 0
    n_r, n_e = len(reader.requests), len(editor.requests)
    assert run("all", *args) == 0
    assert (len(reader.requests), len(editor.requests)) == (n_r, n_e)
    assert run("read", *args, "--force") == 0
    assert len(reader.requests) == n_r + 2


def test_status(tmp_path, pdf_path, servers, capsys):
    work = str(tmp_path / "work")
    run("ingest", "--inputs", pdf_path, "--work", work)
    rows = run_ocr.stage_status(run_ocr.build_parser().parse_args(
        ["status", "--work", work]))
    assert rows[0][1:] == (2, 0, 0, False)


def test_signal_stop_exits_85(tmp_path, pdf_path, servers, monkeypatch):
    work = str(tmp_path / "work")
    run("ingest", "--inputs", pdf_path, "--work", work)
    monkeypatch.setitem(run_ocr.STOP, "flag", True)
    try:
        assert run("read", "--work", work, "--workers", "1") == run_ocr.EXIT_REQUEUE
    finally:
        run_ocr.STOP["flag"] = False
    (doc,) = os.listdir(work)
    assert not os.path.exists(os.path.join(work, doc, "read"))


def test_shards_partition_documents(tmp_path, pdf_path):
    from PIL import Image
    paths = [pdf_path]
    for k in range(4):
        p = tmp_path / f"img{k}.png"
        Image.new("RGB", (100, 100), (k * 40, 0, 0)).save(p)
        paths.append(str(p))
    seen = []
    for i in range(3):
        a = run_ocr.build_parser().parse_args(
            ["ingest", "--inputs", *paths, "--work", str(tmp_path / "w"),
             "--shard", f"{i}/3"])
        seen += run_ocr.select_docs(a)
    assert len(seen) == 5 and len(set(seen)) == 5


def test_parse_description_fallbacks():
    d = parse_description("KIND: spaceship\n<description>x</description>")
    assert d == {"kind": "other", "summary": "x", "structure": ""}
    d = parse_description("KIND: diagram\n<description>y</description>\n"
                          "<structure>\n```mermaid\nflowchart LR\nA-->B\n```\n</structure>")
    assert d["structure"] == "flowchart LR\nA-->B"
