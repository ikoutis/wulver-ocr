"""Round-3 fixes to the core, the SLURM script and the env setup: a folder
in an @listfile, inputs that cannot be read, a partly purged WORK, a rerun
with another profile into the same OUT, a review-only --force after a purge,
the CUDA 12 flavours vLLM really publishes, a USR1 while the login profile
loads, and the choice between read attempts (findings v3-ops-1..5,
reopen-v2-slurm-6, and the run_ocr side of v3-readers-2 / v3-readers-7 /
v3-gate-1: the "repeated" flag)."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from conftest import FakeServer
from src import ingest as ing
from src import run_ocr
from src.schema import Block
from src.validate import validate_block
from test_fix_ops2 import run_setup
from test_fix_slurm import SBATCH, Fakes, finish, make_docs, markdown, wait_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import incomplete  # noqa: E402

needs_bash = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("curl")),
                                reason="needs bash and curl")

DOTS_REPLY = json.dumps([{"bbox": [100, 100, 1500, 300], "category": "Text",
                          "text": "Hello from the reader."}])


@pytest.fixture
def fakes(tmp_path):
    f = Fakes(tmp_path)
    yield f
    f.cleanup()


@pytest.fixture
def servers(monkeypatch):
    """A reader (whose replies a test can swap) and an editor that agrees."""
    s = {"r": FakeServer(lambda p, n: DOTS_REPLY),
         "e": FakeServer(lambda p, n: "VERDICT: correct\n<text>\nunchanged\n</text>")}
    monkeypatch.setattr(run_ocr, "ChatClient", lambda url, *a, **kw:
                        s["r" if url.endswith("8001") else "e"].client())
    return s


def png(path, shade=0, size=(32, 32)) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (shade, 0, 0)).save(path)
    return str(path)


def incomplete_ids(capsys, *argv) -> str:
    capsys.readouterr()
    incomplete.main(list(argv))
    return capsys.readouterr().out.strip()


def run(capsys, *argv) -> tuple[int, str]:
    capsys.readouterr()
    rc = run_ocr.main(list(argv))
    return rc, capsys.readouterr().out


def pixels(path) -> bytes:
    with Image.open(path) as im:
        return im.convert("RGB").tobytes()


# ------------------------------------------------- v3-ops-1: listfile folders


def test_a_folder_in_a_listfile_is_walked_not_a_system_failure(tmp_path, capsys):
    """`ls -d papers/*` or `find papers` puts folders in a listfile. They are
    walked as on the command line; at e52f952 a folder was skipped, and at
    round 2 its EISDIR made ingest exit 3 on every submission (v3-ops-1)."""
    a = png(tmp_path / "papers" / "2019" / "a.png", 1)
    c = png(tmp_path / "papers" / "2019" / "venue" / "c.png", 3)
    b = png(tmp_path / "more" / "b.png", 2)
    lst = tmp_path / "list.txt"
    lst.write_text(f"{tmp_path / 'papers'}\n{tmp_path / 'papers' / '2019'}\n# a note\n{b}\n")
    assert sorted(ing.discover([f"@{lst}"])) == sorted([a, b, c])

    work, out = tmp_path / "work", tmp_path / "out"
    rc, log = run(capsys, "ingest", "--inputs", f"@{lst}", "--work", str(work),
                  "--out", str(out))
    assert rc == 0, log
    assert "SYSTEM FAILURE" not in log and "ingested 3 document(s)" in log
    assert sorted(d.split("-")[0] for d in os.listdir(work)) == ["a", "b", "c"]
    assert incomplete_ids(capsys, "--inputs", f"@{lst}", "--nshards", "2",
                          "--out", str(out)) == "0-1"


def test_an_input_that_cannot_be_read_is_skipped_not_a_system_failure(
        tmp_path, monkeypatch, capsys):
    """An input whose read fails (a damaged file: EIO) is the input's
    problem: skipped with a line naming it, as tools/incomplete.py skips it,
    not exit 3 forever. A system errno while hashing (no file handles left)
    is still the system's (v3-ops-1)."""
    a = png(tmp_path / "in" / "a.png", 1)
    b = png(tmp_path / "in" / "b.png", 2)
    real = ing.sha256_file

    def damaged_b(path):
        if path == b:
            raise OSError(errno.EIO, "Input/output error", path)
        return real(path)

    monkeypatch.setattr(ing, "sha256_file", damaged_b)
    work, out = tmp_path / "work", tmp_path / "out"
    argv = ["ingest", "--inputs", a, b, "--work", str(work), "--out", str(out)]
    rc, log = run(capsys, *argv)
    assert rc == 0, log
    assert f"INGEST ERROR {b}: cannot read it" in log and "SYSTEM FAILURE" not in log
    assert [d.split("-")[0] for d in os.listdir(work)] == ["a"]
    assert not os.path.exists(out) or not os.listdir(out)          # nothing marked

    def no_handles(path):
        raise OSError(errno.EMFILE, "Too many open files", path)

    monkeypatch.setattr(ing, "sha256_file", no_handles)
    rc, log = run(capsys, *argv)
    assert rc == run_ocr.EXIT_SERVER and "SYSTEM FAILURE" in log


def test_an_ingest_error_on_a_directory_is_not_the_systems():
    err = IsADirectoryError(errno.EISDIR, "Is a directory")
    assert not run_ocr._system_error(err, os.path.dirname(os.path.abspath(__file__)))
    assert run_ocr._system_error(OSError(errno.ENOSPC, "No space left on device"), "/nope")
    assert not run_ocr._system_error(OSError("cannot identify image file"), __file__)


@needs_bash
def test_batch_with_a_folder_in_the_listfile_completes(fakes, tmp_path, capsys):
    """Through the real sbatch script: a listfile naming a folder and a file
    processes both, and recovery finds nothing left (v3-ops-1)."""
    make_docs(tmp_path / "papers" / "2019", n=2)
    other = png(tmp_path / "more" / "x.png", 5, size=(850, 1100))
    lst = tmp_path / "list.txt"
    lst.write_text(f"{tmp_path / 'papers' / '2019'}\n{other}\n")
    out = tmp_path / "out"
    p = fakes.sbatch(fakes.env(INPUTS=f"@{lst}", OUT=out, WORK=tmp_path / "work",
                               SLURM_ARRAY_JOB_ID=1000, SLURM_JOB_ID=1001,
                               SLURM_ARRAY_TASK_ID=0, SLURM_ARRAY_TASK_COUNT=1,
                               SLURM_ARRAY_TASK_MAX=0),
                     tmp_path / "run.out")
    assert finish(p) == 0, (tmp_path / "run.out").read_text()
    assert sorted(markdown(out)) == ["doc0", "doc1", "x"]
    assert incomplete_ids(capsys, "--inputs", f"@{lst}", "--nshards", "1",
                          "--out", str(out)) == ""


# --------------------------------------------- v3-ops-2: a partly purged WORK


def test_ingest_renders_lost_page_images_again(tmp_path, pdf_path):
    """/scratch purges file by file: a manifest can outlive its page images.
    ingest renders the lost ones again, at the manifest's dpi (v3-ops-2)."""
    work = tmp_path / "work"
    man = ing.ingest(pdf_path, str(work), dpi=100)
    pages = work / man["doc_id"] / "pages"
    before = [pixels(pages / f"p000{k}.png") for k in (1, 2)]
    os.remove(pages / "p0002.png")
    assert ing.missing_pages(str(work / man["doc_id"]), man) == ["pages/p0002.png"]
    assert ing.ingest(pdf_path, str(work), dpi=300) == man          # not re-rendered at 300
    assert [pixels(pages / f"p000{k}.png") for k in (1, 2)] == before
    shutil.rmtree(pages)
    assert ing.ingest(pdf_path, str(work)) == man
    assert [pixels(pages / f"p000{k}.png") for k in (1, 2)] == before
    assert ing.missing_pages(str(work / man["doc_id"]), man) == []


def test_finished_document_with_lost_page_images_is_not_redone(tmp_path, pdf_path, servers):
    """manifest.json and review/ are read on every resubmission, pages/ and
    read/ once: under an access-time purge they go first. A resubmission then
    saved 'OCR failed' placeholders over the finished Markdown, and the
    --retry-failed recovery did the same. Such a document now counts as
    purged: not redone, unless the reading is forced (v3-ops-2)."""
    work, out = tmp_path / "w", tmp_path / "o"
    argv = ["all", "--inputs", pdf_path, "--work", str(work), "--out", str(out), "--workers", "1"]
    assert run_ocr.main(argv) == 0
    (doc,) = os.listdir(out)
    md = out / doc / f"{doc}.md"
    md.write_text("finished\n")                 # to see whether it is rewritten
    reads = len(servers["r"].requests)
    shutil.rmtree(work / doc / "pages")
    shutil.rmtree(work / doc / "read")
    for extra in ([], ["--retry-failed"]):
        assert run_ocr.main([*argv, *extra]) == 0
        assert len(servers["r"].requests) == reads and md.read_text() == "finished\n"
        assert not os.path.exists(work / doc / "pages")
    assert run_ocr.main([*argv, "--force"]) == 0
    assert len(servers["r"].requests) == reads + 2
    text = md.read_text()
    assert "Hello from the reader." in text and "OCR failed" not in text
    assert json.loads((out / doc / "report.json").read_text())["failed_pages"] == []


def test_unfinished_document_with_lost_page_images_is_read_properly(tmp_path, pdf_path,
                                                                     servers):
    """Not finished yet: its lost pages are rendered again and read, never
    saved as failed placeholders (v3-ops-2)."""
    work, out = tmp_path / "w", tmp_path / "o"
    common = ["--inputs", pdf_path, "--work", str(work), "--out", str(out), "--workers", "1"]
    assert run_ocr.main(["ingest", *common]) == 0
    (doc,) = os.listdir(work)
    os.remove(work / doc / "pages" / "p0001.png")
    assert run_ocr.main(["all", *common]) == 0
    report = json.loads((out / doc / "report.json").read_text())
    assert report["failed_pages"] == [] and report["reviewed_pages"] == 2
    assert "OCR failed" not in (out / doc / f"{doc}.md").read_text()


def test_a_page_image_gone_at_read_time_stays_in_todo(tmp_path, pdf_path, servers,
                                                      monkeypatch, capsys):
    """A page image that vanishes between ingest and read is not a page the
    reader failed on: no placeholder, the page stays in todo, the stage
    exits 3, and the next run renders it again and reads it (v3-ops-2)."""
    work, out = tmp_path / "w", tmp_path / "o"
    common = ["--inputs", pdf_path, "--work", str(work), "--out", str(out), "--workers", "1"]
    real = run_ocr.select_docs

    def select_then_purge(args):
        docs = real(args)
        for d in docs:
            os.remove(os.path.join(d, "pages", "p0002.png"))
        return docs

    monkeypatch.setattr(run_ocr, "select_docs", select_then_purge)
    rc, log = run(capsys, "read", *common)
    assert rc == run_ocr.EXIT_SERVER, log
    assert "SYSTEM FAILURE" in log and "1 page image(s) gone" in log
    (doc,) = os.listdir(work)
    assert sorted(os.listdir(work / doc / "read")) == ["p0001.json"]
    monkeypatch.setattr(run_ocr, "select_docs", real)
    rc, log = run(capsys, "todo", "--stage", "read", *common)
    assert log.strip().splitlines()[-1] == "1"
    assert run_ocr.main(["all", *common]) == 0
    assert json.loads((out / doc / "report.json").read_text())["failed_pages"] == []


# ------------------------------------- v3-ops-3: another profile, the same OUT


def test_a_rerun_into_the_same_out_from_another_work_says_so(tmp_path, pdf_path, servers,
                                                             capsys):
    """OUT is the completion record, so a run with another profile (another
    WORK) into the same OUT does nothing. That is no longer silent, and the
    log no longer blames a purge (v3-ops-3)."""
    out = tmp_path / "out"
    first = ["--inputs", pdf_path, "--out", str(out), "--workers", "1"]
    assert run_ocr.main(["all", *first, "--work", str(tmp_path / "default")]) == 0
    (doc,) = os.listdir(out)
    report = json.loads((out / doc / "report.json").read_text())
    assert report["work"] == str(tmp_path / "default")
    reads = len(servers["r"].requests)

    rc, log = run(capsys, "all", *first, "--work", str(tmp_path / "dots"))
    assert rc == 0 and len(servers["r"].requests) == reads
    assert "with no complete work dir under" in log and "purged" not in log
    assert "made under another WORK" in log and "needs its own OUT" in log

    shutil.rmtree(tmp_path / "default")         # a real purge of the same WORK
    rc, log = run(capsys, "all", *first, "--work", str(tmp_path / "default"))
    assert rc == 0 and len(servers["r"].requests) == reads
    assert "already finished" in log and "another WORK" not in log


def test_sbatch_header_says_another_profile_needs_its_own_out():
    head = Path(SBATCH).read_text().split("#SBATCH", 1)[0]
    assert "really re-reads" not in head
    assert re.search(r"another profile\s+#?\s*needs its own OUT", head)


# ------------------------------------ v3-ops-4: a review-only --force, purged


def test_review_or_assemble_force_does_not_revive_a_purged_document(tmp_path, pdf_path,
                                                                    servers, capsys):
    """REVIEW_ARGS=--force used to re-ingest purged documents without reading
    them; the next plain run then read, reviewed and rewrote them. Only a
    forced read bypasses the purge rule (v3-ops-4)."""
    work, out = tmp_path / "w", tmp_path / "o"
    common = ["--inputs", pdf_path, "--work", str(work), "--out", str(out), "--workers", "1"]
    assert run_ocr.main(["all", *common]) == 0
    (doc,) = os.listdir(out)
    md = out / doc / f"{doc}.md"
    md.write_text("finished\n")
    reads = len(servers["r"].requests)
    shutil.rmtree(work)
    for cmd in (["todo", "--stage", "review", "--force"], ["review", "--force"],
                ["assemble", "--force"]):
        rc, log = run(capsys, *cmd, *common)
        assert rc == 0, log
        assert not os.path.exists(work / doc), cmd
    assert run_ocr.main(["all", *common]) == 0
    assert len(servers["r"].requests) == reads and md.read_text() == "finished\n"
    rc, log = run(capsys, "todo", "--stage", "read", "--force", *common)
    assert log.strip().splitlines()[-1] == "2"            # a forced read redoes it


# ------------------------------------------- v3-ops-5: CUDA 12 flavours


@needs_bash
def test_setup_refuses_a_cuda12_flavour_with_no_published_wheel(fakes, tmp_path):
    """vLLM 0.30.0 publishes its CUDA 12 wheel as +cu129 only (+cu128 is a
    404): cu128 is refused before the env is made, unless the wheel itself
    is given (v3-ops-5)."""
    rc, text, uv = run_setup(fakes, tmp_path, "570.86.10", WOCR_TORCH_BACKEND="cu128")
    assert rc == 2 and "+cu129 only" in text
    assert not any("vllm" in ln for ln in uv) and not (tmp_path / "env").exists()

    wheel = tmp_path / "vllm-0.30.0+cu128-cp38-abi3-manylinux_2_28_x86_64.whl"
    rc, text, uv = run_setup(fakes, tmp_path, "570.86.10", WOCR_TORCH_BACKEND="cu128",
                             WOCR_VLLM_WHEEL=wheel)
    assert rc == 0, text
    assert any(f"{wheel} --extra-index-url https://download.pytorch.org/whl/cu128" in ln
               for ln in uv)

    head = Path(ROOT, "tools", "setup_env.sh").read_text().split("set -euo", 1)[0]
    assert "WOCR_TORCH_BACKEND=cu130 | cu129 (" in head and "| cu128" not in head


# ------------------------- reopen-v2-slurm-6: USR1 during the login profile


def test_sbatch_is_not_a_login_shell():
    """`#!/bin/bash -l` loaded the login profile before the script's first
    line could trap USR1; the script loads it itself, after the trap."""
    lines = Path(SBATCH).read_text().splitlines()
    assert lines[0] == "#!/bin/bash"
    trap = next(i for i, ln in enumerate(lines) if ln.startswith("trap 'WOCR_SIGNALLED=1' USR1"))
    profile = next(i for i, ln in enumerate(lines) if "/etc/profile ] &&" in ln)
    module = next(i for i, ln in enumerate(lines) if ln.startswith("module load"))
    assert trap < profile < module


def spool_copy(tmp_path) -> Path:
    spool = tmp_path / "spool"
    spool.mkdir()
    shutil.copy(SBATCH, spool / "slurm_script")
    os.chmod(spool / "slurm_script", 0o755)
    return spool / "slurm_script"


@needs_bash
def test_usr1_while_the_login_profile_loads_requeues(fakes, tmp_path):
    """A USR1 during a slow login profile (a conda init hook in ~/.bashrc)
    killed the task with no requeue; now it requeues. The spool copy is
    exec'd through its shebang, as slurmstepd does (reopen-v2-slurm-6)."""
    mark = tmp_path / "in_profile"
    profile = tmp_path / "profile.sh"
    profile.write_text(f'touch "{mark}"\nsleep 1.5\n')
    make_docs(tmp_path / "papers", n=1)
    out = tmp_path / "run.out"
    p = fakes.start([str(spool_copy(tmp_path))],
                    fakes.env(INPUTS=tmp_path / "papers", OUT=tmp_path / "out",
                              WORK=tmp_path / "work", SLURM_ARRAY_JOB_ID=1000,
                              SLURM_JOB_ID=1001, SLURM_ARRAY_TASK_ID=0,
                              SLURM_ARRAY_TASK_COUNT=1, SLURM_ARRAY_TASK_MAX=0,
                              SLURM_SUBMIT_DIR=ROOT, WOCR_LOGIN_PROFILE=profile),
                    out)
    end = time.time() + 30
    while not mark.exists() and time.time() < end:
        time.sleep(0.02)
    os.kill(p.pid, signal.SIGUSR1)
    text = wait_for(out, "=== requeueing 1000_0", timeout=20)
    assert "ingested" not in text
    # scontrol runs just after that line is printed
    assert wait_for(fakes.scontrol_log, "\n", timeout=20).split("\n")[0] == "requeue 1000_0"
    assert p.poll() is None                     # alive, awaiting the requeue kill


@needs_bash
def test_login_profile_is_loaded_by_a_batch_task_only(fakes, tmp_path):
    """A batch task loads the login profile (as `bash -l` did; one written
    for an interactive shell, failing under -eu, does not stop it). `bash
    slurm/ocr.sbatch` in your own shell does not, as before."""
    mark = tmp_path / "loaded"
    profile = tmp_path / "profile.sh"
    profile.write_text(f'false\necho "$WOCR_NO_SUCH_VARIABLE" >/dev/null\n'
                       f'echo loaded >> "{mark}"\n')
    make_docs(tmp_path / "papers", n=1)
    env = dict(INPUTS=tmp_path / "papers", OUT=tmp_path / "out", WORK=tmp_path / "work",
               PHASES="assemble", WOCR_LOGIN_PROFILE=profile)
    p = fakes.start([str(spool_copy(tmp_path))],
                    fakes.env(**env, SLURM_JOB_ID=1001, SLURM_SUBMIT_DIR=ROOT),
                    tmp_path / "batch.out")
    assert finish(p) == 0, (tmp_path / "batch.out").read_text()
    assert mark.read_text() == "loaded\n"
    p = fakes.start(["bash", SBATCH], fakes.env(**env, SLURM_JOB_ID=1002),
                    tmp_path / "inplace.out")
    assert finish(p) == 0, (tmp_path / "inplace.out").read_text()
    assert mark.read_text() == "loaded\n"
    p = fakes.start([str(tmp_path / "spool" / "slurm_script")],
                    fakes.env(**dict(env, WOCR_LOGIN_PROFILE=0), SLURM_JOB_ID=1003,
                              SLURM_SUBMIT_DIR=ROOT),
                    tmp_path / "none.out")
    assert finish(p) == 0, (tmp_path / "none.out").read_text()
    assert mark.read_text() == "loaded\n"


# ------------------- the read attempt kept (v3-readers-2, v3-readers-7/gate-1)


class ScriptedReader:
    """A reader whose attempt k returns blocks made from attempts[k]."""

    name = "scripted"
    tag = "reader:scripted:m"

    def __init__(self, attempts):
        self.attempts, self.calls = attempts, []

    def read(self, img, attempt=0):
        self.calls.append(attempt)
        return [Block(**dict(kw, meta=dict(kw.get("meta", {}))))
                for kw in self.attempts[attempt]]


def text(content, bbox=None, **meta):
    return {"type": "text", "content": content, "bbox": bbox, "meta": meta}


def tail(bbox):
    return {"type": "text", "content": "", "bbox": bbox, "meta": {"truncated_tail": True}}


@pytest.fixture
def doc_dir(tmp_path):
    src = png(tmp_path / "in" / "page.png", 0, size=(200, 260))
    man = ing.ingest(src, str(tmp_path / "work"))
    return str(tmp_path / "work" / man["doc_id"])


def read_one(doc_dir, attempts):
    reader = ScriptedReader(attempts)
    page = run_ocr._read_one(reader, doc_dir, 0, retries=len(attempts) - 1)
    return page, reader.calls


def test_a_repeated_element_still_gets_the_page_reread(doc_dir, monkeypatch):
    """validate maps meta["repeated"] to its own flag "repeated" (not
    degenerate at the gate); _read_one still re-reads such a page once and
    keeps the clean reading, or the first of equals (v3-readers-7,
    v3-gate-1)."""
    def flags(b):                   # the "repeated" flag, whichever branch maps it
        f = set(validate_block(b))
        if (b.meta.get("repeated") or 0) >= 2:
            f = f - {"repetition"} | {"repeated"}
        return sorted(f)

    monkeypatch.setattr(run_ocr, "validate_block", flags)
    page, calls = read_one(doc_dir, [[text("Fine text.", repeated=3)], [text("Fine text.")]])
    assert calls == [0, 1] and page.meta["reader_attempt"] == 1
    assert [b.flags for b in page.blocks] == [[]]
    page, calls = read_one(doc_dir, [[text("Fine text.", repeated=3)],
                                     [text("Fine text.", repeated=2)]])
    assert calls == [0, 1] and page.meta["reader_attempt"] == 0
    assert page.blocks[0].flags == ["repeated"]


def test_the_attempt_that_lost_least_is_kept(doc_dir):
    """A left-column cut leaves two tail regions and used to count 2, so an
    attempt cut in the title (one tail, nearly the whole page) won; a cut now
    counts once, then the lost area decides. A complete page with one
    repeated element beats an attempt cut early (v3-readers-2)."""
    left_cut = [text("Title", [0, 0, 1, 0.08]), text("Abstract.", [0, 0.1, 1, 0.25]),
                text("Left column.", [0, 0.3, 0.48, 0.6]),
                tail([0, 0.6, 0.5, 1]), tail([0.5, 0.3, 1, 1])]
    title_cut = [text("Title", [0, 0, 1, 0.08]), tail([0, 0.1, 1, 1])]
    page, _ = read_one(doc_dir, [left_cut, title_cut])
    assert page.meta["reader_attempt"] == 0 and len(page.blocks) == 5
    page, _ = read_one(doc_dir, [title_cut, left_cut])
    assert page.meta["reader_attempt"] == 1

    complete = [text("Title", [0, 0, 1, 0.08]),
                text("Body, written twice by the model.", [0, 0.1, 1, 0.9], repeated=2)]
    page, _ = read_one(doc_dir, [complete, title_cut])
    assert page.meta["reader_attempt"] == 0

    boxless = [text("Some text."), tail(None)]          # a box-less tail: the whole page
    boxed = [text("Some text.", [0, 0, 1, 0.5]), tail([0, 0.5, 1, 1])]
    page, _ = read_one(doc_dir, [boxless, boxed])
    assert page.meta["reader_attempt"] == 1


def test_attempt_loss_is_zero_only_for_a_clean_reading():
    def loss(*blocks):
        bs = [Block(**kw) for kw in blocks]
        for b in bs:
            b.flags = validate_block(b)
        return run_ocr._attempt_loss(bs)

    assert loss(text("Fine.", [0, 0, 1, 1])) == (0, 0)
    assert loss(text("Fine."), tail([0, 0.5, 1, 1]), tail([0.5, 0.2, 1, 0.5])) == (1, 0.65)
    assert loss(text("Fine.", repeated=2))[0] == 1
