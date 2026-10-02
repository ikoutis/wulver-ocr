"""Round-2 fixes to the core, the SLURM scripts and the env setup: batch
retry of failed pages, shards that do not depend on how INPUTS is spelled
and never share a document, system errors at ingest, a purged WORK, recovered
truncated tails in report.json, the readiness race, the wall-clock lead, an
early USR1, the vLLM build for the GPU driver, and the interactive recipe
(findings v2-core-1/2/3, v2-slurm-1..6, v2-e2e-2/6/7, reopen-core-7,
reopen-slurm-1)."""

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

import httpx
import pytest
from PIL import Image

from conftest import FakeServer
from src import ingest as ing
from src import run_ocr
from src.schema import Block, Page
from test_fix_slurm import (FORCE_FIRST_PORT, Fakes, finish, free_port, make_docs, markdown,
                            wait_for)

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


def png(path, shade=0) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32), (shade, 0, 0)).save(path)
    return str(path)


def incomplete_ids(capsys, *argv) -> str:
    capsys.readouterr()
    incomplete.main(list(argv))
    return capsys.readouterr().out.strip()


@pytest.fixture
def servers(monkeypatch):
    """A reader (whose replies a test can swap) and an editor that agrees."""
    s = {"r": FakeServer(lambda p, n: DOTS_REPLY),
         "e": FakeServer(lambda p, n: "VERDICT: correct\n<text>\nunchanged\n</text>")}
    monkeypatch.setattr(run_ocr, "ChatClient", lambda url, *a, **kw:
                        s["r" if url.endswith("8001") else "e"].client())
    return s


def rejecting_reader():
    return FakeServer(lambda p, n: httpx.Response(400, text="maximum context length"))


# ------------------------------------------- failed pages, retried in batch


def test_todo_counts_what_the_stage_will_do(tmp_path, pdf_path, servers, capsys):
    """`todo` honours --retry-failed and --force as the stages do, so the
    sbatch gate no longer skips a phase they would redo (v2-core-1)."""
    servers["r"] = rejecting_reader()
    common = ["--inputs", pdf_path, "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    assert run_ocr.main(["all", *common, "--workers", "1"]) == 0

    def todo(*extra):
        capsys.readouterr()
        run_ocr.main(["todo", *common, *extra])
        return capsys.readouterr().out.strip().splitlines()[-1]

    assert todo("--stage", "read") == "0"                       # placeholders are read JSON
    assert todo("--stage", "read", "--retry-failed") == "2"
    assert todo("--stage", "read", "--force") == "2"
    assert todo("--stage", "review") == "0"
    assert todo("--stage", "review", "--force") == "2"
    servers["r"] = FakeServer(lambda p, n: DOTS_REPLY)
    assert run_ocr.main(["read", *common, "--retry-failed", "--workers", "1"]) == 0
    assert todo("--stage", "read", "--retry-failed") == "0"
    assert todo("--stage", "review") == "2"                     # the re-read pages


@needs_bash
def test_batch_retry_failed_rereads_placeholders(fakes, tmp_path, capsys):
    """READ_ARGS=--retry-failed, as the docs say, starts the reader and
    re-reads failed placeholders; incomplete.py --retry-failed lists such
    shards, and REVIEW_ARGS=--force really reviews again (v2-slurm-1,
    v2-e2e-2, v2-core-1)."""
    make_docs(tmp_path / "papers", n=1)
    out = tmp_path / "out"
    job = dict(INPUTS=tmp_path / "papers", OUT=out, WORK=tmp_path / "work",
               SLURM_ARRAY_JOB_ID=1000, SLURM_ARRAY_TASK_ID=0,
               SLURM_ARRAY_TASK_COUNT=1, SLURM_ARRAY_TASK_MAX=0)
    rec = ["--inputs", str(tmp_path / "papers"), "--nshards", "1", "--out", str(out)]

    p = fakes.sbatch(fakes.env(**job, SLURM_JOB_ID=1001, FAKE_READER_400=1), tmp_path / "1.out")
    assert finish(p) == 0, (tmp_path / "1.out").read_text()
    assert "OCR failed" in markdown(out)["doc0"]
    assert incomplete_ids(capsys, *rec) == ""                   # done, with a marked gap
    assert incomplete_ids(capsys, *rec, "--retry-failed") == "0"
    assert "1 with failed pages" in incomplete_ids(capsys, *rec, "--list")

    p = fakes.sbatch(fakes.env(**job, SLURM_JOB_ID=1002, READ_ARGS="--retry-failed"),
                     tmp_path / "2.out")
    assert finish(p) == 0, (tmp_path / "2.out").read_text()
    text = (tmp_path / "2.out").read_text()
    assert "=== starting chandra_ocr_2" in text and "=== starting qwen3_8_27b" in text
    assert "Read by job 1002." in markdown(out)["doc0"]
    assert "OCR failed" not in markdown(out)["doc0"]
    assert incomplete_ids(capsys, *rec, "--retry-failed") == ""

    p = fakes.sbatch(fakes.env(**job, SLURM_JOB_ID=1003, REVIEW_ARGS="--force"),
                     tmp_path / "3.out")
    assert finish(p) == 0, (tmp_path / "3.out").read_text()
    text = (tmp_path / "3.out").read_text()
    assert "=== starting chandra_ocr_2" not in text and "=== starting qwen3_8_27b" in text


# ---------------------------------------------------------------- shards


def test_relative_and_absolute_inputs_cut_the_same_shards(tmp_path, monkeypatch, capsys):
    """ocr.sbatch hands run_ocr $START/<entry>; incomplete.py gets INPUTS as
    typed. Both now order documents alike, so recovery names the shard that
    is really unfinished (v2-core-2, v2-slurm-2)."""
    submit = tmp_path / "repo"
    png(submit / "papersR" / "x.png", 1)
    png(tmp_path / "zz" / "y.png", 2)
    monkeypatch.chdir(submit)
    typed = ["papersR", str(tmp_path / "zz")]               # '/' sorts before 'p'
    absolute = [f"{submit}/papersR", str(tmp_path / "zz")]  # ocr.sbatch's abspath()
    order = [os.path.realpath(p) for p in ing.discover(absolute)]
    for spelling in (typed, ["./papersR", str(tmp_path / "zz")]):
        assert [os.path.realpath(p) for p in ing.discover(spelling)] == order

    out, work = tmp_path / "out", str(tmp_path / "work")
    a = run_ocr.build_parser().parse_args(
        ["ingest", "--inputs", *absolute, "--work", work, "--out", str(out), "--shard", "0/2"])
    for d in run_ocr.select_docs(a):                    # array task 0 finishes
        doc = os.path.basename(d)
        os.makedirs(out / doc)
        (out / doc / f"{doc}.md").write_text("x")
    rec = ["--inputs", " ".join(typed), "--nshards", "2", "--out", str(out)]
    assert incomplete_ids(capsys, *rec) == "1"


def test_identical_copies_in_two_folders_are_one_document(tmp_path, capsys):
    """Byte-identical files with one name (one doc id) are one document for
    the whole input list, so no two shards work on one doc dir (reopen-core-7).
    Same name and size with other bytes is another document."""
    first = png(tmp_path / "papers" / "2019" / "spielman.png", 7)
    os.makedirs(tmp_path / "papers" / "spectral")
    shutil.copy(first, tmp_path / "papers" / "spectral" / "spielman.png")
    png(tmp_path / "papers" / "zz" / "spielman.png", 8)
    png(tmp_path / "papers" / "zz.png", 9)
    papers = [str(tmp_path / "papers")]
    docs = ing.discover(papers)
    assert first in docs and len(docs) == 3
    assert len({ing.make_doc_id(p, ing.sha256_file(p)) for p in docs}) == 3

    out, work = tmp_path / "out", str(tmp_path / "work")
    shards = []
    for i in range(2):
        a = run_ocr.build_parser().parse_args(
            ["ingest", "--inputs", *papers, "--work", work, "--out", str(out),
             "--shard", f"{i}/2"])
        shards.append({os.path.basename(d) for d in run_ocr.select_docs(a)})
    assert not shards[0] & shards[1] and len(shards[0] | shards[1]) == 3
    for doc in shards[0]:
        os.makedirs(out / doc)
        (out / doc / f"{doc}.md").write_text("x")
    assert incomplete_ids(capsys, "--inputs", *papers, "--nshards", "2", "--out", str(out)) == "1"


# ------------------------------------------------------- ingest failures


def test_system_error_at_ingest_marks_nothing_and_exits_3(tmp_path, monkeypatch, capsys):
    """A disk-full error while ingesting is the system's, not the input's:
    no FAILED.json, the other inputs are ingested, the stage exits 3 and the
    shard stays listed for recovery. An undecodable file is still marked
    (Pillow's OSErrors carry no errno) (v2-core-3)."""
    png(tmp_path / "in" / "a.png", 1)
    png(tmp_path / "in" / "b.png", 2)
    (tmp_path / "in" / "c.png").write_bytes(b"not an image at all")
    real = ing._save_png

    def disk_full_for_b(img, dst):
        if f"{os.sep}b-" in dst:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(img, dst)

    monkeypatch.setattr(ing, "_save_png", disk_full_for_b)
    out, work = tmp_path / "out", tmp_path / "work"
    argv = ["ingest", "--inputs", str(tmp_path / "in"), "--work", str(work), "--out", str(out)]
    assert run_ocr.main(argv) == run_ocr.EXIT_SERVER
    assert "SYSTEM FAILURE" in capsys.readouterr().out
    assert [d.split("-")[0] for d in os.listdir(out)] == ["c"]      # only the bad file
    assert os.path.exists(out / os.listdir(out)[0] / "FAILED.json")
    ingested = {d.split("-")[0]: os.path.exists(work / d / "manifest.json")
                for d in os.listdir(work)}
    assert ingested["a"] and not ingested["b"]
    assert incomplete_ids(capsys, "--inputs", str(tmp_path / "in"), "--nshards", "1",
                          "--out", str(out)) == "0"
    monkeypatch.setattr(ing, "_save_png", real)                     # space freed
    assert run_ocr.main(argv) == 0


# ------------------------------------------------------------ purged WORK


def test_finished_document_is_not_redone_after_work_is_purged(tmp_path, pdf_path, servers):
    """OUT is the completion record: when /scratch purges WORK, a resubmitted
    shard does not read, review and overwrite a finished document again;
    --force does (v2-e2e-7)."""
    work, out = tmp_path / "w", tmp_path / "o"
    argv = ["all", "--inputs", pdf_path, "--work", str(work), "--out", str(out), "--workers", "1"]
    assert run_ocr.main(argv) == 0
    (doc,) = os.listdir(out)
    md = out / doc / f"{doc}.md"
    md.write_text("finished\n")                 # to see whether it is rewritten
    reads = len(servers["r"].requests)
    shutil.rmtree(work)
    assert run_ocr.main(argv) == 0
    assert len(servers["r"].requests) == reads and md.read_text() == "finished\n"
    assert not os.path.exists(work / doc)
    assert run_ocr.main([*argv, "--force"]) == 0
    assert len(servers["r"].requests) == reads + 2 and "Hello from the reader." in md.read_text()


def test_failed_pages_are_retried_after_work_is_purged(tmp_path, pdf_path, servers):
    """... but --retry-failed still re-reads a finished document whose
    report.json lists failed pages, WORK purged or not."""
    servers["r"] = rejecting_reader()
    work, out = tmp_path / "w", tmp_path / "o"
    argv = ["all", "--inputs", pdf_path, "--work", str(work), "--out", str(out), "--workers", "1"]
    assert run_ocr.main(argv) == 0
    (doc,) = os.listdir(out)
    shutil.rmtree(work)
    servers["r"] = FakeServer(lambda p, n: DOTS_REPLY)
    assert run_ocr.main(argv) == 0 and servers["r"].requests == []
    assert run_ocr.main([*argv, "--retry-failed"]) == 0
    report = json.loads((out / doc / "report.json").read_text())
    assert report["failed_pages"] == [] and len(servers["r"].requests) == 2


# ---------------------------------------------------------------- report


def test_report_lists_recovered_and_still_missing_tails():
    """report.json keeps a page whose cut-off tail the reviewer transcribed
    (truncated_recovered), so a recovery of the wrong region is never silent;
    truncated_pages still lists tails that stayed empty."""
    def page(i, *blocks):
        return Page(doc_id="d", index=i, image="", width=10, height=10, blocks=list(blocks))

    pages = [page(0, Block(type="text", content="The reviewer's transcription.",
                           meta={"truncated_tail": True, "tail_recovered": True})),
             page(1, Block(type="text", content="", meta={"truncated_tail": True})),
             page(2, Block(type="text", content="Fine."))]
    report = run_ocr.make_report({"doc_id": "d", "source": "s", "n_pages": 3}, pages)
    assert report["truncated_recovered"] == [1] and report["truncated_pages"] == [2]


# -------------------------------------------------------- serve_lib.sh


@needs_bash
def test_a_same_port_loser_is_not_ready_before_it_exits(fakes, tmp_path):
    """uvicorn logs "Application startup complete" before listen(): the task
    whose server loses a same-port race sees that line, its process alive for
    seconds, and /health answered by the winner. It must wait for its own
    listener and retry on another port (v2-slurm-4, reopen-slurm-1)."""
    port = free_port()
    procs = [fakes.bash(FORCE_FIRST_PORT,
                        fakes.env(FIRST_PORT=port, MARK=tmp_path / f"mark{job}",
                                  SLURM_JOB_ID=job, FAKE_LOAD=1.5, FAKE_UVICORN_ORDER=1,
                                  FAKE_SHUTDOWN=3),
                        tmp_path / f"{job}.out")
             for job in (7001, 7002)]
    texts = []
    for job, p in zip((7001, 7002), procs):
        rc = finish(p)
        texts.append((tmp_path / f"{job}.out").read_text())
        assert rc == 0, texts[-1]
        health = json.loads(texts[-1].split("HEALTH ", 1)[1].splitlines()[0])
        assert health["job"] == str(job)
    assert sum(f"port {port} was taken" in t for t in texts) == 1


# --------------------------------------------------------- ocr.sbatch


@needs_bash
def test_time_limit_within_the_usr1_lead_is_refused(fakes, tmp_path):
    """--time <= 30 min (+10) would get the USR1 at once and requeue forever:
    a batch task refuses it; an interactive run (no --signal) does not
    (v2-slurm-5)."""
    make_docs(tmp_path / "papers", n=1)
    env = dict(INPUTS=tmp_path / "papers", OUT=tmp_path / "out", WORK=tmp_path / "work",
               SLURM_JOB_ID=1001, PHASES="assemble", SLURM_JOB_START_TIME=1000)
    p = fakes.sbatch(fakes.env(**env, SLURM_JOB_END_TIME=1000 + 1800), tmp_path / "short.out")
    assert finish(p) == 2
    text = (tmp_path / "short.out").read_text()
    assert "--time must exceed the 30 min USR1 lead" in text and "=== wocr" not in text
    assert not fakes.scontrol_log.exists()

    p = fakes.sbatch(fakes.env(**env, SLURM_JOB_END_TIME=1000 + 4 * 3600), tmp_path / "ok.out")
    assert finish(p) == 0, (tmp_path / "ok.out").read_text()
    p = fakes.start(["bash", os.path.join(ROOT, "slurm", "ocr.sbatch")],
                    fakes.env(**env, SLURM_JOB_END_TIME=1000 + 1800), tmp_path / "inter.out")
    assert finish(p) == 0, (tmp_path / "inter.out").read_text()


@needs_bash
def test_usr1_before_requeue_lib_is_sourced_requeues(fakes, tmp_path):
    """A USR1 while `conda info --base` runs (before requeue_lib.sh installs
    its trap) used to kill the batch shell; now it requeues (v2-slurm-6)."""
    conda_base, mark = fakes.tmp / "conda", tmp_path / "in_conda"
    slow = fakes.bin / "conda"
    slow.write_text(f'#!/bin/sh\ntouch "{mark}"\nsleep 1.5\necho "{conda_base}"\n')
    make_docs(tmp_path / "papers", n=1)
    out = tmp_path / "run.out"
    p = fakes.sbatch(fakes.env(INPUTS=tmp_path / "papers", OUT=tmp_path / "out",
                               WORK=tmp_path / "work", SLURM_ARRAY_JOB_ID=1000,
                               SLURM_JOB_ID=1001, SLURM_ARRAY_TASK_ID=0,
                               SLURM_ARRAY_TASK_COUNT=1, SLURM_ARRAY_TASK_MAX=0),
                     out)
    end = time.time() + 30
    while not mark.exists() and time.time() < end:
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGUSR1)
    text = wait_for(out, "=== requeueing 1000_0", timeout=20)
    assert "ingested" not in text
    assert p.poll() is None                     # alive, awaiting the requeue kill
    assert fakes.scontrol_log.read_text().split("\n")[0] == "requeue 1000_0"


# ------------------------------------------------------- tools/setup_env.sh


def fake_python_libs(tmp_path, cuda="13.0", ops_so=False) -> Path:
    lib = tmp_path / "pylib"
    (lib / "torch").mkdir(parents=True, exist_ok=True)
    (lib / "vllm").mkdir(exist_ok=True)
    (lib / "torch" / "__init__.py").write_text(
        '__version__ = "2.13.0"\n'
        f'class version:\n    cuda = "{cuda}"\n'
        'class cuda:\n    is_available = staticmethod(lambda: False)\n')
    (lib / "vllm" / "__init__.py").write_text('__version__ = "0.30.0"\n')
    if ops_so:                  # a compiled-ops library that cannot be loaded
        (lib / "vllm" / "_C_stable_libtorch.abi3.so").write_bytes(b"not an ELF file")
    return lib


def run_setup(fakes, tmp_path, driver, *argv, **env):
    """Run tools/setup_env.sh with fake nvidia-smi / uv / pip / npm; return
    (rc, output, the uv command lines)."""
    smi = (f'#!/bin/sh\ncase "$*" in *driver_version*) echo "{driver}";; '
           f'*) echo "NVIDIA A100-SXM4-80GB, {driver}";; esac\n' if driver
           else '#!/bin/sh\necho "NVIDIA-SMI has failed: no driver" ; exit 9\n')
    uv_log = tmp_path / "uv.log"
    (fakes.tmp / "conda" / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() { if [ "$1" = create ]; then mkdir -p "$4"; fi; return 0; }\n')
    for name, text in {"nvidia-smi": smi, "uv": f'#!/bin/sh\necho "$*" >> "{uv_log}"\n',
                       "pip": '#!/bin/sh\n[ "$1" = freeze ] && echo "vllm==0.30.0"\nexit 0\n',
                       "npm": "#!/bin/sh\nexit 0\n"}.items():
        (fakes.bin / name).write_text(text)
        (fakes.bin / name).chmod(0o755)
    out = tmp_path / f"setup{len(list(tmp_path.glob('setup*.out')))}.out"
    env.setdefault("PYTHONPATH", fake_python_libs(tmp_path))
    p = fakes.start(["bash", os.path.join(ROOT, "tools", "setup_env.sh"), *argv],
                    fakes.env(WOCR_CONDA_ENV=tmp_path / "env", **env), out, cwd=tmp_path)
    rc = finish(p)
    return rc, out.read_text(), uv_log.read_text().splitlines() if uv_log.exists() else []


@needs_bash
@pytest.mark.parametrize("driver, flavour", [("590.48.01", "cu130"), ("550.54.15", "cu129"),
                                             (None, "cu130")])
def test_setup_installs_the_vllm_build_the_driver_runs(fakes, tmp_path, driver, flavour):
    """vLLM is pinned (vllm==0.30.0, whose PyPI wheels are CUDA 13); a driver
    older than 580 gets the release's +cu129 wheel with the cu129 torch
    index; with no driver visible (a login node) the CUDA 13 build comes with
    a loud note. A later run that would switch the build in place is refused
    (v2-slurm-3)."""
    rc, text, uv = run_setup(fakes, tmp_path, driver)
    assert rc == 0, text
    vllm_line = next(ln for ln in uv if "vllm" in ln)
    if flavour == "cu130":
        assert vllm_line == "pip install --no-cache vllm==0.30.0"
    else:
        assert re.search(r"releases/download/v0\.30\.0/vllm-0\.30\.0\+cu129-cp38-abi3-"
                         r"manylinux_2_28_\w+\.whl --extra-index-url "
                         r"https://download\.pytorch\.org/whl/cu129$", vllm_line), vllm_line
    assert ("No GPU driver is visible" in text) == (driver is None)
    assert (tmp_path / "env" / "wocr.flavour").read_text().strip() == flavour

    other = "cu129" if flavour == "cu130" else "cu130"
    rc, text, _ = run_setup(fakes, tmp_path, driver, WOCR_TORCH_BACKEND=other)
    assert rc == 2 and "needs a fresh env" in text


@needs_bash
def test_gpu_check_loads_vllm_ops_and_flags_a_driver_too_old(fakes, tmp_path):
    """--gpu-check loads vLLM's compiled ops (a mismatch shows there, not in
    the first job) and says when the driver cannot run the env's build."""
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "wocr.flavour").write_text("cu130\n")
    rc, text, _ = run_setup(fakes, tmp_path, "550.54.15", "--gpu-check",
                            PYTHONPATH=fake_python_libs(tmp_path, ops_so=True))
    assert rc == 0, text
    assert "driver 550.54.15 runs the cu129 build; this env has: cu130" in text
    assert "WOCR_TORCH_BACKEND=cu129" in text
    assert "vllm ops: FAILED to load: vllm._C_stable_libtorch" in text
    assert "Traceback" not in text


# ------------------------------------------------------------------- docs


def test_interactive_recipe_requests_what_the_model_servers_need():
    """`interactive` defaults to 1 core (4 GB) and 1 hour; the documented
    recipe asks for the batch job's 64 GB and a few hours (v2-e2e-6)."""
    for path in ("README.md", "slurm/ocr.sbatch"):
        text = Path(ROOT, path).read_text()
        recipes = re.findall(r"interactive -a ikoutis -q standard -j gpu([^`\n]*)", text)
        assert recipes, path
        for opts in recipes:
            cores, hours = re.search(r"-n (\d+)", opts), re.search(r"-t (\d+)", opts)
            assert cores and int(cores.group(1)) * 4 >= 64, (path, opts)
            assert hours and int(hours.group(1)) >= 2, (path, opts)
