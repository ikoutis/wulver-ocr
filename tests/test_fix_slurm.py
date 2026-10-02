"""Bash-level tests of the Wulver scripts: slurm/ocr.sbatch, serve_lib.sh,
requeue_lib.sh and tools/setup_env.sh, driven with fake executables (vllm,
scontrol, nvidia-smi, module, conda) written into a temp dir. No SLURM, GPU,
model or network is needed: the fake `vllm serve` is a small HTTP server that
binds, "loads", and listens the way vLLM does, and answers chat requests with
canned replies that name the job whose server wrote them.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SBATCH = os.path.join(ROOT, "slurm", "ocr.sbatch")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import stage_models  # noqa: E402

pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("curl")),
                                reason="needs bash and curl")

# Like vLLM's launcher: bind first (SO_REUSEADDR, no SO_REUSEPORT), load the
# model, then listen, so a second server on the port dies with EADDRINUSE at
# bind or at listen; then uvicorn's start-up lines. FAKE_UVICORN_ORDER=1 logs
# "Application startup complete" before listen(), as real uvicorn does, and
# FAKE_SHUTDOWN is how long a server that lost the port takes to exit (vLLM
# shuts its engine down first). FAKE_READER_400=1: the reader rejects pages.
FAKE_VLLM = r'''
import http.server, json, os, socket, sys, threading, time

argv = sys.argv
port = int(argv[argv.index("--port") + 1])
name = argv[argv.index("--served-model-name") + 1]
job = os.environ.get("SLURM_JOB_ID", "local")
timer = threading.Timer(float(os.environ.get("FAKE_LIFETIME", "120")), os._exit, (0,))
timer.daemon = True                     # never outlive a failed test for long
timer.start()
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", port))
time.sleep(float(os.environ.get("FAKE_LOAD", "0.3")))      # "loading the model"
if os.environ.get("FAKE_UVICORN_ORDER"):
    print("INFO:     Application startup complete.", flush=True)
try:
    sock.listen(64)
except OSError as e:
    print(f"OSError: {e}", flush=True)
    time.sleep(float(os.environ.get("FAKE_SHUTDOWN", "0")))
    sys.exit(1)

READER = ('<div data-bbox="100 100 900 200" data-label="Text"><p>Read by job JOB.</p></div>'
          '<div data-bbox="200 300 800 350" data-label="Equation-Block">'
          '<math display="block">L = D - A</math></div>'
          '<div data-bbox="100 400 900 700" data-label="Image"><img alt="a plot"/></div>')


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"job": job, "pid": os.getpid()})
        else:
            self._send({"data": [{"id": name}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        print(f"POST {self.path} max_tokens={req.get('max_tokens')}", flush=True)
        text = json.dumps(req["messages"])
        if name.startswith("chandra") and os.environ.get("FAKE_READER_400"):
            body = b'{"error": {"message": "maximum context length exceeded"}}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if name.startswith("chandra"):
            out = READER.replace("JOB", job)
        elif "KIND:" in text:
            out = "KIND: plot\n<description>A plot.</description>"
        else:
            out = "VERDICT: correct"
        self._send({"choices": [{"message": {"role": "assistant", "content": out},
                                 "finish_reason": "stop"}]})


server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler,
                                         bind_and_activate=False)
server.socket.close()
server.socket = sock
if not os.environ.get("FAKE_UVICORN_ORDER"):
    print(f"INFO:     Started server process [{os.getpid()}]", flush=True)
    print("INFO:     Application startup complete.", flush=True)
server.serve_forever()
'''


class Fakes:
    """Fake cluster commands on PATH, fake model dirs, and the env to run with."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.models = tmp / "models"
        self.vlogs = tmp / "vlogs"
        self.scontrol_log = tmp / "scontrol.log"
        conda_base = tmp / "conda"
        for d in (self.bin, self.models / "chandra_ocr_2", self.models / "qwen3_8_27b",
                  conda_base / "etc" / "profile.d"):
            d.mkdir(parents=True)
        (conda_base / "etc" / "profile.d" / "conda.sh").write_text(
            "conda() { return 0; }\n")
        scripts = {
            "vllm": f"#!{sys.executable}\n{FAKE_VLLM}",
            "python": f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
            "module": "#!/bin/sh\nexit 0\n",
            "conda": f'#!/bin/sh\necho "{conda_base}"\n',
            "nvidia-smi": '#!/bin/sh\necho "NVIDIA A100-SXM4-80GB (fake), 81920 MiB, 550.00"\n',
            "scontrol": '#!/bin/sh\necho "$*" >> "$FAKE_SCONTROL_LOG"\n',
        }
        for name, text in scripts.items():
            path = self.bin / name
            path.write_text(text)
            path.chmod(0o755)
        self.procs: list[subprocess.Popen] = []

    def env(self, **kw) -> dict:
        drop = {"INPUTS", "OUT", "WORK", "NSHARDS", "PHASES", "READ_ARGS", "REVIEW_ARGS",
                "WOCR_PROFILE", "WOCR_CONDA_ENV"}
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SLURM_") and k not in drop}
        env.update(PATH=f"{self.bin}:{os.environ['PATH']}", USER="tester",
                   WOCR_MODELS=str(self.models), WOCR_SERVER_LOGDIR=str(self.vlogs),
                   WOCR_SERVER_POLL="0.2", FAKE_SCONTROL_LOG=str(self.scontrol_log))
        env.update({k: str(v) for k, v in kw.items()})
        return env

    def start(self, argv, env, out, cwd=ROOT) -> subprocess.Popen:
        """Run in its own session, like a job: killed as a group at teardown."""
        with open(out, "w") as f:
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=f,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append(p)
        return p

    def bash(self, script, env, out, cwd=ROOT) -> subprocess.Popen:
        return self.start(["bash", "-c", script], env, out, cwd)

    def sbatch(self, env, out) -> subprocess.Popen:
        """As SLURM runs a batch task: a spool copy of the script, started in
        the submit dir (the repo root)."""
        spool = self.tmp / "spool" / f"job{env['SLURM_JOB_ID']}"
        spool.mkdir(parents=True, exist_ok=True)
        shutil.copy(SBATCH, spool / "slurm_script")
        return self.start(["bash", str(spool / "slurm_script")],
                          dict(env, SLURM_SUBMIT_DIR=ROOT), out)

    def cleanup(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


@pytest.fixture
def fakes(tmp_path):
    f = Fakes(tmp_path)
    yield f
    f.cleanup()


def wait_for(path, text, timeout=30.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        s = path.read_text() if path.exists() else ""
        if text in s:
            return s
        time.sleep(0.05)
    raise AssertionError(f"{text!r} not in {path}:\n{s}")


def finish(p, timeout=90) -> int:
    return p.wait(timeout=timeout)


def make_docs(d, n=2) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n):
        img = Image.new("RGB", (850, 1100), "white")
        ImageDraw.Draw(img).rectangle((100 + 40 * k, 600, 500, 900), outline="black")
        path = d / f"doc{k}.png"
        img.save(path)
        paths.append(str(path))
    return paths


def markdown(out_root) -> dict[str, str]:
    """{doc file stem: assembled Markdown} under an OUT root."""
    found = {}
    for doc_id in os.listdir(out_root):
        md = os.path.join(out_root, doc_id, doc_id + ".md")
        if os.path.exists(md):
            found[doc_id.split("-")[0]] = Path(md).read_text()
    return found


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------ whole sbatch


def test_two_array_tasks_on_one_node_each_use_their_own_servers(fakes, tmp_path):
    """Two tasks of one array run at once on one node: each starts its own
    servers on its own ports, reads its own shard through them, and stops
    them (slurm-1: with fixed ports the second task used the first's server).
    READ_ARGS reaches the read stage (docs-11)."""
    make_docs(tmp_path / "papers")
    out, work = tmp_path / "out", tmp_path / "work"
    common = dict(INPUTS=tmp_path / "papers", OUT=out, WORK=work, SLURM_ARRAY_JOB_ID=1000,
                  SLURM_ARRAY_TASK_COUNT=2, SLURM_ARRAY_TASK_MAX=1,
                  FAKE_LOAD=1.0, READER_WORKERS=2, EDITOR_WORKERS=2)
    a = fakes.sbatch(fakes.env(**common, SLURM_JOB_ID=1001, SLURM_ARRAY_TASK_ID=0,
                               READ_ARGS="--reader-max-tokens 777"),
                     tmp_path / "a.out")
    wait_for(tmp_path / "a.out", "=== starting chandra_ocr_2")
    b = fakes.sbatch(fakes.env(**common, SLURM_JOB_ID=1002, SLURM_ARRAY_TASK_ID=1),
                     tmp_path / "b.out")
    rc_a, rc_b = finish(a), finish(b)
    out_a, out_b = (tmp_path / "a.out").read_text(), (tmp_path / "b.out").read_text()
    assert (rc_a, rc_b) == (0, 0), out_a + out_b

    assert "shard 0/2" in out_a and "shard 1/2" in out_b
    md = markdown(out)
    assert "Read by job 1001." in md["doc0"]           # shard 0 read by task A's server
    assert "Read by job 1002." in md["doc1"]
    for job, text in (("1001", out_a), ("1002", out_b)):
        for name in ("chandra_ocr_2", "qwen3_8_27b"):
            assert re.search(rf"=== {name} ready after \d+s on 127\.0\.0\.1:\d+ ===", text)
            assert f"=== stopped {name} ===" in text
            log = (fakes.vlogs / f"vllm_{name}_{job}.log").read_text()
            assert "Application startup complete" in log and "POST" in log
    reader_a = (fakes.vlogs / "vllm_chandra_ocr_2_1001.log").read_text()
    reader_b = (fakes.vlogs / "vllm_chandra_ocr_2_1002.log").read_text()
    assert "max_tokens=777" in reader_a and "max_tokens=777" not in reader_b


def test_interactive_run_uses_the_repo_and_the_callers_relative_paths(fakes, tmp_path):
    """`interactive` sets SLURM_SUBMIT_DIR to wherever it was started (here a
    fake home). The documented command still runs from the repo, and relative
    INPUTS/OUT/WORK are taken from the caller's directory (slurm-9)."""
    home, user = tmp_path / "home", tmp_path / "user"
    home.mkdir()
    make_docs(user, n=1)
    p = fakes.start(["bash", SBATCH],
                    fakes.env(INPUTS="doc0.png", OUT="out", WORK="work", SLURM_JOB_ID=555,
                              SLURM_SUBMIT_DIR=home),
                    tmp_path / "run.out", cwd=user)
    assert finish(p) == 0, (tmp_path / "run.out").read_text()
    assert "Read by job 555." in markdown(user / "out")["doc0"]
    assert os.listdir(home) == []
    assert os.path.isdir(user / "work")


def test_sparse_resubmission_without_nshards_is_refused(fakes, tmp_path):
    """--array=3,5 without NSHARDS would run shards 3/2 and 5/2; refuse with a
    hint instead. With NSHARDS it runs (slurm-8, core-16, docs-3)."""
    make_docs(tmp_path / "papers")
    env = dict(INPUTS=tmp_path / "papers", OUT=tmp_path / "out", WORK=tmp_path / "work",
               SLURM_ARRAY_JOB_ID=900, SLURM_JOB_ID=903, SLURM_ARRAY_TASK_ID=3,
               SLURM_ARRAY_TASK_COUNT=2, SLURM_ARRAY_TASK_MAX=5)
    p = fakes.sbatch(fakes.env(**env), tmp_path / "bad.out")
    assert finish(p) == 2
    assert "set NSHARDS" in (tmp_path / "bad.out").read_text()
    assert not (tmp_path / "work").exists()

    p = fakes.sbatch(fakes.env(**env, NSHARDS=8, PHASES="assemble"), tmp_path / "ok.out")
    assert finish(p) == 0, (tmp_path / "ok.out").read_text()
    assert "shard 3/8" in (tmp_path / "ok.out").read_text()


def test_usr1_during_server_startup_requeues_without_waiting_for_it(fakes, tmp_path):
    """A USR1 while a server is still loading stops it and requeues at once,
    not after the start-up finishes (slurm-6)."""
    make_docs(tmp_path / "papers", n=1)
    out = tmp_path / "run.out"
    p = fakes.sbatch(fakes.env(INPUTS=tmp_path / "papers", OUT=tmp_path / "out",
                               WORK=tmp_path / "work", SLURM_ARRAY_JOB_ID=1000,
                               SLURM_JOB_ID=1001, SLURM_ARRAY_TASK_ID=0,
                               SLURM_ARRAY_TASK_COUNT=1, SLURM_ARRAY_TASK_MAX=0,
                               FAKE_LOAD=60),
                     out)
    wait_for(out, "=== starting chandra_ocr_2")
    t0 = time.time()
    os.kill(p.pid, signal.SIGUSR1)
    text = wait_for(out, "=== requeueing 1000_0", timeout=15)
    assert time.time() - t0 < 10
    assert "ready" not in text
    assert text.index("=== stopped chandra_ocr_2") < text.index("=== requeueing")
    time.sleep(0.2)
    assert fakes.scontrol_log.read_text().split("\n")[0] == "requeue 1000_0"


def test_sbatch_settings():
    text = Path(SBATCH).read_text()
    # slurm-4: a lead long enough to drain in-flight requests and requeue
    assert "#SBATCH --signal=B:USR1@1800" in text
    # docs-2: the default WORK is per profile, so a rerun with another
    # profile reads the pages again instead of reusing the first reading
    profile = text.index('WOCR_PROFILE="${WOCR_PROFILE:-default}"')
    work = text.index("${WORK:-/scratch/ikoutis/$USER/wocr/work/$WOCR_PROFILE}")
    assert profile < work
    # slurm-1: no fixed ports anywhere in the job
    assert not re.search(r"\b800[12]\b", text)
    # slurm-8 / core-16 / docs-3: the recovery lines pass INPUTS, OUT, NSHARDS
    for path in (SBATCH, os.path.join(ROOT, "slurm", "requeue_lib.sh")):
        recovery = [ln for ln in Path(path).read_text().splitlines()
                    if "sbatch --array=$IDS" in ln]
        assert recovery, path
        for ln in recovery:
            assert 'INPUTS="$INPUTS"' in ln and 'OUT="$OUT"' in ln and "NSHARDS=" in ln


# ------------------------------------------------------------ serve_lib.sh

FORCE_FIRST_PORT = r'''
set -euo pipefail
source slurm/serve_lib.sh
eval "real_$(declare -f free_port)"
free_port() {           # the first start is handed $FIRST_PORT
    if mkdir "$MARK" 2>/dev/null; then echo "$FIRST_PORT"; else real_free_port; fi
}
start_server chandra_ocr_2 "$WOCR_MODELS/chandra_ocr_2"
echo "HEALTH $(curl -sf "$(server_url chandra_ocr_2)/health")"
stop_server chandra_ocr_2
'''


class _Foreign(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"job": "someone else"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_another_jobs_server_on_the_port_is_not_taken_for_ours(fakes, tmp_path):
    """The port handed to vLLM is taken by a healthy server of another job
    before vLLM binds it: ours dies with EADDRINUSE and start_server retries
    on another port, never reporting the other server as ready (slurm-1)."""
    foreign = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Foreign)
    threading.Thread(target=foreign.serve_forever, daemon=True).start()
    try:
        port = foreign.server_address[1]
        p = fakes.bash(FORCE_FIRST_PORT,
                       fakes.env(FIRST_PORT=port, MARK=tmp_path / "mark", SLURM_JOB_ID=7001),
                       tmp_path / "run.out")
        assert finish(p) == 0, (tmp_path / "run.out").read_text()
    finally:
        foreign.shutdown()
        foreign.server_close()
    text = (tmp_path / "run.out").read_text()
    assert f"port {port} was taken" in text
    ready = re.search(r"ready after \d+s on 127\.0\.0\.1:(\d+)", text)
    assert ready and int(ready.group(1)) != port
    health = json.loads(text.split("HEALTH ", 1)[1].splitlines()[0])
    assert health["job"] == "7001"


def test_simultaneous_starts_on_one_port_each_get_their_own_server(fakes, tmp_path):
    """Two tasks start servers on the same port at once. Both bind (as vLLM
    does, before loading), one loses at listen() — and must not report the
    winner's server as its own (slurm-1)."""
    port = free_port()
    procs = [fakes.bash(FORCE_FIRST_PORT,
                        fakes.env(FIRST_PORT=port, MARK=tmp_path / f"mark{job}",
                                  SLURM_JOB_ID=job, FAKE_LOAD=1.5),
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
    ports = [re.search(r"ready after \d+s on 127\.0\.0\.1:(\d+)", t).group(1) for t in texts]
    assert len(set(ports)) == 2


def test_a_server_that_died_during_the_stage_fails_the_task(fakes, tmp_path):
    """stop_server returns 1, with the server log's tail, when the server had
    already exited: under set -e the task fails (slurm-3)."""
    script = r'''
set -euo pipefail
source slurm/serve_lib.sh
start_server chandra_ocr_2 "$WOCR_MODELS/chandra_ocr_2"
kill -KILL "${WOCR_SERVER_PIDS[chandra_ocr_2]}"
sleep 0.5
stop_server chandra_ocr_2
echo NOT REACHED
'''
    p = fakes.bash(script, fakes.env(SLURM_JOB_ID=7001), tmp_path / "run.out")
    assert finish(p) == 1
    text = (tmp_path / "run.out").read_text()
    assert "SERVE ERROR: chandra_ocr_2 had exited on its own" in text
    assert "Application startup complete" in text            # the log tail
    assert "NOT REACHED" not in text


# ---------------------------------------------------------- requeue_lib.sh


def test_usr1_that_kills_the_pipeline_before_its_handler_requeues(fakes, tmp_path):
    """The forwarded USR1 reaches the pipeline before it installed its handler
    (rc 138): requeue, like rc 85, instead of failing the task (slurm-7)."""
    script = r'''
set -euo pipefail
source slurm/requeue_lib.sh
run_with_requeue python -c 'print("child up", flush=True); import time; time.sleep(10)'
echo NOT REACHED
'''
    out = tmp_path / "run.out"
    p = fakes.bash(script, fakes.env(), out)
    wait_for(out, "child up")
    os.kill(p.pid, signal.SIGUSR1)
    assert finish(p) == 85
    text = out.read_text()
    assert "pipeline stopped (rc 138)" in text and "NOT REACHED" not in text


def test_usr1_forwarded_to_a_finishing_step_stops_the_next_server_start(fakes, tmp_path):
    """A USR1 forwarded to a step that then exits 0 anyway is remembered: the
    next server start requeues instead of starting (slurm-6)."""
    script = r'''
set -euo pipefail
source slurm/serve_lib.sh
source slurm/requeue_lib.sh
run_with_requeue python -c '
import signal, time
signal.signal(signal.SIGUSR1, signal.SIG_IGN)
print("child up", flush=True)
time.sleep(1.5)'
echo "step done"
start_server chandra_ocr_2 "$WOCR_MODELS/chandra_ocr_2"
echo NOT REACHED
'''
    out = tmp_path / "run.out"
    p = fakes.bash(script, fakes.env(), out)
    wait_for(out, "child up")
    os.kill(p.pid, signal.SIGUSR1)
    assert finish(p) == 85
    text = out.read_text()
    assert "step done" in text and "signalled outside SLURM" in text
    assert "=== starting" not in text and "NOT REACHED" not in text


def test_model_server_failure_exit_3_fails_the_task(fakes, tmp_path):
    script = r'''
set -euo pipefail
source slurm/requeue_lib.sh
run_with_requeue python -c 'raise SystemExit(3)'
echo NOT REACHED
'''
    p = fakes.bash(script, fakes.env(), tmp_path / "run.out")
    assert finish(p) == 3
    text = (tmp_path / "run.out").read_text()
    assert "NOT REACHED" not in text and "requeue" not in text


# ----------------------------------------------------------------- tools/


def test_gpu_check_reports_cuda_false_instead_of_a_traceback(fakes, tmp_path):
    """When CUDA is unusable, torch.cuda.get_device_name() raises; the check
    prints cuda=False, the symptom the docs describe (docs-15)."""
    lib = tmp_path / "pylib"
    (lib / "torch").mkdir(parents=True)
    (lib / "vllm").mkdir()
    (lib / "torch" / "__init__.py").write_text(
        '__version__ = "2.9.0"\n'
        'class version:\n    cuda = "12.9"\n'
        'class cuda:\n'
        '    is_available = staticmethod(lambda: False)\n'
        '    def get_device_name(i):\n'
        '        raise RuntimeError("Found no NVIDIA driver on your system")\n')
    (lib / "vllm" / "__init__.py").write_text('__version__ = "0.17.0"\n')
    p = fakes.start(["bash", os.path.join(ROOT, "tools", "setup_env.sh"), "--gpu-check"],
                    fakes.env(PYTHONPATH=lib), tmp_path / "run.out", cwd=tmp_path)
    assert finish(p) == 0, (tmp_path / "run.out").read_text()
    text = (tmp_path / "run.out").read_text()
    assert "A100-SXM4-80GB (fake)" in text
    assert "torch 2.9.0 (CUDA 12.9) | vllm 0.17.0 | cuda=False" in text
    assert "Traceback" not in text


def test_stage_models_without_the_env_says_to_activate_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)     # import fails
    with pytest.raises(SystemExit) as e:
        stage_models.main(["--profile", "default", "--models_dir", str(tmp_path)])
    assert "activate the env" in str(e.value)


def test_logs_dir_exists_in_a_fresh_clone():
    """SLURM opens --output=logs/... before the script runs and does not
    create the directory: it must be tracked (slurm-2, docs-1)."""
    assert os.path.exists(os.path.join(ROOT, "logs", ".gitkeep"))
    ignore = Path(ROOT, ".gitignore").read_text().splitlines()
    assert "logs/*" in ignore and "!logs/.gitkeep" in ignore and "logs/" not in ignore
    if shutil.which("git"):
        def ignored(path):
            return subprocess.run(["git", "check-ignore", "--no-index", "-q", path],
                                  cwd=ROOT).returncode == 0
        assert ignored("logs/wocr_1_0.log") and not ignored("logs/.gitkeep")
