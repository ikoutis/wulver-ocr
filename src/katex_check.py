"""Optional validator: does the LaTeX parse in KaTeX?

KaTeX is what olmOCR-Bench renders math with, and what GitHub, Obsidian, and
most Markdown viewers use. So "KaTeX can parse it" is a strong, cheap check
that complements the structural checks in validate.py: it catches unknown
control sequences, bad arguments, and malformed environments.

One persistent `node` process serves every check over a pipe (about a
millisecond per formula), shared by the pipeline's threads behind a lock. If
node or the katex package is missing, the check is silently skipped and the
structural checks stand alone. Locate katex with WOCR_KATEX_DIR, a directory
containing node_modules/katex (tools/setup_env.sh installs one into the
conda env). Set WOCR_KATEX=0 to disable it.

A worker that dies (an OOM kill, a stray signal) is restarted, at most
MAX_RESTARTS times per process, and the formula is asked again. When KaTeX is
installed but no worker can be kept running, every check answers UNCHECKED
(validate.py flags it "latex_unchecked"), so a dead checker never passes a
formula off as parseable. Both events are logged to stderr.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional

UNCHECKED = "KaTeX worker unavailable"   # katex_error's answer when KaTeX cannot run
MAX_RESTARTS = 3

_JS = r"""
const katex = require(process.env.WOCR_KATEX_MODULE);
const rl = require('readline').createInterface({input: process.stdin});
rl.on('line', (line) => {
  let q; try { q = JSON.parse(line); } catch (e) { return; }
  let out = {ok: true};
  try {
    katex.renderToString(q.tex, {displayMode: !!q.display, throwOnError: true,
                                 strict: 'ignore', trust: false});
  } catch (e) { out = {ok: false, error: String(e.message || e).slice(0, 300)}; }
  process.stdout.write(JSON.stringify(out) + '\n');
});
"""


def _katex_module() -> Optional[str]:
    candidates = [os.environ.get("WOCR_KATEX_DIR")]
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "wocr-katex"))
    for d in filter(None, candidates):
        mod = os.path.join(d, "node_modules", "katex")
        if os.path.isdir(mod):
            return mod
    return None


def _log(msg: str) -> None:
    print(f"[katex_check] {msg}", file=sys.stderr, flush=True)


class _KatexServer:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._off = os.environ.get("WOCR_KATEX", "1") == "0"   # disabled or not installed
        self._dead = False          # installed, but no worker could be kept running
        self._starts = 0

    def _start(self) -> bool:
        node, mod = shutil.which("node"), _katex_module()
        if not node or not mod:
            self._off = True
            return False
        self._starts += 1
        try:
            self._proc = subprocess.Popen(
                [node, "-e", _JS], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                bufsize=1, env={**os.environ, "WOCR_KATEX_MODULE": mod})
        except OSError as e:
            _log(f"cannot start the KaTeX worker: {e!r}")
            return False
        # Probe once: a worker that cannot load katex must be found out now,
        # not mistaken for "every formula parses".
        if self._ask("x", True) != {"ok": True}:
            _log("the KaTeX worker failed its probe (cannot load katex?)")
            self._stop()
            return False
        return True

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass

    def _ensure(self) -> bool:
        """True if a worker is running, (re)starting one within the cap."""
        if self._proc is not None:
            return True
        if self._off or self._dead:
            return False
        if self._starts > MAX_RESTARTS:
            self._dead = True
            _log(f"giving up on KaTeX after {self._starts} worker starts: formulas "
                 "are flagged latex_unchecked for the rest of this process")
            return False
        return self._start()

    def _ask(self, tex: str, display: bool) -> Optional[dict]:
        try:
            self._proc.stdin.write(json.dumps({"tex": tex, "display": display}) + "\n")
            self._proc.stdin.flush()
            return json.loads(self._proc.stdout.readline())
        except (OSError, ValueError):
            return None

    def check(self, tex: str, display: bool) -> Optional[str]:
        """None if it parses (or KaTeX is not installed); UNCHECKED if KaTeX
        is installed but cannot be run; else the error."""
        with self._lock:
            for _ in range(2):          # a worker that died is replaced and asked again, once
                if not self._ensure():
                    return None if self._off else UNCHECKED
                res = self._ask(tex, display)
                if res is not None:
                    return None if res.get("ok") else res.get("error", "katex error")
                _log(f"the KaTeX worker stopped answering (exit status "
                     f"{self._proc.poll()}); restarting it")
                self._stop()
            return UNCHECKED

    def available(self) -> bool:
        with self._lock:
            return self._ensure()


_SERVER = _KatexServer()


def katex_error(tex: str, display: bool = True) -> Optional[str]:
    """None if KaTeX parses tex (or KaTeX is not installed), UNCHECKED if
    KaTeX is installed but cannot be run, else KaTeX's error message."""
    return _SERVER.check(tex, display)


def katex_available() -> bool:
    return _SERVER.available()
