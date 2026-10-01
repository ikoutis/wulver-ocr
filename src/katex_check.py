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
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from typing import Optional

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


class _KatexServer:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._dead = os.environ.get("WOCR_KATEX", "1") == "0"

    def _start(self) -> bool:
        node, mod = shutil.which("node"), _katex_module()
        if not node or not mod:
            self._dead = True
            return False
        self._proc = subprocess.Popen([node, "-e", _JS], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, bufsize=1,
                                      env={**os.environ, "WOCR_KATEX_MODULE": mod})
        # Probe once: a worker that cannot load katex must be found out now,
        # not mistaken for "every formula parses".
        if self._ask("x", True) != {"ok": True}:
            self._dead, self._proc = True, None
            return False
        return True

    def _ask(self, tex: str, display: bool) -> Optional[dict]:
        try:
            self._proc.stdin.write(json.dumps({"tex": tex, "display": display}) + "\n")
            self._proc.stdin.flush()
            return json.loads(self._proc.stdout.readline())
        except (OSError, ValueError):
            return None

    def check(self, tex: str, display: bool) -> Optional[str]:
        """None if it parses (or KaTeX is unavailable); else the error."""
        with self._lock:
            if self._dead or (self._proc is None and not self._start()):
                return None
            res = self._ask(tex, display)
            if res is None:
                self._dead = True        # never let the checker break the pipeline
                return None
            return None if res.get("ok") else res.get("error", "katex error")

    def available(self) -> bool:
        with self._lock:
            return not self._dead and (self._proc is not None or self._start())


_SERVER = _KatexServer()


def katex_error(tex: str, display: bool = True) -> Optional[str]:
    return _SERVER.check(tex, display)


def katex_available() -> bool:
    return _SERVER.available()
