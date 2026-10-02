"""The KaTeX worker's life cycle (restart, cap, latex_unchecked), against a
stand-in worker written in Python, so no node or katex is needed."""

import stat
import sys

import pytest

from src import katex_check
from src.validate import check_latex

# Speaks the worker's protocol: one JSON request per line, one JSON answer.
# "\crash" kills it; FAKE_KATEX=broken fails every request (katex not loadable).
FAKE_WORKER = """#!{python}
import json, os, sys
while True:
    line = sys.stdin.readline()
    if not line:
        break
    tex = json.loads(line)["tex"]
    if "\\\\crash" in tex:
        sys.exit(1)
    bad = os.environ.get("FAKE_KATEX") == "broken" or "\\\\bad" in tex
    print(json.dumps({{"ok": False, "error": "bad"}} if bad else {{"ok": True}}), flush=True)
"""


@pytest.fixture
def server(tmp_path, monkeypatch):
    node = tmp_path / "node"
    node.write_text(FAKE_WORKER.format(python=sys.executable))
    node.chmod(node.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(katex_check.shutil, "which", lambda name: str(node))
    monkeypatch.setattr(katex_check, "_katex_module", lambda: str(tmp_path))
    monkeypatch.delenv("WOCR_KATEX", raising=False)
    srv = katex_check._KatexServer()
    monkeypatch.setattr(katex_check, "_SERVER", srv)
    yield srv
    srv._stop()


class TestKatexWorker:
    def test_answers(self, server):
        assert katex_check.katex_error("x") is None
        assert katex_check.katex_error(r"\bad") == "bad"
        assert check_latex(r"\bad{x}") == ["latex_katex"]

    def test_dead_worker_is_restarted(self, server, capsys):
        assert katex_check.katex_error("x") is None
        server._proc.kill()             # as the OOM killer would
        server._proc.wait()
        assert katex_check.katex_error(r"\bad") == "bad"     # asked again, of a new worker
        assert check_latex("x^{2}") == []
        assert server._starts == 2
        assert "restarting" in capsys.readouterr().err

    def test_gives_up_after_the_cap(self, server, capsys):
        # each crash costs a restart; past the cap every formula is unchecked
        for _ in range(1 + katex_check.MAX_RESTARTS):
            assert katex_check.katex_error(r"\crash") == katex_check.UNCHECKED
        assert server._dead and server._starts == 1 + katex_check.MAX_RESTARTS
        assert katex_check.katex_error("x") == katex_check.UNCHECKED
        assert check_latex("x^{2}") == ["latex_unchecked"]
        assert not katex_check.katex_available()
        assert "giving up" in capsys.readouterr().err

    def test_worker_that_cannot_load_katex(self, server, monkeypatch, capsys):
        monkeypatch.setenv("FAKE_KATEX", "broken")
        assert katex_check.katex_error("x") == katex_check.UNCHECKED
        assert check_latex("x^{2}") == ["latex_unchecked"]
        assert "probe" in capsys.readouterr().err

    def test_not_installed_is_silent(self, server, monkeypatch, capsys):
        monkeypatch.setattr(katex_check.shutil, "which", lambda name: None)
        assert katex_check.katex_error("x") is None
        assert check_latex("x^{2}") == []
        assert capsys.readouterr().err == ""

    def test_disabled_is_silent(self, monkeypatch, capsys):
        monkeypatch.setenv("WOCR_KATEX", "0")
        monkeypatch.setattr(katex_check, "_SERVER", katex_check._KatexServer())
        assert katex_check.katex_error("x") is None
        assert capsys.readouterr().err == ""
