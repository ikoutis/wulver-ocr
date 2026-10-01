import os
import subprocess
import sys

from src.ingest import discover, ingest, make_doc_id, sha256_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import incomplete  # noqa: E402
import stage_models  # noqa: E402


def test_profiles_parse_and_names_have_no_dots():
    profiles = [f[:-3] for f in os.listdir(os.path.join(ROOT, "profiles"))
                if f.endswith(".sh")]
    assert "default" in profiles
    for name in profiles:
        p = stage_models.read_profile(name)
        for role in ("READER", "EDITOR"):
            assert p[f"{role}_REPO"].count("/") == 1, (name, role)
            assert "." not in p[f"{role}_NAME"], (name, role)
        assert p["READER_ADAPTER"]


def test_profiles_are_valid_bash():
    for f in os.listdir(os.path.join(ROOT, "profiles")):
        if f.endswith(".sh"):
            path = os.path.join(ROOT, "profiles", f)
            out = subprocess.run(
                ["bash", "-c", f'set -eu; source "{path}"; '
                 'echo "$READER_ADAPTER ${#READER_VLLM_ARGS[@]} ${#EDITOR_VLLM_ARGS[@]}"'],
                capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            adapter, n_r, n_e = out.stdout.split()
            assert int(n_r) > 0 and int(n_e) > 0


def test_incomplete_reports_unassembled_shards(tmp_path, pdf_path, capsys):
    from PIL import Image
    paths = [pdf_path]
    for k in range(3):
        p = tmp_path / f"s{k}.png"
        Image.new("RGB", (64, 64), (k * 60, 0, 0)).save(p)
        paths.append(str(p))
    out = tmp_path / "out"
    # mark the doc of shard 1 (of 2) complete: discover order, index 1 and 3
    docs = discover(paths)
    for p in docs[1::2]:
        d = make_doc_id(p, sha256_file(p))
        os.makedirs(out / d)
        (out / d / f"{d}.md").write_text("x")
    incomplete.main(["--inputs", *paths, "--nshards", "2", "--out", str(out)])
    assert capsys.readouterr().out.strip() == "0"


def test_ingest_is_idempotent_and_renders_pages(tmp_path, pdf_path):
    m1 = ingest(pdf_path, str(tmp_path))
    m2 = ingest(pdf_path, str(tmp_path))
    assert m1 == m2 and m1["n_pages"] == 2
    page = os.path.join(tmp_path, m1["doc_id"], m1["pages"][0])
    from PIL import Image
    with Image.open(page) as im:
        assert im.size == (1700, 2200)        # 8.5x11 in at 200 dpi
