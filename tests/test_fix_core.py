"""Ingest, discovery and tools/incomplete.py: image normalisation, huge pages,
doc ids, --force, hidden/symlinked inputs, and recovery for every INPUTS form
(findings core-3, -4, -5, -6, -8, -10, -17, -18, slurm-5)."""

import json
import os
import sys

import httpx
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import FakeServer
from src import ingest as ing
from src import run_ocr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import incomplete  # noqa: E402


def page_of(manifest, work, k=0):
    with Image.open(os.path.join(work, manifest["doc_id"], manifest["pages"][k])) as im:
        return im.convert("RGB")


def text_page(mode="RGB", size=(200, 100), paper="white", ink="black"):
    img = Image.new(mode, size, paper)
    ImageDraw.Draw(img).rectangle((20, 40, 120, 60), fill=ink)
    return img


# ------------------------------------------------------------ image formats


class TestImageNormalisation:
    def ingest_array(self, tmp_path, a, name="scan.tif"):
        p = tmp_path / name
        Image.fromarray(a).save(p)
        m = ing.ingest(str(p), str(tmp_path / "w"))
        return page_of(m, tmp_path / "w")

    def test_16bit_scan_keeps_its_text(self, tmp_path):
        a = np.full((100, 200), 52000, np.uint16)
        a[40:60, 20:120] = 3000
        img = self.ingest_array(tmp_path, a)
        assert img.getpixel((5, 5)) == (255, 255, 255) and img.getpixel((50, 50)) == (0, 0, 0)

    def test_12bit_data_in_16bit_container(self, tmp_path):
        a = np.full((100, 200), 3250, np.uint16)       # a >> 8 would make this black
        a[40:60, 20:120] = 187
        img = self.ingest_array(tmp_path, a)
        assert img.getpixel((5, 5)) == (255, 255, 255) and img.getpixel((50, 50)) == (0, 0, 0)

    def test_float_image(self, tmp_path):
        a = np.full((100, 200), 0.9, np.float32)
        a[40:60, 20:120] = 0.05
        img = self.ingest_array(tmp_path, a)
        assert img.getpixel((5, 5)) == (255, 255, 255) and img.getpixel((50, 50)) == (0, 0, 0)

    @pytest.mark.parametrize("mode", ["RGBA", "LA"])
    def test_transparent_background_becomes_white(self, tmp_path, mode):
        clear = (0, 0, 0, 0) if mode == "RGBA" else (0, 0)
        ink = (0, 0, 0, 255) if mode == "RGBA" else (0, 255)
        p = tmp_path / "t.png"
        text_page(mode, paper=clear, ink=ink).save(p)
        img = page_of(ing.ingest(str(p), str(tmp_path / "w")), tmp_path / "w")
        assert img.getpixel((5, 5)) == (255, 255, 255) and img.getpixel((50, 50)) == (0, 0, 0)

    def test_palette_with_transparency(self, tmp_path):
        im = Image.new("P", (200, 100), 0)
        im.putpalette([0, 0, 0, 10, 10, 10])
        ImageDraw.Draw(im).rectangle((20, 40, 120, 60), fill=1)
        p = tmp_path / "p.png"
        im.save(p, transparency=0)
        img = page_of(ing.ingest(str(p), str(tmp_path / "w")), tmp_path / "w")
        assert img.getpixel((5, 5)) == (255, 255, 255) and img.getpixel((50, 50)) == (10, 10, 10)

    def test_cmyk(self, tmp_path):
        p = tmp_path / "c.jpg"
        text_page("CMYK", paper=(0, 0, 0, 0), ink=(0, 0, 0, 255)).save(p, quality=95)
        img = page_of(ing.ingest(str(p), str(tmp_path / "w")), tmp_path / "w")
        assert min(img.getpixel((5, 5))) > 245 and max(img.getpixel((50, 50))) < 10

    def test_exif_orientation_applied(self, tmp_path):
        im = text_page(size=(400, 300))
        exif = im.getexif()
        exif[0x0112] = 6                            # stored landscape, shown portrait
        p = tmp_path / "photo.jpg"
        im.save(p, exif=exif.tobytes())
        m = ing.ingest(str(p), str(tmp_path / "w"))
        assert page_of(m, tmp_path / "w").size == (300, 400)

    def test_mpo_second_image_is_not_a_page(self, tmp_path):
        p = tmp_path / "phone.jpg"
        text_page(size=(120, 160)).save(p, format="MPO", save_all=True,
                                         append_images=[Image.new("RGB", (30, 40), "gray")])
        with Image.open(p) as im:
            assert im.format == "MPO" and im.n_frames == 2
        m = ing.ingest(str(p), str(tmp_path / "w"))
        assert m["n_pages"] == 1 and page_of(m, tmp_path / "w").size == (120, 160)

    def test_tiff_pages_without_reduced_resolution_copies(self, tmp_path):
        thumb = Image.new("L", (20, 20), 128)
        thumb.encoderinfo = {"tiffinfo": {254: 1}}  # NewSubfileType: reduced-resolution
        p = tmp_path / "multi.tif"
        Image.new("L", (100, 120), 255).save(p, save_all=True,
                                             append_images=[thumb, Image.new("L", (110, 130), 200)])
        m = ing.ingest(str(p), str(tmp_path / "w"))
        assert m["n_pages"] == 2
        assert [page_of(m, tmp_path / "w", k).size for k in range(2)] == [(100, 120), (110, 130)]

    def test_tiff_orientation_per_page(self, tmp_path):
        turned = Image.new("L", (130, 110), 200)
        turned.encoderinfo = {"tiffinfo": {274: 6}}     # page 2 scanned sideways
        p = tmp_path / "multi.tif"
        Image.new("L", (100, 120), 255).save(p, save_all=True, append_images=[turned])
        m = ing.ingest(str(p), str(tmp_path / "w"))
        assert [page_of(m, tmp_path / "w", k).size for k in range(2)] == [(100, 120), (110, 130)]


# --------------------------------------------------------------- huge pages


class TestPixelCap:
    def test_pdf_page_rendered_within_cap(self, tmp_path, pdf_path, monkeypatch):
        monkeypatch.setattr(ing, "MAX_PAGE_PX", 500_000)    # letter at 200 dpi is 3.7 M
        m = ing.ingest(pdf_path, str(tmp_path / "w"))
        w, h = page_of(m, tmp_path / "w").size
        assert w * h <= 500_000 * 1.01 and abs(w / h - 8.5 / 11) < 0.01
        assert m["dpi"] == 200 and m["page_scale"][0] == pytest.approx(w / 1700, abs=0.01)

    def test_page_huge_in_points(self, tmp_path):
        """A 600-dpi scan wrapped into a PDF at 72 dpi: 5100 x 6600 pt."""
        p = tmp_path / "wrapped.pdf"
        Image.new("L", (510, 660), 255).save(p, resolution=7.2)
        m = ing.ingest(str(p), str(tmp_path / "w"))
        w, h = page_of(m, tmp_path / "w").size
        assert w * h <= ing.MAX_PAGE_PX * 1.01 and m["page_scale"][0] < 0.5

    def test_raster_beyond_pillow_bomb_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2_000)   # opening would raise
        monkeypatch.setattr(ing, "MAX_PAGE_PX", 5_000)
        p = tmp_path / "big.png"
        text_page(size=(300, 200)).save(p)
        m = ing.ingest(str(p), str(tmp_path / "w"))
        assert Image.MAX_IMAGE_PIXELS == 2_000                  # restored
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
        w, h = page_of(m, tmp_path / "w").size
        assert w * h <= 5_000 and (w, h) == (86, 57) and m["page_scale"][0] < 1


# ------------------------------------------------------- doc ids and --force


class TestDocIds:
    def test_doc_id_has_64_bits_of_hash(self, pdf_path):
        doc_id = ing.make_doc_id(pdf_path, ing.sha256_file(pdf_path))
        assert doc_id == "paper_one-" + ing.sha256_file(pdf_path)[:16]

    def test_id_collision_is_refused(self, tmp_path, pdf_path):
        doc_id = ing.make_doc_id(pdf_path, ing.sha256_file(pdf_path))
        os.makedirs(tmp_path / "w" / doc_id)
        (tmp_path / "w" / doc_id / "manifest.json").write_text(json.dumps(
            {"doc_id": doc_id, "source": "/elsewhere/paper one.pdf", "sha256": "0" * 64}))
        with pytest.raises(ValueError, match="different file"):
            ing.ingest(pdf_path, str(tmp_path / "w"))


class TestForce:
    def test_ingest_force_rerenders_and_invalidates(self, tmp_path, pdf_path):
        w = str(tmp_path / "w")
        m = ing.ingest(pdf_path, w)
        d = os.path.join(w, m["doc_id"])
        for sub in ("read", "review", "figures"):
            os.makedirs(os.path.join(d, sub))
            open(os.path.join(d, sub, "p0001.json"), "w").close()
        args = ["--inputs", pdf_path, "--work", w, "--dpi", "100", "--force"]
        run_ocr.select_docs(run_ocr.build_parser().parse_args(["read", *args]))
        assert page_of(m, w).size == (1700, 2200)       # other stages' --force: no re-render
        assert run_ocr.main(["ingest", *args]) == 0
        m2 = json.load(open(os.path.join(d, "manifest.json")))
        assert m2["dpi"] == 100 and page_of(m2, w).size == (850, 1100)
        assert sorted(os.listdir(d)) == ["manifest.json", "pages"]

    def test_dpi_change_without_force_warns(self, tmp_path, pdf_path, capsys):
        ing.ingest(pdf_path, str(tmp_path / "w"))
        m = ing.ingest(pdf_path, str(tmp_path / "w"), dpi=300)
        assert m["dpi"] == 200 and "--force" in capsys.readouterr().err


# ---------------------------------------------------------------- discovery


def png(path, shade=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32), (shade, 0, 0)).save(path)
    return str(path)


class TestDiscover:
    def test_skips_hidden_and_appledouble(self, tmp_path):
        good = png(tmp_path / "papers" / "a.png")
        (tmp_path / "papers" / "._a.png").write_bytes(b"\x00\x05\x16\x07AppleDouble")
        png(tmp_path / "papers" / ".hidden.png")
        png(tmp_path / "papers" / ".git" / "x.png")
        assert ing.discover([str(tmp_path / "papers")]) == [good]

    def test_follows_symlinked_dirs_once(self, tmp_path):
        papers = tmp_path / "papers"
        b = png(papers / "b.png")
        png(tmp_path / "store" / "2019" / "a.png", 50)
        os.symlink(tmp_path / "store" / "2019", papers / "2019")
        os.symlink(papers, papers / "2019" / "loop")    # a cycle
        assert ing.discover([str(papers)]) == [str(papers / "2019" / "a.png"), b]

    def test_same_file_by_two_paths_listed_once(self, tmp_path):
        a = png(tmp_path / "papers" / "a.png")
        os.symlink(tmp_path / "papers", tmp_path / "alias")
        assert ing.discover([str(tmp_path / "papers"), str(tmp_path / "alias" / "a.png"),
                             str(tmp_path / "papers" / "." / "a.png")]) \
            == [str(tmp_path / "alias" / "a.png")]       # the path that sorts first
        assert ing.discover([str(tmp_path / "papers")]) == [a]


# ------------------------------------------------------------ incomplete.py


def incomplete_run(capsys, *argv) -> tuple[str, str]:
    incomplete.main(list(argv))
    res = capsys.readouterr()
    return res.out.strip(), res.err


@pytest.fixture
def corpus(tmp_path):
    """Two input dirs, one of them holding an input that cannot be ingested
    (a corrupt PDF) and one that cannot even be read (a broken symlink)."""
    a, b = tmp_path / "papersA", tmp_path / "papersB"
    for k in range(3):
        png(a / f"a{k}.png", 10 + k)
        png(b / f"b{k}.png", 100 + k)
    (a / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    os.symlink(tmp_path / "gone.pdf", a / "moved.pdf")
    return [str(a), str(b)]


class TestIncomplete:
    def test_space_separated_inputs_and_globs(self, tmp_path, corpus, capsys):
        out = ["--nshards", "3", "--out", str(tmp_path / "o")]
        for extra in ([], ["--list"]):
            several, err = incomplete_run(capsys, "--inputs", *corpus, *out, *extra)
            assert several and "moved.pdf" in err           # skipped, not a crash
            assert incomplete_run(capsys, "--inputs", " ".join(corpus), *out, *extra)[0] \
                == several                                  # --inputs "$INPUTS"
            assert incomplete_run(capsys, "--inputs", str(tmp_path / "papers*"), *out,
                                  *extra)[0] == several
        assert incomplete_run(capsys, "--inputs", *corpus, *out)[0] == "0-2"

    def test_matches_pipeline_shards_failed_and_unreadable_are_terminal(
            self, tmp_path, corpus, capsys):
        out, work = str(tmp_path / "o"), str(tmp_path / "w")

        def assemble_shard(i):          # what array task i of 2 would leave behind
            a = run_ocr.build_parser().parse_args(
                ["ingest", "--inputs", *corpus, "--work", work, "--out", out,
                 "--shard", f"{i}/2"])
            for d in run_ocr.select_docs(a):
                doc = os.path.basename(d)
                os.makedirs(os.path.join(out, doc), exist_ok=True)
                open(os.path.join(out, doc, doc + ".md"), "w").close()
            capsys.readouterr()         # the pipeline's log

        quoted = ["--inputs", " ".join(corpus), "--nshards", "2", "--out", out]
        assert incomplete_run(capsys, *quoted)[0] == "0-1"
        assemble_shard(1)
        failed = [d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "FAILED.json"))]
        assert len(failed) == 1 and failed[0].startswith("broken-")
        assert incomplete_run(capsys, *quoted)[0] == "0"
        assemble_shard(0)
        ids, err = incomplete_run(capsys, *quoted)
        assert ids == "" and "moved.pdf" in err
        listing = incomplete_run(capsys, *quoted, "--list")[0].splitlines()
        assert [line.split(":")[1].strip() for line in listing] == [
            "4/4 done (1 unreadable)", "4/4 done (1 failed)"]

    def test_empty_inputs_is_an_error(self, capsys):
        with pytest.raises(SystemExit) as e:
            incomplete.main(["--inputs", "", "--nshards", "2"])
        assert e.value.code == 2 and "INPUTS" in capsys.readouterr().err

    def test_existing_path_with_spaces_kept_whole(self, tmp_path, pdf_path, capsys):
        assert " " in os.path.basename(pdf_path)
        ids, err = incomplete_run(capsys, "--inputs", pdf_path, "--nshards", "1",
                                  "--out", str(tmp_path / "o"))
        assert ids == "0" and err == ""


def test_failed_pages_visible_in_markdown(tmp_path, pdf_path, monkeypatch):
    """core-3: a placeholder page assembles as a marked gap, not as nothing."""
    servers = {"r": FakeServer(lambda p, n: httpx.Response(400, text="image too large")),
               "e": FakeServer(lambda p, n: "VERDICT: correct\n<text>\n</text>")}
    monkeypatch.setattr(run_ocr, "ChatClient", lambda url, *a, **kw:
                        servers["r" if url.endswith("8001") else "e"].client())
    out = tmp_path / "o"
    assert run_ocr.main(["all", "--inputs", pdf_path, "--work", str(tmp_path / "w"),
                         "--out", str(out), "--workers", "1"]) == 0
    (doc,) = os.listdir(out)
    md = (out / doc / f"{doc}.md").read_text()
    assert md == ("<!-- page 1: OCR failed, see report.json -->\n\n"
                  "<!-- page 2: OCR failed, see report.json -->\n")
