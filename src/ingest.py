"""Stage 0: documents -> page images + a manifest.

Accepts PDFs (scanned or born-digital) and images (PNG/JPEG/TIFF/BMP/WebP,
multi-page TIFF included). Each document gets a work directory

    <work_root>/<doc_id>/
        manifest.json        source path, sha256, page count, render DPI
        pages/p0001.png ...  rendered pages

where doc_id = <sanitised file stem>-<first 16 hex of sha256>, so re-ingesting
the same file is a no-op and two different files with the same name do not
collide (a collision of the 64-bit prefix is detected and refused). Pages are
rendered once at a generous DPI; each reader adapter then resizes to the
resolution its model was trained on. A huge page (a scan wrapped into a PDF
without its real DPI, a 1200-dpi raster) is scaled down to MAX_PAGE_PX, far
above what any reader uses; manifest["page_scale"] records the factor per
page (a PDF page's effective DPI is dpi * page_scale).

Images are normalised to what a scan looks like on paper: EXIF orientation
applied, 16-bit and float samples stretched to 8 bits, transparency composited
onto white. Only TIFF frames are pages; the extra images of other formats (an
MPO JPEG's preview, gain map or depth map, animation frames) are not, nor are
a TIFF's reduced-resolution copies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
import warnings

from PIL import Image, ImageOps

from .schema import atomic_write_text

PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_DPI = 200
MAX_PAGE_PX = 40_000_000        # rendered pixels per page (letter at 300 dpi: 8.4 M)
MAX_SOURCE_PX = 1_000_000_000   # raster inputs decoded up to this size, then capped


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")[:60] or "doc"


def make_doc_id(path: str, digest: str) -> str:
    return f"{_stem(path)}-{digest[:16]}"


def discover(inputs: list[str]) -> list[str]:
    """Expand files / directories / @listfiles into a list of documents.

    Directories, named on the command line or on a line of an @listfile
    (``ls -d papers/*`` lists subfolders too), are walked through symlinks
    (each real directory once), and hidden names are skipped there, macOS
    AppleDouble ``._x.pdf`` files included. Explicitly named files are taken
    as given.

    Every document is listed once. A file reachable by several paths
    (symlinks, ``./``, relative and absolute spellings) is listed under the
    spelling whose absolute path sorts first, and byte-identical copies under
    one name (one doc id) by the first copy, so no two shards share a doc id.
    The list is ordered by real path, not by spelling: run_ocr (handed
    absolute paths by slurm/ocr.sbatch) and tools/incomplete.py (handed the
    user's INPUTS as typed) cut the same shards from it."""
    out: list[str] = []

    def add(item: str) -> None:
        if not os.path.isdir(item):
            out.append(item)
            return
        seen = {os.path.realpath(item)}
        for root, dirs, files in os.walk(item, followlinks=True):
            kept = []
            for d in sorted(dirs):
                real = os.path.realpath(os.path.join(root, d))
                if not d.startswith(".") and real not in seen:   # no cycles
                    seen.add(real)
                    kept.append(d)
            dirs[:] = kept
            out.extend(os.path.join(root, name) for name in files
                       if not name.startswith(".")
                       and os.path.splitext(name)[1].lower() in PDF_EXT | IMAGE_EXT)

    for item in inputs:
        if item.startswith("@"):
            with open(item[1:], encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        add(line.strip())
        else:
            add(item)
    spelling: dict[str, str] = {}
    for p in out:
        real = os.path.realpath(p)
        if real not in spelling or os.path.abspath(p) < os.path.abspath(spelling[real]):
            spelling[real] = p
    return _drop_copies([spelling[real] for real in sorted(spelling)])


def _drop_copies(paths: list[str]) -> list[str]:
    """``paths`` without the later of byte-identical files with one name (the
    same doc id). Only files sharing a name and a size are hashed."""
    groups: dict[tuple, list[str]] = {}
    for p in paths:
        try:
            key: tuple = (_stem(p), os.path.getsize(p))
        except OSError:                 # missing or unreadable: ingest reports it
            key = (p,)
        groups.setdefault(key, []).append(p)
    copies = set()
    for group in (g for g in groups.values() if len(g) > 1):
        seen = set()
        for p in group:
            try:
                doc_id = make_doc_id(p, sha256_file(p))
            except OSError:
                continue
            if doc_id in seen:
                copies.add(p)
            seen.add(doc_id)
    return [p for p in paths if p not in copies]


def _cap(pixels: float) -> float:
    """Scale factor that brings a page of ``pixels`` down to MAX_PAGE_PX."""
    return min(1.0, math.sqrt(MAX_PAGE_PX / pixels)) if pixels > 0 else 1.0


def _save_png(img: Image.Image, dst: str) -> None:
    # A unique temp name: two shards may hold copies of one file (same doc id).
    tmp = f"{dst}.{uuid.uuid4().hex[:12]}.tmp.png"
    try:
        img.save(tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _render_pdf(path: str, pages_dir: str, dpi: int) -> list[float]:
    """Render every page not yet on disk; return each page's scale factor."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        scales = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                w, h = page.get_size()                  # points
                s = _cap(w * h * (dpi / 72) ** 2)
                scales.append(round(s, 4))
                dst = os.path.join(pages_dir, f"p{i + 1:04d}.png")
                if not os.path.exists(dst):
                    _save_png(page.render(scale=dpi / 72 * s).to_pil().convert("RGB"), dst)
            finally:
                page.close()
        return scales
    finally:
        pdf.close()


def _open_source(path: str) -> Image.Image:
    """Image.open, allowing sources up to MAX_SOURCE_PX: Pillow refuses images
    over 179 M pixels as decompression bombs, but a 1200-dpi A3 scan is 280 M.
    (Ingest is single-threaded, so the global limit is safe to lift briefly.)"""
    old = Image.MAX_IMAGE_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        if old is not None:
            Image.MAX_IMAGE_PIXELS = max(old, MAX_SOURCE_PX // 2)  # errors at 2x
        try:
            return Image.open(path)
        finally:
            Image.MAX_IMAGE_PIXELS = old


def _page_frames(im: Image.Image) -> list[int]:
    """Frame indices that are pages: every TIFF frame except reduced-resolution
    copies (NewSubfileType bit 0); for any other format, the first frame."""
    if im.format != "TIFF":
        return [0]
    frames = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        if not int(im.tag_v2.get(254, 0)) & 1:
            frames.append(i)
    return frames or [0]


# EXIF/TIFF Orientation -> the transpose that shows the page upright
_UPRIGHT = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90}


def _upright(im: Image.Image) -> Image.Image:
    """The current frame turned by its Orientation tag. A TIFF keeps one per
    page, which Pillow's EXIF view does not follow from frame to frame."""
    if im.format == "TIFF":
        method = _UPRIGHT.get(im.tag_v2.get(274, 1))
        return im.transpose(method) if method is not None else im.copy()
    try:
        return ImageOps.exif_transpose(im)
    except Exception:   # noqa: BLE001 — malformed EXIF: the pixels are still fine
        return im.copy()


def _to_rgb(im: Image.Image) -> Image.Image:
    """A page as 8-bit RGB on white paper, whatever the source's sample format."""
    if im.mode in ("I", "F") or im.mode.startswith("I;16"):
        # Stretch the values actually used: 12-bit data in a 16-bit container
        # is common, so neither clipping nor a fixed shift is right.
        im = im.convert("F")
        lo, hi = im.getextrema()
        k = 255.0 / (hi - lo) if hi > lo else 0.0
        im = im.point(lambda v: (v - lo) * k + (0 if k else 255)).convert("L")
        im.info.pop("transparency", None)   # a 16-bit colour key means nothing now
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        page = Image.new("RGBA", im.size, "white")
        page.alpha_composite(im.convert("RGBA"))
        im = page
    return im.convert("RGB")


def _render_image(path: str, pages_dir: str) -> list[float]:
    """Write every page not yet on disk; return each page's scale factor."""
    with _open_source(path) as im:
        scales = []
        for k, i in enumerate(_page_frames(im)):
            im.seek(i)
            s = _cap(im.width * im.height)
            scales.append(round(s, 4))
            dst = os.path.join(pages_dir, f"p{k + 1:04d}.png")
            if os.path.exists(dst):
                continue
            page = _to_rgb(_upright(im))
            if s < 1.0:
                size = (max(1, int(page.width * s)), max(1, int(page.height * s)))
                page = page.resize(size, Image.LANCZOS, reducing_gap=3.0)
            _save_png(page, dst)
    return scales


def missing_pages(doc_dir: str, manifest: dict) -> list[str]:
    """The page images a manifest lists that are not on disk (/scratch purges
    file by file, so a work dir can lose its pages and keep its manifest)."""
    pages = manifest.get("pages") or [os.path.join("pages", f"p{i + 1:04d}.png")
                                      for i in range(manifest.get("n_pages") or 0)]
    return [p for p in pages if not os.path.exists(os.path.join(doc_dir, p))]


def _load_manifest(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ingest(path: str, work_root: str, dpi: int = DEFAULT_DPI, force: bool = False,
           digest: str | None = None) -> dict:
    """Render one document into its work directory; return its manifest.

    A document already ingested is returned as is, once any page image it
    lost (a partial purge of /scratch) is rendered again, as it was: the same
    file at the manifest's dpi gives the same page. With ``force`` its pages
    are rendered again (e.g. at a new ``dpi``) and swapped in, and everything
    derived from the old pages (read/, review/, figures/) is deleted.
    ``digest`` is the file's sha256, if the caller has it already."""
    digest = digest or sha256_file(path)
    doc_id = make_doc_id(path, digest)
    doc_dir = os.path.join(work_root, doc_id)
    manifest_path = os.path.join(doc_dir, "manifest.json")
    ext = os.path.splitext(path)[1].lower()
    old = _load_manifest(manifest_path)
    if old is not None and old.get("sha256") != digest:
        raise ValueError(f"doc id {doc_id} is taken by a different file: {old.get('source')}")
    if old is not None and not force:
        if ext in PDF_EXT and dpi != DEFAULT_DPI and old.get("dpi") != dpi:
            print(f"WARNING {doc_id}: pages were rendered at {old.get('dpi')} dpi; "
                  f"--dpi {dpi} takes effect only with `ingest --force`",
                  file=sys.stderr, flush=True)
        if missing_pages(doc_dir, old):
            pages_dir = os.path.join(doc_dir, "pages")
            os.makedirs(pages_dir, exist_ok=True)
            if ext in PDF_EXT:          # both skip the pages still on disk
                _render_pdf(path, pages_dir, old.get("dpi") or dpi)
            elif ext in IMAGE_EXT:
                _render_image(path, pages_dir)
        return old
    if ext not in PDF_EXT | IMAGE_EXT:
        raise ValueError(f"unsupported input type: {path}")

    pages_dir = os.path.join(doc_dir, "pages")
    target = pages_dir + ".new" if force else pages_dir
    if force:
        shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)
    if ext in PDF_EXT:
        scales = _render_pdf(path, target, dpi)
    else:
        scales = _render_image(path, target)
    if force:
        # Swap the new pages in. What was derived from the old pages goes
        # first, then the manifest, so an interruption leaves either the old
        # pages with their manifest or a document that is not ingested yet.
        for sub in ("read", "review", "figures", "pages.old"):
            shutil.rmtree(os.path.join(doc_dir, sub), ignore_errors=True)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        if os.path.exists(pages_dir):
            os.replace(pages_dir, pages_dir + ".old")
        os.replace(target, pages_dir)

    n_pages = len(scales)
    manifest = {
        "doc_id": doc_id,
        "source": os.path.abspath(path),
        "sha256": digest,
        "n_pages": n_pages,
        "dpi": dpi if ext in PDF_EXT else None,
        "page_scale": scales,
        "pages": [os.path.join("pages", f"p{i + 1:04d}.png") for i in range(n_pages)],
    }
    # Written last: its presence means "ingested".
    atomic_write_text(manifest_path, json.dumps(manifest, indent=1))
    shutil.rmtree(pages_dir + ".old", ignore_errors=True)
    return manifest
