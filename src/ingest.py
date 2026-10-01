"""Stage 0: documents -> page images + a manifest.

Accepts PDFs (scanned or born-digital) and images (PNG/JPEG/TIFF, multi-page
TIFF included). Each document gets a work directory

    <work_root>/<doc_id>/
        manifest.json        source path, sha256, page count, render DPI
        pages/p0001.png ...  rendered pages

where doc_id = <sanitised file stem>-<first 8 hex of sha256>, so re-ingesting
the same file is a no-op and two different files with the same name never
collide. Pages are rendered once at a generous DPI; each reader adapter then
resizes to the resolution its model was trained on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from PIL import Image

PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_DPI = 200


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_doc_id(path: str, digest: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")[:60] or "doc"
    return f"{stem}-{digest[:8]}"


def discover(inputs: list[str]) -> list[str]:
    """Expand files / directories / @listfiles into a sorted list of documents."""
    out: list[str] = []
    for item in inputs:
        if item.startswith("@"):
            with open(item[1:], encoding="utf-8") as f:
                out.extend(line.strip() for line in f
                           if line.strip() and not line.startswith("#"))
        elif os.path.isdir(item):
            for root, _, files in os.walk(item):
                for name in files:
                    if os.path.splitext(name)[1].lower() in PDF_EXT | IMAGE_EXT:
                        out.append(os.path.join(root, name))
        else:
            out.append(item)
    return sorted(dict.fromkeys(out))


def _render_pdf(path: str, pages_dir: str, dpi: int) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        n = len(pdf)
        for i in range(n):
            dst = os.path.join(pages_dir, f"p{i + 1:04d}.png")
            if os.path.exists(dst):
                continue
            page = pdf[i]
            img = page.render(scale=dpi / 72).to_pil().convert("RGB")
            page.close()
            img.save(dst + ".tmp.png")
            os.replace(dst + ".tmp.png", dst)
        return n
    finally:
        pdf.close()


def _render_image(path: str, pages_dir: str) -> int:
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for i in range(n):
            dst = os.path.join(pages_dir, f"p{i + 1:04d}.png")
            if os.path.exists(dst):
                continue
            im.seek(i)
            im.convert("RGB").save(dst + ".tmp.png")
            os.replace(dst + ".tmp.png", dst)
    return n


def ingest(path: str, work_root: str, dpi: int = DEFAULT_DPI) -> dict:
    """Render one document into its work directory; return its manifest."""
    digest = sha256_file(path)
    doc_id = make_doc_id(path, digest)
    doc_dir = os.path.join(work_root, doc_id)
    manifest_path = os.path.join(doc_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    pages_dir = os.path.join(doc_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXT:
        n_pages = _render_pdf(path, pages_dir, dpi)
    elif ext in IMAGE_EXT:
        n_pages = _render_image(path, pages_dir)
    else:
        raise ValueError(f"unsupported input type: {path}")

    manifest = {
        "doc_id": doc_id,
        "source": os.path.abspath(path),
        "sha256": digest,
        "n_pages": n_pages,
        "dpi": dpi if ext in PDF_EXT else None,
        "pages": [os.path.join("pages", f"p{i + 1:04d}.png") for i in range(n_pages)],
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, manifest_path)   # written last: its presence means "ingested"
    return manifest
