"""CLI entry point. Stages, each resumable and independently re-runnable:

    ingest    documents -> page images             (CPU)
    read      stage 1: specialist OCR model         (GPU server: --reader-url)
    review    stage 2: general VLM proofreader      (GPU server: --editor-url)
    assemble  pages -> Markdown + figures + report  (CPU)
    all       the four in sequence (both servers up at once)
    status    per-document progress
    todo      number of pages a stage still has to do

Examples (servers started separately, e.g. by slurm/serve_lib.sh):

    python -m src.run_ocr ingest   --inputs papers/ --work work/
    python -m src.run_ocr read     --inputs papers/ --work work/ \\
        --reader chandra --reader-url http://127.0.0.1:8001
    python -m src.run_ocr review   --inputs papers/ --work work/ \\
        --editor-url http://127.0.0.1:8002
    python -m src.run_ocr assemble --inputs papers/ --work work/ --out out/

Work layout, per document (see ingest.py for doc ids):

    <work>/<doc_id>/manifest.json, pages/*.png
                   /read/p0001.json      stage-1 blocks (+ figure crops)
                   /review/p0001.json    stage-2 blocks (history = provenance)
                   /figures/*.png
    <out>/<doc_id>/<doc_id>.md, figures/, report.json
    <out>/<doc_id>/FAILED.json           the input could not be ingested

Resumability. Every page is written atomically when it finishes, and every
stage skips pages already on disk, so a preempted or requeued job resumes
where it stopped. ``--force`` redoes a stage's finished pages; ``ingest
--force`` also re-renders the page images and so discards that document's
read/, review/ and figures/ (``all --force`` does not re-render). Review
requests are pooled across the whole shard and each page is saved as soon as
all its requests are done. A document whose Markdown is in <out> but whose
work dir is gone (WORK on /scratch is purged) counts as finished, as for
tools/incomplete.py: <out> is the completion record (see select_docs).

Failure semantics. A page whose reading fails deterministically (the server
rejects the request, the adapter cannot parse the reply, the image cannot be
decoded) is retried once with different decoding, then saved as a
placeholder flagged ``page_failed`` so its document still assembles (report.json
lists it; ``read --retry-failed`` tries again). A review request that fails
deterministically is recorded as that block's final decision. A SERVER
failure (unreachable, 5xx, timeouts) is never saved as done: the affected
pages stay in ``todo`` and the command exits 3. So does a SYSTEM failure
while ingesting (disk full, quota, an I/O error): only an input that is
itself unusable gets FAILED.json.

Exit codes: 0 done; 85 stopped by SIGUSR1/SIGTERM after saving (requeue,
see slurm/requeue_lib.sh); 3 model server or system failure (unfinished work
remains); 2 usage error.
"""

from __future__ import annotations

import sys

from .stopflag import STOP, Stopped, check_stop, install_signal_handlers

if __name__ == "__main__":
    # Before the heavy imports: a USR1 forwarded during interpreter start-up
    # must set the flag, not kill the process.
    install_signal_handlers()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

from PIL import Image  # noqa: E402

from . import ingest as ing  # noqa: E402
from .assemble import assemble  # noqa: E402
from .backend import ChatClient, ServerError, map_concurrent  # noqa: E402
from .figures import crop_figures, describe_figure, needs_description  # noqa: E402
from .readers import READERS, get_reader  # noqa: E402
from .review import ReviewPolicy, review_block  # noqa: E402
from .schema import Block, Page, atomic_write_text, page_stem  # noqa: E402
from .validate import validate_block  # noqa: E402

EXIT_REQUEUE = 85
EXIT_SERVER = 3


class ServerDown(Exception):
    """Some work could not be done because a model server failed."""


class SystemFailure(Exception):
    """Some input could not be ingested because of the system (disk full,
    quota, an I/O error), not because of the input: nothing is marked."""


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ doc selection


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad --shard {s!r}: need 0 <= i < n")
    return i, n


def select_docs(args) -> list[str]:
    """Doc dirs this invocation works on: from --inputs (ingesting as needed)
    or every ingested doc under --work; then this shard's slice.

    An input that cannot be ingested gets <out>/<doc_id>/FAILED.json (a
    terminal state tools/incomplete.py counts as done) instead of blocking
    its shard; one that cannot be read at all is skipped. A failure of the
    system rather than of the input (disk full, quota, an I/O error) marks
    nothing: the other inputs are ingested, then SystemFailure is raised.
    A document whose Markdown is in <out> but whose work dir is gone (WORK on
    /scratch is purged) is finished, as tools/incomplete.py says: it is not
    read again, unless --force, or --retry-failed and it has failed pages."""
    i, n = parse_shard(args.shard)
    if args.inputs:
        paths = ing.discover(args.inputs)[i::n]
        ids, system, purged = [], [], 0
        for p in paths:
            check_stop()
            try:
                digest = ing.sha256_file(p)
                doc_id = ing.make_doc_id(p, digest)
                if _finished_and_purged(args, doc_id):
                    purged += 1
                    continue
                ing.ingest(p, args.work, dpi=args.dpi, digest=digest,
                           force=args.force and args.cmd == "ingest")
                ids.append(doc_id)
                stale = os.path.join(args.out or args.work, doc_id, "FAILED.json")
                if os.path.exists(stale):   # an earlier failure was transient
                    os.remove(stale)
            except Exception as e:      # noqa: BLE001 — a corrupt file is marked, loudly
                log(f"INGEST ERROR {p}: {e!r}")
                # A failed system call on a readable input is the system's
                # fault. (Pillow's decoding errors are OSErrors without errno.)
                if isinstance(e, OSError) and e.errno is not None and os.access(p, os.R_OK):
                    system.append(f"{p}: {e!r}")
                else:
                    _mark_ingest_failed(args, p, e)
        if purged:
            log(f"{purged} document(s) already finished in {args.out or args.work} "
                "(work dirs purged): not redone")
        if system:
            raise SystemFailure(f"{len(system)} input(s) not ingested, e.g. {system[0]}")
        ids = list(dict.fromkeys(ids))  # the same file twice is one document
        return [os.path.join(args.work, d) for d in ids]
    docs = sorted(d for d in os.listdir(args.work)
                  if os.path.exists(os.path.join(args.work, d, "manifest.json")))
    return [os.path.join(args.work, d) for d in docs[i::n]]


def _finished_and_purged(args, doc_id: str) -> bool:
    """True if the document is finished in <out> but its work dir is gone:
    then it is not ingested, read and reviewed again (see select_docs)."""
    if args.force or os.path.exists(os.path.join(args.work, doc_id, "manifest.json")):
        return False
    out_dir = os.path.join(args.out or args.work, doc_id)
    if not os.path.exists(os.path.join(out_dir, doc_id + ".md")):
        return False
    return not (getattr(args, "retry_failed", False) and _report_failed_pages(out_dir))


def _report_failed_pages(out_dir: str) -> list:
    try:
        with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
            return json.load(f).get("failed_pages") or []
    except (OSError, ValueError):
        return []


def _mark_ingest_failed(args, path: str, err: Exception) -> None:
    try:
        doc_id = ing.make_doc_id(path, ing.sha256_file(path))
    except OSError as e:                # unreadable: nothing to name it by
        log(f"  (cannot hash {path} either: {e!r}; not marked)")
        return
    d = os.path.join(args.out or args.work, doc_id)
    if os.path.exists(os.path.join(d, doc_id + ".md")):
        log(f"  (not marking {doc_id} failed: its output already exists)")
        return
    os.makedirs(d, exist_ok=True)
    atomic_write_text(os.path.join(d, "FAILED.json"), json.dumps(
        {"doc_id": doc_id, "source": os.path.abspath(path), "stage": "ingest",
         "error": repr(err)[:800]}, indent=1))


def load_manifest(doc_dir: str) -> dict:
    with open(os.path.join(doc_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def load_page(doc_dir: str, stage: str, idx: int) -> Page | None:
    path = os.path.join(doc_dir, stage, page_stem(idx) + ".json")
    if not os.path.exists(path):
        return None
    page = Page.load(path)
    page.image = os.path.join(doc_dir, "pages", page_stem(idx) + ".png")  # relocatable
    return page


def latest_pages(doc_dir: str, n_pages: int) -> list[Page]:
    out = []
    for k in range(n_pages):
        p = load_page(doc_dir, "review", k) or load_page(doc_dir, "read", k)
        if p is not None:
            out.append(p)
    return out


# ------------------------------------------------------------------ stages


def stage_ingest(args):
    docs = select_docs(args)
    log(f"ingested {len(docs)} document(s) under {args.work}")


def _read_one(reader, doc_dir: str, idx: int, retries: int) -> Page:
    """Read one page. Stopped/ServerError propagate (nothing is saved);
    deterministic failures end in a saved placeholder page (see module doc)."""
    check_stop()
    img_path = os.path.join(doc_dir, "pages", page_stem(idx) + ".png")
    errors: list[str] = []
    try:
        with Image.open(img_path) as im:
            img = im.convert("RGB")
    except Exception as e:              # noqa: BLE001 — undecodable page image
        return _save_failed_page(reader, doc_dir, idx, img_path, (0, 0), [repr(e)[:800]])
    best = None
    for attempt in range(retries + 1):
        if attempt:
            check_stop()                # no new generation after a stop request
        try:
            blocks = reader.read(img, attempt=attempt)
        except (Stopped, ServerError):
            raise
        except Exception as e:          # noqa: BLE001 — rejected request / unparseable reply
            errors.append(f"attempt {attempt}: {e!r}"[:800])
            continue
        for b in blocks:
            b.flags = validate_block(b)
        degenerate = sum(1 for b in blocks
                         if {"truncated", "repetition"} & set(b.flags))
        if best is None or degenerate < best[0]:
            best = (degenerate, blocks, attempt)
        if degenerate == 0:
            break
    if best is None:
        return _save_failed_page(reader, doc_dir, idx, img_path, img.size, errors)
    _, blocks, attempt = best
    page = Page(doc_id=os.path.basename(doc_dir), index=idx, image=img_path,
                width=img.width, height=img.height, blocks=blocks,
                reader=reader.tag, stage="read",
                meta={"reader_attempt": attempt, **({"errors": errors} if errors else {})})
    crop_figures([page], doc_dir)
    _save_read(doc_dir, page)
    return page


def _save_failed_page(reader, doc_dir, idx, img_path, size, errors) -> Page:
    log(f"read FAILED {os.path.basename(doc_dir)} page {idx + 1}: {errors[-1]}")
    page = Page(doc_id=os.path.basename(doc_dir), index=idx, image=img_path,
                width=size[0], height=size[1], reader=reader.tag, stage="read",
                blocks=[Block(type="other", content="", source=reader.tag,
                              flags=["page_failed"], meta={"error": errors[-1]})],
                meta={"failed": True, "errors": errors})
    _save_read(doc_dir, page)
    return page


def _save_read(doc_dir: str, page: Page) -> None:
    os.makedirs(os.path.join(doc_dir, "read"), exist_ok=True)
    page.save(os.path.join(doc_dir, "read", page_stem(page.index) + ".json"))
    stale = os.path.join(doc_dir, "review", page_stem(page.index) + ".json")
    if os.path.exists(stale):          # a re-read invalidates the old review
        os.remove(stale)


def _page_failed(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return bool(json.load(f).get("meta", {}).get("failed"))
    except (OSError, ValueError):
        return True                     # unreadable JSON: read it again


def _to_read(args, doc_dir: str, k: int) -> bool:
    """Whether the read stage reads page k (stage_todo counts the same)."""
    path = os.path.join(doc_dir, "read", page_stem(k) + ".json")
    return args.force or not os.path.exists(path) or (args.retry_failed and _page_failed(path))


def _to_review(args, doc_dir: str, k: int) -> bool:
    """Whether the review stage reviews page k: only a page that was read."""
    stem = page_stem(k) + ".json"
    return os.path.exists(os.path.join(doc_dir, "read", stem)) and (
        args.force or not os.path.exists(os.path.join(doc_dir, "review", stem)))


def stage_read(args):
    client = ChatClient(args.reader_url, args.reader_model, timeout=args.timeout,
                        default_extra=_json_arg(args.reader_extra))
    kw = {} if args.reader_max_tokens is None else {"max_tokens": args.reader_max_tokens}
    reader = get_reader(args.reader)(client, **kw)
    log(f"reader {reader.tag} @ {args.reader_url}")
    docs = select_docs(args)
    todo = []
    for d in docs:
        todo += [(d, k) for k in range(load_manifest(d)["n_pages"]) if _to_read(args, d, k)]
    log(f"read: {len(todo)} page(s) to do across {len(docs)} document(s)")
    t0 = time.time()
    results = map_concurrent(lambda dk: _read_one(reader, dk[0], dk[1], args.retries),
                             todo, args.workers)
    server = [r for r in results if isinstance(r, ServerError)]
    for it, r in zip(todo, results):
        if isinstance(r, Exception) and not isinstance(r, (Stopped, ServerError)):
            log(f"read ERROR {it}: {r!r}")      # a bug: the page stays in todo
    ok = [r for r in results if isinstance(r, Page)]
    failed = sum(bool(p.meta.get("failed")) for p in ok)
    log(f"read: {len(ok)}/{len(todo)} pages saved ({failed} as failed placeholders) "
        f"in {time.time() - t0:.0f}s")
    if server:
        raise ServerDown(f"{len(server)} page(s) not read: {server[0]}")


class _ImageCache:
    """Small thread-safe LRU of decoded page images: review items are pooled
    across the shard, and decoding every page up front would not fit in RAM."""

    def __init__(self, maxsize: int):
        self.maxsize, self._d, self._lock = maxsize, OrderedDict(), threading.Lock()

    def get(self, path: str) -> Image.Image:
        with self._lock:
            if path in self._d:
                self._d.move_to_end(path)
                return self._d[path]
        with Image.open(path) as im:
            img = im.convert("RGB")
        with self._lock:
            self._d[path] = img
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        return img


def stage_review(args):
    client = ChatClient(args.editor_url, args.editor_model, timeout=args.timeout,
                        default_extra=_json_arg(args.editor_extra))
    tag = f"reviewer:{client.model}"
    policy = ReviewPolicy(
        review_types=set(filter(None, args.review_types.split(","))),
        flagged=not args.no_flagged, max_change=args.max_change,
        max_change_flagged=args.max_change_flagged)
    log(f"reviewer {tag} @ {args.editor_url}; types={sorted(policy.review_types)} "
        f"flagged={policy.flagged} figures={args.describe_figures}")

    # One pool for the whole shard: every wanted block and every figure of
    # every page not yet reviewed. A page is saved the moment its last item
    # finishes, so a stop or a server failure loses only unfinished pages.
    pages: dict[tuple, Page] = {}
    pending: dict[tuple, int] = {}
    items: list[tuple] = []
    for d in select_docs(args):
        for k in range(load_manifest(d)["n_pages"]):
            if not _to_review(args, d, k):
                continue
            p = load_page(d, "read", k)
            its = [("block", i) for i, b in enumerate(p.blocks) if policy.wants(b)]
            if args.describe_figures:
                its += [("figure", i) for i, b in enumerate(p.blocks) if needs_description(b)]
            pages[(d, k)], pending[(d, k)] = p, len(its)
            items += [((d, k), kind, i) for kind, i in its]
    log(f"review: {len(items)} request(s) over {len(pages)} page(s)")

    def save(key):
        p = pages[key]
        p.stage = "review"
        os.makedirs(os.path.join(key[0], "review"), exist_ok=True)
        p.save(os.path.join(key[0], "review", page_stem(p.index) + ".json"))

    saved, incomplete, server = 0, set(), []
    for key, n in pending.items():
        if n == 0:
            save(key)
            saved += 1
    images = _ImageCache(maxsize=max(16, 2 * args.workers))

    def run(item):
        key, kind, i = item
        p = pages[key]
        if kind == "block":
            review_block(client, images.get(p.image), p.blocks[i], policy, tag)
        else:
            describe_figure(client, p, i, tag)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(run, it): it for it in items}
        for fut in as_completed(futures):
            key, kind, i = futures[fut]
            exc = fut.exception()
            if isinstance(exc, ServerError):
                server.append(exc)
                incomplete.add(key)
            elif isinstance(exc, Stopped):
                incomplete.add(key)
            elif exc is not None:       # review_block/describe_figure record their own
                log(f"review ERROR {key} {kind} {i}: {exc!r}")
                incomplete.add(key)     # a bug: keep the page in todo, visibly
            pending[key] -= 1
            if pending[key] == 0 and key not in incomplete:
                save(key)
                saved += 1
    decisions = Counter(b.meta.get("reviewed") for key, p in pages.items()
                        if key not in incomplete for b in p.blocks if b.meta.get("reviewed"))
    errors = sum(1 for key, p in pages.items() if key not in incomplete
                 for b in p.blocks if b.meta.get("description_error"))
    log(f"review: {saved}/{len(pages)} pages saved in {time.time() - t0:.0f}s; "
        f"decisions {dict(decisions)}; figure errors {errors}; "
        f"{len(incomplete)} page(s) left for the next run")
    if server:
        raise ServerDown(f"{len(server)} request(s) failed: {server[0]}")


def stage_assemble(args):
    out_root = args.out or args.work
    for d in select_docs(args):
        man = load_manifest(d)
        pages = latest_pages(d, man["n_pages"])
        if len(pages) < man["n_pages"]:
            log(f"assemble {man['doc_id']}: only {len(pages)}/{man['n_pages']} "
                "pages read — skipping")
            continue
        out_dir = os.path.join(out_root, man["doc_id"])
        os.makedirs(out_dir, exist_ok=True)
        md = assemble(pages, page_markers=not args.no_page_markers)
        # Figures and report first; the .md is the completion marker
        # (tools/incomplete.py, status), so it is written last.
        src_fig, dst_fig = os.path.join(d, "figures"), os.path.join(out_dir, "figures")
        if os.path.isdir(src_fig) and os.path.abspath(src_fig) != os.path.abspath(dst_fig):
            shutil.copytree(src_fig, dst_fig, dirs_exist_ok=True)
        atomic_write_text(os.path.join(out_dir, "report.json"),
                          json.dumps(make_report(man, pages), indent=1))
        atomic_write_text(os.path.join(out_dir, man["doc_id"] + ".md"), md)
        log(f"assembled {out_dir}/{man['doc_id']}.md")


def make_report(man: dict, pages: list[Page]) -> dict:
    blocks = [b for p in pages for b in p.blocks]
    return {
        "doc_id": man["doc_id"], "source": man["source"], "n_pages": man["n_pages"],
        "readers": sorted({p.reader for p in pages}),
        "reviewed_pages": sum(p.stage == "review" for p in pages),
        "failed_pages": [p.index + 1 for p in pages if p.meta.get("failed")],
        "truncated_pages": [p.index + 1 for p in pages if any(
            b.meta.get("truncated_tail") and not b.content.strip() for b in p.blocks)],
        "truncated_recovered": [p.index + 1 for p in pages if any(
            b.meta.get("tail_recovered") for b in p.blocks)],
        "review_errors": sum(1 for b in blocks if b.meta.get("reviewed") == "error"
                             or b.meta.get("description_error")),
        "block_types": dict(Counter(b.type for b in blocks)),
        "review": dict(Counter(b.meta.get("reviewed") for b in blocks
                               if b.meta.get("reviewed"))),
        "open_flags": dict(Counter(f for b in blocks for f in b.flags)),
        "flagged_blocks": [
            {"page": p.index + 1, "block": k, "type": b.type, "flags": b.flags}
            for p in pages for k, b in enumerate(p.blocks) if b.flags],
    }


def stage_status(args):
    docs = select_docs(args)
    out_root = args.out or args.work
    rows = []
    for d in docs:
        man = load_manifest(d)
        n = man["n_pages"]
        count = lambda s: sum(os.path.exists(os.path.join(d, s, page_stem(k) + ".json"))
                              for k in range(n))
        md = os.path.exists(os.path.join(out_root, man["doc_id"], man["doc_id"] + ".md"))
        rows.append((man["doc_id"], n, count("read"), count("review"), md))
    print(f"{'doc_id':50s} {'pages':>5s} {'read':>5s} {'review':>6s} md")
    for r in rows:
        print(f"{r[0][:50]:50s} {r[1]:5d} {r[2]:5d} {r[3]:6d} {'yes' if r[4] else '-'}")
    done = sum(r[4] for r in rows)
    print(f"{done}/{len(rows)} documents assembled")
    return rows


def stage_todo(args):
    """Print the number of pages a stage still has to do, counted as that
    stage counts them with the same --force / --retry-failed (used by the
    sbatch script to skip starting a model server for a finished phase)."""
    wanted = _to_read if args.stage == "read" else _to_review
    n = sum(wanted(args, d, k) for d in select_docs(args)
            for k in range(load_manifest(d)["n_pages"]))
    print(n)
    return n


def _json_arg(s: str | None) -> dict:
    if not s:
        return {}
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"bad JSON {s!r}: {e}")
    if not isinstance(d, dict):
        raise SystemExit(f"expected a JSON object, got {s!r}")
    return d


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.run_ocr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--inputs", nargs="+", help="files, dirs, or @listfile")
        p.add_argument("--work", default="work", help="work root (page images, JSON)")
        p.add_argument("--out", help="output root for Markdown (default: --work)")
        p.add_argument("--shard", help="i/N: this process's slice of the documents")
        p.add_argument("--dpi", type=int, default=ing.DEFAULT_DPI)
        p.add_argument("--workers", type=int, default=32,
                       help="concurrent requests to the model server")
        p.add_argument("--timeout", type=float, default=900.0)
        p.add_argument("--force", action="store_true", help="redo finished pages")

    def reader_args(p):
        p.add_argument("--reader", default="dots", choices=sorted(READERS))
        p.add_argument("--reader-url", default="http://127.0.0.1:8001")
        p.add_argument("--reader-model", help="served model name (default: ask server)")
        p.add_argument("--reader-max-tokens", type=int,
                       help="output token cap per page (default: the adapter's)")
        p.add_argument("--reader-extra", help="JSON merged into every reader request")
        p.add_argument("--retries", type=int, default=1,
                       help="re-reads of a page whose output looped, was cut off, "
                            "or failed (then a placeholder is saved)")
        p.add_argument("--retry-failed", action="store_true",
                       help="also re-read pages saved as failed placeholders")

    def editor_args(p):
        p.add_argument("--editor-url", default="http://127.0.0.1:8002")
        p.add_argument("--editor-model", help="served model name (default: ask server)")
        p.add_argument("--editor-extra", help="JSON merged into every reviewer request, "
                       "e.g. '{\"chat_template_kwargs\": {\"enable_thinking\": false}}'")
        p.add_argument("--review-types", default="formula",
                       help="block types always reviewed, comma-separated "
                            "(flagged blocks are reviewed regardless)")
        p.add_argument("--no-flagged", action="store_true",
                       help="do not review blocks just because validators flagged them")
        p.add_argument("--max-change", type=float, default=0.35)
        p.add_argument("--max-change-flagged", type=float, default=0.6)
        p.add_argument("--no-describe-figures", dest="describe_figures",
                       action="store_false")

    def assemble_args(p):
        p.add_argument("--no-page-markers", action="store_true")

    for name, extra in (("ingest", []), ("read", [reader_args]),
                        ("review", [editor_args]), ("assemble", [assemble_args]),
                        ("all", [reader_args, editor_args, assemble_args]),
                        ("status", []), ("todo", [])):
        p = sub.add_parser(name)
        common(p)
        for f in extra:
            f(p)
        if name == "todo":
            p.add_argument("--stage", choices=["read", "review"], required=True)
            p.add_argument("--retry-failed", action="store_true",
                           help="count failed placeholders too, as read --retry-failed does")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    install_signal_handlers()
    stages = {"ingest": [stage_ingest], "read": [stage_read],
              "review": [stage_review], "assemble": [stage_assemble],
              "all": [stage_ingest, stage_read, stage_review, stage_assemble],
              "status": [stage_status], "todo": [stage_todo]}[args.cmd]
    try:
        for st in stages:
            st(args)
            check_stop()
    except Stopped:
        log(f"stopped by signal; finished pages saved — exit {EXIT_REQUEUE} (requeue)")
        return EXIT_REQUEUE
    except SystemFailure as e:
        log(f"SYSTEM FAILURE: {e}. Nothing was marked failed; fix the cause (space, "
            f"quota, file system) and run again — exit {EXIT_SERVER}")
        return EXIT_SERVER
    except (ServerDown, ServerError) as e:
        if STOP.is_set():   # at preemption the server is killed too: requeue
            log(f"stopped by signal (server gone too: {e}) — exit {EXIT_REQUEUE}")
            return EXIT_REQUEUE
        log(f"MODEL SERVER FAILURE: {e}. Finished pages are saved; the rest stay "
            f"in todo — exit {EXIT_SERVER}")
        return EXIT_SERVER
    return 0


if __name__ == "__main__":
    sys.exit(main())
