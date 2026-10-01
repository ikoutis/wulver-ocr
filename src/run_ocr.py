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
        --reader dots --reader-url http://127.0.0.1:8001
    python -m src.run_ocr review   --inputs papers/ --work work/ \\
        --editor-url http://127.0.0.1:8002
    python -m src.run_ocr assemble --inputs papers/ --work work/ --out out/

Work layout, per document (see ingest.py for doc ids):

    <work>/<doc_id>/manifest.json, pages/*.png
                   /read/p0001.json      stage-1 blocks (+ figure crops)
                   /review/p0001.json    stage-2 blocks (history = provenance)
                   /figures/*.png
    <out>/<doc_id>/<doc_id>.md, figures/, report.json

Every page is written atomically when it finishes, and every stage skips
pages already on disk (``--force`` redoes them), so a preempted or requeued
job resumes where it stopped. On SIGUSR1/SIGTERM the current stage finishes
the requests in flight, saves, and exits with code 85 — the convention
slurm/requeue_lib.sh turns into a self-requeue.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from collections import Counter

from PIL import Image

from . import ingest as ing
from .assemble import assemble
from .backend import ChatClient, map_concurrent
from .figures import crop_figures, describe_figures
from .readers import READERS, get_reader
from .review import ReviewPolicy, review_pages
from .schema import Page, page_stem
from .validate import validate_block

EXIT_REQUEUE = 85
STOP = {"flag": False}


class Stopped(Exception):
    pass


def _on_signal(signum, _frame):
    if not STOP["flag"]:
        print(f"[signal {signum}] finishing in-flight work, then exiting "
              f"{EXIT_REQUEUE}", file=sys.stderr, flush=True)
    STOP["flag"] = True


def _check_stop():
    if STOP["flag"]:
        raise Stopped()


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
    or every ingested doc under --work; then this shard's slice."""
    i, n = parse_shard(args.shard)
    if args.inputs:
        paths = ing.discover(args.inputs)[i::n]
        ids = []
        for p in paths:
            _check_stop()
            try:
                ids.append(ing.ingest(p, args.work, dpi=args.dpi)["doc_id"])
            except Exception as e:      # noqa: BLE001 — a corrupt file skips, loudly
                log(f"INGEST ERROR {p}: {e!r}")
        return [os.path.join(args.work, d) for d in ids]
    docs = sorted(d for d in os.listdir(args.work)
                  if os.path.exists(os.path.join(args.work, d, "manifest.json")))
    return [os.path.join(args.work, d) for d in docs[i::n]]


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
    _check_stop()
    img_path = os.path.join(doc_dir, "pages", page_stem(idx) + ".png")
    with Image.open(img_path) as im:
        img = im.convert("RGB")
    best = None
    for attempt in range(retries + 1):
        blocks = reader.read(img, attempt=attempt)
        for b in blocks:
            b.flags = validate_block(b)
        degenerate = sum(1 for b in blocks
                         if {"truncated", "repetition"} & set(b.flags))
        if best is None or degenerate < best[0]:
            best = (degenerate, blocks, attempt)
        if degenerate == 0:
            break
    _, blocks, attempt = best
    page = Page(doc_id=os.path.basename(doc_dir), index=idx, image=img_path,
                width=img.width, height=img.height, blocks=blocks,
                reader=reader.tag, stage="read",
                meta={"reader_attempt": attempt})
    crop_figures([page], doc_dir)
    os.makedirs(os.path.join(doc_dir, "read"), exist_ok=True)
    page.save(os.path.join(doc_dir, "read", page_stem(idx) + ".json"))
    stale = os.path.join(doc_dir, "review", page_stem(idx) + ".json")
    if os.path.exists(stale):          # a re-read invalidates the old review
        os.remove(stale)
    return page


def stage_read(args):
    client = ChatClient(args.reader_url, args.reader_model, timeout=args.timeout,
                        default_extra=_json_arg(args.reader_extra))
    kw = {} if args.reader_max_tokens is None else {"max_tokens": args.reader_max_tokens}
    reader = get_reader(args.reader)(client, **kw)
    log(f"reader {reader.tag} @ {args.reader_url}")
    docs = select_docs(args)
    todo = []
    for d in docs:
        n = load_manifest(d)["n_pages"]
        for k in range(n):
            done = os.path.exists(os.path.join(d, "read", page_stem(k) + ".json"))
            if args.force or not done:
                todo.append((d, k))
    log(f"read: {len(todo)} page(s) to do across {len(docs)} document(s)")
    t0 = time.time()
    results = map_concurrent(lambda dk: _read_one(reader, dk[0], dk[1], args.retries),
                             todo, args.workers)
    _report_failures("read", todo, results)
    ok = sum(1 for r in results if isinstance(r, Page))
    log(f"read: {ok}/{len(todo)} pages in {time.time() - t0:.0f}s")


def stage_review(args):
    client = ChatClient(args.editor_url, args.editor_model, timeout=args.timeout,
                        default_extra=_json_arg(args.editor_extra))
    tag = f"reviewer:{client.model}"
    policy = ReviewPolicy(
        review_types=set(filter(None, args.review_types.split(","))),
        flagged=not args.no_flagged, max_change=args.max_change,
        max_change_flagged=args.max_change_flagged)
    log(f"reviewer {tag} @ {args.editor_url}; types={sorted(policy.review_types)} "
        f"flagged={policy.flagged}")
    docs = select_docs(args)
    for d in docs:
        _check_stop()
        n = load_manifest(d)["n_pages"]
        pages = []
        for k in range(n):
            if not args.force and os.path.exists(
                    os.path.join(d, "review", page_stem(k) + ".json")):
                continue
            p = load_page(d, "read", k)
            if p is not None:
                pages.append(p)
        if not pages:
            continue
        review_pages(client, pages, policy, tag, workers=args.workers)
        if args.describe_figures:
            describe_figures(client, pages, tag, workers=args.workers)
        # A stop mid-document discards this document's review (it is redone
        # whole on resume) rather than saving pages reviewed only in part.
        _check_stop()
        os.makedirs(os.path.join(d, "review"), exist_ok=True)
        for p in pages:
            p.save(os.path.join(d, "review", page_stem(p.index) + ".json"))
        c = Counter(b.meta.get("reviewed") for p in pages for b in p.blocks
                    if b.meta.get("reviewed"))
        log(f"review {os.path.basename(d)}: {len(pages)} pages, {dict(c)}")


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
        tmp = os.path.join(out_dir, man["doc_id"] + ".md.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(md)
        os.replace(tmp, os.path.join(out_dir, man["doc_id"] + ".md"))
        src_fig, dst_fig = os.path.join(d, "figures"), os.path.join(out_dir, "figures")
        if os.path.isdir(src_fig) and os.path.abspath(src_fig) != os.path.abspath(dst_fig):
            shutil.copytree(src_fig, dst_fig, dirs_exist_ok=True)
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(make_report(man, pages), f, indent=1)
        log(f"assembled {out_dir}/{man['doc_id']}.md")


def make_report(man: dict, pages: list[Page]) -> dict:
    blocks = [b for p in pages for b in p.blocks]
    return {
        "doc_id": man["doc_id"], "source": man["source"], "n_pages": man["n_pages"],
        "readers": sorted({p.reader for p in pages}),
        "reviewed_pages": sum(p.stage == "review" for p in pages),
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
    """Print the number of pages a stage still has to do (used by the sbatch
    script to skip starting a model server for a finished phase)."""
    n = 0
    for d in select_docs(args):
        man = load_manifest(d)
        for k in range(man["n_pages"]):
            stem = page_stem(k) + ".json"
            read = os.path.exists(os.path.join(d, "read", stem))
            if args.stage == "read":
                n += not read
            else:   # only pages that have been read can be reviewed
                n += read and not os.path.exists(os.path.join(d, "review", stem))
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


def _report_failures(stage, items, results):
    for it, r in zip(items, results):
        if isinstance(r, Stopped):
            continue
        if isinstance(r, Exception):
            log(f"{stage} ERROR {it}: {r!r}")


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
                       help="re-reads of a page whose output looped or was cut off")

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
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    stages = {"ingest": [stage_ingest], "read": [stage_read],
              "review": [stage_review], "assemble": [stage_assemble],
              "all": [stage_ingest, stage_read, stage_review, stage_assemble],
              "status": [stage_status], "todo": [stage_todo]}[args.cmd]
    try:
        for st in stages:
            st(args)
            _check_stop()
    except Stopped:
        log(f"stopped by signal; state saved — exit {EXIT_REQUEUE} (requeue)")
        return EXIT_REQUEUE
    return 0


if __name__ == "__main__":
    sys.exit(main())
