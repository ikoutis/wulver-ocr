# wulver-ocr — scanned papers → Markdown + LaTeX, on Wulver's A100s

A self-hosted OCR pipeline for research papers: scans (or PDFs) in,
Markdown out, with display and inline math as LaTeX, tables as GFM/HTML, and
figures cropped, linked, and described. Graph drawings come out twice, marked
as such: as simple Markdown (vertex and edge lists) and as TikZ. Diagrams get
Mermaid, and commutative diagrams get tikz-cd.

It pairs two models that are good at different things:

| role | model class | job |
|---|---|---|
| **reader** (stage 1) | document-OCR specialist: **Chandra OCR 2** (4B), or the MIT-licensed **dots.mocr** (3B) | reads every page in one pass: layout boxes, reading order, text, LaTeX, and HTML tables. It is fast and pixel-faithful. |
| **reviewer** (stage 2) | general VLM: **Qwen3.8-27B** (fallback: Qwen3-VL-32B) | proofreads only what needs it, one image crop at a time: every display formula, every block a validator flagged, and every figure (to describe it). |

The reviewer never rewrites a page. A proposed edit replaces the reader's text
only if it passes a **gate**: it must add no validator flag the draft didn't
already have, and it must change at most a bounded fraction of the draft. The
validators include a KaTeX parse of every formula.
Every decision (agreed, edited, rejected, unreadable) is kept in the page
JSON, so each output block can be traced to the model that wrote it.

**Start here:**
- [`dev-communication/design.md`](dev-communication/design.md) covers the proposal: why
  two models, the model choices and the evidence behind them, the Wulver
  deployment, the output format, and the evaluation plan.
- [`dev-communication/log.md`](dev-communication/log.md) is the running task/reply log
  (`[O-00N]` IDs, same conventions as the dml repo).

## Repository layout

```
src/
  ├── ingest.py       PDF/images → page PNGs + manifest (content-addressed doc ids)
  ├── readers/        stage-1 adapters → common Block schema: chandra (HTML layout),
  │                   dots (layout JSON), markdown (whole-page Markdown, any VLM)
  ├── validate.py     CPU checks: LaTeX structure, repetition loops, table shape, …
  ├── katex_check.py  optional KaTeX parse of every formula (persistent node worker)
  ├── review.py       stage-2 gated proofreading (prompts, gate, provenance)
  ├── figures.py      figure crops + generated descriptions / structure
  ├── tikz.py         graph TikZ: canonical form, parser, checks, Markdown rendering
  ├── assemble.py     blocks → Markdown (running heads dropped, page-break joins)
  ├── backend.py      OpenAI-compatible HTTP client (talks to `vllm serve`)
  ├── schema.py       Page / Block data model (the JSON every stage reads/writes)
  └── run_ocr.py      CLI: ingest | read | review | assemble | all | status | todo
profiles/             model pairs: default (Chandra 2 + Qwen3.8-27B), dots (MIT reader),
                      conservative (dots.mocr + Qwen3-VL-32B); repo ids, vLLM flags
slurm/                ocr.sbatch (sharded array), serve_lib.sh, requeue_lib.sh
tools/                setup_env.sh, stage_models.py, incomplete.py
eval/                 degrade.py (scan simulation for the arXiv eval set)
tests/                CPU-only unit + end-to-end tests (fake model servers)
dev-communication/    design doc + dated task/reply log
```

## Install

Local (for development and tests; no GPU or model needed):

```bash
pip install -r requirements.txt
pytest tests/
```

Wulver (once; follows the dml repo's conventions):

```bash
bash tools/setup_env.sh                          # conda env at /project/ikoutis/conda_env/wocr (+ vLLM, KaTeX)
python tools/stage_models.py --profile default   # ~65 GB of weights → /project/ikoutis/wocr_models
```

## Running

Batch, sharded across array tasks. Each task takes 1/N of the documents and
runs ingest → read → review → assemble on one A100, loading one model at a
time. Export the settings, so the recovery commands below see the same
values (INPUTS may list several paths, separated by spaces):

```bash
export INPUTS=/project/ikoutis/$USER/papers
export OUT=/project/ikoutis/$USER/wocr/out        # the default; set it per run, e.g. .../runs/o00N_<what>
sbatch --array=0-7 slurm/ocr.sbatch
# outputs: $OUT/<doc_id>/<doc_id>.md (+ figures/, report.json)
```

Recover from anything (preemption, the wall-clock limit, node failure, a
model server crash) by resubmitting only the unfinished shards. Every stage
skips pages that are already done:

```bash
IDS=$(python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT")
[ -n "$IDS" ] && NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
```

(`$INPUTS` is deliberately unquoted: the sbatch script splits it on spaces
the same way.)

Interactive, from a GPU session (`interactive -a ikoutis -q standard -j gpu`),
from the repo root:

```bash
INPUTS=paper.pdf OUT=out WORK=work bash slurm/ocr.sbatch
```

Batch settings are environment variables read by `slurm/ocr.sbatch`:

| variable | meaning |
|---|---|
| `INPUTS` | documents: dirs, files, or `@listfile`, space-separated (required) |
| `OUT`, `WORK` | output root (Markdown) and work root (page images, JSON); `WORK` defaults to a per-profile directory on `/scratch` |
| `WOCR_PROFILE` | model pair: `default` (Chandra 2 + Qwen3.8-27B), `dots` (MIT reader), `conservative` |
| `PHASES` | subset of `read review assemble`, e.g. `PHASES="read assemble"` runs the reader only |
| `READ_ARGS`, `REVIEW_ARGS` | extra flags for the read / review stage, e.g. `REVIEW_ARGS="--review-types formula,table"` |
| `NSHARDS` | total shard count when resubmitting a subset of array indices |

Stage by stage, against servers you started yourself (any OpenAI-compatible
endpoint works):

```bash
python -m src.run_ocr read     --inputs papers/ --work work --reader chandra --reader-url http://127.0.0.1:8001
python -m src.run_ocr review   --inputs papers/ --work work --editor-url http://127.0.0.1:8002 \
    --editor-extra '{"chat_template_kwargs": {"enable_thinking": false}}'
python -m src.run_ocr assemble --inputs papers/ --work work --out out
python -m src.run_ocr status   --work work --out out
```

Useful stage flags (pass them through `READ_ARGS` / `REVIEW_ARGS` in batch):

- read: `--retries 1` (re-reads of a page whose output looped, was cut off, or failed) and `--retry-failed` (re-read pages saved as failed placeholders);
- review: `--review-types formula,table` (review all tables too), `--no-flagged`, `--max-change 0.35`, and `--no-describe-figures`.

**When things fail.** A page the reader cannot read is retried once, then
saved as a placeholder flagged `page_failed`, so its document still
assembles (report.json lists it). An input file that cannot be opened gets
`$OUT/<doc_id>/FAILED.json`. A review request the server rejects becomes that
block's final decision. If a model server dies, nothing unfinished is saved
as done: the stage exits with code 3 and the next submission picks up where
it stopped. On preemption or the wall clock the stage exits 85 after saving
every finished page, and the task requeues itself.

## Output

```
out/<doc_id>/
  <doc_id>.md      Markdown: $..$ / $$..$$ math, <!-- page N --> markers;
                   tables as HTML (merged cells survive; GFM from the
                   markdown reader); graph drawings twice, marked:
                   "Graph — Markdown (simple)" and "Graph — TikZ"
  figures/*.png    figure crops, linked from the Markdown
  report.json      per-document summary: block types, review decisions,
                   failed pages, review errors, remaining flags (where a
                   human should look first)
out/<doc_id>/FAILED.json   instead, if the input could not be ingested
```

The per-page provenance (every block's model, flags, and edit history) stays
in `$WORK/<doc_id>/{read,review}/p*.json`.
