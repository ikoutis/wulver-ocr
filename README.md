# wulver-ocr — scanned papers → Markdown + LaTeX, on Wulver's A100s

A self-hosted OCR pipeline for research papers: scans (or PDFs) in,
Markdown out, with display and inline math as LaTeX, tables as GFM/HTML, and
figures cropped, linked, and described. Graph drawings get an edge list,
diagrams get Mermaid, and commutative diagrams get tikz-cd.

It pairs two models that are good at different things:

| role | model class | job |
|---|---|---|
| **reader** (stage 1) | small document-OCR specialist (~1–3B) | reads every page in one pass: layout boxes, reading order, text, LaTeX, and HTML tables. It is fast and pixel-faithful. |
| **reviewer** (stage 2) | large general VLM (~30B) | proofreads only what needs it, one image crop at a time: every display formula, every block a validator flagged, and every figure (to describe it). |

The reviewer never rewrites a page. A proposed edit replaces the reader's text
only if it passes a **gate**: it must add no validator flag the draft didn't
already have, and it must change at most a bounded fraction of the draft.
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
  ├── readers/        stage-1 adapters, one per OCR model → common Block schema
  ├── validate.py     CPU checks: LaTeX structure, repetition loops, table shape, …
  ├── review.py       stage-2 gated proofreading (prompts, gate, provenance)
  ├── figures.py      figure crops + generated descriptions / structure
  ├── assemble.py     blocks → Markdown (running heads dropped, page-break joins)
  ├── backend.py      OpenAI-compatible HTTP client (talks to `vllm serve`)
  ├── schema.py       Page / Block data model (the JSON every stage reads/writes)
  └── run_ocr.py      CLI: ingest | read | review | assemble | all | status | todo
profiles/             model pairs (repo ids, served names, vLLM flags)
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
bash tools/setup_env.sh                          # conda env at /project/ikoutis/conda_env/wocr (+ vLLM)
python tools/stage_models.py --profile default   # weights → /project/ikoutis/wocr_models
```

## Running

Batch, sharded across array tasks. Each task takes 1/N of the documents and
runs ingest → read → review → assemble on one A100, loading one model at a
time:

```bash
INPUTS=/project/ikoutis/$USER/papers sbatch --array=0-7 slurm/ocr.sbatch
# outputs: /project/ikoutis/$USER/wocr/out/<doc_id>/<doc_id>.md (+ figures/, report.json)
```

Recover from anything (preemption, the wall-clock limit, node failure) by
resubmitting only the unfinished shards. Every stage skips pages that are
already done:

```bash
IDS=$(python tools/incomplete.py --inputs "$INPUTS" --nshards 8)
[ -n "$IDS" ] && NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
```

Interactive, from a GPU session (`interactive -a ikoutis -q standard -j gpu`):

```bash
INPUTS=paper.pdf OUT=out WORK=work bash slurm/ocr.sbatch
```

Stage by stage, against servers you started yourself (any OpenAI-compatible
endpoint works):

```bash
python -m src.run_ocr read     --inputs papers/ --work work --reader dots --reader-url http://127.0.0.1:8001
python -m src.run_ocr review   --inputs papers/ --work work --editor-url http://127.0.0.1:8002
python -m src.run_ocr assemble --inputs papers/ --work work --out out
python -m src.run_ocr status   --work work
```

Useful knobs: `--review-types formula,table` (review all tables too),
`--no-flagged`, `--max-change 0.35`, `--no-describe-figures`, `--retries 1`
(re-read a page whose output looped), and `PHASES="read assemble"` (run
the reader only).

## Output

```
out/<doc_id>/
  <doc_id>.md      Markdown: $..$ / $$..$$ math, <!-- page N --> markers
  figures/*.png    figure crops, linked from the Markdown
  report.json      per-document summary: block types, review decisions,
                   remaining flags (where a human should look first)
```

The per-page provenance (every block's model, flags, and edit history) stays
in `work/<doc_id>/{read,review}/p*.json`.
