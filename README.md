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
  │                   dots (layout JSON), markdown (whole-page Markdown, any VLM), olmocr
  ├── validate.py     CPU checks: LaTeX structure, repetition loops, table shape, …
  ├── katex_check.py  optional KaTeX parse of every formula (persistent node worker)
  ├── review.py       stage-2 gated proofreading (prompts, gate, provenance)
  ├── figures.py      figure crops + generated descriptions / structure
  ├── tikz.py         graph TikZ: canonical form, parser, checks, Markdown rendering
  ├── assemble.py     blocks → Markdown (running heads dropped, page-break joins)
  ├── backend.py      OpenAI-compatible HTTP client (talks to `vllm serve`): typed
  │                   server/request errors, retries, circuit breaker
  ├── stopflag.py     cooperative stop on SIGUSR1/SIGTERM (requeue without losing work)
  ├── schema.py       Page / Block data model (the JSON every stage reads/writes)
  └── run_ocr.py      CLI: ingest | read | review | assemble | all | status | todo
profiles/             model pairs: default (Chandra 2 + Qwen3.8-27B), dots (MIT reader),
                      conservative (dots.mocr + Qwen3-VL-32B), olmocr (baseline); vLLM flags
slurm/                ocr.sbatch (sharded array), serve_lib.sh, requeue_lib.sh
tools/                setup_env.sh, stage_models.py, incomplete.py
eval/                 degrade.py (scan simulation for the arXiv eval set)
tests/                CPU-only unit + end-to-end tests (fake model servers, fake
                      vllm/scontrol for the SLURM scripts)
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
# first: keep conda/pip/uv/HF caches off the 50 GB $HOME (they fill it fast)
mkdir -p /project/ikoutis/conda_pkgs /project/ikoutis/$USER/cache
printf 'pkgs_dirs:\n  - /project/ikoutis/conda_pkgs\n' >> ~/.condarc
cat >> ~/.bashrc <<'EOF'
export PIP_CACHE_DIR=/project/ikoutis/$USER/cache/pip
export UV_CACHE_DIR=/project/ikoutis/$USER/cache/uv
export HF_HOME=/project/ikoutis/$USER/cache/huggingface
EOF
source ~/.bashrc
bash tools/setup_env.sh                          # conda env at /project/ikoutis/conda_env/wocr (+ vLLM, KaTeX)
module load Miniforge3 && source "$(conda info --base)/etc/profile.d/conda.sh" \
    && conda activate /project/ikoutis/conda_env/wocr
srun --account=ikoutis --qos=debug --partition=debug_gpu --gres=gpu:a100_10g:1 \
    --time=00:10:00 bash -l tools/setup_env.sh --gpu-check   # does the A100 run this vLLM?
# ~65 GB of weights → /project/ikoutis/wocr_models, as a CPU batch job (the login
# node's per-user limits can kill a download this size; the job resumes if cut off)
sbatch --job-name=wocr_stage --partition=general --qos=low --account=ikoutis \
       --cpus-per-task=4 --mem=16G --time=06:00:00 --output=logs/stage_%j.log \
       --wrap "module load Miniforge3 && source \$(conda info --base)/etc/profile.d/conda.sh \
               && conda activate /project/ikoutis/conda_env/wocr && cd $PWD \
               && python tools/stage_models.py --profile default"
```

vLLM is pinned to a tested release (`WOCR_VLLM_SPEC`, default `vllm==0.30.0`).
Its PyPI build is CUDA 13 and needs a GPU driver ≥ 580; for an older driver,
the setup installs the release's CUDA 12.9 wheel instead. A login node has no
driver to look at, so there the CUDA 13 build is installed with a warning,
and `--gpu-check` on a GPU node says whether that node's driver runs it. It
also loads vLLM's compiled kernels, so a mismatch shows up there rather than
in the first job. If the driver is too old, it prints the reinstall command
(a fresh env with `WOCR_TORCH_BACKEND=cu129`). A release has no other CUDA 12
build, so `cu129` is also the one for a 570-series driver.

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
model server crash, a full disk) by resubmitting only the unfinished shards.
Every stage skips pages that are already done:

```bash
IDS=$(python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT")
[ -n "$IDS" ] && NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
```

(`$INPUTS` is deliberately unquoted: the sbatch script splits it on spaces
the same way. Run it from the directory you submitted from, so that relative
paths name the same files.)

A document counts as done once its Markdown is in `$OUT`. `$OUT` is the
completion record: `/scratch` deletes files after 30 days, and a resubmitted
shard does not read again a finished document whose `$WORK` is gone or has
lost page images (unless `READ_ARGS=--force`; a `REVIEW_ARGS=--force` leaves
it alone). So a rerun with another profile needs its own `OUT`, as every run
should have: in another profile's `OUT` it finds everything done, and its log
says so.

Pages the reader could not read are assembled as marked gaps and listed in
`report.json` (`failed_pages`). Their documents count as done, so the line
above does not list them. To retry them, for example after raising a
server limit:

```bash
IDS=$(python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT" --retry-failed)
[ -n "$IDS" ] && READ_ARGS=--retry-failed NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
```

Interactive, from a GPU session with the batch job's memory (16 cores × 4 GB =
64 GB; `interactive` alone gives 1 core, 4 GB and 1 hour, too little for the
model servers), from the repo root:

```bash
interactive -a ikoutis -q standard -j gpu -n 16 -t 4      # a shell on a GPU node, 4 hours
INPUTS=paper.pdf OUT=out WORK=work bash slurm/ocr.sbatch  # then, inside it
```

A batch `--time` must stay well above the 30-minute warning lead: the
script refuses 40 minutes or less, because such a job would be signalled at
once and requeue itself forever. The default is 24 hours.

Batch settings are environment variables read by `slurm/ocr.sbatch`:

| variable | meaning |
|---|---|
| `INPUTS` | documents: dirs, files, or `@listfile` (one path per line; folders in it are walked too), space-separated (required) |
| `OUT`, `WORK` | output root (Markdown) and work root (page images, JSON); `WORK` defaults to a per-profile directory on `/scratch` |
| `WOCR_PROFILE` | model pair: `default` (Chandra 2 + Qwen3.8-27B), `dots` (MIT reader), `conservative` (most mature on A100), `olmocr` (olmOCR-2 baseline) |
| `PHASES` | subset of `read review assemble`, e.g. `PHASES="read assemble"` runs the reader only |
| `READ_ARGS`, `REVIEW_ARGS` | extra flags for the read / review stage, e.g. `READ_ARGS="--retry-failed"`, `REVIEW_ARGS="--review-types formula,table"`. They apply to the pages a stage still has to do; `--force` redoes the finished ones too, at every start of the task (a requeued start included). A finished document whose `WORK` was purged is redone only by `READ_ARGS=--force` |
| `NSHARDS` | total shard count when resubmitting a subset of array indices |
| `WOCR_LOGIN_PROFILE` | what a batch task loads as its login environment once its USR1 trap is set: `1` (default) `/etc/profile` and your `~/.bash_profile` (or `~/.bash_login`, `~/.profile`), as `bash -l` would; `0` nothing; or a file to load instead |

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

- read: `--retries 1` (re-reads of a page whose output looped, repeated an element, was cut off, or failed; the attempt that lost least of the page is kept) and `--retry-failed` (re-read pages saved as failed placeholders);
- review: `--review-types formula,table` (review all tables too), `--no-flagged`, `--max-change 0.35`, and `--no-describe-figures`.

**When things fail.** A page the reader cannot read is retried once, then
saved as a placeholder flagged `page_failed`, so its document still
assembles (report.json lists it; `--retry-failed` above reads it again). An
input file that cannot be opened (corrupt, unsupported) gets
`$OUT/<doc_id>/FAILED.json`. One that cannot be read at all (missing,
damaged, no permission) is skipped with a warning, and `tools/incomplete.py`
skips it too. A review request the server rejects becomes that block's final
decision. If a model server dies, nothing unfinished is saved as done: the
stage exits with code 3 and the next submission picks up where it stopped.
The same holds when the system fails during ingest (disk full, quota, a
failed write), or when a page image vanishes from `$WORK` while the reader
runs: nothing is marked failed, and the stage exits 3 (the next run renders
lost page images again). On preemption or the wall clock the stage exits 85
after saving every finished page, and the task requeues itself.

## Output

```
out/<doc_id>/
  <doc_id>.md      Markdown: $..$ / $$..$$ math, <!-- page N --> markers;
                   tables as HTML (merged cells survive; GFM from the
                   markdown reader); graph drawings twice, marked:
                   "Graph — Markdown (simple)" and "Graph — TikZ"
  figures/*.png    figure crops, linked from the Markdown
  report.json      per-document summary: readers and the WORK it was made
                   from, block types, review decisions, failed pages,
                   truncated pages (cut-off text still
                   missing, and the regions the reviewer transcribed:
                   "truncated_recovered"), review errors, remaining flags
                   (where a human should look first)
out/<doc_id>/FAILED.json   instead, if the input could not be ingested
```

The per-page provenance (every block's model, flags, and edit history) stays
in `$WORK/<doc_id>/{read,review}/p*.json` until `/scratch` purges it; copy
it to `/project` if you need it for longer.
