# wulver-ocr — design proposal

*Status: proposal, 2026-10-01. Opened by [O-001] in [`log.md`](log.md).
Decisions marked **(open)** are for Ioannis.*

## 1. Goal

The goal is to turn scanned research papers into faithful, readable Markdown on Wulver's A100s.
The emphasis is mathematics. Every display equation should come out as LaTeX that compiles
and matches the page symbol for symbol, and inline math should come out as `$...$` in running
text. Tables should be GFM when they are simple and HTML when they have merged cells. Figures
should be cropped and linked, with a generated description. For the figure kinds common in our
papers, the description should carry structure. A drawn graph becomes an edge list, a flowchart
becomes Mermaid, and a commutative diagram becomes tikz-cd.

The pipeline must run in batch on Wulver's SLURM setup, using the same conventions as the dml
repo: a conda env on `/project`, pre-staged weights, `qos=low` arrays that requeue themselves,
and one-line recovery. It should also work interactively on a single paper. Everything is
self-hosted. No paper and no page image leaves the cluster.

Out of scope for v0: handwriting, non-Latin scripts, chemistry (SMILES), and reconstructing a
compilable `.tex` file. Markdown with LaTeX math is the target.

## 2. Why two models

Two families of model now do document OCR well, and they fail in complementary ways.

**Specialist document-OCR models** (~1–3B parameters, trained on page→structure data) read a
whole page in one pass. They produce layout boxes, reading order, text, LaTeX, and HTML tables.
They are fast (roughly a page per second on an A100) and stay close to the pixels. Their
characteristic failures are of a few known kinds:

- **Repetition loops:** the model emits the same token or line until it hits the length limit,
  and the rest of the page is lost.
- **Truncated output** on dense pages.
- **Locally wrong LaTeX:** a subscript read as a superscript, a dropped prime, `\leq` for `\geq`,
  or an unbalanced brace.
- **Mis-split tables.**
- **No understanding of figures.** They box a figure but cannot say what it shows.

**General VLMs** (~30B, instruction-tuned) are slower and less pixel-faithful on a full page,
but they are good at a different job: *comparing* a transcription with an image and spotting the
mismatch. They can also describe a figure. Their characteristic failure is fluency. Asked to
transcribe, they paraphrase, "improve" unusual notation, and silently complete what they cannot
read.

The design gives each model the job it is good at and contains each one's failure mode.

1. The **reader** (specialist) reads every page.
2. Deterministic **validators** (CPU) check every block. They look for LaTeX structure
   (braces, `\begin`/`\end`, `\left`/`\right`, stray delimiters), repetition loops, truncation,
   unbalanced inline `$`, and inconsistent table shapes.
3. The **reviewer** (generalist) sees only what needs it. That means every display formula (the
   project's emphasis, so all of them are reviewed by default), every flagged block, and every
   figure. Each request carries **one image crop plus the reader's draft**. The reviewer answers
   `correct`, `fixed` (with the corrected text), or `unreadable`.
4. A **gate** decides whether a `fixed` proposal replaces the draft:
   - the proposal must not add a validator flag the draft did not have;
   - it must change at most 35% of the draft (1 − difflib ratio), or 60% if the draft was
     flagged;
   - there is no change limit when the draft was degenerate (empty, looping, or truncated). Then
     the reviewer acts as the fallback reader for that block.

   Rejected proposals are kept in the block's history.
5. **Assembly** (CPU) builds the Markdown. It drops running headers, footers, and page numbers,
   re-joins paragraphs split by a page break, and links the figure crops with their descriptions.

The point of the gate is that the reviewer's fluency is only allowed to *repair*, never to
*rewrite*. Its thresholds are deliberately conservative starting values, and they are calibrated
on the evaluation set (§7). Every decision is recorded per block (agreed, edited, rejected,
unreadable), so calibration is a matter of reading the JSON.

The alternatives considered:

- **One big VLM for everything** (e.g. a 30B+ model transcribing whole pages). It is about 10×
  slower per page, and it has the paraphrase and hallucination problem at full strength, with
  no second opinion to catch it.
- **Specialist only.** It is the fastest option and a strong baseline. It is also the A0 arm of
  the evaluation. But nobody catches its local LaTeX errors, which are exactly the errors that
  matter for papers.
- **Two independent readings with adjudication** (specialist and generalist each read every
  formula, and the VLM picks between them). It is more expensive. It is kept as an evaluation
  arm (A4) to test whether proofreading leaves accuracy on the table.

## 3. Architecture

```
  PDF / TIFF / PNG
        │  ingest (CPU): pages at 200 dpi, content-addressed doc_id, manifest
        ▼
  pages/p0001.png …
        │  READ (GPU, reader server): one request per page → JSON blocks
        │    {type, bbox, content}; validators flag; looping/truncated pages
        │    are re-read once with sampling + repetition penalty; figure crops saved
        ▼
  read/p0001.json …
        │  REVIEW (GPU, reviewer server): per wanted block, crop + draft →
        │    verdict + proposal → gate → accept/reject (history kept);
        │    per figure, crop + caption → kind + description + structure
        ▼
  review/p0001.json …
        │  ASSEMBLE (CPU): Markdown, figures/, report.json
        ▼
  out/<doc_id>/<doc_id>.md
```

All model traffic goes over localhost HTTP to `vllm serve`, in the OpenAI chat format
(`src/backend.py`). The pipeline never imports vLLM. As a result, the serving stack can be
upgraded or swapped (pip wheel, Apptainer container, SGLang) without touching the pipeline. The
same code also runs against any endpoint, so a laptop can talk to a server through a tunnel.
Throughput comes from concurrency: 32–64 requests in flight, batched by vLLM.

Every stage reads and writes the same page JSON (`src/schema.py`). Pages are written atomically
and stages skip finished pages, so preemption costs at most the pages in flight. The JSON is
also the provenance record. Every block names the model that wrote it, its flags, and its edit
history.

**Readers are adapters** (`src/readers/`). Each one maps a model's native output onto the common
block vocabulary: title, heading, text, list, formula, table, figure, caption, footnote,
header, footer, page_number, code, reference, other. Two are implemented:

- `dots`: layout JSON with boxes (the dots.ocr output format).
- `markdown`: whole-page Markdown, split back into blocks. It works for olmOCR-style models or
  any VLM. There are no boxes, so the reviewer sees the full page and figures are not cropped.

Adding a model means adding one adapter module and one profile.

## 4. Model choices

*Provisional.* The default pair in [`profiles/default.sh`](../profiles/default.sh) is
**dots.ocr** (reader: a 1.7B layout-plus-content model that produces boxes, LaTeX, and HTML
tables in one pass, served natively by vLLM) and **Qwen3-VL-32B-Instruct** (reviewer: Apache
2.0, strong on LaTeX and documents, fits one A100-80GB in bf16). A survey of the late-2026
candidates, with current benchmark numbers, will replace this paragraph. Adapters are cheap,
so the choice is revisable from data (§7, arm A5).

## 5. Running on Wulver

**Hardware.** These facts are from the NJIT HPC docs, read 2026-10-01.

- Wulver's `gpu` partition has 25 nodes, each with 4× A100-SXM4-**80 GB**, 128 cores, and
  512 GB RAM.
- 16 of those A100s are split into MIG slices, requested as `gpu:a100_10g`, `a100_20g`, and
  `a100_40g`. A job gets at most one MIG instance, so there is no tensor parallelism across
  slices.
- A100s are Ampere (sm_80). bf16 is native, but there are **no FP8 tensor cores**. FP8 or
  int4 checkpoints still run in vLLM as weight-only (Marlin) kernels: memory is saved, but
  there is no FP8 compute speed-up.

**One full A100 per shard, one model at a time.** `slurm/ocr.sbatch` is an array. Each task:

1. takes its 1/N slice of the documents;
2. starts the reader server and runs `read`, then stops it;
3. starts the reviewer server and runs `review`, then stops it;
4. assembles the Markdown.

Each model gets the whole 80 GB, which goes to KV cache and therefore batch size. A ~30B bf16
reviewer (~62–66 GB of weights) fits on one card. Co-locating it with the reader would not fit
comfortably. The cost of phasing is one server start per phase (1–3 min with warm caches), so
shards should be sized to run for hours. A phase with nothing left to do starts no server at
all.

**QOS and cost.** The default follows dml: `--account=ikoutis --qos=low`.

- `low` is free and preemptable.
- The scripts carry `--requeue` and `--signal=B:USR1@600`. On the signal, the pipeline finishes
  the requests in flight, saves, and exits 85, and `slurm/requeue_lib.sh` (adapted from dml)
  requeues the task.
- A signal that lands between steps (during ingest or a server start) is recorded and acted on
  before the next step. In dml's version, it would have killed the batch shell.
- Resubmitting exactly the unfinished shards is one line with `tools/incomplete.py`.

If `low` queues too long, `--qos=standard` on the command line overrides the script. `standard`
is charged at SU/h = max(CPUs, RAM_GB/4) + 16 × (GPU memory / 80 GB). The script's 8 CPUs,
64 GB, and one full A100 come to 32 SU/h.

**Smaller jobs on MIG.** A reader-only pass fits a `a100_20g` slice (4 SU/h of GPU on
`standard`): `PHASES=read sbatch --gres=gpu:a100_20g:1 …`. Slices may also schedule sooner.
The reviewer needs a full card.

**Storage.**

| what | where | why |
|---|---|---|
| conda env | `/project/ikoutis/conda_env/wocr` | same convention as dml (`tools/setup_env.sh`) |
| model weights | `/project/ikoutis/wocr_models/<name>` | persistent (scratch purges after 30 days); ~70 GB for the default pair, inside the 2 TB group quota |
| page images, page JSON | `/scratch/ikoutis/$USER/wocr/work` | large (1–3 MB per page at 200 dpi), regenerable |
| Markdown, figures, reports | `/project/ikoutis/$USER/wocr/out` | the product; backed up |

Weights are staged once with `tools/stage_models.py`, either on a login node (the dml
convention) or in an interactive CPU session if the login node's per-user limits get in the way.
Jobs run with `HF_HUB_OFFLINE=1`.

**Environment.** One conda env holds the pipeline client (httpx, pillow, pypdfium2) and vLLM,
whose wheels bundle torch and the CUDA runtime. The GPU nodes' driver version is not published.
The setup script therefore ends with a one-line `srun` on the `debug_gpu` partition that prints
`nvidia-smi` and checks that torch sees the A100. If the newest vLLM wheel needs a newer CUDA
than the driver supports, there are two fallbacks: pin an older vLLM (`WOCR_VLLM_SPEC`) or run
the official container under Apptainer (`module load apptainer`; `apptainer pull
docker://vllm/vllm-openai:<tag>` on a compute node).

**Throughput (to be measured in [O-002]).** The rough expectation for one A100:

- the reader at ~1–2 pages/s;
- the reviewer on ~5–15 crops per page of a math paper, at a few crops per second with batching.

That puts a 20-page paper at about a minute of GPU time end to end, and 10,000 pages at a few
GPU-hours per stage. These are guesses. The first smoke run replaces them with measurements.

## 6. Output format

```
out/<doc_id>/<doc_id>.md     the document
out/<doc_id>/figures/*.png   figure crops (linked)
out/<doc_id>/report.json     review decisions + open flags (where a human should look)
work/<doc_id>/{read,review}/p*.json   per-block provenance
```

The Markdown conventions are chosen to render unmodified on GitHub, Obsidian, Pandoc, Jupyter,
VS Code, and MkDocs with arithmatex:

- Display math is `$$ … $$` on its own lines; equation numbers are kept as `\tag{n}`.
- Inline math is `$…$`.
- Tables are GFM, or HTML when cells are merged.
- Headings are `#` for the title and `##`/`###` for sections.
- Each figure is `![alt](figures/p0003_b05.png)` followed by a collapsible
  `<details><summary>Figure description (generated)</summary>`. Inside it are a kind, a
  description, and, where the kind has structure, a fenced block: an edge list in `text`,
  `mermaid`, or tikz-cd in `latex`. Generated descriptions are always marked as generated and
  never replace the crop.
- `<!-- page N -->` comments mark page boundaries (invisible when rendered; `--no-page-markers`
  turns them off).
- Running headers, footers, and page numbers are dropped from the Markdown but kept in the JSON.

## 7. Evaluation plan

The question to answer with numbers is whether the second model pays for itself, and with which
thresholds.

**Ground truth.** We build an evaluation set from **arXiv papers with LaTeX source**: math, CS
theory, and physics, including the group's own papers.

1. Compile each source to a PDF and render its pages.
2. Degrade the page images at four seeded levels with `eval/degrade.py`: clean, office scanner,
   old photocopy, and bad phone photo. The degradations are skew, blur, noise, contrast loss,
   binarisation, and JPEG compression.
3. Recover the ground truth for every display equation, table, and paragraph from the source.
   Equations come from their environments, with macros expanded by a small `\newcommand`
   resolver.

This gives exact ground truth at controlled difficulty, which real scans never provide. A
smaller set of **real scans** (older journal papers we hold in print, transcribed once by hand
for ~20 pages) checks that the synthetic degradation is not flattering. The public
**OmniDocBench** and **olmOCR-Bench** results give an outside anchor for the reader choice.

**Metrics.**

- Per display formula:
  - exact match after LaTeX normalisation;
  - normalised edit distance;
  - **CDM**-style render-and-compare: render both formulas and match symbols. This avoids
    penalising `\frac{a}{b}` vs `{a \over b}`.
- Tables: TEDS.
- Text: normalised edit distance and reading-order accuracy.
- Figures: crop recall, and for graph drawings, edge-list exact match.
- Cost: GPU-seconds per page per stage.

**Arms.** All arms run on the same pages, so the comparisons are paired.

| arm | reader | reviewer | question |
|---|---|---|---|
| A0 | specialist | — | the baseline |
| A1 | specialist | formulas + flagged (default) | does proofreading help, and where? |
| A2 | specialist | every block | is reviewing unflagged text worth its cost? |
| A3 | generalist reads whole pages | — | is a specialist needed at all? |
| A4 | specialist + generalist both read each formula | adjudicates | does independent reading beat proofreading? |
| A5 | the other specialist candidates (§4) | as A1 | reader choice, under the same reviewer |

**Gate calibration.** For A1/A2, take every proposal with its draft, the ground truth, and its
change fraction. That gives, per threshold, how many correct fixes are accepted and how many
harmful rewrites are rejected. The 0.35/0.6 defaults are replaced by the thresholds that
maximise net formula accuracy. Because of degrading levels, this can be done per scan quality.

**Registered expectation**, so the result can surprise us:

- A1 should lower the formula error rate relative to A0 by a clear margin on degraded scans
  (levels 2–3), and by little on clean renders.
- The gate should reject a non-trivial fraction of reviewer proposals. If it rejects almost
  none, the reviewer is either very good or the gate is too loose. The ground truth tells which.
- A3 should lose to A1 on formulas and on cost.

## 8. Milestones

| # | milestone | content | entry |
|---|---|---|---|
| M0 | scaffold *(this commit)* | pipeline, two reader adapters, gated reviewer, figure describer, assembly, Wulver tooling, 70+ CPU tests | [O-001] |
| M1 | smoke run on Wulver | env + weights staged; 3–5 papers end to end; measured pages/s, GPU memory, review decisions; fix whatever the real models do differently from their docs (prompt formats, box frames, served names) | [O-002] |
| M2 | evaluation set + metrics | arXiv-source builder (`eval/build_arxiv.py`), formula normaliser, CDM via a headless KaTeX render, TEDS, edit distance; A0 vs A1 on ~50 papers × 4 degradation levels | — |
| M3 | bake-off + calibration | A2–A5; pick the default reader/reviewer pair from data; calibrate the gate thresholds | — |
| M4 | production | the real corpus in sharded arrays; a single-paper command for interactive use (OnDemand session) | — |
| later | | document-level pass (heading hierarchy, cross-page macros and references), born-digital text-layer hints, `.tex` export | — |

## 9. Open questions for Ioannis

1. **Corpus.** What are we converting first, and how much of it? Era and scan quality (300 dpi
   office scans or old photocopies), languages, and page count all matter. They set the shard
   size and whether robustness to older typography is a priority.
2. **"Graphs and diagrams."** I read this as both *graph drawings* (vertices and edges, which get
   an edge list) and *plots* (which get axes, series, and trends), plus flowcharts (Mermaid) and
   commutative diagrams (tikz-cd). Is an edge list the right structured form for graph drawings,
   or would TikZ (or a GraphML/JSON node-edge dump) serve better?
3. **Output target.** Is Markdown for reading and search (Obsidian, GitHub, LLM ingestion)
   enough? Or should M4 also produce compilable `.tex` or HTML?
4. **Allocation.** Should we use `ikoutis`/`low` as in dml (free, preemptable; the default
   here) or `dept_dms`/`high_dept_dms`?
5. **Real-scan ground truth.** Are there ~20 printed pages (ideally old, math-dense) that someone
   could check by hand, to validate the synthetic degradations?
6. **Licences.** Everything runs in-house and nothing is redistributed. Still, some candidate
   readers carry AGPL-style or custom licences (see §4). Confirm this is fine for research use,
   or restrict to Apache/MIT models.
