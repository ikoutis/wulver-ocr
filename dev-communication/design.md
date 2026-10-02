# wulver-ocr — design proposal

*Status: proposal, 2026-10-01. Opened by [O-001] in [`log.md`](log.md).
Decisions marked **(open)** are for Ioannis.*

## 1. Goal

The goal is to turn scanned research papers into faithful, readable Markdown on Wulver's A100s.
The emphasis is mathematics. Every display equation should come out as LaTeX that compiles
and matches the page symbol for symbol, and inline math should come out as `$...$` in running
text. Tables should be GFM when they are simple and HTML when they have merged cells. Figures
should be cropped and linked, with a generated description. For the figure kinds common in our
papers, the description should carry structure. A drawn graph comes out **twice, marked as
such**: as simple Markdown (vertex list and edge list) and as TikZ that redraws it. A flowchart
becomes Mermaid, and a commutative diagram becomes tikz-cd.

The pipeline must run in batch on Wulver's SLURM setup, using the same conventions as the dml
repo: a conda env on `/project`, pre-staged weights, `qos=low` arrays that requeue themselves,
and one-line recovery. It should also work interactively on a single paper. Everything is
self-hosted. No paper and no page image leaves the cluster.

Out of scope for v0: handwriting, non-Latin scripts, and chemistry (SMILES). **Markdown with
LaTeX math is the target.** A compilable `.tex` export is deferred to later, per Ioannis
([O-003]). Choices made now keep that door open: math stays verbatim LaTeX, and graphs are
already TikZ.

## 2. Why two models

Two families of model now do document OCR well, and they fail in complementary ways.

**Specialist document-OCR models** (~1–4B parameters, trained on page→structure data) read a
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

**General VLMs** (~27–32B, instruction-tuned) are slower and less pixel-faithful on a full page,
but they are good at a different job: *comparing* a transcription with an image and spotting the
mismatch. They can also describe a figure. Their characteristic failure is fluency. Asked to
transcribe, they paraphrase, "improve" unusual notation, and silently complete what they cannot
read. This has been measured, not just observed. One 2026 study found that on perturbed input,
general VLMs' word error rate rose by up to 6.9 points, against 0.1–3.4 for OCR-specialised
VLMs (arXiv 2607.21617†).

The design gives each model the job it is good at and contains each one's failure mode.

1. The **reader** (specialist) reads every page.
2. Deterministic **validators** (CPU) check every block. They look for LaTeX structure
   (braces, `\begin`/`\end`, `\left`/`\right`, stray delimiters), repetition loops, truncation,
   unbalanced inline `$`, and inconsistent table shapes. They also check that **KaTeX can parse
   every formula**. KaTeX is the renderer olmOCR-Bench grades math with, and the one GitHub and
   Obsidian display it with.
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
  arm (A4a) to test whether proofreading leaves accuracy on the table.
- **Two specialists, with escalation on disagreement.** Two architecturally different readers
  both read every page, and only blocks where they disagree go to the reviewer, with both
  candidates. This is the routing idea behind Consensus Entropy (CVPR 2026†): multi-model
  agreement sent just 7.3% of inputs to the stronger model and beat VLM-as-judge by 15 F1
  points. It roughly doubles stage-1 cost, but stage 1 is the cheap stage. It is arm A4b, and
  the strongest candidate to become the default if it wins. Our adapters already put both
  readers' output in the same block schema, so the extra code is block alignment and a
  comparison.

Related evidence for the proofreading loop: OCR-EDR (arXiv 2609.03445†) runs edit → render →
reassess on formula crops. It fixes 86% of erroneous inputs and adds up to 4.6 CDM on the hard
subsets of four OCR systems. Our gate is the cheap first version of that loop (structure + KaTeX
parse + bounded change). A render-and-compare check is the natural next validator (M3).

*(† = read by our survey from search snippets or abstracts, not from the full paper; to be
confirmed before we cite it in anything.)*

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
Throughput comes from concurrency: 32–64 requests in flight, batched by vLLM. The review
stage pools every block and figure of every unreviewed page in the shard into one queue, so
the reviewer stays busy across document boundaries.

Every stage reads and writes the same page JSON (`src/schema.py`). Pages are written atomically
and stages skip finished pages, so preemption costs at most the pages in flight. The JSON is
also the provenance record. Every block names the model that wrote it, its flags, and its edit
history.

**Failure semantics** decide what counts as done:

- **Stop request** (SIGUSR1 at preemption or before the wall clock). No new request starts, the
  ones in flight finish, every finished page is saved, and the command exits 85 so the task
  requeues itself.
- **Model server failure** (unreachable, 5xx, timeouts). Nothing unfinished is saved as done.
  A circuit breaker stops a dead server from costing every remaining item its full retry
  schedule. The stage exits 3, and the pages stay in `todo`.
- **Deterministic failure of one request** (a 4xx with the server's reason, or an unparseable
  reply). A review request that fails this way becomes that block's final decision. A page the
  reader cannot read is retried once with different decoding, then saved as a placeholder
  flagged `page_failed`, so its document still assembles; `read --retry-failed` retries such
  pages later.
- **An input that cannot be ingested** gets `<out>/<doc_id>/FAILED.json`, a terminal state that
  `tools/incomplete.py` counts as done.

None of these can block a shard forever or pass off unfinished work as finished.

**Readers are adapters** (`src/readers/`). Each one maps a model's native output onto the common
block vocabulary: title, heading, text, list, formula, table, figure, caption, footnote,
header, footer, page_number, code, reference, other. Three are implemented:

- `chandra`: HTML layout blocks with boxes (Chandra OCR 2). Math comes back in `<math>` tags,
  which `src/readers/htmlmd.py` converts to `$…$`/`$$…$$`. A trailing equation number becomes
  `\tag{n}`. The model's own figure description, including Mermaid for diagrams, is kept
  alongside ours.
- `dots`: layout JSON with boxes (the dots.ocr / dots.mocr format, prompt verified against the
  dots.mocr repository).
- `markdown`: whole-page Markdown, split back into blocks. It works for olmOCR-style models or
  any VLM. There are no boxes, so the reviewer sees the full page and figures are not cropped.

Adding a model means adding one adapter module and one profile.

## 4. Model choices

**How these were chosen.** A survey on 2026-10-01 read the project READMEs and leaderboards on
GitHub. Hugging Face and arXiv were not reachable from the survey's sandbox. I then re-checked
against primary sources the facts the defaults rest on: Chandra 2's and dots.mocr's prompts,
output formats, and licences; Qwen3.8's release and serving instructions; and vLLM's CUDA
wheels. Numbers marked † come from search snippets and are unconfirmed. "n/r" means not
reported.

**The landscape, briefly.**

- The best page readers are now 0.8–4B specialists.
- Standalone formula recognizers (UniMERNet, PP-FormulaNet) are no longer competitive as the
  main path. On OmniDocBench v1.6, a classic detector-plus-formula-recognizer pipeline scores
  CDM 83, against 97+ for the region VLMs.
- **The leaderboards disagree, and the disagreement matters for us.**
  - OmniDocBench v1.6 is mostly clean pages. It also ignores headers, footers, and footnotes.
  - PureDocBench (May 2026) adds digitally and physically degraded tracks of the same pages.
    It reorders the field. PaddleOCR-VL-1.6 is third on OmniDocBench (96.34; an independent
    rerun gives 95.25) but averages 59.3 on PureDocBench, and 54.2 on real degradation.
  - olmOCR-Bench has an "old scans math" category: printed math, scanned. It is the closest
    public proxy for our inputs.
- Even the best systems score only about 51–58 on olmOCR-Bench's general "old scans" category.
  Old scans are not solved by anyone.

**Readers (stage 1).**

| model | size | licence (weights) | boxes | olmOCR-Bench overall / old-scans-math | OmniDocBench v1.6 | PureDocBench avg (real-degraded) | here |
|---|---|---|---|---|---|---|---|
| **Chandra OCR 2** (Datalab, 2026-03) | 4B | modified OpenRAIL-M: free for research/personal use ✓ | yes (0–1000) | **85.8 / 89.1** ✓ | n/r | n/r | **default** (`chandra`) |
| **dots.mocr** (rednote, 2026-03) | 3B | MIT ✓ | yes (pixels) | 83.9 / 85.5 ✓ | n/r (dots.ocr: 90.8) | 70.4 | profile `dots` (`dots`) |
| Infinity-Parser2-Flash (2026-05) | 2B | Apache-2.0† | yes (JSON) | 86.0 | 92.0 (self) | n/r | candidate (A5) |
| TeleOCR (China Telecom, 2026-08/09) | 1.2B | unclear: no licence file | yes | n/r | **96.91** | **78.6 (69.1)** | candidate once the licence is clear |
| WeVisDoc-4B (Tencent, 2026-09) | 4B | Apache-2.0† | no | n/r | 95.4 (self)† | 75.6 (69.1) | candidate (`markdown`) |
| OvisOCR2 (Alibaba, 2026-07) | 0.8B | Apache-2.0 | no | n/r | 96.47 | 75.1 (66.5) | candidate (`markdown`) |
| MinerU2.5-Pro (2026-04/05) | 1.2B | Apache-2.0† (toolkit: custom) | yes | n/r | 95.75 | 70.1 (62.6) | candidate (needs adapter) |
| PaddleOCR-VL-1.6 (2026-05) | 0.9B + detector | Apache-2.0 | yes | n/r | 96.34 | 59.3 (54.2) | not chosen: weakest under degradation |
| olmOCR-2 (Ai2, 2025-10) | 7B | Apache-2.0 | no | 82.4 / 82.3 | 85.7 | 63.8 | baseline (`markdown`) |

✓ = checked against the project's own repository.

**Default reader: Chandra OCR 2.** Three reasons:

1. It has the best published score on the category closest to our inputs: 89.1 on old scans
   with math, against 85.5 for the next open model.
2. It returns layout boxes with a 19-label vocabulary. That vocabulary includes Equation-Block,
   Diagram, Bibliography, and Page-Header/Footer, so figures can be cropped and running heads
   dropped.
3. It already describes images, turns charts into data, and turns diagrams into Mermaid. That
   complements the reviewer's figure pass.

The adapter uses Chandra's own prompt, image scaling, and decoding settings, copied from its
Apache-2.0 code. The trade-offs:

- **The weights licence.** It is fine for research. It is not fine for commercial use or for use
  that competes with Datalab's API (question 6 in §9).
- **It is not on PureDocBench.** So its robustness to *real* degradation is known only through
  olmOCR-Bench.
- **It is larger than the 1B models.** Datalab measures 1.44 pages/s on an H100 on a hard mix.
  Expect about 1 page/s on an A100.

**dots.mocr** is the MIT-licensed alternative (`WOCR_PROFILE=dots`). It is two points behind on
olmOCR-Bench, its adapter is the same code, and it is reader B for arm A4b.

**Leading on clean pages is not enough.** The OmniDocBench leaders (TeleOCR, OvisOCR2,
PaddleOCR-VL) were not taken on that score alone. TeleOCR is also the strongest specialist on
PureDocBench, and it will join the bake-off (A5) once its licence is published.

**Reviewer (stage 2): Qwen3.8-27B** (Qwen, 2026-08-14; ✓ release and serving instructions).

- It is a unified vision-language model.
- It is a 27B dense hybrid: Gated-DeltaNet layers plus full attention. The hybrid layers keep
  the KV cache small, so in bf16 (~56 GB) it fits one A100-80GB with room for batching.
- It has the best open general-model score on PureDocBench, 77.6 (73.6 on the real-degraded
  track). That is above every specialist on that track, and judging degraded crops is exactly
  the reviewer's job.
- It is served with `--reasoning-parser qwen3`. Thinking is off by default (one short comparison
  per crop). The profile sets this through `EDITOR_REQUEST_EXTRA` (the stage-by-stage CLI takes
  the same JSON as `--editor-extra`), and whether thinking buys formula accuracy is an M3
  measurement.
- As a second line of defence, the client strips inline reasoning: `<think>…</think>`
  blocks, a bare `…</think>` (when the chat template opened `<think>` itself), and an
  unterminated `<think>`.
- Licence: per its model card; the survey reports Apache-2.0†.

The fallback is Qwen3-VL-32B-Instruct (Apache-2.0, standard attention, the most mature vLLM path
on Ampere) in `WOCR_PROFILE=conservative`. Not chosen:

- GLM-5.3-Flash. It scores best on PureDocBench (79.7), but vLLM runs its sparse attention only
  on Hopper/Blackwell, and the community A100 port needs 8 GPUs.
- gpt-oss. It is text-only, so it cannot compare a draft against an image.
- Qwen3.5-122B. It needs 4 GPUs, or int4.

**A100 specifics.**

- bf16 is native; FP8 checkpoints run only weight-only (Marlin), and FP8 on hybrid and MoE
  models has open issues. Hence bf16 everywhere.
- The hybrid Qwen models need vLLM ≥ 0.17.
- vLLM's default wheels are built for CUDA 12.9, with 12.8 and 13.0 builds also published
  (✓ vLLM install docs).
- `tools/setup_env.sh` installs the cu129 build by default (`WOCR_TORCH_BACKEND` overrides it).

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

Each model gets the whole 80 GB, which goes to KV cache and therefore batch size. The 27B bf16
reviewer (~56 GB of weights) fits on one card. Co-locating it with the reader would not fit
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
| model weights | `/project/ikoutis/wocr_models/<name>` | persistent (scratch purges after 30 days); ~65 GB for the default pair, inside the 2 TB group quota |
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

- the reader at ~1 page/s for Chandra 2, more for the smaller readers;
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
- Tables are HTML from the layout readers (so merged cells survive), and GFM from the
  Markdown reader. Converting simple HTML tables to GFM is a possible later nicety.
- Headings are `#` for the title and `##`/`###` for sections.
- Each figure is `![alt](figures/p0003_b05.png)` followed by a collapsible
  `<details><summary>Figure description (generated)</summary>`. Inside it are a kind, a
  description, and, where the kind has structure, the structure: Mermaid for a flowchart,
  tikz-cd in a `latex` fence for a commutative diagram. Generated descriptions are always
  marked as generated and never replace the crop.
- **Graph drawings get two marked versions of the same graph:**
  - `**Graph — Markdown (simple):**`, a vertex list plus one bullet per edge (`—`
    undirected, `→` directed, `↔` both, edge labels in parentheses);
  - `**Graph — TikZ:**`, a `latex` fence with a `tikzpicture` that redraws the vertices at
    their drawn positions.

  The reviewer writes only the TikZ, in a fixed pattern (`src/tikz.py`), and the pipeline
  parses it back. The parsed vertices and edges go into the page JSON (`meta["graph"]`), and the
  Markdown version is rendered from them, so the two cannot disagree. TikZ that does not parse,
  or whose edges reference undeclared vertices, is sent back to the reviewer once with the
  problems listed. If it still fails, the figure is flagged (`tikz_*`) and the Markdown version
  says it is unavailable.
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
- Figures: crop recall. For graph drawings, exact match of the edge set parsed back from the
  TikZ: by vertex label, or up to isomorphism when the drawing has no labels. arXiv sources that
  draw their graphs in TikZ give exact ground truth for this.
- Footnotes: recall. olmOCR-Bench rewards dropping headers and footers, and readers tuned on it
  may drop footnotes with them.
- Cost: GPU-seconds per page per stage.

**Arms.** All arms run on the same pages, so the comparisons are paired.

| arm | reader | reviewer | question |
|---|---|---|---|
| A0 | specialist | — | the baseline |
| A1 | specialist | formulas + flagged (default) | does proofreading help, and where? |
| A2 | specialist | every block | is reviewing unflagged text worth its cost? |
| A3 | generalist reads whole pages | — | is a specialist needed at all? |
| A4a | specialist + generalist both read each formula | adjudicates | does independent reading beat proofreading? |
| A4b | two specialists (Chandra 2 + dots.mocr) read every page | sees only blocks where they disagree, with both candidates | does consensus routing beat validator routing, at what cost? |
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
| M0 | scaffold | pipeline, reader adapters (Chandra 2, dots, Markdown, olmOCR), gated reviewer, KaTeX validator, figure describer with TikZ graphs, assembly, Wulver tooling, CPU test suite; hardened by an adversarial code review ([O-004]) | [O-001], [O-004] |
| M1 | smoke run on Wulver | env + weights staged; 3–5 papers end to end; measured pages/s, GPU memory, review decisions; fix whatever the real models do differently from their docs (prompt formats, box frames, served names) | [O-002] |
| M2 | evaluation set + metrics | arXiv-source builder (`eval/build_arxiv.py`), formula normaliser, CDM via a headless KaTeX render, TEDS, edit distance; A0 vs A1 on ~50 papers × 4 degradation levels | — |
| M3 | bake-off + calibration | A2–A5; pick the default reader/reviewer pair from data; calibrate the gate thresholds | — |
| M4 | production | the real corpus in sharded arrays; a single-paper command for interactive use (OnDemand session) | — |
| later | | document-level pass (heading hierarchy, cross-page macros and references), born-digital text-layer hints, compilable `.tex` export (deferred, [O-003]), an optional `pdflatex` compile check for the TikZ | — |

## 9. Open questions for Ioannis

1. **Corpus.** What are we converting first, and how much of it? Era and scan quality (300 dpi
   office scans or old photocopies), languages, and page count all matter. They set the shard
   size and whether robustness to older typography is a priority.
2. ~~**"Graphs and diagrams."**~~ *Answered ([O-003]):* graph drawings come out as TikZ,
   **and** as simple Markdown, the two marked as such (§6).
3. ~~**Output target.**~~ *Answered ([O-003]):* Markdown. Compilable LaTeX can wait.
4. ~~**Allocation.**~~ *Answered ([O-003]):* `ikoutis` / `qos=low`, as in dml (already the
   script default).
5. **Real-scan ground truth.** Are there ~20 printed pages (ideally old, math-dense) that someone
   could check by hand, to validate the synthetic degradations?
6. **Licences.** Everything runs in-house and nothing is redistributed. Still, the default
   reader's weights (Chandra 2) are under a modified OpenRAIL-M licence, which is free for
   research and personal use but not for commercial use or use competing with Datalab's API.
   Is research use the only use we foresee? If not, the MIT pair is `WOCR_PROFILE=dots`. (Other
   candidates are also out unless the answer changes: HunyuanOCR's licence excludes the EU,
   UK, and Korea and forbids using its outputs to improve other models.)
