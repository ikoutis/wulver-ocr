# Communication log

Reverse-chronological task/reply log for wulver-ocr. The **newest entry is
at the top**. Each entry is headed `## YYYY-MM-DD [HH:MM TZ] — <kind>: …`,
where `<kind>` is `Task`, `Note`, or a `<name>` reply. See
[`README.md`](README.md) for the format and the `[O-00N]` ID convention. To
add an entry or reply, insert a new dated section at the top. A reply cites
the entry it answers.

<!-- Add your next entry or reply here, above the older ones. -->

---

## 2026-10-01 — Task [O-002]: first smoke run on Wulver (M1)

**For:** whoever has a Wulver login under `ikoutis`. **Why:** everything in [O-001] has been
tested only against fake model servers. Real models always differ from their documentation in
small ways: the exact prompt template, the frame the boxes are in, a JSON wrapper, a default
`max_tokens`. This run finds those differences on three papers, before anything is run at
scale. It also replaces the throughput guesses in `design.md` §5 with measurements.

**Steps** (from the repo root, on Wulver):

```bash
bash tools/setup_env.sh                                  # login node; ~15 min
# GPU check: the srun one-liner in tools/setup_env.sh's header (debug_gpu, free).
# Paste its output. The driver's max CUDA version decides the vLLM wheel.
python tools/stage_models.py --profile default           # ~65 GB to /project/ikoutis/wocr_models
pytest tests/                                            # CPU, seconds

mkdir -p /project/ikoutis/$USER/wocr/o002_in
# copy in 3 PDFs: one clean born-digital math paper, one older scanned
# paper (journal photocopy if possible), one with graph drawings/diagrams
INPUTS=/project/ikoutis/$USER/wocr/o002_in \
OUT=/project/ikoutis/$USER/wocr/runs/o002_smoke \
    sbatch --array=0-0 slurm/ocr.sbatch
```

**Report back:**

1. The `nvidia-smi` line printed at the top of `logs/wocr_<job>_0.log`, plus torch, CUDA, and
   vLLM versions (from `/project/ikoutis/conda_env/wocr/wocr.lock.txt`), and whether
   `setup_env.sh`'s sanity line said `katex check: True`.
2. Server start-up times (`=== reader ready after …s`) and the `read:`/`review` summary lines
   (pages, seconds, review decisions).
3. For each paper, `report.json`'s `review` and `open_flags` counts. Also a look at the Markdown
   next to the PDF: which formulas are wrong, whether the reading order is right on two-column
   pages, and whether figures were cropped and described sensibly.
4. Any traceback in `logs/*.err` or the `logs/vllm_*.log` files. If a server failed to start,
   the last 40 lines of its log are copied into the `.err`.
5. If the reviewer (Qwen3.8-27B, a hybrid-attention model) fails to start or misbehaves on the
   A100s, rerun the same three papers with `WOCR_PROFILE=conservative` (dots.mocr +
   Qwen3-VL-32B). Report both runs. That is also a first, informal reader comparison.

No numbers are expected to be final here. The point is that the plumbing works with the real
models, and a list of what to fix before M2.

---

## 2026-10-01 — Note [O-001]: project opened — two-model OCR for papers, scaffold on `main`

This entry opens the project. The proposal is in [`design.md`](design.md), and the first
working version of the pipeline is on `main`. In short:

**What it does.** It turns scanned (or born-digital) research papers into Markdown:

- display math as `$$…$$` LaTeX and inline math as `$…$`;
- tables as GFM, or HTML when cells are merged;
- figures cropped, linked, and described. A graph drawing also gets an edge list, a flowchart
  gets Mermaid, and a commutative diagram gets tikz-cd.

**How: two models with a gate between them.**

- A small document-OCR specialist (the *reader*) reads every page in one pass. It is fast and
  faithful to the pixels.
- A large general VLM (the *reviewer*) then proofreads one image crop at a time. It sees every
  display formula, every block that a CPU validator flagged (unbalanced LaTeX, repetition
  loops, truncation, ragged tables, odd `$` counts), and every figure.
- The reviewer can only *repair*, not rewrite. Its proposed edit is accepted only if it adds no
  validator flag and changes no more than a bounded fraction of the reader's text (35%, or 60%
  for flagged blocks, unbounded when the reader's output was degenerate).
- Every decision is kept in the page JSON, which serves as provenance and as the data for
  calibrating those thresholds.

The full reasoning, including the alternatives considered and why they lose, is in `design.md`
§2.

**On Wulver**, the dml conventions carry over:

- conda env at `/project/ikoutis/conda_env/wocr`, weights in `/project/ikoutis/wocr_models`;
- sharded `qos=low` arrays that requeue themselves (the requeue lib is adapted from dml). It now
  also survives a signal that arrives between steps, which in dml's version would have killed the
  batch shell;
- one-line recovery with `tools/incomplete.py`.

Each array task serves one model at a time on a full A100, so each model gets all 80 GB.

**Status.**

- 86 CPU-only tests pass. They include end-to-end runs against fake model servers, resume
  without rework, the exit-85 signal path, and the KaTeX check (3 tests skip without node).
- The validators include a KaTeX parse of every formula (the renderer olmOCR-Bench grades
  math with).
- Nothing has run on a GPU yet. That is [O-002].
- The default pair is **Chandra OCR 2** (reader) and **Qwen3.8-27B** (reviewer). Chandra 2
  has the best published score on scanned math (olmOCR-Bench "old scans math", 89.1). Qwen3.8
  is the strongest open general model on degraded documents (PureDocBench) that fits one A100.
  `design.md` §4 has the survey behind the choice, the alternatives, and the one licence caveat:
  Chandra's weights are free for research, not for commercial use. `WOCR_PROFILE=dots` swaps in
  the MIT-licensed dots.mocr.
- The evaluation plan (§7) builds exact ground truth from arXiv LaTeX sources plus seeded scan
  degradation (`eval/degrade.py`). That is what will tell us, in numbers, whether the second
  model earns its GPU time.

**Questions for Ioannis** are collected in `design.md` §9: the target corpus, what structure
"graphs" should come out as, the output target, the allocation, and licences.
