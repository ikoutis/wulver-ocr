# Communication log

Reverse-chronological task/reply log for wulver-ocr. The **newest entry is
at the top**. Each entry is headed `## YYYY-MM-DD [HH:MM TZ] — <kind>: …`,
where `<kind>` is `Task`, `Note`, or a `<name>` reply. See
[`README.md`](README.md) for the format and the `[O-00N]` ID convention. To
add an entry or reply, insert a new dated section at the top. A reply cites
the entry it answers.

<!-- Add your next entry or reply here, above the older ones. -->

---

## 2026-10-02 — Ioannis / Claude: [O-002] first contact with Wulver — three environment problems, then the servers start

The set-up steps of [O-002] were run on Wulver today. The GPU nodes have driver 580.159.04
(CUDA 13.0), so the default CUDA 13 build of vLLM 0.30.0 is the right one; the `--gpu-check`
confirmed it. Three things stood in the way before the first model server started, none of
them in the pipeline itself:

1. **A full home directory.** `conda create` crashed with an unhelpful report because `$HOME`
   was at its 50 GB quota (26 GB of old conda package caches, 16 GB of `~/.cache`). Caches now
   live on `/project`: `pkgs_dirs` in `~/.condarc`, and `PIP_CACHE_DIR`, `UV_CACHE_DIR`,
   `HF_HOME` in `~/.bashrc`. The README's install section says so.
2. **Weights on the login node.** 65 GB through the login node's per-user limits is a kill
   waiting to happen; a CPU batch job (`sbatch --wrap` on `general`/`low`) did it instead, and
   `snapshot_download` resumes if interrupted.
3. **The C++ runtime.** Every server start died with
   `libstdc++.so.6: version CXXABI_1.3.15 not found (required by libicui18n.so.78)`. The
   `nodejs` that `setup_env.sh` installs for KaTeX brings ICU 78, which needs a newer C++
   runtime than the nodes' RHEL 9 `/lib64` (GCC 11). The env's own `python` finds the env's
   `libstdc++` through its RUNPATH, so `python -c "import sqlite3"` and the GPU check passed
   on the login node, on `debug_gpu`, and in a batch job. The `vllm` console script does not:
   its extension modules resolve against the loader's default path, where `/lib64` wins.
   This took four diagnostic jobs to isolate, because the GPU check tested `python`, not
   `vllm`. Fixed three ways: `setup_env.sh` installs `libstdcxx-ng` with `nodejs`;
   `ocr.sbatch` puts the env's `lib` first on `LD_LIBRARY_PATH` after activation (this
   alone fixed it, confirmed on a `debug_gpu` job); and `--gpu-check` now runs the `vllm`
   command itself.
4. **A compiler at run time.** With the C++ runtime fixed, `vllm serve` loaded Chandra OCR 2
   (a Qwen3.5 architecture: `Qwen3_5ForConditionalGeneration`, 4B), profiled 55 GB of KV cache,
   captured its CUDA graphs and warmed its Triton kernels in 5 minutes, then died in the
   sampler warm-up: vLLM 0.30's sampler calls FlashInfer, which compiles its top-k/top-p
   kernels with `nvcc` on first use (`gen_sampling_module().build_and_load()`), and that build
   fails on the GPU nodes. `ocr.sbatch` now sets `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM's own
   Triton sampler, which needs no toolkit (and the reader samples greedily anyway). Finding
   this took three round trips because `serve_lib.sh` reported only the last 40 lines of the
   server log, all of them the API server's traceback; the engine's root cause was 180 lines
   up. The report now shows the log's error lines first, then the tail.

The smoke run is resubmitted with the fixes. Its inputs are the two public samples now in
`samples/` (a born-digital IEEE paper with numbered equations; a 19th-century book scan) and a
handwritten quiz with graph drawings that stays on `/project`. Results follow in the next
entry.

---

## 2026-10-02 — Note [O-006]: a third pass over the fixes — 21 more findings fixed

A third independent pass checked the [O-005] fixes against the merged code, again with a
skeptic reproducing each finding in a scratch copy. It filed 21 findings: 19 new ones and 2
earlier ones that were only partly fixed. Two of them are the same defect. Three were rated
medium, the rest low: these are edge cases, but several would have cost GPU time or a finished
document without saying so. All were fixed in three parallel branches with separate files
(core and cluster scripts; reader adapters; gate and figures). This time, each branch was also
re-checked adversarially before it was merged. Those re-checks found seven small regressions in
the new code, and the integration commit fixed them:

- a column edge chosen by an overhanging box;
- a below-columns region added for a deep bottom margin;
- plain-operator display math (`(1+x)(1-x)`) no longer lifted;
- notes about "the page" accepted as transcriptions, and real sentences mentioning "only"
  rejected;
- escaped tags in table cells;
- a paragraph's own indentation counted as a change;
- finished documents with only their page images purged counted as purged.

The merged suite went from 473 to 591 tests (584 run without node/KaTeX; the other 7 need it).

**Recovery that could still go wrong.**

- A folder named in an `@listfile` (as `ls -d papers/*` or `find papers` write them) made
  ingest exit 3 on every submission, so its shard was never processed and recovery never
  converged. [O-005]'s rule took any failed system call on a readable path for a system
  failure, and a folder is readable. Folders in a listfile are now walked, as folders in
  `INPUTS` are. An input that cannot be read at all (missing, damaged, no permission) is
  skipped with a warning, as `tools/incomplete.py` skips it. Only a failed write, or an error
  that is the system's whatever the file (no space, quota, a read-only or stale file system),
  still exits 3.
- `/scratch` deletes files one by one. If the purge goes by access time, the page images and
  read JSON go first, because every resubmission reads the manifest and the review JSON again
  and the others only once. A finished document left with its manifest and no page images had
  its Markdown replaced by "OCR failed" placeholders at the next resubmission, and the
  documented `--retry-failed` recovery did the same. Such a document now counts as purged and
  is left alone. An unfinished one gets its lost page images rendered again, and a page image
  that vanishes while the reader runs leaves its page in `todo` (exit 3) instead of becoming a
  placeholder. Wulver's purge rule is not published, so this guards against either kind.
- `REVIEW_ARGS=--force` re-ingested purged documents without reading them, and the next plain
  resubmission then read, reviewed and rewrote them. Only a forced read now redoes a purged
  document.
- A rerun with another profile into the same `OUT` did nothing, while the sbatch header
  promised a re-read and the log blamed a purge. The header now says another profile needs its
  own `OUT`. `report.json` records the `WORK` it was made from, and a run from another `WORK`
  into the same `OUT` says so in its log.

**The cluster scripts.**

- `#!/bin/bash -l` loaded the login profile (Lmod, a `conda init` hook in `~/.bashrc`) before
  the script's first line, so a USR1 in those seconds still killed the task without a
  requeue. The script now traps USR1 first and then loads the profile itself
  (`WOCR_LOGIN_PROFILE`).
- `WOCR_TORCH_BACKEND=cu128` was documented, but vLLM publishes its CUDA 12 wheel as `+cu129`
  only, so that install failed on a 404. Only `cu130` and `cu129` are accepted now; `cu129`
  also runs on a 570-series driver.

**Which reading of a page is kept.** A page whose reading was cut off or looped is read twice,
and the attempt with fewer flagged blocks was kept. Since [O-005] a left-column cut leaves two
tail regions, and Chandra flagged every block of a repeated element, so the count could favour
the attempt that lost more: one cut in the title over one cut low in the left column, or a
cut-off retry over a complete page with one repeated element. Attempts are now compared by the
page area they lost, then by repeated elements, then by the text they kept
(`readers.base.reading_loss`).

**A repeated element is no longer a degenerate draft.** The kept copy of an element the model
wrote several times was flagged `repetition`, which the gate treats as degenerate: no change
limit and no escape check, so the reviewer could rewrite correct text freely. It now gets its
own flag, `repeated`, and keeps the clean-block change limit (0.35), since the kept copy is an
ordinary reading. The page is still re-read once. A loop inside one block's own text is still
`repetition`.

**The readers.**

- The column rule for cut-off pages also fired on one-column pages whose cut fell in a short
  left-aligned element (a list item, a heading). The rest of the page was then split into a
  left and a right half, which the reviewer transcribed separately. Two columns now need left
  boxes that look like a column.
- A tail region fitted between kept columns ran to the page bottom, so a full-width float
  below the columns was cut in half. It now ends where the columns end, with a full-width
  region below them. After a cut in the *left* column, the regions still run to the page
  bottom: nothing read tells where the columns end. That limitation is documented in
  `design.md` §3.
- olmOCR: a LaTeX row break with spacing (`\\[4pt]`) was taken for the start of display math,
  which mangled the formula and swallowed the following paragraphs. An escaped bracket around
  a citation key (`\[ABC+20\]`) could also become a formula block; only a body that looks
  like math is lifted now.
- dots: one single-backslash slip in a JSON string turned its correctly doubled `\\{` or
  `\\|` into row breaks; the repair now works escape by escape. A text string whose only
  single-backslash commands start with `\n` (`\nabla`, `\nu`, `\ne`) is now repaired
  instead of decoding a newline.

**The gate and figures.**

- Agreement ignored whitespace that renders (inside `\text{…}`, a list item's indentation),
  so a reviewer fix that restored it was thrown away.
- The escape check counted a table's own HTML tags as bare `<`, which rejected correct table
  fixes. The check for "not a transcription" rejected short real answers, such as "(8)" or a
  parenthetical remark that mentions "empty".
- For a page olmOCR read turned, the reviewer now sees the turned page.
- TikZ: a mid-arrow drawn through decorations (`->-`) now counts as directed, and the repair
  instruction for a multi-panel graph asks for vertex names unique across panels.

None of this changes what [O-002] asks for.

---

## 2026-10-02 — Note [O-005]: verifying the [O-004] fixes — 41 more findings fixed

After [O-004], an independent pass checked its fixes against the merged code and reviewed the
code afresh, again with a skeptic reproducing each finding in a scratch copy. It filed 41
findings: 36 new ones and 5 earlier ones that were only partly fixed. Several are the same
defect seen by more than one reviewer, so there were 29 distinct defects. All were fixed in
three parallel branches with separate files (core and cluster scripts; reader adapters; gate
and figures), then merged. The merged suite went from 328 to 473 tests (466 run without
node/KaTeX; the other 7 need it).

**Recovery that did not recover.** These would have cost work on the cluster without saying so.

- `READ_ARGS="--retry-failed"`, the documented way to re-read pages saved as failed
  placeholders, did nothing in batch: the sbatch script skips a phase when `todo` reports
  nothing left, and `todo` counted a placeholder as read. `todo` now counts as the stage will,
  with the same `--retry-failed` and `--force`, and `tools/incomplete.py --retry-failed` lists
  the shards that have failed pages.
- With relative and absolute paths mixed in `INPUTS`, `tools/incomplete.py` named the wrong
  shards, because the sbatch script hands the pipeline absolute paths and the document list was
  sorted by spelling. It is now ordered by real path. Byte-identical copies of a file in two
  folders are now one document, so two array tasks no longer work on the same one.
- A full or over-quota `/scratch` during ingest marked documents `FAILED.json`, a terminal
  state, and the task ended COMPLETED. A system error now marks nothing and exits 3.
- Once `/scratch` purged a work directory, resubmitting its shard read, reviewed and rewrote
  finished documents. `OUT` is now the completion record for the stages, as it already was for
  `tools/incomplete.py`.

**The cluster scripts.**

- vLLM's current wheels are CUDA 13 builds, but the setup installed an unpinned vLLM with
  torch's CUDA 12.9 build. vLLM is now pinned (0.30.0), the build follows the GPU driver
  (CUDA 13 for a driver ≥ 580, the release's CUDA 12.9 wheel otherwise), and `--gpu-check`
  loads vLLM's compiled kernels, so a mismatch shows there rather than in the first job.
- uvicorn logs "Application startup complete" before it listens, so a server that lost a
  same-port race could still be taken for ready. Readiness now requires our process to hold
  the listening socket.
- A `--time` of 30 minutes or less would be signalled at once and requeue forever; anything up
  to 40 minutes is now refused. A USR1 while the job was still loading conda killed it; it now
  requeues.
- The interactive recipe now asks for 16 cores (64 GB) and 4 hours: `interactive`'s
  defaults are 1 core, 4 GB and 1 hour.

**Cut-off output on two-column pages was lost without a trace.** The region assumed lost was
the strip below the lowest kept block. On a two-column page that missed the lost column, and a
transcription of the strip cleared every record of the cut. The region now starts at the
element the reader was writing when it was cut off, with the right column added when the cut
fell in the left one. A transcribed region stays listed in `report.json`
(`truncated_recovered`), and `correct` or an empty answer no longer counts as a transcription.

**Smaller fixes to the readers and the gate.**

- dots replies with single-backslash LaTeX lost `\frac`, `\theta` or `\nabla` to control
  characters; they are repaired and flagged `json_repaired`.
- A model's element-level loops came out as dozens of duplicate blocks; they are kept once and
  flagged `repetition`.
- Left-numbered equations got shifted `\tag`s. `<br>` ran lines together. The Markdown readers
  demoted the title, and olmOCR's `\[…\]` and multi-line `\(…\)` math and its rotation
  verdict were dropped.
- At the gate:
  - a reviewer edit that only dropped the readers' Markdown escapes passed;
  - a no-op answer was misclassified;
  - empty box-less blocks were sent as whole-page requests;
  - blocks flagged `latex_unchecked` during a KaTeX outage were never checked again;
  - an unclosed `$$` went unflagged.
- For figures, a multi-panel graph, arrows set through TikZ styles, and a graph answer with no
  usable TikZ each gave an incomplete or empty graph without a flag.

None of this changes what [O-002] asks for. Its GPU-check step now prints which vLLM build the
node's driver runs.

---

## 2026-10-02 — Note [O-004]: adversarial code review before the first GPU run — 82 defects fixed

Nothing in this repo has met a real model yet, so before [O-002] spends GPU hours, the whole
codebase went through a structured review. Five independent reviewers each took one area:

- the pipeline core and resume logic;
- the reader adapters;
- the validators, gate, and TikZ;
- the SLURM and environment scripts;
- the docs against the code.

A separate skeptic then tried to reproduce or refute every finding in a scratch copy of the
repo. Of 83 findings, 82 were confirmed and 1 was refuted. The fixes were made in five
parallel branches with strictly separate files, then merged. The test suite went from 101 to
328 tests (322 run without node/KaTeX; the other 6 need it).

**What would have gone wrong on the cluster.** These are the ones that matter most:

- **Two array tasks on one node shared a model server.** The ports were fixed (8001/8002), and
  Wulver packs up to four of our one-GPU tasks onto a 4-GPU node. The second task's vLLM failed
  to bind, but its readiness check found the *first* task's server. It then ran on another
  task's GPU, and failed when that task shut its server down.
  - *Now:* each server takes a free port, and readiness is tied to our own process and log.
- **A crashed reviewer passed off unreviewed pages as reviewed.** A vLLM crash mid-review
  turned every remaining request into an error entry. Each page was still saved as reviewed,
  the job ended COMPLETED, and nothing showed it.
  - *Now:* server failures and successful-but-rejected requests are different errors.
  - Pages hit by a server failure are not saved, and the stage exits 3.
  - A circuit breaker stops a dead server from costing every remaining item its full retry
    schedule.
- **Preemption threw away work and could overrun the warning.** The stop signal was ignored
  until the end of the current document, and then that document's review was discarded. A long
  document could outlast the 10-minute warning and be killed without a requeue.
  - *Now:* review is one pool across the whole shard, with each page saved when its last request
    finishes.
  - After a stop, no new request starts.
  - The warning is 30 minutes (as in dml).
- **A fresh clone could not submit.** `#SBATCH --output=logs/...` needs `logs/` to exist, and it
  was git-ignored. *Now:* `logs/.gitkeep`.
- **One bad page or file blocked its document, and its shard, forever.** *Now:* a page the reader
  cannot read is retried once, then saved as a visible placeholder. An input that cannot be
  ingested gets `FAILED.json`, which the recovery tool treats as final.

**What would have come out wrong in the Markdown.** The reader adapters lost or garbled content
on realistic replies:

- equation numbers and prose dropped from equation blocks;
- emphasis glued to the next word;
- `a < b` inside math swallowing the rest of the paragraph;
- a hallucinated image description injected into body text;
- a truncated reply leaking a literal `<<TRUNCATED>>` marker.

A cut-off reply is now handled by an explicit *truncated-tail* region. The page is re-read once.
If it is still cut off, the reviewer transcribes the missing region from its crop, and anything
still missing is marked in the Markdown and in `report.json`. Other fixes:

- The validators no longer send legitimate tables to review as "repetition loops".
- The KaTeX check restarts its worker, or says `latex_unchecked`, instead of silently passing
  everything.
- The TikZ parser now reads the forms models actually write (picture-level arrow styles, node
  options in any order, automata loops).
- A graph whose TikZ still fails its checks no longer gets a partial Markdown version that
  contradicts it.

**New:** an `olmocr` reader and profile reproduce olmOCR-2's own pipeline settings, so the
published baseline (arms A0/A5) can be run as is.

None of this changes what [O-002] asks for. Its instructions were corrected where they would
have failed: activate the env, the GPU check command, the per-profile work directory.

## 2026-10-01 — Note [O-003]: decisions from Ioannis — TikZ graphs, Markdown target, qos=low

Ioannis answered three of the open questions in `design.md` §9.

**Graph drawings come out twice, marked as such.** The first version is simple Markdown: a
vertex list and one bullet per edge, with arrows for direction and edge labels in parentheses.
The second is TikZ: a `tikzpicture` that redraws the vertices at their drawn positions.

The reviewer is asked only for the TikZ, in a fixed pattern (`src/tikz.py`). The pipeline then
parses it back and renders the Markdown version from the parsed vertices and edges. That means
the two versions cannot disagree. It also means the graph is available as data in the page JSON
(`meta["graph"]`), which is what the evaluation will score against ground truth.

TikZ that fails to parse, or that draws an edge to a vertex it never declared, is sent back to
the reviewer once with the problems listed. If it still fails, the figure is flagged in
`report.json`, and the Markdown version says it is unavailable rather than guessing. Flowcharts
stay Mermaid, because it renders on GitHub. Commutative diagrams stay tikz-cd.

**Markdown is the target.** A compilable LaTeX export can wait. It stays on the "later" row of
the milestones. Math is kept verbatim as LaTeX, and graphs are already TikZ, so nothing done now
will need redoing for it.

**The allocation is `ikoutis` with `qos=low`**, as in dml. It was already the default in
`slurm/ocr.sbatch`, so nothing changes.

Still open: the target corpus (question 1), a hand-checked set of real scans (5), and licences
(6).

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
module load Miniforge3 && source "$(conda info --base)/etc/profile.d/conda.sh" \
    && conda activate /project/ikoutis/conda_env/wocr    # the env, in your own shell
srun --account=ikoutis --qos=debug --partition=debug_gpu --gres=gpu:a100_10g:1 \
    --time=00:10:00 bash -l tools/setup_env.sh --gpu-check   # GPU check (free)
# Paste its output. If it says the driver cannot run the env's build, reinstall as it says.
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
   vLLM versions (from `/project/ikoutis/conda_env/wocr/wocr.lock.txt`), whether
   `setup_env.sh`'s sanity line said `katex check: True`, and the GPU check's `driver …` and
   `vllm ops: …` lines.
2. Server start-up times (the `=== chandra_ocr_2 ready after …s on 127.0.0.1:<port> ===` line,
   and the same for `qwen3_8_27b`) and the `read:`/`review:` summary lines (pages saved,
   seconds, review decisions).
3. For each paper, `report.json`'s `review` and `open_flags` counts. Also a look at the Markdown
   next to the PDF: which formulas are wrong, whether the reading order is right on two-column
   pages, and whether figures were cropped and described sensibly.
4. Any traceback in `logs/*.err` or the `logs/vllm_*.log` files. If a server failed to start,
   the last 40 lines of its log are copied into the `.err`.
5. If the reviewer (Qwen3.8-27B, a hybrid-attention model) fails to start or misbehaves on the
   A100s, stage the conservative pair (`python tools/stage_models.py --profile conservative`)
   and rerun the same three papers with `WOCR_PROFILE=conservative` (dots.mocr +
   Qwen3-VL-32B) and `OUT=/project/ikoutis/$USER/wocr/runs/o002_smoke_conservative`. The
   default `WORK` is per profile, so this run reads the pages again with dots.mocr instead
   of reusing Chandra's reading. Report both runs. That is also a first, informal reader
   comparison.

No numbers are expected to be final here. The point is that the plumbing works with the real
models, and a list of what to fix before M2.

---

## 2026-10-01 — Note [O-001]: project opened — two-model OCR for papers, scaffold on `main`

This entry opens the project. The proposal is in [`design.md`](design.md), and the first
working version of the pipeline is on `main`. In short:

**What it does.** It turns scanned (or born-digital) research papers into Markdown:

- display math as `$$…$$` LaTeX and inline math as `$…$`;
- tables as GFM, or HTML when cells are merged;
- figures cropped, linked, and described. A flowchart also gets Mermaid, and a commutative
  diagram gets tikz-cd. *(Graph drawings: see [O-003] — Markdown and TikZ, marked.)*

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
