# dev-communication

A dated, async communication channel for the wulver-ocr project. It uses the
same mechanism as the dml and knowledge-diffusion repos: tasks are handed to
collaborators here, and what comes back is recorded here, so the thread
survives across sessions and machines.

## How it works

- **[`log.md`](log.md)** is the running log: one rolling document of dated
  entries, **newest at the top**.
- Every entry is headed `## YYYY-MM-DD [HH:MM TZ] — <kind>: <short title>`.
  The time is optional.
- A **task** (`Task:`) says who it is for, why, exactly what to run, and what
  to report back.
- A **note** (`Note:`) is informational and asks nothing of the reader.
- A **reply** (`<name>:`) is a new entry at the top that cites the task it
  answers, with numbers, errors, and questions pasted in.
- Keep entries self-contained. Link code by repo-relative path.
- **Write findings in plain, uncompressed prose.** Spell out what was
  expected, what was observed, and what it means. Tables carry the numbers;
  the meaning goes in the prose.

## Communication IDs

Task entries carry an incrementing ID: `[O-001]`, `[O-002]`, …. The `O`
prefix keeps them distinct from dml's `[D-00N]` and knowledge-diffusion's
`[C-00N]`.

A batch launched because of an entry embeds its ID in the output root, for
example `OUT=/project/ikoutis/$USER/wocr/runs/o00N_<what>/`. That way
anyone holding an output directory can find the instruction that produced it.

## Index

| Document | What |
|---|---|
| [`log.md`](log.md) | The running, dated task/reply log. Start here. |
| [`design.md`](design.md) | The proposal: architecture, model choices, Wulver deployment, output format, evaluation plan, milestones. |
