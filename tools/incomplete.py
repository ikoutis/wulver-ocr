"""Print the shard indices whose documents are not all finished, as an
--array expression, so recovery after ANY interruption is one line
(same convention as the dml repo's tools/incomplete.py):

    IDS=$(python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT")
    [ -n "$IDS" ] && INPUTS="$INPUTS" OUT="$OUT" NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch

    python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT" --list

INPUTS is read the way slurm/ocr.sbatch reads it (IN=($INPUTS): split on
whitespace, globs expanded), so a quoted "$INPUTS" works too; an argument
that names an existing path is kept whole. The documents and their shards
are exactly run_ocr's: ingest.discover(inputs)[i::nshards].

A document is finished when <OUT>/<doc_id>/<doc_id>.md exists, or when
<OUT>/<doc_id>/FAILED.json says it could not be ingested (terminal: running
it again cannot help). An input that cannot be read at all (missing,
unreadable) is skipped with a warning, as the pipeline skips it. Doc ids are
computed from file contents (sha256), so this works before ingestion too.
OUT defaults to $OUT, then to the sbatch script's default.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import discover, make_doc_id, sha256_file  # noqa: E402


def compress(ids: list[int]) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8'."""
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def split_inputs(args: list[str]) -> list[str]:
    """The sbatch script's IN=($INPUTS), applied to each argument that is not
    an existing path: split on whitespace, expand globs (a pattern matching
    nothing stays as it is, as in bash)."""
    out = []
    for a in args:
        if os.path.exists(a[1:] if a.startswith("@") else a):
            out.append(a)
            continue
        for w in a.split():
            hits = sorted(glob.glob(w)) if any(c in w for c in "*?[") else []
            out.extend(hits or [w])
    return out


def status(path: str, out_root: str) -> str:
    """'done', 'failed' (terminal), 'unreadable' (skipped) or 'todo'."""
    try:
        doc_id = make_doc_id(path, sha256_file(path))
    except OSError as e:
        print(f"incomplete.py: skipping unreadable input {path}: {e}", file=sys.stderr)
        return "unreadable"
    d = os.path.join(out_root, doc_id)
    if os.path.exists(os.path.join(d, doc_id + ".md")):
        return "done"
    if os.path.exists(os.path.join(d, "FAILED.json")):
        return "failed"
    return "todo"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", default=os.environ.get(
        "OUT", f"/project/ikoutis/{os.environ.get('USER', 'user')}/wocr/out"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    inputs = split_inputs(args.inputs)
    if not inputs:
        ap.error("--inputs is empty (is INPUTS set?)")
    if args.nshards < 1:
        ap.error("--nshards must be at least 1")

    try:
        paths = discover(inputs)    # sliced as run_ocr.select_docs slices it
    except OSError as e:            # an @listfile that cannot be read
        ap.error(str(e))
    bad = []
    for i in range(args.nshards):
        shard = paths[i::args.nshards]
        n = {"done": 0, "failed": 0, "unreadable": 0, "todo": 0}
        for p in shard:
            n[status(p, args.out)] += 1
        if n["todo"]:
            bad.append(i)
        if args.list:
            extra = ", ".join(f"{n[k]} {k}" for k in ("failed", "unreadable") if n[k])
            print(f"shard {i:4d}: {len(shard) - n['todo']:5d}/{len(shard)} done"
                  + (f" ({extra})" if extra else ""))
    if not args.list:
        print(compress(bad))


if __name__ == "__main__":
    main()
