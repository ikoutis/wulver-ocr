"""Print the shard indices whose documents are not all assembled, as an
--array expression, so recovery after ANY interruption is one line
(same convention as the dml repo's tools/incomplete.py):

    IDS=$(python tools/incomplete.py --inputs "$INPUTS" --nshards 8)
    [ -n "$IDS" ] && INPUTS="$INPUTS" NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch

    python tools/incomplete.py --inputs "$INPUTS" --nshards 8 --list   # per shard

A document is complete when <OUT>/<doc_id>/<doc_id>.md exists. Doc ids are
computed from file contents (sha256), so this works before ingestion too.
OUT defaults to the sbatch script's default.
"""

from __future__ import annotations

import argparse
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", default=os.environ.get(
        "OUT", f"/project/ikoutis/{os.environ.get('USER', 'user')}/wocr/out"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    paths = discover(args.inputs)
    bad = []
    for i in range(args.nshards):
        shard = paths[i::args.nshards]
        missing = []
        for p in shard:
            doc_id = make_doc_id(p, sha256_file(p))
            if not os.path.exists(os.path.join(args.out, doc_id, doc_id + ".md")):
                missing.append(p)
        if missing:
            bad.append(i)
        if args.list:
            print(f"shard {i:4d}: {len(shard) - len(missing):5d}/{len(shard)} done")
    if not args.list:
        print(compress(bad))


if __name__ == "__main__":
    main()
