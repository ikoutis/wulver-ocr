"""Download a profile's model weights to project storage (once).

    python tools/stage_models.py --profile default
    python tools/stage_models.py --profile default --dry-run

Reads READER_REPO / READER_NAME / EDITOR_REPO / EDITOR_NAME (and optional
*_REVISION) from profiles/<profile>.sh and snapshot-downloads each repo into
$WOCR_MODELS/<NAME> (default /project/ikoutis/wocr_models). Weights go to
/project, not /scratch: scratch is purged after 30 days. Local directory names
never contain dots — some trust-remote-code models (dots.ocr among them)
break when their directory name is not a valid Python module name.

Where to run: the dml convention is a login node (no GPU needed). Large
downloads can hit the login node's per-user limits; if so, run it in an
interactive CPU session (compute nodes reach Hugging Face too). Gated repos
need `huggingface-cli login` (or HF_TOKEN) first.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_profile(name: str) -> dict:
    path = os.path.join(ROOT, "profiles", f"{name}.sh")
    if not os.path.exists(path):
        sys.exit(f"no profile {path}")
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r'^\s*(?:export\s+)?([A-Z_]+)=["\']?([^"\'\s#(]*)["\']?\s*(#.*)?$', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--models_dir",
                    default=os.environ.get("WOCR_MODELS", "/project/ikoutis/wocr_models"))
    ap.add_argument("--only", choices=["reader", "editor"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    prof = read_profile(args.profile)
    roles = [r for r in ("READER", "EDITOR") if args.only in (None, r.lower())]
    for role in roles:
        repo, name = prof.get(f"{role}_REPO"), prof.get(f"{role}_NAME")
        if not repo or not name:
            sys.exit(f"profile {args.profile} lacks {role}_REPO / {role}_NAME")
        if "." in name:
            sys.exit(f"{role}_NAME={name!r} must not contain dots (see docstring)")
        dst = os.path.join(args.models_dir, name)
        rev = prof.get(f"{role}_REVISION") or None
        print(f"[*] {role.lower()}: {repo}{'@' + rev if rev else ''} -> {dst}")
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, revision=rev, local_dir=dst)
    print("[*] done" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
