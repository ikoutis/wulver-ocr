#!/bin/bash -l
# =============================================================================
# ocr.sbatch — scanned papers -> Markdown, one array task per shard.
#
# Each task takes its 1/N slice of the input documents and runs the stages in
# phases on ONE full A100, loading one model at a time so each gets the whole
# 80 GB (bigger KV cache = bigger batches = faster):
#
#   ingest    CPU   render pages                       (skipped if done)
#   read      GPU   reader server up -> stage 1 -> down (skipped if done)
#   review    GPU   editor server up -> stage 2 -> down (skipped if done)
#   assemble  CPU   Markdown + figures + report
#
# Configure by environment variables (sbatch exports your shell by default):
#   INPUTS      REQUIRED. dir(s), PDF/image file(s), or @listfile, space-separated
#   OUT         Markdown output root   (default /project/ikoutis/$USER/wocr/out)
#   WORK        page images + JSON     (default /scratch/ikoutis/$USER/wocr/work/<profile>:
#               per profile, so a rerun with another profile really re-reads)
#   WOCR_PROFILE  model pair in profiles/<name>.sh: default (Chandra 2 +
#               Qwen3.8-27B), dots (dots.mocr, MIT), conservative
#   READ_ARGS   extra read flags, e.g. "--retry-failed"
#   REVIEW_ARGS extra review flags, e.g. "--review-types formula,table"
#   PHASES      subset of "read review assemble"     (default: all three)
#   NSHARDS     shards in total (default: the array size; REQUIRED when you
#               resubmit only some indices, e.g. via tools/incomplete.py)
# Relative paths are taken relative to the directory you submit (or run) from
# (paths inside an @listfile: make them absolute).
#
# Submit from the repo root, e.g. 8 shards:
#   export INPUTS=/project/ikoutis/$USER/papers OUT=/project/ikoutis/$USER/wocr/runs/o00N_<what>
#   sbatch --array=0-7 slurm/ocr.sbatch
# Recovery after anything: resubmit exactly the unfinished shards, with the
# same INPUTS, OUT and shard count (and WOCR_PROFILE / WORK, if you set them):
#   IDS=$(python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT")
#   [ -n "$IDS" ] && INPUTS="$INPUTS" OUT="$OUT" NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
# Interactive (inside `interactive -a ikoutis -q standard -j gpu`): run
#   INPUTS=paper.pdf OUT=out WORK=work bash slurm/ocr.sbatch   # one shard, same phases
#
# Model servers listen on 127.0.0.1 at ports picked when they start, so
# several tasks can share a node (see slurm/serve_lib.sh). If one dies, the
# stage exits 3 and the task FAILS (its finished pages are saved): resubmit
# with the recovery line above. On the USR1 below (wall clock or preemption)
# the task saves and requeues itself (slurm/requeue_lib.sh). The 30 min lead
# covers the requests in flight (a long generation on the 27B reviewer takes
# minutes), stopping the server, and the requeue: the dml convention.
#
# Throughput planning (to be measured, see dev-communication/design.md §5):
# a 3–4B reader does roughly 1 page/s on an A100; the reviewer only sees
# formulas, flagged blocks and figures. Size shards at a few thousand pages
# so each task spends minutes, not most of its time, starting servers.
# =============================================================================

#SBATCH --job-name=wocr
#SBATCH --output=logs/wocr_%A_%a.log
#SBATCH --error=logs/wocr_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-0

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${WOCR_CONDA_ENV:-/project/ikoutis/conda_env/wocr}"
set -u

# Where this was started, and the repo. `bash slurm/ocr.sbatch` runs this
# file in place, from anywhere (in an `interactive` session SLURM_SUBMIT_DIR
# is wherever that session began); sbatch runs a spool copy of it, started
# from the submit dir.
if [ -f "$(dirname "$0")/serve_lib.sh" ]; then
    START=$PWD
    REPO="$(cd "$(dirname "$0")/.." && pwd)"
else
    START="${SLURM_SUBMIT_DIR:-$PWD}"
    REPO=$START
fi
abspath() {                     # settings' relative paths are relative to START
    case "$1" in
        /* | @/*) echo "$1" ;;
        @*) echo "@$START/${1#@}" ;;
        *) echo "$START/$1" ;;
    esac
}

: "${INPUTS:?set INPUTS to the documents to process (dir, files, or @listfile)}"
WOCR_PROFILE="${WOCR_PROFILE:-default}"
OUT="$(abspath "${OUT:-/project/ikoutis/$USER/wocr/out}")"
WORK="$(abspath "${WORK:-/scratch/ikoutis/$USER/wocr/work/$WOCR_PROFILE}")"
IN=()
# shellcheck disable=SC2086  # INPUTS is a space-separated list
for x in $INPUTS; do IN+=("$(abspath "$x")"); done
PHASES="${PHASES:-read review assemble}"
if [ -z "${NSHARDS:-}" ] && [ "${SLURM_ARRAY_TASK_MAX:-0}" -ge "${SLURM_ARRAY_TASK_COUNT:-1}" ]; then
    echo "ERROR: --array is not 0..N-1, so the shard count is unknown: set NSHARDS" \
         "to the original submission's (see tools/incomplete.py)" >&2
    exit 2
fi
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}/$NSHARDS"

cd "$REPO"
if [ ! -f "profiles/$WOCR_PROFILE.sh" ]; then
    echo "ERROR: no profiles/$WOCR_PROFILE.sh in $PWD (submit from the repo root)" >&2
    exit 2
fi
mkdir -p logs "$OUT" "$WORK"

# shellcheck source=/dev/null
source "profiles/$WOCR_PROFILE.sh"
source slurm/serve_lib.sh
source slurm/requeue_lib.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are pre-staged
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

COMMON=(--inputs "${IN[@]}" --work "$WORK" --out "$OUT" --shard "$SHARD")

echo "=== wocr | shard $SHARD | profile $WOCR_PROFILE | phases: $PHASES | job ${SLURM_JOB_ID:-local} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
run_with_requeue python -m src.run_ocr ingest "${COMMON[@]}"

has_phase() { [[ " $PHASES " == *" $1 "* ]]; }

# Each stage exits 85 on USR1 (requeue) and 3 if its model server fails
# (set -e then fails the task); stop_server fails it too if the server died.
if has_phase read && [ "$(python -m src.run_ocr todo --stage read "${COMMON[@]}" | tail -n1)" != 0 ]; then
    start_server "$READER_NAME" "$WOCR_MODELS/$READER_NAME" "${READER_VLLM_ARGS[@]}"
    # shellcheck disable=SC2086  # READ_ARGS is a list of flags
    run_with_requeue python -m src.run_ocr read "${COMMON[@]}" \
        --reader "$READER_ADAPTER" --reader-url "$(server_url "$READER_NAME")" \
        --reader-model "$READER_NAME" --reader-extra "${READER_REQUEST_EXTRA:-}" \
        --workers "${READER_WORKERS:-64}" ${READ_ARGS:-}
    stop_server "$READER_NAME"
fi

if has_phase review && [ "$(python -m src.run_ocr todo --stage review "${COMMON[@]}" | tail -n1)" != 0 ]; then
    start_server "$EDITOR_NAME" "$WOCR_MODELS/$EDITOR_NAME" "${EDITOR_VLLM_ARGS[@]}"
    # shellcheck disable=SC2086  # REVIEW_ARGS is a list of flags
    run_with_requeue python -m src.run_ocr review "${COMMON[@]}" \
        --editor-url "$(server_url "$EDITOR_NAME")" --editor-model "$EDITOR_NAME" \
        --editor-extra "${EDITOR_REQUEST_EXTRA:-}" \
        --workers "${EDITOR_WORKERS:-32}" ${REVIEW_ARGS:-}
    stop_server "$EDITOR_NAME"
fi

if has_phase assemble; then
    run_with_requeue python -m src.run_ocr assemble "${COMMON[@]}"
fi
echo "=== done $(date) ==="
