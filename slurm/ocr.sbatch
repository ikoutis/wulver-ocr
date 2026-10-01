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
#   WORK        page images + JSON     (default /scratch/ikoutis/$USER/wocr/work)
#   WOCR_PROFILE  model pair in profiles/<name>.sh: default (Chandra 2 +
#               Qwen3.8-27B), dots (dots.mocr, MIT), conservative
#   REVIEW_ARGS extra review flags, e.g. "--review-types formula,table"
#   PHASES      subset of "read review assemble"     (default: all three)
#   NSHARDS     shards in total (default: the array size; set it when you
#               resubmit only some indices, e.g. via tools/incomplete.py)
#
# Submit from the repo root, e.g. 8 shards:
#   INPUTS=/project/ikoutis/$USER/papers sbatch --array=0-7 slurm/ocr.sbatch
# Recovery after anything: resubmit exactly the unfinished shards —
#   IDS=$(python tools/incomplete.py --inputs "$INPUTS" --nshards 8)
#   [ -n "$IDS" ] && INPUTS=... NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
# Interactive (inside `interactive -a ikoutis -q standard -j gpu`): run
#   INPUTS=paper.pdf bash slurm/ocr.sbatch           # one shard, same phases
#
# Throughput planning (to be measured, see dev-communication/design.md §8):
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
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-0

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${WOCR_CONDA_ENV:-/project/ikoutis/conda_env/wocr}"
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

: "${INPUTS:?set INPUTS to the documents to process (dir, files, or @listfile)}"
OUT="${OUT:-/project/ikoutis/$USER/wocr/out}"
WORK="${WORK:-/scratch/ikoutis/$USER/wocr/work}"
WOCR_PROFILE="${WOCR_PROFILE:-default}"
PHASES="${PHASES:-read review assemble}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}/$NSHARDS"
mkdir -p "$OUT" "$WORK"

# shellcheck source=/dev/null
source "profiles/$WOCR_PROFILE.sh"
source slurm/serve_lib.sh
source slurm/requeue_lib.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are pre-staged
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# shellcheck disable=SC2206
IN=($INPUTS)
COMMON=(--inputs "${IN[@]}" --work "$WORK" --out "$OUT" --shard "$SHARD")

echo "=== wocr | shard $SHARD | profile $WOCR_PROFILE | phases: $PHASES | job ${SLURM_JOB_ID:-local} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -m src.run_ocr ingest "${COMMON[@]}"
requeue_if_signalled

has_phase() { [[ " $PHASES " == *" $1 "* ]]; }

if has_phase read && [ "$(python -m src.run_ocr todo --stage read "${COMMON[@]}" | tail -n1)" != 0 ]; then
    start_server "$READER_NAME" "$WOCR_MODELS/$READER_NAME" 8001 "${READER_VLLM_ARGS[@]}"
    run_with_requeue python -m src.run_ocr read "${COMMON[@]}" \
        --reader "$READER_ADAPTER" --reader-url http://127.0.0.1:8001 \
        --reader-model "$READER_NAME" --reader-extra "${READER_REQUEST_EXTRA:-}" \
        --workers "${READER_WORKERS:-64}"
    stop_server "$READER_NAME"
fi

if has_phase review && [ "$(python -m src.run_ocr todo --stage review "${COMMON[@]}" | tail -n1)" != 0 ]; then
    start_server "$EDITOR_NAME" "$WOCR_MODELS/$EDITOR_NAME" 8002 "${EDITOR_VLLM_ARGS[@]}"
    run_with_requeue python -m src.run_ocr review "${COMMON[@]}" \
        --editor-url http://127.0.0.1:8002 --editor-model "$EDITOR_NAME" \
        --editor-extra "${EDITOR_REQUEST_EXTRA:-}" \
        --workers "${EDITOR_WORKERS:-32}" ${REVIEW_ARGS:-}
    stop_server "$EDITOR_NAME"
fi

if has_phase assemble; then
    run_with_requeue python -m src.run_ocr assemble "${COMMON[@]}"
fi
echo "=== done $(date) ==="
