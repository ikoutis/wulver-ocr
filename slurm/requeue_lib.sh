# requeue_lib.sh — wall-clock / preemption self-requeue for array tasks.
#
# Adapted from the dml repo's slurm/requeue_lib.sh, where it is
# battle-tested ([D-002], [D-006] there). Changes: (1) a USR1 that lands
# BETWEEN steps (between two pipeline commands, or during a server start-up,
# when no pipeline process is running to forward it to) no longer kills the
# batch shell — it is recorded, and the task requeues before the next step
# starts (serve_lib.sh also checks it while waiting for a server); (2) a
# non-array job requeues itself by its plain job id; (3) a pipeline killed by
# the forwarded USR1 before it installed its handler (rc 138) requeues too.
#
#   * scripts request --signal=B:USR1@<secs> so SLURM signals the batch shell
#     before the wall (and, under qos=low, at preemption with GraceTime);
#   * run_with_requeue forwards the signal to the pipeline (src/run_ocr.py),
#     which starts no new model request, lets those in flight finish, saves
#     every finished page atomically, and exits with code 85;
#   * on 85 this calls `scontrol requeue` on THIS array task; the requeued task
#     re-enters the script and every stage skips pages already on disk.
# OCR state is per page, so a resume loses at most the pages in flight.
#
# Manual recovery after anything else (node failure, a model server crash,
# cancelled array, ...), with the same INPUTS, OUT, shard count (and
# WOCR_PROFILE / WORK, if you set them) as the original submission:
#   IDS=$(python tools/incomplete.py --inputs $INPUTS --nshards 8 --out "$OUT")
#   [ -n "$IDS" ] && INPUTS="$INPUTS" OUT="$OUT" NSHARDS=8 sbatch --array=$IDS slurm/ocr.sbatch
#
# Usage in a script (from the repo root):
#   source slurm/requeue_lib.sh
#   run_with_requeue python -m src.run_ocr read ...

WOCR_SIGNALLED=${WOCR_SIGNALLED:-0}   # keep a USR1 recorded before sourcing (ocr.sbatch)
_wocr_flag_usr1() {
    WOCR_SIGNALLED=1
    echo "=== USR1 between steps: will requeue before the next step | $(date) ==="
}
trap _wocr_flag_usr1 USR1          # armed as soon as this file is sourced

requeue_self() {
    # NOTE: USR1 fires not only near the wall — preemption with GraceTime
    # resets the job's end time to now+grace, which triggers --signal
    # immediately. Either way the correct move is: request a requeue, then
    # WAIT to be killed. Never exit 0 here: a clean exit races the requeue and
    # SLURM can record COMPLETED and drop the task from the queue with the
    # work unfinished (observed in production, dml [D-006]). If the
    # requeue-kill (or the preemption kill) doesn't arrive, exit 75 — a loud
    # FAILED that tools/incomplete.py surfaces for resubmission.
    local target="${SLURM_JOB_ID:-}"
    [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    if [ -z "$target" ]; then
        echo "=== signalled outside SLURM: state saved; rerun to resume ==="
        exit 85
    fi
    echo "=== requeueing $target and awaiting kill | $(date) ==="
    scontrol requeue "$target" || true
    sleep 300
    echo "=== requeue kill never arrived; exiting 75 for manual resubmission ==="
    exit 75
}

requeue_if_signalled() {
    if [ "$WOCR_SIGNALLED" = 1 ]; then
        requeue_self
    fi
    return 0
}

run_with_requeue() {
    requeue_if_signalled
    local pid rc
    # Forward the signal, and remember it: if the step ends without acting on
    # it (it was finishing anyway), the next step must not start.
    trap 'WOCR_SIGNALLED=1; kill -USR1 "$pid" 2>/dev/null' USR1
    "$@" &
    pid=$!
    set +e
    wait "$pid"; rc=$?
    # A trapped signal interrupts `wait` (rc > 128); wait again until the
    # child has actually exited.
    while [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; do
        wait "$pid"; rc=$?
    done
    set -e
    trap _wocr_flag_usr1 USR1
    # rc 85 = the pipeline saved its state after USR1. rc 138 = 128 + USR1:
    # the signal arrived before the pipeline installed its handler, so it
    # died before saving anything new (pages are written atomically).
    if [ "$rc" -eq 85 ] || [ "$rc" -eq 138 ]; then
        echo "=== USR1: pipeline stopped (rc $rc), its finished pages are saved | $(date) ==="
        requeue_self
    fi
    return "$rc"
}
