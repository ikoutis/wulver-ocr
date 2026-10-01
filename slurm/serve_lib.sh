# serve_lib.sh — start / health-check / stop vLLM servers inside a job.
#
# The pipeline talks to its models over localhost HTTP (OpenAI-compatible),
# so a job runs `vllm serve` in the background, waits for /health, runs the
# stage, and stops the server. Servers bind 127.0.0.1 only: nothing is
# exposed off the node.
#
# Usage (after sourcing a profile, see profiles/):
#   source slurm/serve_lib.sh
#   start_server reader "$WOCR_MODELS/$READER_NAME" 8001 "${READER_VLLM_ARGS[@]}"
#   ... python -m src.run_ocr read --reader-url http://127.0.0.1:8001 ...
#   stop_server reader
# An EXIT trap stops anything still running, so a failed stage never leaves
# an orphaned server holding the GPU.

declare -A WOCR_SERVER_PIDS=()
WOCR_SERVER_LOGDIR="${WOCR_SERVER_LOGDIR:-logs}"
WOCR_SERVER_TIMEOUT="${WOCR_SERVER_TIMEOUT:-1200}"   # s; first start compiles + captures CUDA graphs

start_server() {
    local name=$1 model=$2 port=$3
    shift 3
    mkdir -p "$WOCR_SERVER_LOGDIR"
    local log="$WOCR_SERVER_LOGDIR/vllm_${name}_${SLURM_JOB_ID:-local}.log"
    if [ ! -e "$model" ]; then
        echo "SERVE ERROR: model path $model not found." >&2
        echo "Stage it first: python tools/stage_models.py --profile $WOCR_PROFILE" >&2
        return 1
    fi
    echo "=== starting $name: vllm serve $model on :$port | log: $log | $(date) ==="
    vllm serve "$model" --host 127.0.0.1 --port "$port" \
        --served-model-name "$name" "$@" >>"$log" 2>&1 &
    WOCR_SERVER_PIDS[$name]=$!
    local t=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        if ! kill -0 "${WOCR_SERVER_PIDS[$name]}" 2>/dev/null; then
            echo "SERVE ERROR: $name exited during startup; last lines of $log:" >&2
            tail -n 40 "$log" >&2
            unset "WOCR_SERVER_PIDS[$name]"
            return 1
        fi
        if [ "$t" -ge "$WOCR_SERVER_TIMEOUT" ]; then
            echo "SERVE ERROR: $name not healthy after ${t}s" >&2
            stop_server "$name"
            return 1
        fi
        sleep 5
        t=$((t + 5))
    done
    echo "=== $name ready after ${t}s ==="
}

stop_server() {
    local name=$1 pid="${WOCR_SERVER_PIDS[$1]:-}"
    [ -z "$pid" ] && return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    unset "WOCR_SERVER_PIDS[$name]"
    echo "=== stopped $name ==="
}

stop_all_servers() {
    local n
    for n in "${!WOCR_SERVER_PIDS[@]}"; do
        stop_server "$n"
    done
}
trap stop_all_servers EXIT
