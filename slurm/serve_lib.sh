# serve_lib.sh — start / health-check / stop vLLM servers inside a job.
#
# The pipeline talks to its models over localhost HTTP (OpenAI-compatible),
# so a job runs `vllm serve` in the background, waits until it is ready, runs
# the stage, and stops the server. Servers bind 127.0.0.1, which keeps them
# off the network but NOT away from other jobs on the same node: SLURM packs
# up to four of our 1-GPU array tasks (and other users' jobs) onto one
# 4-GPU node. Hence:
#   * each start takes a port the kernel reports free at that moment, never a
#     fixed one (a lost race for it is retried on another port);
#   * "ready" means OUR server: our process is alive, our own log shows
#     uvicorn's "Application startup complete" (an INFO line: keep vLLM's
#     --uvicorn-log-level at its default, info), and /health answers on our
#     port. Another job's server answering on that port does not count.
#
# Usage (after sourcing a profile, see profiles/):
#   source slurm/serve_lib.sh
#   start_server "$READER_NAME" "$WOCR_MODELS/$READER_NAME" "${READER_VLLM_ARGS[@]}"
#   ... python -m src.run_ocr read --reader-url "$(server_url "$READER_NAME")" ...
#   stop_server "$READER_NAME"     # returns 1 if the server had died on its own
# The first argument is also the served model name, so it is what every
# block's provenance ("reader:chandra:chandra_ocr_2") records.
# With slurm/requeue_lib.sh sourced, a USR1 that arrives before or during a
# start-up requeues the task at once instead of after the (long) start-up.
# An EXIT trap stops anything still running, so a failed stage never leaves
# an orphaned server holding the GPU.

declare -A WOCR_SERVER_PIDS=() WOCR_SERVER_PORTS=() WOCR_SERVER_LOGS=()
WOCR_SERVER_LOGDIR="${WOCR_SERVER_LOGDIR:-logs}"
WOCR_SERVER_TIMEOUT="${WOCR_SERVER_TIMEOUT:-1200}"   # s; first start compiles + captures CUDA graphs
WOCR_SERVER_POLL="${WOCR_SERVER_POLL:-5}"            # s between readiness checks

free_port() {
    python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

server_url() {
    echo "http://127.0.0.1:${WOCR_SERVER_PORTS[$1]}"
}

# True if the log, from byte $2 on (this start's output only: the log is
# appended to across requeues), contains $3.
_wocr_log_has() {
    grep -q -- "$3" < <(tail -c +"$2" "$1")
}

# A USR1 recorded by requeue_lib.sh: requeue now rather than after start-up.
_wocr_requeue_if_signalled() {
    [ "${WOCR_SIGNALLED:-0}" = 1 ] || return 0
    stop_server "$1" || true
    requeue_self
}

start_server() {
    local name=$1 model=$2
    shift 2
    if [ ! -e "$model" ]; then
        echo "SERVE ERROR: model path $model not found." >&2
        echo "Stage it first: python tools/stage_models.py --profile ${WOCR_PROFILE:-default}" >&2
        return 1
    fi
    mkdir -p "$WOCR_SERVER_LOGDIR"
    local log="$WOCR_SERVER_LOGDIR/vllm_${name}_${SLURM_JOB_ID:-local}.log"
    WOCR_SERVER_LOGS[$name]=$log
    local attempt port from pid t0
    for attempt in 1 2 3; do
        _wocr_requeue_if_signalled "$name"
        port=$(free_port)
        : >>"$log"
        from=$(( $(wc -c <"$log") + 1 ))
        echo "=== starting $name: vllm serve $model on 127.0.0.1:$port | log: $log | $(date) ==="
        vllm serve "$model" --host 127.0.0.1 --port "$port" \
            --served-model-name "$name" "$@" >>"$log" 2>&1 &
        pid=$!
        WOCR_SERVER_PIDS[$name]=$pid
        t0=$SECONDS
        while :; do
            _wocr_requeue_if_signalled "$name"
            if ! kill -0 "$pid" 2>/dev/null; then
                unset "WOCR_SERVER_PIDS[$name]"
                if _wocr_log_has "$log" "$from" "Address already in use"; then
                    echo "=== port $port was taken before $name bound it; trying another ==="
                    continue 2
                fi
                echo "SERVE ERROR: $name exited during startup; last lines of $log:" >&2
                tail -n 40 "$log" >&2
                return 1
            fi
            if _wocr_log_has "$log" "$from" "Application startup complete" &&
                    curl -sf --max-time 10 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
                WOCR_SERVER_PORTS[$name]=$port
                echo "=== $name ready after $((SECONDS - t0))s on 127.0.0.1:$port ==="
                return 0
            fi
            if [ $((SECONDS - t0)) -ge "$WOCR_SERVER_TIMEOUT" ]; then
                echo "SERVE ERROR: $name not healthy after $((SECONDS - t0))s" >&2
                stop_server "$name" || true
                return 1
            fi
            sleep "$WOCR_SERVER_POLL"
        done
    done
    echo "SERVE ERROR: $name lost the race for a free port $attempt times; last lines of $log:" >&2
    tail -n 40 "$log" >&2
    return 1
}

stop_server() {
    local name=$1 pid="${WOCR_SERVER_PIDS[$1]:-}" rc=0
    [ -z "$pid" ] && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
        # Died while a stage was using it. The pipeline exits 3 when its
        # requests fail; this also catches a death it did not notice.
        echo "SERVE ERROR: $name had exited on its own; last lines of ${WOCR_SERVER_LOGS[$name]}:" >&2
        tail -n 40 "${WOCR_SERVER_LOGS[$name]}" >&2
        rc=1
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    unset "WOCR_SERVER_PIDS[$name]" "WOCR_SERVER_PORTS[$name]"
    echo "=== stopped $name ==="
    return "$rc"
}

stop_all_servers() {
    local n
    for n in "${!WOCR_SERVER_PIDS[@]}"; do
        stop_server "$n" || true
    done
}
trap stop_all_servers EXIT
