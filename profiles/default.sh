# profiles/default.sh — the default reader/reviewer pair (design.md §4).
# Sourced by slurm/ocr.sbatch; scalar *_REPO / *_NAME / *_REVISION lines are
# also parsed by tools/stage_models.py. Names must not contain dots.
#
#   reader   Chandra OCR 2 (4B; datalab-to). Best published olmOCR-Bench
#            "old scans math" score (89.1). Weights: modified OpenRAIL-M —
#            free for research/personal use; see design.md §4 before other use.
#   reviewer Qwen3.8-27B (Aug 2026; unified vision-language, hybrid
#            Gated-DeltaNet attention). bf16 fits one A100-80GB.
#
# Serving flags follow each model's own published launch settings; the
# reader's are datalab's H100-80GB baseline (same memory as our A100s).
# Confirmed in [O-002] on one A100-80GB: the reader is ready 266 s after
# start (55 GB of KV cache), the reviewer 320 s after; see the log entry
# for the measured throughput. Both need vLLM's own sampler (ocr.sbatch).

# ---- reader (stage 1) ----------------------------------------------------------
READER_ADAPTER=chandra                    # src/readers/chandra.py
READER_REPO=datalab-to/chandra-ocr-2
READER_NAME=chandra_ocr_2
READER_VLLM_ARGS=(
    --dtype bfloat16
    --max-model-len 20480
    --max-num-batched-tokens 8192
    --max-num-seqs 64
    --gpu-memory-utilization 0.85
    --enable-prefix-caching
    --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
)
READER_REQUEST_EXTRA=''

# ---- reviewer (stage 2) ---------------------------------------------------------
EDITOR_REPO=Qwen/Qwen3.8-27B
EDITOR_NAME=qwen3_8_27b
EDITOR_VLLM_ARGS=(
    --max-model-len 32768
    --gpu-memory-utilization 0.92
    --max-num-seqs 64
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
    --reasoning-parser qwen3
)
# Thinking off: proofreading one crop is a short comparison. Whether a little
# reasoning buys accuracy on formulas is an [O-002]/M3 question.
EDITOR_REQUEST_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'

WOCR_MODELS="${WOCR_MODELS:-/project/ikoutis/wocr_models}"
