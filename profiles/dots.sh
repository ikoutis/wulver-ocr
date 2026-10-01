# profiles/dots.sh — permissively licensed reader: dots.mocr (MIT, 3B,
# rednote-hilab, Mar 2026) + the default reviewer. dots.mocr scores 83.9 on
# olmOCR-Bench (old scans math 85.5) vs Chandra 2's 85.8 (89.1), returns layout
# JSON with boxes, and is the reader for arm A5 of the evaluation.
# Serving flags from the dots.mocr README.

READER_ADAPTER=dots                       # src/readers/dots.py
READER_REPO=rednote-hilab/dots.mocr
READER_NAME=dots_mocr
READER_VLLM_ARGS=(
    --trust-remote-code
    --chat-template-content-format string
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --max-num-seqs 64
)
READER_REQUEST_EXTRA=''

EDITOR_REPO=Qwen/Qwen3.8-27B
EDITOR_NAME=qwen3_8_27b
EDITOR_VLLM_ARGS=(
    --max-model-len 32768
    --gpu-memory-utilization 0.92
    --max-num-seqs 64
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
    --reasoning-parser qwen3
)
EDITOR_REQUEST_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'

WOCR_MODELS="${WOCR_MODELS:-/project/ikoutis/wocr_models}"
