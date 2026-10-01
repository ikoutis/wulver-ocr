# profiles/conservative.sh — the most mature serving path on A100s: dots.mocr
# (MIT) + Qwen3-VL-32B-Instruct (Apache-2.0, Oct 2025, standard attention).
# Use if the default pair's newer architectures misbehave on Ampere / the
# installed vLLM (the hybrid-attention reviewer is the likelier suspect).

READER_ADAPTER=dots
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

EDITOR_REPO=Qwen/Qwen3-VL-32B-Instruct
EDITOR_NAME=qwen3_vl_32b
EDITOR_VLLM_ARGS=(
    --max-model-len 16384
    --gpu-memory-utilization 0.92
    --max-num-seqs 32
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
)
EDITOR_REQUEST_EXTRA=''

WOCR_MODELS="${WOCR_MODELS:-/project/ikoutis/wocr_models}"
