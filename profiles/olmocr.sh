# profiles/olmocr.sh — the olmOCR-2 baseline: allenai/olmOCR-2-7B-1025
# (Apache-2.0; Qwen2.5-VL-7B fine-tuned; whole-page Markdown, no layout boxes)
# as the reader, with the default reviewer. It is the published baseline for
# arms A0/A5 of the evaluation (design.md §7). Without boxes, figures are not
# cropped and the reviewer sees the whole page for each block. The adapter
# (src/readers/markdown.py, OlmOCRReader) reproduces olmOCR's own prompt,
# 1288 px rendering, 8000-token cap, and per-attempt temperatures.

READER_ADAPTER=olmocr
READER_REPO=allenai/olmOCR-2-7B-1025
READER_NAME=olmocr_2_7b
READER_VLLM_ARGS=(
    --max-model-len 16384
    --gpu-memory-utilization 0.90
    --max-num-seqs 64
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
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
