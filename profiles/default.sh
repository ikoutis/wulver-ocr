# profiles/default.sh — the default reader/reviewer pair.
# Sourced by slurm/ocr.sbatch; parsed (scalar *_REPO / *_NAME / *_REVISION
# lines only) by tools/stage_models.py. Names must not contain dots.
#
# PROVISIONAL (2026-10-01): pending the model survey in
# dev-communication/design.md §4 and the [O-002] smoke run.

# ---- reader: specialist document-OCR model (stage 1) ------------------------
READER_ADAPTER=dots                      # src/readers/dots.py
READER_REPO=rednote-hilab/dots.ocr
READER_NAME=dots_ocr
READER_VLLM_ARGS=(
    --trust-remote-code
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --max-num-seqs 128
    --limit-mm-per-prompt '{"image": 1}'
)

# ---- reviewer: general VLM (stage 2) -----------------------------------------
EDITOR_REPO=Qwen/Qwen3-VL-32B-Instruct
EDITOR_NAME=qwen3_vl_32b
EDITOR_VLLM_ARGS=(
    --max-model-len 16384
    --gpu-memory-utilization 0.92
    --max-num-seqs 32
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
)

WOCR_MODELS="${WOCR_MODELS:-/project/ikoutis/wocr_models}"
