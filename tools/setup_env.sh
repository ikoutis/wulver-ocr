#!/bin/bash
# =============================================================================
# tools/setup_env.sh — create the pipeline + serving conda env on Wulver.
#
# Run ONCE from the repo root (same convention as the dml repo):
#     bash tools/setup_env.sh
#
# Creates a prefix env at /project/ikoutis/conda_env/wocr (override by
# exporting WOCR_CONDA_ENV first) with python 3.12, vLLM, node + KaTeX (for
# the formula validator), and requirements.txt. One env holds both the
# pipeline client (light: httpx, pillow, pypdfium2) and vLLM, whose wheels
# bundle torch + the CUDA runtime, so no cluster CUDA module is needed — only
# a recent enough driver on the GPU nodes. `pip freeze` is saved next to the
# env as wocr.lock.txt.
#
# CUDA flavour. vLLM's default wheels are built for CUDA 12.9 (12.8 and 13.0
# builds also exist). Login nodes have no driver to auto-detect, so the
# flavour is explicit: WOCR_TORCH_BACKEND=cu129 (default) | cu128 | cu130.
# CUDA 13 needs a driver >= 580; check `nvidia-smi` on a GPU node (the srun
# line below) before choosing it. To pin vLLM: WOCR_VLLM_SPEC="vllm==<ver>".
# If the wheels and the node driver cannot be reconciled, use the official
# container instead: `apptainer pull docker://vllm/vllm-openai:<tag>` on a
# compute node (module load apptainer) and run serve_lib.sh's command
# through `apptainer exec --nv`.
#
# Login nodes have no GPU (cuda=False below is expected) and cap memory at
# 20 GB per user; if the vLLM install is killed there, run this script inside
# an interactive CPU session instead. Verify on a GPU (debug QOS, free):
#     srun --account=ikoutis --qos=debug --partition=debug_gpu \
#          --gres=gpu:a100_10g:1 --time=00:10:00 \
#          bash -lc 'module load Miniforge3 && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${WOCR_CONDA_ENV:-/project/ikoutis/conda_env/wocr}" && nvidia-smi && python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
# If torch reports cuda=False on the GPU node, the wheel's CUDA is newer than
# the node driver (nvidia-smi prints the max CUDA it supports): reinstall with
# a matching wheel, e.g. WOCR_VLLM_SPEC="vllm==<older>" or the cu12x variant.
# =============================================================================
set -euo pipefail

ENV_PREFIX="${WOCR_CONDA_ENV:-/project/ikoutis/conda_env/wocr}"
VLLM_SPEC="${WOCR_VLLM_SPEC:-vllm}"
TORCH_BACKEND="${WOCR_TORCH_BACKEND:-cu129}"
mkdir -p "$(dirname "$ENV_PREFIX")"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -d "$ENV_PREFIX" ]; then
    echo "[*] env already exists at $ENV_PREFIX — updating packages in place"
else
    echo "[*] creating conda env at $ENV_PREFIX"
    conda create -y --prefix "$ENV_PREFIX" python=3.12
fi

set +u
conda activate "$ENV_PREFIX"
set -u

pip install --no-cache-dir --upgrade pip uv
# uv picks the torch build matching the requested CUDA flavour (vLLM's
# recommended install path); plain pip would take whatever torch PyPI has.
uv pip install --no-cache "$VLLM_SPEC" --torch-backend="$TORCH_BACKEND"
uv pip install --no-cache -r requirements.txt

# KaTeX for the formula validator (src/katex_check.py); optional — the
# pipeline skips the check if this step fails.
if conda install -y -q -c conda-forge nodejs >/dev/null; then
    mkdir -p "$ENV_PREFIX/share/wocr-katex"
    npm install --silent --no-audit --no-fund --prefix "$ENV_PREFIX/share/wocr-katex" katex \
        || echo "[!] katex install failed — formula KaTeX check disabled"
else
    echo "[!] nodejs install failed — formula KaTeX check disabled"
fi
pip freeze > "$ENV_PREFIX/wocr.lock.txt"

echo
echo "[*] sanity check (login nodes have no GPU — cuda=False is expected here):"
python - <<'EOF'
import torch, vllm, httpx, pypdfium2, PIL
from src.katex_check import katex_available
print(f"    torch {torch.__version__} (CUDA {torch.version.cuda}) | vllm {vllm.__version__} "
      f"| cuda available: {torch.cuda.is_available()} | katex check: {katex_available()}")
EOF

echo
echo "[*] done; package versions saved to $ENV_PREFIX/wocr.lock.txt. Next:"
echo "      python tools/stage_models.py --profile default     # download weights (~65 GB)"
echo "      pytest tests/                                     # CPU-only, seconds"
echo "    then e.g.: INPUTS=<dir of PDFs> sbatch --array=0-3 slurm/ocr.sbatch"
if [ "$ENV_PREFIX" != "/project/ikoutis/conda_env/wocr" ]; then
    echo "    NOTE: non-default env path — export WOCR_CONDA_ENV=$ENV_PREFIX"
    echo "    in the shell you sbatch from."
fi
