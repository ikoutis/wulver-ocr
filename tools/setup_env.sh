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
# vLLM version and CUDA flavour. vLLM is pinned to a tested release,
# WOCR_VLLM_SPEC (default vllm==0.30.0). Its PyPI wheel, and the torch 2.13
# it pins, are CUDA 13 builds, which need a GPU driver >= 580. For older
# drivers each release also has a +cu129 wheel on GitHub, installed with
# torch from the PyTorch cu129 index (vLLM's own install docs); CUDA 12
# builds run on drivers from R525 on. The flavour follows the driver when
# nvidia-smi sees one: >= 580 -> cu130 (PyPI), older -> cu129. Login nodes
# have no driver: there the CUDA 13 build is installed, with a loud note,
# and --gpu-check (below) says whether the GPU nodes' driver can run it.
# Overrides: WOCR_TORCH_BACKEND=cu130 | cu129 (the flavour; a release has
# no other CUDA 12 wheel, e.g. +cu128 is a 404, and cu129 runs on 570-series
# drivers too), WOCR_VLLM_WHEEL=<url or file> (the CUDA 12 wheel, if its
# name differs; with it another cu12x flavour is accepted too). A different
# flavour needs a fresh env (rm -rf the env first): uv keeps the installed
# torch when only its CUDA build differs. If no wheel matches the
# node driver, use the official container instead: `apptainer pull
# docker://vllm/vllm-openai:<tag>` on a compute node (module load apptainer)
# and run serve_lib.sh's command through `apptainer exec --nv`.
#
# Login nodes have no GPU (cuda=False below is expected) and cap memory at
# 20 GB per user; if the vLLM install is killed there, run this script inside
# an interactive CPU session instead. Then verify on a GPU (debug QOS, free):
#     srun --account=ikoutis --qos=debug --partition=debug_gpu \
#          --gres=gpu:a100_10g:1 --time=00:10:00 bash -l tools/setup_env.sh --gpu-check
# --gpu-check runs nvidia-smi, says which flavour the node's driver needs,
# checks that torch sees the GPU, and loads vLLM's compiled ops (vLLM loads
# them lazily, so a wheel built for another CUDA would otherwise fail only
# inside the first job). If it reports a mismatch, reinstall as it says.
#
# The script activates the env only for itself. In your own shell, before
# tools/stage_models.py, pytest, or anything else that needs the env, run:
#     module load Miniforge3 && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate /project/ikoutis/conda_env/wocr
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."             # requirements.txt, src/ and logs/ are in the repo root

ENV_PREFIX="${WOCR_CONDA_ENV:-/project/ikoutis/conda_env/wocr}"
VLLM_SPEC="${WOCR_VLLM_SPEC:-vllm==0.30.0}"
ACTIVATE="module load Miniforge3 && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate $ENV_PREFIX"

# The GPU driver's version (e.g. 550.54.15), or nothing when none is visible.
driver_version() {
    local v
    v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1) || true
    [[ "$v" =~ ^[0-9]+\. ]] && echo "$v"
    return 0
}

# The vLLM build a driver runs: CUDA 13 needs >= 580.
flavour_for() {
    if [ "${1%%.*}" -ge 580 ]; then echo cu130; else echo cu129; fi
}

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "${1:-}" = --gpu-check ]; then
    set +u
    conda activate "$ENV_PREFIX"
    set -u
    export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # as ocr.sbatch
    nvidia-smi || echo "[!] nvidia-smi failed: no GPU or no driver on this node"
    DRIVER=$(driver_version)
    INSTALLED=$(cat "$ENV_PREFIX/wocr.flavour" 2>/dev/null || echo unknown)
    if [ -n "$DRIVER" ]; then
        echo "driver $DRIVER runs the $(flavour_for "$DRIVER") build; this env has: $INSTALLED"
        if [ "$INSTALLED" = cu130 ] && [ "$(flavour_for "$DRIVER")" != cu130 ]; then
            echo "[!] this driver cannot run the env's CUDA 13 build. Reinstall:"
            echo "    rm -rf $ENV_PREFIX && WOCR_TORCH_BACKEND=cu129 bash tools/setup_env.sh"
        fi
    fi
    python - <<'EOF'
import importlib
import importlib.util

import torch
import vllm

ok = torch.cuda.is_available()      # first: get_device_name() raises when CUDA is unusable
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | vllm {vllm.__version__} "
      f"| cuda={ok}" + (f" | {torch.cuda.get_device_name(0)}" if ok else ""))


def present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# vLLM loads its compiled ops lazily; load them now, so that a wheel built
# for another CUDA than torch's or the driver's fails here.
ops = [m for m in ("vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C",
                   "vllm._moe_C_stable_libtorch") if present(m)]
failed = []
for m in ops:
    try:
        importlib.import_module(m)
    except Exception as e:          # e.g. libcudart.so.13: cannot open shared object file
        failed.append(f"{m}: {e}")
if failed:
    print("vllm ops: FAILED to load: " + "; ".join(failed))
    print("[!] vLLM's compiled ops do not match torch or the driver: see the CUDA flavour "
          "notes at the top of tools/setup_env.sh")
else:
    print("vllm ops: " + (", ".join(ops) + " load" if ops else "none found"))
EOF
    # What a job actually runs is the `vllm` console script, not `python -c`:
    # it resolves C++ runtimes differently (see ocr.sbatch), so test it as is.
    CLI_ERR=$(mktemp)
    if vllm --help >/dev/null 2>"$CLI_ERR"; then
        echo "vllm cli: starts"
    else
        echo "vllm cli: FAILED to start:"; tail -n 3 "$CLI_ERR"
        echo "[!] every model server would fail the same way; see the C++ runtime"
        echo "    note in tools/setup_env.sh (libstdcxx-ng) and ocr.sbatch (LD_LIBRARY_PATH)"
    fi
    rm -f "$CLI_ERR"
    exit 0
fi

DRIVER=$(driver_version)
if [ -n "${WOCR_TORCH_BACKEND:-}" ]; then
    FLAVOUR=$WOCR_TORCH_BACKEND
    echo "[*] vLLM build: $FLAVOUR (WOCR_TORCH_BACKEND)"
elif [ -n "$DRIVER" ]; then
    FLAVOUR=$(flavour_for "$DRIVER")
    echo "[*] vLLM build: $FLAVOUR (for driver $DRIVER)"
else
    FLAVOUR=cu130
    echo "[!] ========================================================================"
    echo "[!] No GPU driver is visible here (a login node), so the vLLM build cannot be"
    echo "[!] matched to the GPU nodes' driver. Installing the CUDA 13 build ($VLLM_SPEC"
    echo "[!] from PyPI), which needs driver >= 580. Run --gpu-check on a GPU node (see"
    echo "[!] the end of this output): it says whether that driver can run it, and if"
    echo "[!] not, how to reinstall (WOCR_TORCH_BACKEND=cu129, in a fresh env)."
    echo "[!] ========================================================================"
fi
case "$FLAVOUR" in
    cu130) ;;
    cu129 | cu12[0-8])
        if [ "$FLAVOUR" != cu129 ] && [ -z "${WOCR_VLLM_WHEEL:-}" ]; then
            echo "ERROR: WOCR_TORCH_BACKEND=$FLAVOUR: vLLM releases publish their CUDA 12" \
                 "wheel as +cu129 only (it runs on drivers from R525 on): use cu129, or give" \
                 "a $FLAVOUR wheel in WOCR_VLLM_WHEEL" >&2
            exit 2
        fi
        VLLM_VERSION="${VLLM_SPEC#vllm==}"
        if [ -z "${WOCR_VLLM_WHEEL:-}" ] && [ "$VLLM_VERSION" = "$VLLM_SPEC" ]; then
            echo "ERROR: the $FLAVOUR build needs an exact pin, WOCR_VLLM_SPEC=vllm==<version>" \
                 "(or the wheel itself, WOCR_VLLM_WHEEL)" >&2
            exit 2
        fi ;;
    *)
        echo "ERROR: WOCR_TORCH_BACKEND=$FLAVOUR: expected cu130 or cu129" >&2
        exit 2 ;;
esac
if [ -f "$ENV_PREFIX/wocr.flavour" ] && [ "$(cat "$ENV_PREFIX/wocr.flavour")" != "$FLAVOUR" ]; then
    echo "ERROR: $ENV_PREFIX holds the $(cat "$ENV_PREFIX/wocr.flavour") build; the $FLAVOUR" \
         "build needs a fresh env: rm -rf $ENV_PREFIX, then run this again" >&2
    exit 2
fi

mkdir -p "$(dirname "$ENV_PREFIX")"
mkdir -p logs                       # sbatch's --output dir (tracked; a copy may lack it)

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
if [ "$FLAVOUR" = cu130 ]; then
    uv pip install --no-cache "$VLLM_SPEC"          # PyPI's vLLM and torch: CUDA 13
else
    # The release's CUDA 12 wheel, with torch from the matching PyTorch index
    # (uv prefers the extra index), as vLLM's install docs do it.
    uv pip install --no-cache "${WOCR_VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+$FLAVOUR-cp38-abi3-manylinux_2_28_$(uname -m).whl}" \
        --extra-index-url "https://download.pytorch.org/whl/$FLAVOUR"
fi
echo "$FLAVOUR" > "$ENV_PREFIX/wocr.flavour"
uv pip install --no-cache -r requirements.txt

# KaTeX for the formula validator (src/katex_check.py); optional — the
# pipeline skips the check if this step fails. nodejs brings a recent ICU,
# which needs a newer C++ runtime than the nodes' /lib64 has (RHEL 9:
# GCC 11): install the env's own libstdc++ with it, and ocr.sbatch puts it
# first on the loader path. Without that, `vllm` dies at start-up with
# "libstdc++.so.6: version CXXABI_1.3.15 not found" ([O-002]).
if conda install -y -q -c conda-forge nodejs "libstdcxx-ng>=14" "libgcc-ng>=14" >/dev/null; then
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
echo "[*] done ($FLAVOUR build); package versions saved to $ENV_PREFIX/wocr.lock.txt. Next:"
echo "    activate the env in your own shell (this script's activation ended with it):"
echo "      $ACTIVATE"
echo "      srun --account=ikoutis --qos=debug --partition=debug_gpu --gres=gpu:a100_10g:1 \\"
echo "           --time=00:10:00 bash -l tools/setup_env.sh --gpu-check   # GPU check (free)"
echo "      python tools/stage_models.py --profile default     # download weights (~65 GB)"
echo "      pytest tests/                                     # CPU-only, seconds"
echo "    then e.g.: INPUTS=<dir of PDFs> sbatch --array=0-3 slurm/ocr.sbatch"
if [ "$ENV_PREFIX" != "/project/ikoutis/conda_env/wocr" ]; then
    echo "    NOTE: non-default env path — export WOCR_CONDA_ENV=$ENV_PREFIX"
    echo "    in the shell you sbatch from."
fi
