#!/bin/bash
# Idempotent setup for the llamafactory conda env.
# Safe to source on every devbox/worker boot — already-installed steps no-op.
# Usage:
#   source /mnt/bn/ecom-govern-models/qijiahe/setup_env.sh
# After sourcing, the `llamafactory` conda env is active.

set -e

QJ_ROOT=/mnt/bn/ecom-govern-models/qijiahe
CONDA_DIR=/home/tiger/miniconda3
ENV_NAME=llamafactory
PY_VERSION=3.11

# 1. miniconda (local disk — BF-fuse breaks conda's parallel extractor)
if [ ! -d "${CONDA_DIR}" ]; then
  echo "[setup_env] installing miniconda -> ${CONDA_DIR}"
  bash "${QJ_ROOT}/installers/Miniconda3-latest-Linux-x86_64.sh" -b -p "${CONDA_DIR}"
fi
# shellcheck disable=SC1091
source "${CONDA_DIR}/bin/activate"

# 2. accept Anaconda default-channel ToS (required by conda 24.x+; no-op if already accepted)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# 3. llamafactory env
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup_env] creating env ${ENV_NAME} (python ${PY_VERSION})"
  conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
fi
conda activate "${ENV_NAME}"

# 4. llamafactory (editable — source on BF-fuse, code edits don't need reinstall)
if ! python -c "import llamafactory" 2>/dev/null; then
  echo "[setup_env] pip install -e LLaMA-Factory"
  pip install -e "${QJ_ROOT}/LLaMA-Factory"
fi

# 5. torch must match driver. Driver 535/CUDA 12.2 → need torch+cu124 (cu126/cu128/cu130 builds
# fail with "driver too old"). LLaMA-Factory pins only torch>=2.4 so pip will pull cu130 wheels
# by default; we force-downgrade if cuda runtime isn't usable.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[setup_env] reinstalling torch with cu124 build (driver-compatible)"
  pip install --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
  # pytorch index ships newer fsspec/pillow that violate datasets/gradio pins — pin them back
  pip install --force-reinstall --no-deps fsspec==2025.3.0 pillow==11.3.0
fi

# 6. wandb
if ! python -c "import wandb" 2>/dev/null; then
  echo "[setup_env] pip install wandb"
  pip install wandb
fi

# 6b. deepspeed (required for full FT with ZeRO sharding; LLaMA-Factory doesn't pull it in)
if ! python -c "import deepspeed" 2>/dev/null; then
  echo "[setup_env] pip install deepspeed"
  pip install deepspeed
fi

# 7. wandb api key — load from local-disk secret file because byted infra periodically
# rewrites ~/.netrc and strips the wandb entry. After devbox redeploy this file is gone;
# recreate it with: echo "$WANDB_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
WANDB_KEY_FILE="${HOME}/.wandb_api_key"
if [ -s "${WANDB_KEY_FILE}" ]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
fi

echo "[setup_env] ready — python=$(python -V 2>&1) env=${CONDA_DEFAULT_ENV} wandb=$([ -n "${WANDB_API_KEY:-}" ] && echo configured || echo MISSING)"
