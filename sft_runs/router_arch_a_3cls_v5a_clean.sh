#!/bin/bash
set -uxo pipefail

# Arch A 3cls v2 — post-proceed router, 5498 examples, 2-GPU run
# Dataset: router_no_reason_v1_3cls (labels: reflect/replan/escalate)
# Output: sft_runs/outputs/router_arch_a_3cls_v2/

echo "===== ENV $(date -u) ====="
hostname
nvidia-smi -L || echo "[warn] nvidia-smi failed"

WORKSPACE=/mnt/bn/ecom-govern-models/qijiahe
LF_DIR=$WORKSPACE/LLaMA-Factory
YAML=$WORKSPACE/sft_runs/router_arch_a_3cls_v5a_clean.yaml

echo "===== SETUP llamafactory env ====="
source "$WORKSPACE/setup_env.sh"
SETUP_RC=$?
echo "[setup rc] $SETUP_RC"

which python && python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" \
  || echo "[warn] torch import failed"

if [ ! -d "$LF_DIR/.git" ]; then
  echo "[warn] LLaMA-Factory source missing at $LF_DIR — cloning"
  git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
  pip install -e "$LF_DIR"
fi

echo "===== SMOKE TEST ====="
python -c "import llamafactory; print('llamafactory:', llamafactory.__file__)" \
  || echo "[err] import llamafactory failed"
llamafactory-cli version 2>&1 | head -5 || echo "[err] cli version failed"

echo "===== WANDB ====="
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

echo "===== TRAIN (2 GPU torchrun) ====="
FORCE_TORCHRUN=1 llamafactory-cli train ${YAML}

echo "===== DONE $(date -u) ====="
