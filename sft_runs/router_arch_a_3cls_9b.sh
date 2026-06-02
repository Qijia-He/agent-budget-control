#!/bin/bash
set -euo pipefail

# Arch A 3cls — Qwen3.5-9B LoRA variant (vs baseline 4B LoRA)
# Dataset: router_no_reason_v1_3cls (1583 examples, labels: reflect/replan/escalate)

# activate env (idempotent: installs miniconda + env on first run, no-ops after)
source /mnt/bn/ecom-govern-models/qijiahe/setup_env.sh

# wandb — project/entity here; API key from ~/.wandb_api_key (auto-exported by setup_env.sh)
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

YAML=/mnt/bn/ecom-govern-models/qijiahe/sft_runs/router_arch_a_3cls_9b.yaml

# multi-GPU? set FORCE_TORCHRUN=1 before invoking llamafactory-cli, e.g.:
#   FORCE_TORCHRUN=1 bash router_arch_a_3cls_9b.sh
llamafactory-cli train ${YAML}
