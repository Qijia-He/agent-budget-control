#!/bin/bash
set -euo pipefail

# Arch A 4cls — Qwen3.5-4B FULL fine-tune (no LoRA, ZeRO-3 sharded)
# Dataset: router_no_reason_v1_4cls (4813 examples, labels: reflect/replan/escalate/unsolvable)
# lr 1e-5 (full FT needs ~10x lower lr than LoRA to avoid catastrophic forgetting)

# activate env (idempotent: installs miniconda + env on first run, no-ops after)
source /mnt/bn/ecom-govern-models/qijiahe/setup_env.sh

# wandb — project/entity here; API key from ~/.wandb_api_key (auto-exported by setup_env.sh)
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

YAML=/mnt/bn/ecom-govern-models/qijiahe/sft_runs/router_arch_a_4cls_full.yaml

# multi-GPU? set FORCE_TORCHRUN=1 before invoking llamafactory-cli, e.g.:
#   FORCE_TORCHRUN=1 bash router_arch_a_4cls_full.sh
llamafactory-cli train ${YAML}
