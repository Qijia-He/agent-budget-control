#!/bin/bash
set -euo pipefail

# Arch A 3cls — post-proceed router, drops verdict=pass rows
# Dataset: router_no_reason_v1_3cls (1583 examples, labels: reflect/replan/escalate)

# activate env (idempotent: installs miniconda + env on first run, no-ops after)
source /mnt/bn/ecom-govern-models/qijiahe/setup_env.sh

# wandb — project/entity here; API key from `wandb login` (~/.netrc)
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

YAML=/mnt/bn/ecom-govern-models/qijiahe/sft_runs/router_arch_a_3cls.yaml

# multi-GPU? set FORCE_TORCHRUN=1 before invoking llamafactory-cli, e.g.:
#   FORCE_TORCHRUN=1 bash router_arch_a_3cls.sh
llamafactory-cli train ${YAML}
