#!/bin/bash
set -euxo pipefail

# Arch A 4cls — post-proceed router, keeps unsolvable label
# Dataset: router_no_reason_v1_4cls (4813 examples, labels: reflect/replan/escalate/unsolvable)

# activate env (idempotent: installs miniconda + env on first run, no-ops after)
source /mnt/bn/ecom-govern-models/qijiahe/setup_env.sh

# wandb — project/entity here; API key from `wandb login` (~/.netrc)
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

# reduce CUDA memory fragmentation (helps bs>1 with variable seq lengths)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YAML=/mnt/bn/ecom-govern-models/qijiahe/sft_runs/router_arch_a_4cls.yaml

# multi-GPU? set FORCE_TORCHRUN=1 before invoking llamafactory-cli, e.g.:
#   FORCE_TORCHRUN=1 bash router_arch_a_4cls.sh
llamafactory-cli train ${YAML}
