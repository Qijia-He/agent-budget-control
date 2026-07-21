#!/bin/bash
#SBATCH --job-name=router_3cls_v6_v4meta_qwen35_4b_full
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/agent-budget-control/sft_runs/logs/%x_%j.out
#SBATCH --error=/path/to/agent-budget-control/sft_runs/logs/%x_%j.err

#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=ALL

# Backbone/finetuning-method ablation on the BEST 3cls data recipe (v6_v4meta,
# holdout solve_rate=0.697 on the original Qwen3.5-4B LoRA run -- see SUMMARY.md).
# This run: Qwen3.5-4B FULL fine-tune -- completes the 2x2 grid alongside
# router_arch_a_3cls_v6_v4meta_qwen3_4b_full.sh (Qwen3-4B full, test cls_acc=0.564/
# solve_rate=0.644) and the two LoRA runs (Qwen3-4B/-8B).
# Dataset: router_arch_a_3cls_v6_v4meta (n=4656, labels: reflect/replan/escalate)
# Holdout: datasets/benchmarks/holdout_3cls_{calib,test}.json (NOT in this dataset)
#
# Uses DeepSpeed ZeRO-2 (examples/deepspeed/ds_z2_config.json) -- deepspeed
# already installed in the conda env from the earlier full-FT runs.

module purge
module load cuda/12.4.1

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "Activated conda env: $CONDA_DEFAULT_ENV"
which python
which llamafactory-cli

LF_DIR=/path/to/LLaMA-Factory
cd "$LF_DIR"

export HF_HOME=/path/to/hf_cache
export WANDB_DIR=/path/to/agent-budget-control/sft_runs/wandb_logs
export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT="router-sft"
export WANDB_ENTITY="your-wandb-entity"

FORCE_TORCHRUN=1 llamafactory-cli train /path/to/agent-budget-control/sft_runs/qwen35_4b_full.yaml
