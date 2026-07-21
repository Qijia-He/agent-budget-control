#!/bin/bash
#SBATCH --job-name=gemini25_router_3cls_full
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

# Gemini 2.5 Flash→Pro router, 3cls, Qwen3.5-4B full fine-tune
# Dataset: gemini_25_flash_pro_3cls (n=683, Flash→Pro only)

module purge
module load cuda/12.4.1

source /sw/contrib/foster-src/python/miniconda/3.8/etc/profile.d/conda.sh
conda activate /path/to/conda_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/path/to/hf_cache
export WANDB_DIR=/path/to/agent-budget-control/sft_runs/wandb_logs
export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT="router-sft"
export WANDB_ENTITY="your-wandb-entity"

LF_DIR=/path/to/LLaMA-Factory
cd "$LF_DIR"

FORCE_TORCHRUN=1 llamafactory-cli train \
  /path/to/agent-budget-control/sft_runs/gemini_qwen35_4b_full.yaml
