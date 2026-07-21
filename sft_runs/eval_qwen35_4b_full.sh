#!/bin/bash
#SBATCH --job-name=eval_3cls_v6_v4meta_qwen35_full
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/agent-budget-control/sft_runs/logs/%x_%j.out
#SBATCH --error=/path/to/agent-budget-control/sft_runs/logs/%x_%j.err

#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=ALL

# Eval the Qwen3.5-4B FULL fine-tune checkpoint (3cls v6_v4meta backbone
# ablation, completes the 2x2 backbone/method grid) on the real 3cls holdout.

set -uxo pipefail

module purge
module load cuda/12.4.1

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "Activated conda env: $CONDA_DEFAULT_ENV"
which python

ROOT=/path/to/agent-budget-control
cd "$ROOT"

export HF_HOME=/path/to/hf_cache

BENCH_GLOB="datasets/benchmarks/holdout_3cls_*.json"

echo "===== EVAL: Qwen3.5-4B full fine-tune ====="
python data_analysis/scripts/eval_router_benchmark.py \
  --base_model "${ROOT}/sft_runs/outputs/router_arch_a_3cls_v6_v4meta_qwen35_4b_full/checkpoint-524" \
  --adapter_path "${ROOT}/sft_runs/outputs/router_arch_a_3cls_v6_v4meta_qwen35_4b_full/checkpoint-524" \
  --no_adapter \
  --bench_glob "${BENCH_GLOB}" \
  --out_dir "${ROOT}/datasets/eval_results/holdout_3cls_v6_v4meta_qwen35_4b_full" \
  --candidates 3cls

echo "===== DONE $(date -u) ====="
