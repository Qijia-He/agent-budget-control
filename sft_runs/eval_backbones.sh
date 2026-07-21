#!/bin/bash
#SBATCH --job-name=eval_3cls_v6_v4meta_backbones
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

# Eval the 3-way backbone/finetuning-method ablation on the BEST 3cls data
# recipe (v6_v4meta) on the real holdout: Qwen3-4B LoRA, Qwen3-8B LoRA,
# Qwen3-4B full fine-tune. Compare against the original Qwen3.5-4B LoRA
# baseline for this recipe (holdout test solve_rate=0.697, see SUMMARY.md).

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

echo "===== EVAL: Qwen3-4B LoRA ====="
python data_analysis/scripts/eval_router_benchmark.py \
  --adapter_path "${ROOT}/sft_runs/outputs/router_arch_a_3cls_v6_v4meta_qwen3_4b_lora/checkpoint-1050" \
  --base_model "Qwen/Qwen3-4B" \
  --bench_glob "${BENCH_GLOB}" \
  --out_dir "${ROOT}/datasets/eval_results/holdout_3cls_v6_v4meta_qwen3_4b_lora" \
  --candidates 3cls

echo "===== EVAL: Qwen3-8B LoRA ====="
python data_analysis/scripts/eval_router_benchmark.py \
  --adapter_path "${ROOT}/sft_runs/outputs/router_arch_a_3cls_v6_v4meta_qwen3_8b_lora/checkpoint-1050" \
  --base_model "Qwen/Qwen3-8B" \
  --bench_glob "${BENCH_GLOB}" \
  --out_dir "${ROOT}/datasets/eval_results/holdout_3cls_v6_v4meta_qwen3_8b_lora" \
  --candidates 3cls

echo "===== DONE $(date -u) ====="
