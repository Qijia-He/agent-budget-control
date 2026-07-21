#!/bin/bash
#SBATCH --job-name=eval_gemini25fp
#SBATCH --account=gpu-a100-cse
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/agent-budget-control/sft_runs/logs/%x_%j.out
#SBATCH --error=/path/to/agent-budget-control/sft_runs/logs/%x_%j.err
#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=ALL

set -euo pipefail
module purge
module load cuda/12.4.1
source /sw/contrib/foster-src/python/miniconda/3.8/etc/profile.d/conda.sh
conda activate web-agent-llama
export LD_LIBRARY_PATH=/path/to/conda_env/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_OFFLINE=1

REPO=/path/to/agent-budget-control
CKPT=${REPO}/sft_runs/outputs/gemini_25_flash_pro_3cls_qwen35_4b_full
BENCH_DIR=${REPO}/datasets/gemini_25_flash_pro
EVAL_DIR=${REPO}/datasets/eval_results/gemini_25_fp_3cls_qwen35_4b_full
CRC_DIR=${REPO}/datasets/eval_results
COSTS=${BENCH_DIR}/gemini_25_hf_action_costs_usd.json
mkdir -p ${EVAL_DIR}

echo "=== Step 1: Router inference (227 calib + 229 test) ==="
python ${REPO}/data_analysis/scripts/eval_router_benchmark.py \
    --no_adapter \
    --adapter_path ${CKPT} \
    --base_model ${CKPT} \
    --bench_glob "${BENCH_DIR}/gemini_25_holdout_3cls_*_for_eval.json" \
    --out_dir ${EVAL_DIR} \
    --dtype bf16

echo ""
echo "=== Step 2: Attach USD costs ==="
for split in calib test; do
    python ${REPO}/conformal/scripts/attach_usd_costs.py \
        --eval_json  ${EVAL_DIR}/gemini_25_holdout_3cls_${split}_for_eval_eval.json \
        --bench_json ${BENCH_DIR}/gemini_25_holdout_3cls_${split}_for_eval.json \
        --costs_json ${COSTS} \
        --out        ${EVAL_DIR}/gemini_25_holdout_3cls_${split}_eval_usd.json
done

echo ""
echo "=== Step 3: Augment SA (add escalate to solvable examples) ==="
python3 - <<'PYEOF'
import json; from pathlib import Path
EVAL_DIR = Path("/path/to/agent-budget-control/datasets/eval_results/gemini_25_fp_3cls_qwen35_4b_full")
n_aug = 0
for split in ["calib", "test"]:
    path = EVAL_DIR / f"gemini_25_holdout_3cls_{split}_eval_usd.json"
    d = json.load(open(path))
    for ex in d["examples"]:
        sa = ex.get("successful_actions") or []
        if sa and "escalate" not in sa:
            ex["successful_actions"] = sa + ["escalate"]; n_aug += 1
    json.dump(d, open(path, "w"), indent=2)
    print(f"{split}: n={len(d['examples'])}")
print(f"Augmented {n_aug} with escalate SA")
PYEOF

echo ""
echo "=== Step 4: CRC evaluation ==="
python ${REPO}/conformal/scripts/crc_on_holdout_usd.py \
    --calib_eval ${EVAL_DIR}/gemini_25_holdout_3cls_calib_eval_usd.json \
    --test_eval  ${EVAL_DIR}/gemini_25_holdout_3cls_test_eval_usd.json \
    --out        ${CRC_DIR}/crc_gemini_25_fp_3cls_qwen35_4b_full_usd.json \
    --budgets    0.005,0.010,0.015,0.020,0.025,0.030,0.040,0.050,0.060,0.075,0.090,0.120,0.150,0.200,0.250,0.300

echo "=== Done ==="
