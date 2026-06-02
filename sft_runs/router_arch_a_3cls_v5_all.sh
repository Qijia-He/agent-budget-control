#!/bin/bash
set -uxo pipefail
# Run all 4 v5 experiments sequentially, then fix LoRA key bug for each.

WORKSPACE=/mnt/bn/ecom-govern-models/qijiahe
LF_DIR=$WORKSPACE/LLaMA-Factory

echo "===== SETUP ====="
source "$WORKSPACE/setup_env.sh"
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

fix_keys() {
    local out_dir="$1"
    local fixed_dir="${out_dir}_fixed"
    echo "[fix_keys] checking $out_dir ..."
    python - << PYEOF
import os, glob, shutil
from safetensors import safe_open
from safetensors.torch import save_file

out_dir = "$out_dir"
fixed_dir = "$fixed_dir"

bug_found = False
for ckpt in sorted(glob.glob(os.path.join(out_dir, 'checkpoint-*'))):
    st = os.path.join(ckpt, 'adapter_model.safetensors')
    if not os.path.exists(st): continue
    with safe_open(st, framework='pt') as f:
        first_key = list(f.keys())[0]
    if 'language_model' not in first_key:
        continue
    bug_found = True
    os.makedirs(os.path.join(fixed_dir, os.path.basename(ckpt)), exist_ok=True)
    for fn in os.listdir(ckpt):
        if fn != 'adapter_model.safetensors':
            shutil.copy2(os.path.join(ckpt, fn), os.path.join(fixed_dir, os.path.basename(ckpt), fn))
    tensors = {}
    with safe_open(st, framework='pt') as f:
        for key in f.keys():
            tensors[key.replace('model.language_model.', 'model.')] = f.get_tensor(key)
    save_file(tensors, os.path.join(fixed_dir, os.path.basename(ckpt), 'adapter_model.safetensors'))

if bug_found:
    print(f"[fix_keys] fixed -> {fixed_dir}")
else:
    print("[fix_keys] no bug found, keys OK")
PYEOF
}

for variant in v5b_clean v5b_full v5a_clean v5a_full; do
    YAML=$WORKSPACE/sft_runs/router_arch_a_3cls_${variant}.yaml
    echo ""
    echo "===== TRAINING: $variant ====="
    mkdir -p $WORKSPACE/sft_runs/outputs/router_arch_a_3cls_${variant}
    FORCE_TORCHRUN=1 llamafactory-cli train ${YAML}
    echo "===== DONE: $variant ====="
    fix_keys "$WORKSPACE/sft_runs/outputs/router_arch_a_3cls_${variant}"
done

echo "===== ALL DONE ====="
