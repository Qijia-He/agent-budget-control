#!/bin/bash
set -uxo pipefail

# Arch A 3cls v4 — post-proceed router, 4656 examples (v3_train after holdout split)
# Dataset: router_arch_a_3cls_v3_train (deduped + holdout removed)
# Output: sft_runs/outputs/router_arch_a_3cls_v4/
# Note: after training, run fix_lora_keys.py to correct language_model path bug if present

echo "===== ENV $(date -u) ====="
hostname
nvidia-smi -L || echo "[warn] nvidia-smi failed"

WORKSPACE=/mnt/bn/ecom-govern-models/qijiahe
LF_DIR=$WORKSPACE/LLaMA-Factory
YAML=$WORKSPACE/sft_runs/router_arch_a_3cls_v4.yaml
OUTPUT=$WORKSPACE/sft_runs/outputs/router_arch_a_3cls_v4

echo "===== SETUP llamafactory env ====="
source "$WORKSPACE/setup_env.sh"
SETUP_RC=$?
echo "[setup rc] $SETUP_RC"

which python && python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" \
  || echo "[warn] torch import failed"

if [ ! -d "$LF_DIR/.git" ]; then
  echo "[warn] LLaMA-Factory source missing at $LF_DIR — cloning"
  git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
  pip install -e "$LF_DIR"
fi

echo "===== SMOKE TEST ====="
python -c "import llamafactory; print('llamafactory:', llamafactory.__file__)" \
  || echo "[err] import llamafactory failed"
llamafactory-cli version 2>&1 | head -5 || echo "[err] cli version failed"

echo "===== WANDB ====="
export WANDB_ENTITY=heqj3-university-of-washington
export WANDB_PROJECT=router-sft

echo "===== TRAIN (2 GPU torchrun) ====="
FORCE_TORCHRUN=1 llamafactory-cli train ${YAML}

echo "===== POST-TRAINING: detect & fix LoRA key path bug ====="
python - << 'PYEOF'
import os, glob, json
from safetensors import safe_open
from safetensors.torch import save_file

output_dir = os.environ.get('OUTPUT', '/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls_v4')
fixed_dir  = output_dir + '_fixed'

bug_found = False
for ckpt in sorted(glob.glob(os.path.join(output_dir, 'checkpoint-*'))):
    st = os.path.join(ckpt, 'adapter_model.safetensors')
    if not os.path.exists(st):
        continue
    with safe_open(st, framework='pt') as f:
        first_key = list(f.keys())[0]
    if 'language_model' not in first_key:
        print(f'[ok] {os.path.basename(ckpt)}: keys look correct')
        continue
    bug_found = True
    print(f'[fix] {os.path.basename(ckpt)}: language_model path bug detected, fixing...')
    import shutil
    os.makedirs(os.path.join(fixed_dir, os.path.basename(ckpt)), exist_ok=True)
    for f2 in os.listdir(ckpt):
        if f2 != 'adapter_model.safetensors':
            shutil.copy2(os.path.join(ckpt, f2), os.path.join(fixed_dir, os.path.basename(ckpt), f2))
    tensors = {}
    with safe_open(st, framework='pt') as f:
        for key in f.keys():
            tensors[key.replace('model.language_model.', 'model.')] = f.get_tensor(key)
    save_file(tensors, os.path.join(fixed_dir, os.path.basename(ckpt), 'adapter_model.safetensors'))

if bug_found:
    print(f'[fix] Fixed checkpoints saved to {fixed_dir}')
else:
    print('[ok] No path bug found — checkpoints are ready to use as-is')

# Verify: check first-token logit diff between base and best ckpt
import glob as g2
ckpts = sorted(g2.glob(os.path.join(output_dir if not bug_found else fixed_dir, 'checkpoint-*')))
if ckpts:
    best = ckpts[0]  # just verify any checkpoint
    print(f'[verify] Checking adapter effect on {os.path.basename(best)}...')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')
    base  = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', torch_dtype=torch.float16, device_map='cuda')
    model = PeftModel.from_pretrained(base, best)
    model.eval()
    sample = '{"instruction": "You are a router.", "input": "Problem: test"}'
    ids = tok('test prompt', return_tensors='pt').input_ids.to('cuda')
    with torch.no_grad():
        bl = base(ids).logits[0, -1]
        al = model(ids).logits[0, -1]
    diff = (al - bl).abs().max().item()
    print(f'[verify] max logit diff base vs adapted: {diff:.4f}')
    if diff < 0.01:
        print('[WARN] adapter has near-zero effect — path bug may still be present')
    else:
        print('[ok] adapter is working correctly')
PYEOF

echo "===== DONE $(date -u) ====="
