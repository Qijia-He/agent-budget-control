"""Inspect what modules the base Qwen3.5-4B exposes and whether PEFT can find
the targets the adapter was trained for.
"""
import json
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from safetensors import safe_open

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = "/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200"

# load base
print(f"loading {BASE}...")
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

# inspect first 2 layers' module structure
print("\n=== layer 0 submodule names ===")
for name, mod in m.model.layers[0].named_modules():
    cls = type(mod).__name__
    if cls not in {"Identity", "Dropout"} and name and "." not in name:  # top-level only
        print(f"  layers[0].{name}: {cls}")
print("\n=== layer 0 LINEAR submodules (full paths) ===")
for name, mod in m.model.layers[0].named_modules():
    if hasattr(mod, "weight") and len(mod.weight.shape) == 2:
        print(f"  layers[0].{name}: {type(mod).__name__} weight={tuple(mod.weight.shape)}")

print("\n=== layer 31 LINEAR submodules ===")
for name, mod in m.model.layers[31].named_modules():
    if hasattr(mod, "weight") and len(mod.weight.shape) == 2:
        print(f"  layers[31].{name}: {type(mod).__name__} weight={tuple(mod.weight.shape)}")

# load adapter weights and list their keys
print("\n=== adapter target module names (from safetensors) ===")
adapter_keys = set()
with safe_open(f"{ADAPTER}/adapter_model.safetensors", framework="pt") as f:
    for k in f.keys():
        # canonicalise: strip lora_A.default.weight / lora_B.default.weight
        if "lora_A" in k:
            base_path = k.replace(".lora_A.default.weight", "")
            adapter_keys.add(base_path)

# count by layer-position keyword
from collections import Counter
sub_kinds = Counter()
for k in adapter_keys:
    parts = k.split(".")
    if "linear_attn" in parts:
        sub_kinds["linear_attn." + parts[parts.index("linear_attn") + 1]] += 1
    elif "self_attn" in parts:
        sub_kinds["self_attn." + parts[parts.index("self_attn") + 1]] += 1
    elif "mlp" in parts:
        sub_kinds["mlp." + parts[parts.index("mlp") + 1]] += 1
    else:
        sub_kinds["other"] += 1
print("adapter targets (count of LoRA pairs per target kind):")
for k, v in sub_kinds.most_common():
    print(f"  {k}: {v}")
print(f"total adapter target modules: {len(adapter_keys)}")

# now apply PEFT and check which got attached
print("\n=== apply PEFT adapter to base ===")
peft_model = PeftModel.from_pretrained(m, ADAPTER)
attached = [name for name, mod in peft_model.named_modules() if hasattr(mod, "lora_A") and len(mod.lora_A) > 0]
print(f"PEFT-attached modules: {len(attached)}")
attached_kinds = Counter()
for n in attached:
    parts = n.split(".")
    if "linear_attn" in parts:
        attached_kinds["linear_attn." + parts[parts.index("linear_attn") + 1]] += 1
    elif "self_attn" in parts:
        attached_kinds["self_attn." + parts[parts.index("self_attn") + 1]] += 1
    elif "mlp" in parts:
        attached_kinds["mlp." + parts[parts.index("mlp") + 1]] += 1
    else:
        attached_kinds["other"] += 1
print("attached LoRA targets:")
for k, v in attached_kinds.most_common():
    print(f"  {k}: {v}")

print("\n=== adapter vs attached mismatches ===")
for kind in sub_kinds:
    diff = sub_kinds[kind] - attached_kinds.get(kind, 0)
    if diff != 0:
        print(f"  {kind}: adapter has {sub_kinds[kind]}, attached {attached_kinds.get(kind, 0)} -> MISSING {diff}")
