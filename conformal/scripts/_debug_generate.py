"""Quick sanity check: load model + adapter, call generate() on a few calib
examples, see what the model *actually* outputs greedily.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER = "/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200"
BASE = "Qwen/Qwen3.5-4B"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

with open("/mnt/bn/ecom-govern-models/qijiahe/conformal/data/calib.jsonl") as f:
    examples = [json.loads(l) for l in f]

print("greedy generation on first 10 calib examples:")
for i, ex in enumerate(examples[:10]):
    messages = [
        {"role": "system", "content": ex["instruction"]},
        {"role": "user", "content": ex["input"]},
    ]
    prefix_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    input_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    gen_ids = out[0, input_ids.size(1):].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    sign = "✓" if ex["output"] in gen_text else "✗"
    print(f"  [{i}] {sign} true={ex['output']:10s} gen={gen_text!r:30s} gen_ids={gen_ids}")
