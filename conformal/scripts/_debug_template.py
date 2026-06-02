"""Compare what LLaMA-Factory's qwen3_nothink template produces vs the
tokenizer's apply_chat_template output. If they differ, the LoRA was trained on
a different prompt prefix than what we're feeding at inference, which explains
why the model falls back to base behavior.
"""
import json
import sys
sys.path.insert(0, "/mnt/bn/ecom-govern-models/qijiahe/LLaMA-Factory/src")

from transformers import AutoTokenizer
from llamafactory.data.template import TEMPLATES

ADAPTER = "/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200"
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)

with open("/mnt/bn/ecom-govern-models/qijiahe/conformal/data/calib.jsonl") as f:
    ex = json.loads(f.readline())

messages_train_format = [
    {"role": "user", "content": ex["input"]},
    {"role": "assistant", "content": ex["output"]},
]

print("=" * 80)
print("=== LLaMA-Factory qwen3_nothink template output (the way training saw it) ===")
print("=" * 80)
tmpl = TEMPLATES["qwen3_nothink"]
# encode_oneturn returns (prompt_ids, response_ids)
try:
    # In LF, the API is `template.encode_supervised(...)`. Check what's available.
    # The Template object has `format_user`, `format_assistant`, `format_system`, `default_system`.
    sys_msg = ex["instruction"]
    user_msg = ex["input"]
    asst_msg = ex["output"]
    # Build the parts
    sys_part = tmpl.format_system.apply(content=sys_msg)
    user_part = tmpl.format_user.apply(content=user_msg, idx=0)
    asst_part = tmpl.format_assistant.apply(content=asst_msg)
    # concatenate as the training collator would
    print("SYS:", sys_part)
    print()
    print("USER:", user_part)
    print()
    print("ASST:", asst_part)
    print()
except Exception as e:
    print(f"could not call LF formatter via direct method: {e}")
    print(f"tmpl attrs: {dir(tmpl)}")

print("=" * 80)
print("=== tokenizer.apply_chat_template output (inference path) ===")
print("=" * 80)
messages_full = [
    {"role": "system", "content": ex["instruction"]},
    {"role": "user", "content": ex["input"]},
    {"role": "assistant", "content": ex["output"]},
]
out = tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, tokenize=False)
print(out[:1500])
print("...")
print(out[-500:])

print("=" * 80)
print("=== apply_chat_template with add_generation_prompt=True (what router_predict uses) ===")
print("=" * 80)
messages_q = [
    {"role": "system", "content": ex["instruction"]},
    {"role": "user", "content": ex["input"]},
]
out2 = tokenizer.apply_chat_template(messages_q, add_generation_prompt=True, tokenize=False)
print(out2[-500:])  # just the tail
