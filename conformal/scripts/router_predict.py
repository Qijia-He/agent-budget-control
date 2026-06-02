"""
Run the LoRA router on a data split and dump per-example label probabilities.

Output format (jsonl, one example per line):
  {"idx": int, "true_label": str, "probs": {"reflect": p, "replan": p, "escalate": p}}

The probabilities are normalised over the three candidate labels only
(softmax over log P(label_tokens | input) for each candidate). This is what
CRC needs: a categorical distribution over the action space, conditioned on
the input.
"""
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


CANDIDATES = ["reflect", "replan", "escalate"]


def build_messages(ex):
    return [
        {"role": "system", "content": ex["instruction"]},
        {"role": "user", "content": ex["input"]},
    ]


def score(model, tokenizer, ex, device, label_token_ids_cache):
    """Return {label: prob} for an example, normalised across candidates."""
    messages = build_messages(ex)
    # `add_generation_prompt=True` appends "<|im_start|>assistant\n" so the next
    # token to predict is the first token of the label.
    # IMPORTANT: pass enable_thinking=False — Qwen3.5's chat_template.jinja inserts
    # a `<think>...</think>` block by default, but the router was SFT'd via LF's
    # qwen3_nothink template (no <think> block). Without this kwarg, the model
    # falls into chain-of-thought mode and never emits the label.
    prefix_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    log_p_per_label = {}
    for label in CANDIDATES:
        label_ids = label_token_ids_cache[label].to(device)  # [1, K]
        full = torch.cat([prefix_ids, label_ids], dim=1)
        L = prefix_ids.size(1)
        K = label_ids.size(1)
        with torch.no_grad():
            logits = model(full).logits  # [1, L+K, V]
        # logit at position (L-1) predicts token at L (first label token)
        # logit at position (L+K-2) predicts token at L+K-1 (last label token)
        slice_logits = logits[0, L - 1:L + K - 1, :]  # [K, V]
        log_probs = F.log_softmax(slice_logits, dim=-1)  # [K, V]
        target = label_ids[0]  # [K]
        # length-normalised: mean log-prob per token, so labels with different
        # token counts are comparable (otherwise single-token "reflect" always
        # beats multi-token "replan"/"escalate" by sheer accumulated -log P).
        log_p_seq = log_probs.gather(1, target.unsqueeze(1)).sum().item() / K
        log_p_per_label[label] = log_p_seq

    # softmax across candidate log-probs -> probability over the 3-action set
    vals = torch.tensor([log_p_per_label[l] for l in CANDIDATES])
    probs = F.softmax(vals, dim=0).tolist()
    return {l: p for l, p in zip(CANDIDATES, probs)}, log_p_per_label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to split jsonl (calib.jsonl / test.jsonl)")
    p.add_argument("--output", required=True, help="path to write predictions jsonl")
    p.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter_path", default="/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200-renamed",
                   help="use the -renamed dir (key paths stripped of 'language_model.') so PEFT can load weights correctly")
    p.add_argument("--no_adapter", action="store_true",
                   help="skip LoRA adapter; use base Qwen3.5-4B with the same prompt. For ablation.")
    p.add_argument("--max_examples", type=int, default=-1, help="cap for debugging; -1 = all")
    p.add_argument("--dtype", default="bf16")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"device={device} dtype={dtype}")

    # ---- load tokenizer + model (base + LoRA adapter) ----
    # We use the tokenizer that lives in the checkpoint dir (includes the chat
    # template LLaMA-Factory shipped with the trained model).
    tokenizer_src = args.adapter_path if not args.no_adapter else args.base_model
    print(f"loading tokenizer from {tokenizer_src} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    print(f"loading base model {args.base_model}...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    print(f"  base model loaded in {time.time()-t0:.1f}s")
    if args.no_adapter:
        print("  [no_adapter=True] skipping LoRA — using base model directly for ablation")
        model = base
        model.eval()
    else:
        print(f"applying LoRA adapter from {args.adapter_path}...")
        t0 = time.time()
        model = PeftModel.from_pretrained(base, args.adapter_path)
        model.eval()
        print(f"  adapter loaded in {time.time()-t0:.1f}s")
        # ---- debug: report how many LoRA adapters were actually applied ----
        n_lora_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0)
        print(f"  LoRA modules attached: {n_lora_modules}")
        if n_lora_modules == 0:
            print("  [WARN] no LoRA modules attached — adapter weights are not being used!")

    # ---- precompute label token IDs ----
    # We tokenize each label *without* leading special tokens. The chat template
    # ends with "...<|im_start|>assistant\n" so the next token should be the
    # bare label string (no leading space in Qwen3 tokenizer).
    label_token_ids = {}
    print("label tokenizations:")
    for label in CANDIDATES:
        ids = tokenizer(label, return_tensors="pt", add_special_tokens=False).input_ids
        tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
        print(f"  {label!r:12s} -> ids={ids[0].tolist()}  tokens={tokens}")
        label_token_ids[label] = ids

    # ---- load split ----
    examples = []
    with open(args.input) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"scoring {len(examples)} examples from {args.input}")

    # ---- score loop ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct_argmax = 0
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, ex in enumerate(examples):
            probs, log_ps = score(model, tokenizer, ex, device, label_token_ids)
            argmax = max(probs, key=probs.get)
            if argmax == ex["output"]:
                correct_argmax += 1
            rec = {
                "idx": i,
                "true_label": ex["output"],
                "probs": probs,
                "log_p_per_label": log_ps,
                "argmax": argmax,
            }
            fout.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0 or (i + 1) == len(examples):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(examples) - i - 1) / rate
                acc = correct_argmax / (i + 1)
                print(f"  [{i+1}/{len(examples)}] argmax_acc={acc:.3f} rate={rate:.2f}/s eta={eta:.0f}s")

    acc = correct_argmax / len(examples)
    print(f"done. argmax_acc={acc:.4f} ({correct_argmax}/{len(examples)})")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
