"""Evaluate a LoRA router checkpoint on benchmark JSON files.

Computes:
  1. cls_acc   — argmax prediction == GT label
  2. solve_rate — argmax prediction in successful_actions (only for examples
                  where successful_actions is not None)

Usage:
    python scripts/eval_router_benchmark.py \\
        --adapter_path /mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls_v2/checkpoint-1000 \\
        --bench_glob "/mnt/bn/ecom-govern-models/qijiahe/datasets/benchmarks/*3cls*" \\
        --out_dir /mnt/bn/ecom-govern-models/qijiahe/conformal/results/bench_eval_3cls_v2
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CANDIDATES_3CLS = ["reflect", "replan", "escalate"]
CANDIDATES_4CLS = ["reflect", "replan", "escalate", "unsolvable"]
COST = {"proceed": 1, "reflect": 2, "replan": 2, "escalate": 13, "unsolvable": 0}


def build_prefix(ex, tokenizer):
    # Match qwen3_nothink training template exactly:
    # <|im_start|>system\n{instruction}<|im_end|>\n
    # <|im_start|>user\n{input}<|im_end|>\n
    # <|im_start|>assistant\n
    # (no <think> tokens — those are only in qwen3, not qwen3_nothink)
    text = (
        f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return text


def score(model, tokenizer, ex, device, label_token_ids_cache, candidates):
    prefix_text = build_prefix(ex, tokenizer)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)

    log_p_per_label = {}
    for label in candidates:
        label_ids = label_token_ids_cache[label].to(device)
        full = torch.cat([prefix_ids, label_ids], dim=1)
        L = prefix_ids.size(1)
        K = label_ids.size(1)
        with torch.no_grad():
            logits = model(full).logits
        slice_logits = logits[0, L - 1:L + K - 1, :]
        log_probs = F.log_softmax(slice_logits, dim=-1)
        target = label_ids[0]
        log_p_per_label[label] = log_probs.gather(
            1, target.unsqueeze(1)).sum().item() / K

    vals = torch.tensor([log_p_per_label[l] for l in candidates])
    probs = F.softmax(vals, dim=0).tolist()
    return {l: p for l, p in zip(candidates, probs)}, log_p_per_label


def eval_file(bpath, model, tokenizer, device, label_token_ids_cache,
              candidates, out_dir):
    data = json.load(open(bpath))
    print(f"\n[{bpath.name}] {len(data)} examples")

    results = []
    cls_correct = 0
    solve_correct = 0
    solve_total = 0
    t0 = time.time()

    for i, ex in enumerate(data):
        probs, log_ps = score(model, tokenizer, ex, device,
                              label_token_ids_cache, candidates)
        pred = max(probs, key=probs.get)
        true_label = ex["output"]
        sa = ex.get("successful_actions")

        cls_ok = int(pred == true_label)
        cls_correct += cls_ok

        solve_ok = None
        if sa is not None:
            solve_ok = int(pred in sa)
            solve_correct += solve_ok
            solve_total += 1

        results.append({
            "idx": i,
            "true_label": true_label,
            "pred": pred,
            "probs": probs,
            "cls_ok": cls_ok,
            "solve_ok": solve_ok,
            "successful_actions": sa,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(data):
            elapsed = time.time() - t0
            cls_acc = cls_correct / (i + 1)
            sr = solve_correct / solve_total if solve_total else float("nan")
            print(f"  [{i+1}/{len(data)}] cls_acc={cls_acc:.3f} "
                  f"solve_rate={sr:.3f} ({solve_total} with SA) "
                  f"elapsed={elapsed:.0f}s")

    n = len(data)
    cls_acc = cls_correct / n
    solve_rate = solve_correct / solve_total if solve_total else float("nan")

    # per-label breakdown
    from collections import defaultdict
    per_label = defaultdict(lambda: {"cls_correct": 0, "total": 0,
                                     "solve_correct": 0, "solve_total": 0})
    for r in results:
        lbl = r["true_label"]
        per_label[lbl]["total"] += 1
        per_label[lbl]["cls_correct"] += r["cls_ok"]
        if r["solve_ok"] is not None:
            per_label[lbl]["solve_total"] += 1
            per_label[lbl]["solve_correct"] += r["solve_ok"]

    print(f"\n  === {bpath.name} ===")
    print(f"  cls_acc   = {cls_acc:.4f}  ({cls_correct}/{n})")
    print(f"  solve_rate= {solve_rate:.4f}  ({solve_correct}/{solve_total})")
    print(f"  per-label cls_acc:")
    for lbl in candidates:
        if lbl in per_label:
            d = per_label[lbl]
            ca = d["cls_correct"] / d["total"] if d["total"] else 0
            sr = d["solve_correct"] / d["solve_total"] if d["solve_total"] else float("nan")
            print(f"    {lbl:12s}  cls={ca:.3f} ({d['cls_correct']}/{d['total']})  "
                  f"solve={sr:.3f} ({d['solve_correct']}/{d['solve_total']})")

    out = {
        "file": bpath.name,
        "n_total": n,
        "n_with_sa": solve_total,
        "cls_acc": cls_acc,
        "solve_rate": solve_rate,
        "per_label": {k: dict(v) for k, v in per_label.items()},
        "examples": results,
    }
    out_path = Path(out_dir) / (bpath.stem + "_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  saved -> {out_path}")
    return cls_acc, solve_rate, n, solve_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--bench_glob", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--no_adapter", action="store_true")
    p.add_argument("--candidates", default="3cls",
                   choices=["3cls", "4cls"],
                   help="3cls=reflect/replan/escalate, 4cls adds unsolvable")
    args = p.parse_args()

    candidates = CANDIDATES_3CLS if args.candidates == "3cls" else CANDIDATES_4CLS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    print(f"device={device} dtype={dtype}")
    print(f"candidates: {candidates}")

    tokenizer_src = args.adapter_path if not args.no_adapter else args.base_model
    print(f"loading tokenizer from {tokenizer_src} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src, trust_remote_code=True)

    print(f"loading base model {args.base_model} ...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device,
        trust_remote_code=True)
    print(f"  loaded in {time.time()-t0:.1f}s")

    if args.no_adapter:
        model = base.eval()
    else:
        print(f"applying LoRA from {args.adapter_path} ...")
        t0 = time.time()
        model = PeftModel.from_pretrained(base, args.adapter_path)
        model.eval()
        n_lora = sum(1 for _, m in model.named_modules()
                     if hasattr(m, "lora_A") and len(m.lora_A) > 0)
        print(f"  loaded in {time.time()-t0:.1f}s, LoRA modules: {n_lora}")
        if n_lora == 0:
            print("  [WARN] no LoRA modules attached!")

    label_token_ids = {}
    print("label token ids:")
    for label in candidates:
        ids = tokenizer(label, return_tensors="pt",
                        add_special_tokens=False).input_ids
        tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
        print(f"  {label!r:14s} -> {ids[0].tolist()}  {tokens}")
        label_token_ids[label] = ids

    import glob
    bench_files = sorted(glob.glob(args.bench_glob))
    print(f"\nfound {len(bench_files)} benchmark files")

    summary = []
    for bpath in bench_files:
        cls_acc, solve_rate, n, n_sa = eval_file(
            Path(bpath), model, tokenizer, device,
            label_token_ids, candidates, args.out_dir)
        summary.append({
            "file": Path(bpath).name,
            "cls_acc": round(cls_acc, 4),
            "solve_rate": round(solve_rate, 4),
            "n": n, "n_sa": n_sa,
        })

    print("\n\n=== SUMMARY ===")
    print(f"{'file':<55} {'cls_acc':>8} {'solve_rate':>10} {'n_sa':>6}")
    print("-" * 85)
    for s in summary:
        print(f"{s['file']:<55} {s['cls_acc']:>8.4f} {s['solve_rate']:>10.4f} {s['n_sa']:>6}")

    json.dump(summary, open(Path(args.out_dir) / "summary.json", "w"), indent=2)
    print(f"\nsummary saved -> {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
