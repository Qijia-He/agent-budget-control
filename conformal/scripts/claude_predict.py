"""Use Claude Sonnet 4.6 as the router classifier (argmax-only baseline).

The Anthropic Messages API does not expose logprobs, so unlike
nano_predict.py this cannot produce a calibrated probability distribution
over {reflect, replan, escalate} -- only a single argmax completion. probs
is therefore a 0/1 indicator on the parsed argmax, mirroring nano_predict.py's
own no-logprobs fallback path. This means CRC/Mode A analysis is NOT possible
on this output (no real probability mass to sweep lambda over) -- only
cls_acc / solve_rate comparisons against the SFT router and other baselines.

API key is read from a file path (--key_file), never passed on the command
line or hardcoded, to avoid exposing it in shell history / `ps` on a shared
cluster.

Input: a benchmark JSON file (list of {instruction, input, output,
successful_actions, ...}), e.g. datasets/benchmarks/holdout_3cls_test.json.

Output: jsonl with {"idx", "true_label", "probs", "argmax", "raw_completion"}.

Usage:
    python conformal/scripts/claude_predict.py \\
        --input datasets/benchmarks/holdout_3cls_test.json \\
        --output datasets/eval_results/bench_eval_claude/holdout_test_preds.jsonl \\
        --key_file /path/to/.anthropic_key
"""
import argparse
import json
import time
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
CANDIDATES = ["reflect", "replan", "escalate"]


def parse_argmax(text: str) -> str:
    """Pick the candidate word that appears LAST in the text (the model's
    conclusion, if it reasoned through other actions first), not just the
    first one found in CANDIDATES order."""
    t = text.strip().lower()
    best_pos, best_c = -1, None
    for c in CANDIDATES:
        pos = t.rfind(c)
        if pos > best_pos:
            best_pos, best_c = pos, c
    return best_c if best_c is not None else "escalate"


def call_with_retry(client, ex, max_retries=6, base_wait=4):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=300,
                temperature=0.0,
                system=ex["instruction"],
                messages=[{"role": "user", "content": ex["input"]}],
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate = "429" in msg or "rate" in msg.lower() or "overloaded" in msg.lower()
            if not is_rate:
                raise
            time.sleep(base_wait + 2 * attempt)
    raise RuntimeError(f"exhausted retries: {last_err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key_file", required=True)
    p.add_argument("--max_examples", type=int, default=-1)
    p.add_argument("--pace_sec", type=float, default=0.1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    api_key = Path(args.key_file).read_text().strip()
    client = anthropic.Anthropic(api_key=api_key)

    examples = json.load(open(args.input))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    start = 0
    mode = "w"
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            start = sum(1 for _ in f)
        mode = "a"
        print(f"resume from index {start}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    seen = 0
    t0 = time.time()

    with open(args.output, mode) as fout:
        for i, ex in enumerate(examples):
            if i < start:
                continue
            try:
                resp = call_with_retry(client, ex)
                content = resp.content[0].text if resp.content else ""
            except Exception as e:
                print(f"  [{i}] FAILED after retries: {e}", flush=True)
                rec = {"idx": i, "true_label": ex["output"], "probs": None,
                       "argmax": None, "error": str(e)}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue

            argmax = parse_argmax(content)
            probs = {c: (1.0 if c == argmax else 0.0) for c in CANDIDATES}
            if argmax == ex["output"]:
                correct += 1
            seen += 1
            rec = {
                "idx": i,
                "true_label": ex["output"],
                "successful_actions": ex.get("successful_actions"),
                "probs": probs,
                "argmax": argmax,
                "raw_completion": content,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0 or (i + 1) == len(examples):
                elapsed = time.time() - t0
                rate = seen / elapsed if elapsed > 0 else 0
                eta = (len(examples) - (i + 1)) / rate if rate > 0 else 0
                acc = correct / seen if seen else 0
                print(f"  [{i+1}/{len(examples)}] argmax_acc={acc:.3f} rate={rate:.2f}/s eta={eta:.0f}s", flush=True)
            time.sleep(args.pace_sec)

    acc = correct / seen if seen else 0
    print(f"done. argmax_acc={correct}/{seen}={acc:.4f}")
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()
