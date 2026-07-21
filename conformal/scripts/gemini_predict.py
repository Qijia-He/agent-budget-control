"""Use Gemini 3.1 Pro as the router classifier (argmax-only baseline).

Like claude_predict.py: the Gemini API rejects responseLogprobs for the
gemini-3.x model family ("Logprobs is not enabled for this model"), so this
only produces an argmax completion, not a calibrated probability
distribution. probs is a 0/1 indicator on the parsed argmax. No CRC/Mode A
analysis is possible on this output -- cls_acc / solve_rate only.

gemini-3.1-pro-preview is a reasoning model: thinkingConfig.thinkingLevel is
set to "low" to keep latency/cost down while still letting it finish
thinking before the output token budget runs out (default budget burns
output tokens on thinking and can return empty content otherwise).

API key is read from the GOOGLE_GENAI_API_KEY / GEMINI_API_KEY env var
(already set in this shell) -- never passed on the command line.

Input: a benchmark JSON file (list of {instruction, input, output,
successful_actions, ...}), e.g. datasets/benchmarks/holdout_3cls_test.json.

Output: jsonl with {"idx", "true_label", "probs", "argmax", "raw_completion"}.

Usage:
    python conformal/scripts/gemini_predict.py \\
        --input datasets/benchmarks/holdout_3cls_test.json \\
        --output datasets/eval_results/bench_eval_gemini/holdout_test_preds.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

MODEL = "gemini-3.1-pro-preview"
CANDIDATES = ["reflect", "replan", "escalate"]
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


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


def call_with_retry(api_key, ex, max_retries=6, base_wait=4):
    url = f"{API_BASE}/{MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"{ex['instruction']}\n\n{ex['input']}"}]}],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.0,
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    last_err = None
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        last_err = f"{resp.status_code}: {resp.text[:300]}"
        if resp.status_code not in (429, 503):
            raise RuntimeError(last_err)
        time.sleep(base_wait + 2 * attempt)
    raise RuntimeError(f"exhausted retries: {last_err}")


def extract_text(resp_json):
    try:
        parts = resp_json["candidates"][0]["content"].get("parts", [])
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_examples", type=int, default=-1)
    p.add_argument("--pace_sec", type=float, default=0.2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("GOOGLE_GENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("set GOOGLE_GENAI_API_KEY or GEMINI_API_KEY in the environment")

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
                resp_json = call_with_retry(api_key, ex)
                content = extract_text(resp_json)
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
