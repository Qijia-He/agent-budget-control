"""Use GPT-5.4-nano as the router classifier.

Pipeline parallel to router_predict.py but via Azure OpenAI logprobs API.

Output format: same jsonl schema so calibrate_crc.py / eval_crc.py work unchanged.
  {"idx": int, "true_label": str, "probs": {"reflect": p, "replan": p, "escalate": p},
   "argmax": str, "raw_completion": str}

Scoring:
  - First-generation-token's top_logprobs (capped at 5 by the API)
  - Map each top-K token to a candidate label using a disjoint prefix table
    that resolves the only ambiguity ("re" prefix shared by reflect & replan)
    by attributing "re" → "replan", because the tokenizer encodes "reflect"
    as its OWN single token — so a bare "re" first token almost certainly
    initiates "replan", not "reflect".
  - Take the max logprob seen per candidate; softmax across the 3 candidates.
"""
import argparse
import json
import math
import os
import time
import sys
from pathlib import Path

from openai import AzureOpenAI


API_KEY = os.environ.get("AZURE_NANO_KEY", "VxmXTg4dzQ6qwnfsgdFHT4OS75nVY9up_GPT_AK")
ENDPOINT = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
MODEL = "gpt-5.4-nano-2026-03-17"  # default; override with --model

CANDIDATES = ["reflect", "replan", "escalate"]

# Disjoint token→label table for first-generation-token disambiguation.
# Construction logic:
#   - any prefix that uniquely identifies one candidate maps to it
#   - "re" is shared by reflect & replan; "reflect" is its own single token,
#     so a standalone "re" first token must be initiating "replan".
TOKEN_TO_LABEL = {
    # reflect: only the full single-token form (since prefixes "r"/"re" collide with replan)
    "reflect": "reflect",
    "refl": "reflect",
    "ref": "reflect",
    # replan
    "re": "replan",
    "rep": "replan",
    "repl": "replan",
    "repla": "replan",
    "replan": "replan",
    # escalate: only candidate starting with "e"
    "e": "escalate",
    "es": "escalate",
    "esc": "escalate",
    "esca": "escalate",
    "escal": "escalate",
    "escala": "escalate",
    "escalat": "escalate",
    "escalate": "escalate",
}


def make_client():
    return AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-02-01",
        azure_endpoint=ENDPOINT,
        default_headers={"X-TT-LOGID": "router-crc"},
    )


def call_with_retry(client, ex, model=MODEL, max_retries=8, base_wait=4):
    """Call the API with backoff on rate limits."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ex["instruction"]},
                    {"role": "user", "content": ex["input"]},
                ],
                max_tokens=150,  # mini endpoint requires >=150 to return logprobs; nano works with smaller but 150 is harmless
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate = "429" in msg or "qpm" in msg.lower() or "rate" in msg.lower()
            if not is_rate:
                raise
            wait = base_wait + 2 * attempt
            time.sleep(wait)
    raise RuntimeError(f"exhausted retries: {last_err}")


def parse_probs(resp):
    """Return ({label: prob}, argmax, raw_completion) for one API response."""
    content = resp.choices[0].message.content or ""
    lp = resp.choices[0].logprobs
    if lp is None or not lp.content:
        # fall back: assign all prob to the candidate matching the content text
        cand = None
        for c in CANDIDATES:
            if c in content.lower():
                cand = c
                break
        if cand is None:
            cand = "escalate"  # safe fallback
        probs = {c: (1.0 if c == cand else 0.0) for c in CANDIDATES}
        return probs, cand, content

    first = lp.content[0]
    cand_lp = {c: -math.inf for c in CANDIDATES}
    for entry in first.top_logprobs:
        tok = entry.token.strip().lower()
        if not tok:
            continue
        c = TOKEN_TO_LABEL.get(tok)
        if c is not None:
            cand_lp[c] = max(cand_lp[c], entry.logprob)

    # if no top-K token matched any candidate, fall back to content parse
    if all(v == -math.inf for v in cand_lp.values()):
        cand = None
        for c in CANDIDATES:
            if c in content.lower():
                cand = c
                break
        if cand is None:
            cand = "escalate"
        probs = {c: (1.0 if c == cand else 0.0) for c in CANDIDATES}
        return probs, cand, content

    # softmax (with -inf rows getting 0)
    mx = max(v for v in cand_lp.values() if v != -math.inf)
    exps = {c: (math.exp(v - mx) if v != -math.inf else 0.0) for c, v in cand_lp.items()}
    tot = sum(exps.values())
    probs = {c: e / tot for c, e in exps.items()}
    argmax = max(probs, key=probs.get)
    return probs, argmax, content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_examples", type=int, default=-1)
    p.add_argument("--model", default=MODEL, help="Azure model id (e.g. gpt-5.4-mini-2026-03-17)")
    p.add_argument("--pace_sec", type=float, default=0.2,
                   help="sleep between successful calls to keep QPM under limit")
    p.add_argument("--resume", action="store_true",
                   help="if output already has N records, skip first N")
    args = p.parse_args()

    examples = []
    with open(args.input) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    # resume
    start = 0
    mode = "w"
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            done = sum(1 for _ in f)
        start = done
        mode = "a"
        print(f"resume from index {start}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    client = make_client()
    correct = 0
    seen = 0
    t0 = time.time()

    with open(args.output, mode) as fout:
        for i, ex in enumerate(examples):
            if i < start:
                continue
            try:
                resp = call_with_retry(client, ex, model=args.model)
            except Exception as e:
                print(f"  [{i}] FAILED after retries: {e}", flush=True)
                # write a sentinel record so we keep index alignment
                rec = {"idx": i, "true_label": ex["output"], "probs": None,
                       "argmax": None, "error": str(e)}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue
            probs, argmax, content = parse_probs(resp)
            if argmax == ex["output"]:
                correct += 1
            seen += 1
            rec = {
                "idx": i,
                "true_label": ex["output"],
                "probs": probs,
                "argmax": argmax,
                "raw_completion": content,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or (i + 1) == len(examples):
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
