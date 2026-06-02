"""Smoke test for GPT-5.4-nano as the router classifier.

Goal: confirm
  (1) API responds with bare label content,
  (2) logprobs are available and the first token is one of "reflect"/"re"/"esc" etc.,
  (3) we can build a 3-way softmax over candidate first-tokens.
"""
import json
import math
import os
import sys
from openai import AzureOpenAI

API_KEY = os.environ.get("AZURE_NANO_KEY", "VxmXTg4dzQ6qwnfsgdFHT4OS75nVY9up_GPT_AK")
ENDPOINT = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
MODEL = "gpt-5.4-nano-2026-03-17"

client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
    default_headers={"X-TT-LOGID": "router-crc-smoke"},
)

CANDIDATES = ["reflect", "replan", "escalate"]


def call_one(ex, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ex["instruction"]},
                    {"role": "user", "content": ex["input"]},
                ],
                max_tokens=8,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )
            return resp
        except Exception as e:
            msg = str(e)
            if "429" in msg or "qpm" in msg.lower() or "rate" in msg.lower():
                wait = 5 + 2 * attempt
                print(f"  rate-limited, sleeping {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("exhausted retries")


def main():
    with open("/mnt/bn/ecom-govern-models/qijiahe/conformal/data/calib.jsonl") as f:
        examples = [json.loads(l) for l in f][:3]

    import time
    for i, ex in enumerate(examples):
        print(f"\n========== example {i}  true_label={ex['output']} ==========")
        if i > 0:
            time.sleep(2)  # gentle pacing to avoid QPM
        try:
            resp = call_one(ex)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        msg = resp.choices[0].message
        print(f"  content: {msg.content!r}")
        # reasoning field if present
        if hasattr(msg, "reasoning") and msg.reasoning:
            print(f"  reasoning: {msg.reasoning!r}")

        lp = resp.choices[0].logprobs
        if lp is None or not lp.content:
            print("  NO LOGPROBS")
            continue
        first = lp.content[0]
        print(f"  first token: {first.token!r}  logprob={first.logprob:.4f}")
        print(f"  top_logprobs (first {min(8, len(first.top_logprobs))} of {len(first.top_logprobs)}):")
        for entry in first.top_logprobs[:8]:
            print(f"    {entry.token!r:20s}  logprob={entry.logprob:8.4f}  prob={math.exp(entry.logprob):.4f}")

        # map to candidates by prefix
        cand_lp = {c: -math.inf for c in CANDIDATES}
        for entry in first.top_logprobs:
            tok = entry.token.strip().lower()
            if not tok:
                continue
            for cand in CANDIDATES:
                # exact match OR token is a prefix of candidate (e.g. "re" matches "reflect"/"replan")
                if tok == cand or cand.startswith(tok):
                    cand_lp[cand] = max(cand_lp[cand], entry.logprob)
        print(f"  per-candidate best first-token logprob:")
        for c, l in cand_lp.items():
            print(f"    {c:10s}  {l:8.4f}")
        # softmax
        if all(l == -math.inf for l in cand_lp.values()):
            print("  no candidate matched any top-K token!")
        else:
            mx = max(cand_lp.values())
            exps = {c: math.exp(l - mx) if l != -math.inf else 0.0 for c, l in cand_lp.items()}
            tot = sum(exps.values())
            probs = {c: e / tot for c, e in exps.items()}
            print(f"  softmax probs: {probs}")
            argmax = max(probs, key=probs.get)
            sign = "✓" if argmax == ex["output"] else "✗"
            print(f"  argmax={argmax} {sign}")


if __name__ == "__main__":
    main()
