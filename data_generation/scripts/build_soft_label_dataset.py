# -*- coding: utf-8 -*-
"""Convert vote JSONL to soft-label SFT dataset.

Input:  outputs/votes/votes_3cls_v3.jsonl  (one record per problem)
Output: datasets/router_soft_label_3cls.json  (LlamaFactory Alpaca format,
        with 'output' replaced by a soft label dict)

Soft label semantics (3cls):
  p_reflect  = votes["reflect"]  / n_solvable
  p_replan   = votes["replan"]   / n_solvable
  p_escalate = votes["escalate"] / n_solvable
  (normalized over solvable trials only; sums to 1)

  p_unsolvable = votes["unsolvable"] / n_votes  (stored separately, not trained)

Examples where n_solvable == 0 (all 10 trials unsolvable) are dropped from
the 3cls dataset — no signal for action selection.

Examples where n_solvable < MIN_SOLVABLE are optionally dropped or down-weighted.

Output format (LlamaFactory Alpaca with soft label):
  {
    "instruction": "...",
    "input": "...",
    "output": "escalate",            <- argmax hard label (for baseline / SFT comparison)
    "soft_label": {                  <- normalized soft label (sum=1 over 3 actions)
      "reflect": 0.1,
      "replan": 0.2,
      "escalate": 0.7
    },
    "p_unsolvable": 0.1,             <- fraction of trials with no passing action
    "n_solvable": 9,
    "n_votes": 10,
    "problem_id": "...",
    "dataset": "..."
  }

Usage:
    python scripts/build_soft_label_dataset.py \\
        --votes outputs/votes/votes_3cls_v3.jsonl \\
        --sft_data datasets/router_arch_a_3cls_v3.json \\
        --out datasets/router_soft_label_3cls.json \\
        --min_solvable 1
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ACTIONS = ["reflect", "replan", "escalate"]
PRIO    = ["reflect", "replan", "escalate"]

INSTRUCTION = (
    "You are a cost-aware coding router. A small fast model just attempted the "
    "coding problem below and FAILED. Based on the problem and the failure trace, "
    "choose the cheapest recovery action that will solve it.\n\n"
    "Recovery actions (in increasing cost):\n"
    "  reflect  — small model fixes its own code given the failure trace (cost ~1.2)\n"
    "  replan   — small model discards the attempt and re-plans from scratch (cost ~1.5)\n"
    "  escalate — strong model solves from scratch (cost ~12)\n\n"
    "Reply with exactly one word: reflect, replan, or escalate. Do not explain."
)


def _argmax_label(votes):
    """Cheapest action with most votes (tie-break by PRIO)."""
    best, best_v = "unsolvable", -1
    for action in PRIO:
        v = votes.get(action, 0)
        if v > best_v:
            best_v = v
            best = action
    if best_v == 0:
        return "unsolvable"
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes",    default="outputs/votes/votes_3cls_v3.jsonl")
    parser.add_argument("--sft_data", default=None,
                        help="Original SFT data to copy instruction/input from. "
                             "If omitted, instruction/input are reconstructed from votes JSONL.")
    parser.add_argument("--out",      default="../../datasets/router_soft_label_3cls.json")
    parser.add_argument("--min_solvable", type=int, default=1,
                        help="Drop examples with n_solvable < this value.")
    args = parser.parse_args()

    # --- Load votes ---
    votes_path = Path(args.votes)
    if not votes_path.exists():
        raise FileNotFoundError(f"Votes file not found: {votes_path}")

    vote_records = {}
    with open(votes_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            vote_records[rec["problem_id"]] = rec

    print(f"Loaded {len(vote_records)} vote records")

    # --- Load original SFT data for instruction/input text (if provided) ---
    pid_to_sft = {}
    if args.sft_data:
        # Build a prompt-prefix -> sft example index to match by text
        # (vote records have problem_prompt, sft has instruction+input)
        with open(args.sft_data, encoding="utf-8") as f:
            sft_examples = json.load(f)
        # We'll match by building a prompt-prefix index of the sft data
        # and looking up each vote record's problem_prompt
        def _sft_key(inp):
            if inp.startswith("Problem:\n"):
                inp = inp[len("Problem:\n"):]
            marker = "\nInitial attempt verdict:"
            idx = inp.find(marker)
            if idx >= 0:
                inp = inp[:idx]
            return inp.strip()[:200]

        sft_by_prompt = {_sft_key(ex["input"]): ex for ex in sft_examples}
        print(f"SFT index: {len(sft_by_prompt)} entries")

    # --- Build dataset ---
    out_data = []
    stats = Counter()

    for pid, rec in vote_records.items():
        votes    = rec["votes"]
        n_votes  = rec["n_votes"]
        n_solv   = rec["n_solvable"]

        # Drop fully unsolvable
        if n_solv < args.min_solvable:
            stats["dropped_unsolvable"] += 1
            continue

        # Soft label (normalized over solvable trials)
        soft = {a: votes.get(a, 0) / n_solv for a in ACTIONS}
        assert abs(sum(soft.values()) - 1.0) < 1e-6, f"soft label doesn't sum to 1: {soft}"

        # Hard argmax label (most-voted action, cheapest wins ties)
        hard = _argmax_label(votes)
        if hard == "unsolvable":
            stats["dropped_unsolvable"] += 1
            continue

        p_unsolvable = votes.get("unsolvable", 0) / n_votes

        # instruction / input text
        if args.sft_data:
            key = rec["problem_prompt"][:200]
            sft_ex = sft_by_prompt.get(key)
            if sft_ex:
                instruction = sft_ex["instruction"]
                inp         = sft_ex["input"]
            else:
                # Fallback: reconstruct from vote record
                instruction = INSTRUCTION
                inp = f"Problem:\n{rec['problem_prompt']}\n\nInitial attempt verdict: {rec.get('proceed_verdict','fail')}\n\nError:\n{rec.get('stderr_0','')}"
                stats["sft_fallback"] += 1
        else:
            instruction = INSTRUCTION
            inp = f"Problem:\n{rec['problem_prompt']}\n\nInitial attempt verdict: {rec.get('proceed_verdict','fail')}\n\nError:\n{rec.get('stderr_0','')}"

        out_data.append({
            "instruction":   instruction,
            "input":         inp,
            "output":        hard,
            "soft_label":    soft,
            "p_unsolvable":  round(p_unsolvable, 4),
            "n_solvable":    n_solv,
            "n_votes":       n_votes,
            "problem_id":    pid,
            "dataset":       rec.get("dataset", ""),
        })
        stats[f"hard_{hard}"] += 1

    # --- Write ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {out_path}  ({len(out_data)} examples)")
    print(f"\nLabel distribution (hard argmax):")
    total = len(out_data)
    for a in ACTIONS:
        k = f"hard_{a}"
        n = stats.get(k, 0)
        print(f"  {a:10s}: {n:5d} ({n/total:.1%})")
    print(f"\nDropped (n_solvable < {args.min_solvable}): {stats['dropped_unsolvable']}")
    if stats.get("sft_fallback"):
        print(f"SFT fallback (reconstructed input): {stats['sft_fallback']}")

    # --- Soft label distribution summary ---
    print(f"\nSoft label stats (mean p per action):")
    for a in ACTIONS:
        mean_p = sum(ex["soft_label"][a] for ex in out_data) / len(out_data)
        print(f"  {a:10s}: mean p = {mean_p:.3f}")

    frac_mixed = sum(
        1 for ex in out_data
        if sum(1 for a in ACTIONS if ex["soft_label"][a] > 0.05) > 1
    )
    print(f"\nExamples with >1 action having p>5%: {frac_mixed} ({frac_mixed/total:.1%})")
    print("  (these are the 'uncertain' examples where soft labels add value over hard labels)")


if __name__ == "__main__":
    main()
