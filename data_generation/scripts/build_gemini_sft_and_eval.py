"""Build SFT training data and CRC eval benchmarks from Gemini cascade-router rollouts.

Source: whisperle/cascade-router-gemini-rollouts on HuggingFace
Two model pairs (20260712 versions):
  gemini_25_flash_pro  google/gemini-2.5-flash (cheap) + gemini-2.5-pro (strong)
  gemini_35_flash_pro  google/gemini-3.5-flash (cheap) + gemini-2.5-pro (strong)

SFT outputs (LlamaFactory Alpaca, same schema as build_sft_router_v1.py):
  outputs/sft/gemini_router_3cls.json
  outputs/sft/gemini_router_4cls.json

Eval outputs (same pipeline structure as GPT benchmarks):
  Benchmark JSONs (no costs embedded, same format as holdout_3cls_calib.json):
    outputs/eval/gemini_25_holdout_3cls_calib.json
    outputs/eval/gemini_25_holdout_3cls_test.json
    outputs/eval/gemini_35_holdout_3cls_calib.json
    outputs/eval/gemini_35_holdout_3cls_test.json

  Cost tables (same format as conformal/data/action_costs_usd.json):
    outputs/eval/gemini_25_action_costs_usd.json
    outputs/eval/gemini_35_action_costs_usd.json

Pipeline reuse:
  attach_usd_costs.py --bench_json gemini_25_holdout_3cls_calib.json \
                       --costs_json gemini_25_action_costs_usd.json ...
  crc_on_holdout_usd.py   (unchanged)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID = "whisperle/cascade-router-gemini-rollouts"

MODEL_PAIRS = {
    "gemini_25": {
        "dir": "gemini_25_flash_pro_dataset_20260712",
        "label_prefix": "gemini25_flash_pro_router_labels",
        "raw_prefix": "gemini25_flash_pro_rollouts",
        "cheap_model": "google/gemini-2.5-flash",
        "strong_model": "gemini-2.5-pro",
    },
    "gemini_35": {
        "dir": "gemini_35_flash_pro_dataset_20260712",
        "label_prefix": "gemini35_flash_pro_router_labels",
        "raw_prefix": "gemini35_flash_pro_rollouts",
        "cheap_model": "google/gemini-3.5-flash",
        "strong_model": "gemini-2.5-pro",
    },
}

TRAIN_SPLITS = [
    "bench_4cls_random",
    "bench_archB_4cls_random",
    "bench_archB_5cls_random",
    "holdout_3cls_bench",
    "holdout_4cls_bench",
]

EVAL_SPLITS = ["holdout_3cls_calib", "holdout_3cls_test"]

# Gemini token pricing (USD per 1M tokens).
# Thinking models: reasoning_tokens billed at 'thinking' rate;
# remaining completion tokens billed at 'output' rate.
# gemini-3.5-flash pricing estimated (same tier as 2.5-flash).
GEMINI_PRICING: Dict[str, Dict[str, float]] = {
    "google/gemini-2.5-flash": {"input": 0.15, "output": 0.60, "thinking": 3.50},
    "google/gemini-3.5-flash": {"input": 0.15, "output": 0.60, "thinking": None},
    "gemini-2.5-pro":          {"input": 1.25, "output": 10.00, "thinking": 3.50},
}

PROMPT_CHARS = 6000
STDERR_CAP_FAIL = 800
STDERR_CAP_OTHER = 400

INSTRUCTION_3CLS = (
    "You are a cost-aware coding router. A small fast model just attempted the "
    "coding problem below and FAILED. Based on the problem and the failure trace, "
    "choose the cheapest recovery action that will solve it.\n\n"
    "Recovery actions (in increasing cost):\n"
    "  reflect  — small model fixes its own code given the failure trace (cost ~1.2)\n"
    "  replan   — small model discards the attempt and re-plans from scratch (cost ~1.5)\n"
    "  escalate — strong model solves from scratch (cost ~12)\n\n"
    "Reply with exactly one word: reflect, replan, or escalate. Do not explain."
)

INSTRUCTION_4CLS = (
    "You are a cost-aware coding router. A small fast model just attempted the "
    "coding problem below and FAILED. Based on the problem and the failure trace, "
    "choose the cheapest recovery action that will solve it, or label the problem "
    "as unsolvable if no action in our cascade is likely to succeed.\n\n"
    "Recovery actions (in increasing cost):\n"
    "  reflect    — small model fixes its own code given the failure trace (cost ~1.2)\n"
    "  replan     — small model discards the attempt and re-plans from scratch (cost ~1.5)\n"
    "  escalate   — strong model solves from scratch (cost ~12)\n"
    "  unsolvable — no action in our cascade will solve this; do not spend budget\n\n"
    "Reply with exactly one word: reflect, replan, escalate, or unsolvable. Do not explain."
)

RECOVERY_ACTIONS = ["reflect", "replan", "escalate"]

# ---------------------------------------------------------------------------
# Metadata prefix — same format as v6_v4meta GPT data
# "[source=X | difficulty=Y | len=Z | algo=A]"
# ---------------------------------------------------------------------------

SOURCE_TO_DIFFICULTY = {
    "apps": "interview", "apps_functional": "interview", "apps_lc": "interview",
    "taco_hard": "competitive_hard", "taco_medhard": "competitive_medhard",
    "taco_medium": "competitive_medium", "taco_veryhard": "competitive_veryhard",
    "taco_all": "competitive_mixed",
    "bcb": "easy", "lcb": "contest_recent",
    "codecontests_train": "competitive_hard",
    "unknown": "unknown",
}

# Benchmark v6_meta files: problem_id -> metadata string (loaded once)
_PID_TO_META: Optional[Dict[str, str]] = None

V6META_BENCH_GLOB = (
    "/path/to/agent-budget-control"
    "/datasets/benchmarks/v6_meta/*.json"
)


def _load_pid_meta_lookup() -> Dict[str, str]:
    global _PID_TO_META
    if _PID_TO_META is not None:
        return _PID_TO_META
    lookup: Dict[str, str] = {}
    for bf in glob.glob(V6META_BENCH_GLOB):
        for r in json.load(open(bf)):
            pid = r.get("_problem_id")
            if not pid:
                continue
            m = re.match(r"^\[(.+?)\]\n", r["input"])
            if m:
                lookup[pid] = m.group(1)
    _PID_TO_META = lookup
    return lookup


def build_meta_prefix(problem_id: str, dataset: str, problem_text: str) -> str:
    """Return '[source=X | difficulty=Y | len=Z | algo=A]' string.

    Priority:
      1. Exact problem_id match in v6_meta benchmark files (preserves algo tags)
      2. Fallback: derive source/difficulty from dataset field, algo=general
    """
    lookup = _load_pid_meta_lookup()
    if problem_id in lookup:
        # Reuse existing metadata but recompute len from actual text
        existing = lookup[problem_id]
        parts = dict(p.strip().split("=", 1) for p in existing.split("|"))
        length = round(len(problem_text) / 200) * 200
        return (f"[source={parts['source']} | difficulty={parts['difficulty']}"
                f" | len={length} | algo={parts.get('algo', 'general')}]")

    source = dataset if dataset in SOURCE_TO_DIFFICULTY else "unknown"
    difficulty = SOURCE_TO_DIFFICULTY.get(source, "unknown")
    length = round(len(problem_text) / 200) * 200
    return f"[source={source} | difficulty={difficulty} | len={length} | algo=general]"


OUT_DIR = Path("outputs")
SFT_DIR = OUT_DIR / "sft"
EVAL_DIR = OUT_DIR / "eval"
SFT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def compute_call_cost(model: str, usage: Optional[Dict]) -> Optional[float]:
    if usage is None or model not in GEMINI_PRICING:
        return None
    price = GEMINI_PRICING[model]
    prompt_tok  = usage.get("prompt_tokens", 0) or 0
    completion_tok = usage.get("completion_tokens", 0) or 0
    reasoning_tok  = usage.get("reasoning_tokens") or 0

    input_cost = prompt_tok * price["input"] / 1_000_000
    if price["thinking"] is not None and reasoning_tok > 0:
        think_cost = reasoning_tok * price["thinking"] / 1_000_000
        out_cost   = max(completion_tok - reasoning_tok, 0) * price["output"] / 1_000_000
    else:
        think_cost = 0.0
        out_cost   = completion_tok * price["output"] / 1_000_000

    return input_cost + think_cost + out_cost


# ---------------------------------------------------------------------------
# Input construction (matches build_sft_router_v1.py exactly)
# ---------------------------------------------------------------------------

def trim_stderr(verdict: str, stderr: str) -> str:
    s = (stderr or "").strip()
    if not s:
        return ""
    cap = STDERR_CAP_FAIL if verdict == "fail" else STDERR_CAP_OTHER
    if len(s) > cap:
        s = s[:cap] + "\n... [truncated]"
    return s


def build_input(problem: str, verdict: str, stderr: str,
                problem_id: str = "", dataset: str = "") -> str:
    if len(problem) > PROMPT_CHARS:
        problem = problem[:PROMPT_CHARS] + "\n... [problem truncated]"
    stderr_trimmed = trim_stderr(verdict, stderr)
    parts = ["Problem:", problem, "", f"Initial attempt verdict: {verdict}"]
    if stderr_trimmed:
        parts.append("Initial attempt stderr:")
        parts.append(stderr_trimmed)
    body = "\n".join(parts)
    if problem_id or dataset:
        prefix = build_meta_prefix(problem_id, dataset, problem)
        return f"{prefix}\n{body}"
    return body


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def _hf_file(pair_dir: str, subdir: str, filename: str) -> Path:
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{pair_dir}/{subdir}/{filename}.jsonl",
        repo_type="dataset",
    ))


def load_labels(pair_dir: str, label_prefix: str, split: str) -> List[Dict]:
    path = _hf_file(pair_dir, "labels", f"{label_prefix}_{split}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_raw_index(pair_dir: str, raw_prefix: str, split: str) -> Dict[str, Dict]:
    path = _hf_file(pair_dir, "raw", f"{raw_prefix}_{split}")
    index: Dict[str, Dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            index[rec["problem_id"]] = rec
    return index


# ---------------------------------------------------------------------------
# SFT builder
# ---------------------------------------------------------------------------

def build_sft_records(pair_key: str, records_3cls: List, records_4cls: List):
    cfg = MODEL_PAIRS[pair_key]
    n3 = n4 = 0
    for split in TRAIN_SPLITS:
        print(f"  [{pair_key}] SFT split: {split}")
        for lab in load_labels(cfg["dir"], cfg["label_prefix"], split):
            label = lab["label"]
            inp = build_input(lab["problem"], lab["initial_verdict"],
                              lab.get("initial_stderr") or "",
                              lab["problem_id"], lab["dataset"])
            if label in RECOVERY_ACTIONS:
                records_3cls.append({"instruction": INSTRUCTION_3CLS, "input": inp, "output": label})
                n3 += 1
            records_4cls.append({"instruction": INSTRUCTION_4CLS, "input": inp, "output": label})
            n4 += 1
    return n3, n4


# ---------------------------------------------------------------------------
# Eval builder — benchmark JSON + cost table
# ---------------------------------------------------------------------------

def build_eval_for_pair(pair_key: str) -> Dict[str, float]:
    """Build benchmark JSONs and return per_problem cost dict for cost table."""
    cfg = MODEL_PAIRS[pair_key]
    per_problem: Dict[str, Dict] = {}

    for split in EVAL_SPLITS:
        print(f"  [{pair_key}] eval split: {split}")
        labels    = load_labels(cfg["dir"], cfg["label_prefix"], split)
        raw_index = load_raw_index(cfg["dir"], cfg["raw_prefix"], split)

        # Collect actual per-action costs from raw rollouts
        raw_costs_map: Dict[str, Dict[str, Optional[float]]] = {}
        esc_tested: Dict[str, bool] = {}
        for lab in labels:
            pid = lab["problem_id"]
            if pid not in raw_index:
                continue
            action_costs: Dict[str, Optional[float]] = {}
            has_escalate = False
            for call in raw_index[pid].get("calls", []):
                action = call.get("action")
                if action not in RECOVERY_ACTIONS:
                    continue
                cost = compute_call_cost(call.get("model", ""), call.get("usage"))
                action_costs[action] = cost
                if action == "escalate":
                    has_escalate = True
            raw_costs_map[pid] = action_costs
            esc_tested[pid] = has_escalate

        # Compute per-action means for imputation
        action_vals: Dict[str, List[float]] = defaultdict(list)
        for costs in raw_costs_map.values():
            for a, c in costs.items():
                if c is not None:
                    action_vals[a].append(c)
        mean_cost = {a: (sum(v)/len(v) if v else None) for a, v in action_vals.items()}

        # Build benchmark records (no costs embedded) and accumulate cost table
        bench: List[Dict] = []
        n_imputed = 0
        for lab in labels:
            if lab["oracle_unsolvable"]:
                continue  # 3cls eval only keeps solvable problems
            pid     = lab["problem_id"]
            dataset = lab["dataset"]
            inp     = build_input(lab["problem"], lab["initial_verdict"],
                                  lab.get("initial_stderr") or "")

            # Benchmark record — same keys as GPT holdout_3cls_calib.json
            bench.append({
                "instruction":       INSTRUCTION_3CLS,
                "input":             inp,
                "output":            lab["label"],
                "successful_actions": lab["successful_actions"],
                "oracle_unsolvable": lab["oracle_unsolvable"],
                "_dataset":          dataset,
                "_problem_id":       pid,
                "_escalate_tested":  esc_tested.get(pid, False),
            })

            # Accumulate cost table entry
            costs_row = raw_costs_map.get(pid, {})
            cost_key  = f"{dataset}::{pid}"
            entry: Dict = {"dataset": dataset, "problem_id": pid}
            for a in RECOVERY_ACTIONS:
                c = costs_row.get(a)
                if c is None:
                    c = mean_cost.get(a)
                    n_imputed += 1
                entry[f"{a}_usd"] = c
            entry["proceed_usd"]    = None  # not collected for Gemini rollouts
            entry["unsolvable_usd"] = 0.0
            per_problem[cost_key]   = entry

        out = EVAL_DIR / f"{pair_key}_{split}.json"
        out.write_text(json.dumps(bench, ensure_ascii=False, indent=2))
        print(f"    {len(bench)} benchmark records, {n_imputed} costs imputed → {out}")
        for a in RECOVERY_ACTIONS:
            c = mean_cost.get(a)
            print(f"    mean {a}: {'${:.6f}'.format(c) if c else 'N/A'}")

    return per_problem


def write_cost_table(pair_key: str, per_problem: Dict) -> None:
    """Write cost table in the same format as action_costs_usd.json."""
    # Compute dataset-level means
    dataset_vals: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in per_problem.values():
        ds = entry["dataset"]
        for a in RECOVERY_ACTIONS:
            c = entry.get(f"{a}_usd")
            if c is not None:
                dataset_vals[ds][a].append(c)

    dataset_means: Dict[str, Dict[str, Optional[float]]] = {}
    for ds, av in dataset_vals.items():
        dataset_means[ds] = {a: (sum(v)/len(v) if v else None) for a, v in av.items()}

    table = {
        "dataset_means_usd": dataset_means,
        "per_problem":        per_problem,
        "_counts":            {
            "total": len(per_problem),
            "pair":  pair_key,
        },
    }
    out = EVAL_DIR / f"{pair_key}_action_costs_usd.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"  Cost table → {out} ({len(per_problem)} problems)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-only",  action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--pairs", nargs="+", choices=list(MODEL_PAIRS), default=list(MODEL_PAIRS))
    args = ap.parse_args()

    do_sft  = not args.eval_only
    do_eval = not args.sft_only

    if do_sft:
        print("=== Building SFT data ===")
        all_3cls: List[Dict] = []
        all_4cls: List[Dict] = []
        for pair_key in args.pairs:
            n3, n4 = build_sft_records(pair_key, all_3cls, all_4cls)
            print(f"  [{pair_key}] +{n3} 3cls, +{n4} 4cls")

        (SFT_DIR / "gemini_router_3cls.json").write_text(
            json.dumps(all_3cls, ensure_ascii=False, indent=2))
        (SFT_DIR / "gemini_router_4cls.json").write_text(
            json.dumps(all_4cls, ensure_ascii=False, indent=2))

        from collections import Counter
        c3 = Counter(r["output"] for r in all_3cls)
        c4 = Counter(r["output"] for r in all_4cls)
        print(f"\n3cls ({len(all_3cls)}): " + " | ".join(f"{a}={c3[a]}" for a in RECOVERY_ACTIONS))
        print(f"4cls ({len(all_4cls)}): " + " | ".join(f"{a}={c4[a]}" for a in RECOVERY_ACTIONS + ["unsolvable"]))
        print(f"Written: {SFT_DIR}/gemini_router_3cls.json  gemini_router_4cls.json")

    if do_eval:
        print("\n=== Building eval data ===")
        for pair_key in args.pairs:
            per_problem = build_eval_for_pair(pair_key)
            write_cost_table(pair_key, per_problem)


if __name__ == "__main__":
    main()
