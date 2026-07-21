"""Build SFT data for the no-reasoning router (paper #1, v1 ablation).

Router only fires when proceed FAILS (proceed-pass exits the cascade). So we
filter to records where the proceed call did not pass — those are the only
inputs the router will ever see at deployment.

Resumable: tracks processed (dataset_tag, problem_id) pairs in a state file.
Re-running picks up only new rollout records appended since the last run.

Produces 2 variants, both in LlamaFactory Alpaca format (instruction/input/output):

  router_no_reason_v1_3cls.json   — drop oracle_unsolvable; only solvable-by-recovery
                                    classes: reflect / replan / escalate
  router_no_reason_v1_4cls.json   — keep oracle_unsolvable as its own class
                                    classes: reflect / replan / escalate / unsolvable

Input construction (both variants identical):
  - Problem prompt (truncated to PROMPT_CHARS = 6000)
  - "Initial attempt verdict: <fail|timeout|compile_error>"
  - stderr, capped:
      * verdict=fail  → cap at 800 chars
      * verdict=compile_error/timeout → cap at 400 chars

Output: one word — reflect / replan / escalate (or unsolvable in 4cls).

Usage:
    python scripts/build_sft_router_v1.py            # incremental (default)
    python scripts/build_sft_router_v1.py --rebuild  # full rebuild from scratch

Outputs (in outputs/sft/):
  router_no_reason_v1_4cls.json
  router_no_reason_v1_5cls.json
  .build_state.json              — resume state (do not delete unless rebuilding)
  dataset_info.snippet.json      — paste into LlamaFactory data/dataset_info.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

ROLLOUT_DIR = Path("outputs/rollouts/v55")
OUT_DIR = Path("outputs/sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_3CLS = OUT_DIR / "router_no_reason_v1_3cls.json"
OUT_4CLS = OUT_DIR / "router_no_reason_v1_4cls.json"
STATE_PATH = OUT_DIR / ".build_state.json"
INFO_PATH = OUT_DIR / "dataset_info.snippet.json"

PROMPT_CHARS = 6000
STDERR_CAP_FAIL = 800
STDERR_CAP_OTHER = 400

COST = {"proceed": 1.0, "reflect": 2.0, "replan": 2.0, "escalate": 13.0}
PRIO = ["proceed", "reflect", "replan", "escalate"]

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


def cheapest_action(rec):
    """Cheapest action whose verdict==pass. None if oracle_unsolvable."""
    succ = rec["summary"]["successful_actions"]
    if not succ:
        return None
    return min(succ, key=lambda a: (COST[a], PRIO.index(a)))


def trim_stderr(verdict: str, stderr: str) -> str:
    """Router only sees failure cases — verdict=pass is filtered upstream."""
    s = (stderr or "").strip()
    if not s:
        return ""
    cap = STDERR_CAP_FAIL if verdict == "fail" else STDERR_CAP_OTHER
    if len(s) > cap:
        s = s[:cap] + "\n... [truncated]"
    return s


def build_input(rec) -> str:
    prompt = rec["problem_prompt"]
    if len(prompt) > PROMPT_CHARS:
        prompt = prompt[:PROMPT_CHARS] + "\n... [problem truncated]"
    c0 = rec["calls"][0]   # always proceed
    verdict = c0["verdict"]
    stderr = trim_stderr(verdict, c0.get("stderr") or "")

    parts = [
        "Problem:",
        prompt,
        "",
        f"Initial attempt verdict: {verdict}",
    ]
    if stderr:
        parts.append("Initial attempt stderr:")
        parts.append(stderr)
    return "\n".join(parts)


def iter_records():
    """Yield (dataset_tag, line_idx, rec) in deterministic file × line-order order."""
    for f in sorted(ROLLOUT_DIR.glob("*.jsonl")):
        with f.open() as fp:
            for i, line in enumerate(fp):
                if not line.strip():
                    continue
                r = json.loads(line)
                if "calls" not in r or not r["calls"]:
                    continue   # error stub
                yield f.stem, i, r


def load_existing(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore .build_state.json and rebuild from scratch")
    args = ap.parse_args()

    # ---- Load state + existing outputs ----
    if args.rebuild or not STATE_PATH.exists():
        state = {"processed_keys": []}
        existing_3 = []
        existing_4 = []
        mode = "rebuild" if args.rebuild else "fresh"
    else:
        state = json.loads(STATE_PATH.read_text())
        existing_3 = load_existing(OUT_3CLS)
        existing_4 = load_existing(OUT_4CLS)
        mode = "incremental"

    processed = set(state.get("processed_keys", []))
    t0 = time.time()
    print(f"[{mode}] state has {len(processed)} previously-processed keys; "
          f"existing 3cls={len(existing_3)} 4cls={len(existing_4)}")

    new_3 = []
    new_4 = []
    counter_new_3 = Counter()
    counter_new_4 = Counter()
    by_dataset_new = Counter()
    n_seen = 0
    n_skip = 0
    n_filt_proceed = 0

    for dataset_tag, line_idx, rec in iter_records():
        n_seen += 1
        key = f"{dataset_tag}::{rec['problem_id']}"
        if key in processed:
            n_skip += 1
            continue
        # Router never invoked on proceed-pass — drop these examples.
        verdict = rec["calls"][0]["verdict"]
        if verdict == "pass":
            processed.add(key)   # remember it so resume skips it next time too
            n_filt_proceed += 1
            continue

        action = cheapest_action(rec)
        input_text = build_input(rec)

        if action is not None:
            # solvable by some recovery action (reflect/replan/escalate) — both variants
            new_3.append({
                "instruction": INSTRUCTION_3CLS,
                "input": input_text,
                "output": action,
            })
            counter_new_3[action] += 1
            new_4.append({
                "instruction": INSTRUCTION_4CLS,
                "input": input_text,
                "output": action,
            })
            counter_new_4[action] += 1
        else:
            # oracle_unsolvable — only 4cls keeps these
            new_4.append({
                "instruction": INSTRUCTION_4CLS,
                "input": input_text,
                "output": "unsolvable",
            })
            counter_new_4["unsolvable"] += 1

        by_dataset_new[dataset_tag] += 1
        processed.add(key)

    # ---- Combine + write ----
    out_3 = existing_3 + new_3
    out_4 = existing_4 + new_4

    OUT_3CLS.write_text(json.dumps(out_3, ensure_ascii=False, indent=2))
    OUT_4CLS.write_text(json.dumps(out_4, ensure_ascii=False, indent=2))

    state["processed_keys"] = sorted(processed)
    state["last_run_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["last_run_added_3cls"] = len(new_3)
    state["last_run_added_4cls"] = len(new_4)
    state["last_run_filtered_proceed"] = n_filt_proceed
    STATE_PATH.write_text(json.dumps(state, indent=2))

    # ---- LlamaFactory dataset_info snippet ----
    snippet = {
        "router_no_reason_v1_3cls": {
            "file_name": "router_no_reason_v1_3cls.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"}
        },
        "router_no_reason_v1_4cls": {
            "file_name": "router_no_reason_v1_4cls.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"}
        }
    }
    INFO_PATH.write_text(json.dumps(snippet, indent=2))

    # ---- Stats ----
    print(f"\n[{mode}] seen={n_seen}, skipped={n_skip} (already processed), "
          f"filtered-proceed={n_filt_proceed}, "
          f"added 3cls={len(new_3)}, added 4cls={len(new_4)}")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    print(f"\n=== {OUT_3CLS.name}: {len(out_3)} examples ===")
    full_3 = Counter(s["output"] for s in out_3)
    for lab in ["reflect", "replan", "escalate"]:
        n = full_3[lab]
        pct = 100*n/len(out_3) if out_3 else 0
        delta = counter_new_3[lab]
        print(f"  {lab:10s} {n:>6d} ({pct:5.1f}%)" +
              (f"   (+{delta} new)" if delta else ""))

    print(f"\n=== {OUT_4CLS.name}: {len(out_4)} examples ===")
    full_4 = Counter(s["output"] for s in out_4)
    for lab in ["reflect", "replan", "escalate", "unsolvable"]:
        n = full_4[lab]
        pct = 100*n/len(out_4) if out_4 else 0
        delta = counter_new_4[lab]
        print(f"  {lab:10s} {n:>6d} ({pct:5.1f}%)" +
              (f"   (+{delta} new)" if delta else ""))

    if by_dataset_new:
        print(f"\n  New records by source dataset:")
        for ds, c in by_dataset_new.most_common():
            print(f"    {ds:18s} {c:>5d}")

    # ---- Input length stats (full datasets) ----
    def pct(lst, q):
        s = sorted(lst); return s[min(int(len(s)*q), len(s)-1)]
    if out_3:
        lengths = [len(r["input"]) for r in out_3]
        print(f"\nInput length (chars) — 3cls: "
              f"p50={pct(lengths,0.5)}, p90={pct(lengths,0.9)}, "
              f"p99={pct(lengths,0.99)}, max={max(lengths)}")
    if out_4:
        lengths = [len(r["input"]) for r in out_4]
        print(f"Input length (chars) — 4cls: "
              f"p50={pct(lengths,0.5)}, p90={pct(lengths,0.9)}, "
              f"p99={pct(lengths,0.99)}, max={max(lengths)}")

    print(f"\nWritten:")
    print(f"  {OUT_3CLS}")
    print(f"  {OUT_4CLS}")
    print(f"  {STATE_PATH}  (resume state; --rebuild to ignore)")
    print(f"  {INFO_PATH}  (paste into LlamaFactory data/dataset_info.json)")


if __name__ == "__main__":
    main()
