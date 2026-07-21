# -*- coding: utf-8 -*-
"""Collect N-vote rollouts for soft-label router training (Arch A, 3cls).

For each problem in the training set (router_arch_a_3cls_v3.json):
  1. Retrieve the fixed proceed failure (code_0 + stderr_0):
       - Matched (76%): reuse existing rollout record from cloudide/rollout/*.jsonl
       - Unmatched (24%): find problem via text index, run one fresh proceed call
  2. With (code_0, stderr_0) fixed, run reflect / replan / escalate N_VOTES times each.
     Per-trial label = cheapest action that passes (reflect > replan > escalate priority).
  3. Write vote counts per problem to a JSONL output file.

Output (one JSON line per problem):
  {
    "problem_id": "...",
    "dataset": "...",
    "problem_prompt": "...",
    "code_0": "...",
    "stderr_0": "...",
    "proceed_verdict": "fail" | "compile_error" | "timeout",
    "votes": {"reflect": k, "replan": k, "escalate": k, "unsolvable": k},
    "n_votes": 10,
    "n_solvable": k       # trials where at least one action passed
  }

Resumable: skips problem_ids already present in the output file.

Usage (from /mnt/bn/ecom-govern-models/qijiahe/cloudide/code):
    python scripts/collect_vote_rollouts.py \\
        --n_votes 10 \\
        --concurrency 50 \\
        --out outputs/votes/votes_3cls_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROLLOUT_DIR = Path("/mnt/bn/ecom-govern-models/qijiahe/cloudide/rollout")
SFT_DATA    = Path("/mnt/bn/ecom-govern-models/qijiahe/datasets/router_arch_a_3cls_v3.json")

DATASET_MAP = {
    "bcb":                ("bigcodebench", {}),
    "apps":               ("apps", {}),
    "apps_functional":    ("apps_functional", {}),
    "apps_lc":            ("apps", {}),
    "lcb":                ("livecodebench", {}),
    "taco_medhard":       ("taco", {"taco_difficulties": ["MEDIUM_HARD"]}),
    "taco_hard":          ("taco", {"taco_difficulties": ["HARD"]}),
    "taco_veryhard":      ("taco", {"taco_difficulties": ["VERY_HARD"]}),
    "taco_medium":        ("taco", {"taco_difficulties": ["MEDIUM"]}),
    "codecontests_train": ("codecontests", {}),
}

ACTIONS = ["reflect", "replan", "escalate"]
COST    = {"reflect": 2.0, "replan": 2.0, "escalate": 13.0}
PRIO    = ["reflect", "replan", "escalate"]   # tie-break order (same cost → reflect first)

# Thread-local storage for per-thread CodeAgent instances.
# CodeAgent.last_call_meta is not thread-safe — each thread needs its own agent.
_thread_local = threading.local()

# Semaphore to cap concurrent gpt-5.4 (escalate) calls.
# Set at startup via _init_escalate_semaphore(); default 5.
_escalate_sem: Optional[threading.Semaphore] = None

def _init_escalate_semaphore(max_concurrent: int):
    global _escalate_sem
    _escalate_sem = threading.Semaphore(max_concurrent)


# ---------------------------------------------------------------------------
# Rollout index: load existing failed-proceed rollouts
# ---------------------------------------------------------------------------

def _build_rollout_index() -> Dict[str, dict]:
    """Build prompt_prefix → rollout_record for all existing failed-proceed rollouts."""
    index: Dict[str, dict] = {}
    for fpath in sorted(ROLLOUT_DIR.glob("*.jsonl")):
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                calls = rec.get("calls") or []
                proceed = next((c for c in calls if c["action"] == "proceed"), None)
                if proceed and proceed.get("verdict") != "pass":
                    key = rec["problem_prompt"][:200]
                    if key not in index:   # first occurrence wins (matches dedup keep-first)
                        index[key] = rec
    return index


# ---------------------------------------------------------------------------
# Environment / problem index (for unmatched examples)
# ---------------------------------------------------------------------------

def _load_envs():
    from benchmarks.code_env import CodeEnv
    envs: Dict[str, object] = {}
    for tag, (ds_name, kwargs) in DATASET_MAP.items():
        try:
            envs[tag] = CodeEnv(ds_name, **kwargs)
        except Exception as e:
            print(f"[warn] could not load {tag}: {e}")
    return envs


def _build_problem_index(envs) -> Dict[str, Tuple]:
    """problem_id → (CodeProblem, env)"""
    idx: Dict[str, Tuple] = {}
    for env in envs.values():
        for p in env.problems:
            idx[p.task_id] = (p, env)
    return idx


def _build_text_index(envs) -> Dict[str, Tuple]:
    """prompt[:200] → (CodeProblem, env)"""
    idx: Dict[str, Tuple] = {}
    for env in envs.values():
        for p in env.problems:
            idx[p.prompt[:200]] = (p, env)
    return idx


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _get_agent(base_model: str, escalate_to: str):
    """Return a thread-local CodeAgent (creates on first access per thread)."""
    key = f"agent_{base_model}_{escalate_to}"
    if not hasattr(_thread_local, key):
        from agents.code_agent import CodeAgent
        setattr(_thread_local, key, CodeAgent(
            base_model=base_model,
            escalate_to=escalate_to,
        ))
    return getattr(_thread_local, key)


def _run_action(action: str, agent, env, problem,
                code_0: str, stderr_0: str) -> bool:
    """Run one recovery action and return True if verdict == pass.

    Retries up to 4 times on 429 rate-limit errors with exponential backoff.
    """
    import time
    from core.short_circuit_runner import _build_call_record
    if action == "reflect":
        fn = lambda: agent.reflect(problem.prompt, code_0, stderr_0)
    elif action == "replan":
        fn = lambda: agent.replan(problem.prompt, [code_0], [stderr_0])
    elif action == "escalate":
        fn = lambda: agent.escalate(problem.prompt)
    else:
        raise ValueError(f"unknown action: {action}")

    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            if action == "escalate" and _escalate_sem is not None:
                with _escalate_sem:
                    rec = _build_call_record(action, agent, env, problem, fn)
            else:
                rec = _build_call_record(action, agent, env, problem, fn)
            return rec["verdict"] == "pass"
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                time.sleep(wait)
                continue
            raise


def _run_proceed(agent, env, problem) -> Tuple[str, str, str]:
    """Run proceed; return (code_output, stderr, verdict)."""
    from core.short_circuit_runner import _build_call_record
    rec = _build_call_record("proceed", agent, env, problem,
                             lambda: agent.propose(problem.prompt))
    return rec["code_output"] or "", rec["stderr"] or "", rec["verdict"]


# ---------------------------------------------------------------------------
# Single-trial vote: run all 3 actions independently, return cheapest passing
# ---------------------------------------------------------------------------

def _one_trial(base_model, escalate_to, env, problem, code_0, stderr_0):
    """Short-circuit trial: try reflect → replan → escalate, stop at first pass.

    Cheaper than running all 3 independently: escalate is only called when
    both reflect and replan fail (~43% of trials on average).
    Vote = cheapest passing action, or 'unsolvable'.
    """
    for action in PRIO:
        agent = _get_agent(base_model, escalate_to)
        try:
            if _run_action(action, agent, env, problem, code_0, stderr_0):
                return action
        except Exception as e:
            print(f"[warn] {problem.task_id} {action}: {e}")
    return "unsolvable"


# ---------------------------------------------------------------------------
# Per-problem vote collection
# ---------------------------------------------------------------------------

def _collect_votes_for_problem(
    problem_id: str,
    dataset: str,
    problem_prompt: str,
    code_0: str,
    stderr_0: str,
    proceed_verdict: str,
    problem,        # CodeProblem
    env,
    n_votes: int,
    base_model: str,
    escalate_to: str,
) -> dict:
    """Run N_VOTES trials for one problem and return the vote record."""
    votes: Dict[str, int] = {"reflect": 0, "replan": 0, "escalate": 0, "unsolvable": 0}

    for _ in range(n_votes):
        result = _one_trial(base_model, escalate_to, env, problem, code_0, stderr_0)
        votes[result] += 1

    n_solvable = n_votes - votes["unsolvable"]
    return {
        "problem_id":      problem_id,
        "dataset":         dataset,
        "problem_prompt":  problem_prompt,
        "code_0":          code_0,
        "stderr_0":        stderr_0,
        "proceed_verdict": proceed_verdict,
        "votes":           votes,
        "n_votes":         n_votes,
        "n_solvable":      n_solvable,
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_done_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    done: Set[str] = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["problem_id"])
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Extract problem prompt from SFT input field
# ---------------------------------------------------------------------------

def _extract_problem_text(sft_input: str) -> str:
    """Strip 'Problem:\n' header and verdict/stderr tail, return pure problem text."""
    if sft_input.startswith("Problem:\n"):
        sft_input = sft_input[len("Problem:\n"):]
    # The verdict line is "\nInitial attempt verdict: ..." — strip it and everything after
    verdict_marker = "\nInitial attempt verdict:"
    idx = sft_input.find(verdict_marker)
    if idx >= 0:
        sft_input = sft_input[:idx]
    return sft_input.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", default=str(SFT_DATA))
    parser.add_argument("--n_votes", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Number of parallel problem workers.")
    parser.add_argument("--out", default="outputs/votes/votes_3cls_v3.jsonl")
    parser.add_argument("--base_model", default="gpt-5.4-nano")
    parser.add_argument("--escalate_to", default="gpt-5.4")
    parser.add_argument("--escalate_concurrency", type=int, default=0,
                        help="Max concurrent gpt-5.4 escalate calls (0=no limit, rely on worker count).")
    parser.add_argument("--filter_datasets", nargs="+", default=None,
                        help="Only process examples from these datasets (e.g. apps taco_hard).")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load SFT data ---
    with open(args.sft_data, encoding="utf-8") as f:
        sft_data = json.load(f)
    print(f"SFT examples: {len(sft_data)}")

    # --- Build rollout index ---
    print("Building rollout index ...")
    rollout_index = _build_rollout_index()
    print(f"Rollout index size: {len(rollout_index)}")

    # --- Match SFT examples to rollout records ---
    matched_items   = []   # (sft_ex, rollout_rec) — have existing code_0/stderr_0
    unmatched_items = []   # (sft_ex,) — need fresh proceed

    for ex in sft_data:
        prob_text = _extract_problem_text(ex["input"])
        key = prob_text[:200]
        rec = rollout_index.get(key)
        if rec:
            matched_items.append((ex, rec))
        else:
            unmatched_items.append(ex)

    print(f"Matched:   {len(matched_items)}")
    print(f"Unmatched: {len(unmatched_items)}")

    # --- Resume: skip already-done problem_ids ---
    done_ids = _load_done_ids(out_path)
    print(f"Already done: {len(done_ids)}")

    if args.dry_run:
        n_todo = sum(1 for _, rec in matched_items
                     if rec["problem_id"] not in done_ids)
        print(f"[dry_run] Would process {n_todo} matched examples "
              f"× {args.n_votes} votes ≈ {n_todo * args.n_votes * 2} API calls "
              f"(short-circuit: ~2 avg per trial)")
        print(f"Skipping {len(unmatched_items)} unmatched examples (no rollout data)")
        return

    # --- Init escalate semaphore (only if explicitly requested) ---
    if args.escalate_concurrency > 0:
        _init_escalate_semaphore(args.escalate_concurrency)
        print(f"Escalate concurrency cap: {args.escalate_concurrency}", flush=True)
    else:
        print(f"Escalate concurrency: unlimited (controlled by --concurrency)", flush=True)

    # --- Group matched items by dataset, ordered small→large for fast startup ---
    DATASET_ORDER = [
        "bcb", "codecontests_train", "apps_lc",
        "lcb", "apps_functional",
        "taco_medium", "taco_medhard", "taco_hard", "taco_veryhard",
        "apps",
    ]

    filter_ds = set(args.filter_datasets) if args.filter_datasets else None

    by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for ex, rollout_rec in matched_items:
        pid = rollout_rec["problem_id"]
        if pid in done_ids:
            continue
        if filter_ds and rollout_rec["dataset"] not in filter_ds:
            continue
        calls = rollout_rec.get("calls") or []
        proceed_call = next((c for c in calls if c["action"] == "proceed"), None)
        if not proceed_call:
            continue
        by_dataset[rollout_rec["dataset"]].append({
            "problem_id":      pid,
            "dataset":         rollout_rec["dataset"],
            "problem_prompt":  rollout_rec["problem_prompt"],
            "code_0":          proceed_call.get("code_output") or "",
            "stderr_0":        proceed_call.get("stderr") or "",
            "proceed_verdict": proceed_call.get("verdict", "fail"),
        })

    total_items = sum(len(v) for v in by_dataset.values())
    print(f"\nWork items: {total_items}  (skipping {len(unmatched_items)} unmatched)")
    print(f"Unmatched {len(unmatched_items)} examples skipped — no rollout data available\n")

    # --- Output file (append mode, line-buffered) ---
    out_file = out_path.open("a", encoding="utf-8", buffering=1)

    # --- Progress counters (shared across dataset batches) ---
    lock    = threading.Lock()
    n_done  = [0]
    n_err   = [0]
    t_start = time.time()

    def _process_one(item: dict) -> Optional[dict]:
        return _collect_votes_for_problem(
            problem_id=item["problem_id"],
            dataset=item["dataset"],
            problem_prompt=item["problem_prompt"],
            code_0=item["code_0"],
            stderr_0=item["stderr_0"],
            proceed_verdict=item["proceed_verdict"],
            problem=item["problem_obj"],
            env=item["env_obj"],
            n_votes=args.n_votes,
            base_model=args.base_model,
            escalate_to=args.escalate_to,
        )

    def _on_done(result: Optional[dict]):
        with lock:
            if result:
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
                n_done[0] += 1
            else:
                n_err[0] += 1
            total = n_done[0] + n_err[0]
            if total % 10 == 0 or total == total_items:
                elapsed = time.time() - t_start
                rate = total / elapsed * 60 if elapsed > 0 else 0
                eta = int((total_items - total) / rate * 60) if rate > 0 else 0
                print(f"  [{total}/{total_items}] done={n_done[0]} err={n_err[0]} "
                      f"rate={rate:.1f}/min ETA={eta}s", flush=True)

    # --- Load one dataset at a time, submit to shared pool immediately ---
    all_futures: List[Future] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        datasets_present = [ds for ds in DATASET_ORDER if ds in by_dataset]
        # also catch any dataset not in our order list
        datasets_present += [ds for ds in by_dataset if ds not in DATASET_ORDER]

        for ds_tag in datasets_present:
            items = by_dataset[ds_tag]
            print(f"Loading env: {ds_tag} ({len(items)} problems) ...", flush=True)
            try:
                from benchmarks.code_env import CodeEnv
                ds_name, kwargs = DATASET_MAP[ds_tag]
                env = CodeEnv(ds_name, **kwargs)
            except Exception as e:
                print(f"  [warn] failed to load {ds_tag}: {e}, skipping")
                continue

            # Build minimal pid_index for this env
            pid_index = {p.task_id: (p, env) for p in env.problems}
            print(f"  {ds_tag}: {len(pid_index)} problems indexed", flush=True)

            n_submitted = 0
            for item in items:
                result = pid_index.get(item["problem_id"])
                if result is None:
                    # text fallback within this env
                    key = item["problem_prompt"][:200]
                    result = next(
                        ((p, env) for p in env.problems if p.prompt[:200] == key),
                        None
                    )
                if result is None:
                    print(f"  [warn] {item['problem_id']} not in {ds_tag}, skipping")
                    continue
                item["problem_obj"], item["env_obj"] = result
                all_futures.append(pool.submit(_process_one, item))
                n_submitted += 1

            print(f"  {ds_tag}: submitted {n_submitted} jobs", flush=True)

        # Collect matched-batch results
        for fut in as_completed(all_futures):
            try:
                result = fut.result()
            except Exception as e:
                result = None
                print(f"[err] {e}\n{traceback.format_exc()}", flush=True)
            _on_done(result)

        # --- Phase 2: unmatched examples (need fresh proceed) ---
        # Reload done_ids to skip anything already written
        done_ids_now = _load_done_ids(out_path)
        unmatched_todo = [ex for ex in unmatched_items
                          if ex.get("_pid") not in done_ids_now]  # _pid unknown yet

        if unmatched_items:
            print(f"\n--- Phase 2: {len(unmatched_items)} unmatched examples "
                  f"(fresh proceed needed) ---", flush=True)
            print("Loading all envs for text search (from local cache) ...", flush=True)
            try:
                from benchmarks.code_env import CodeEnv
                all_envs = {}
                for tag, (ds_name, kwargs) in DATASET_MAP.items():
                    try:
                        all_envs[tag] = CodeEnv(ds_name, **kwargs)
                        print(f"  loaded {tag}", flush=True)
                    except Exception as e:
                        print(f"  [warn] {tag}: {e}", flush=True)

                # Build text index across all envs
                text_index: Dict[str, tuple] = {}
                for env in all_envs.values():
                    for p in env.problems:
                        text_index[p.prompt[:200]] = (p, env)
                print(f"Text index: {len(text_index)} problems", flush=True)

                def _process_unmatched(item):
                    agent = _get_agent(args.base_model, args.escalate_to)
                    try:
                        code_0, stderr_0, verdict = _run_proceed(
                            agent, item["env_obj"], item["problem_obj"])
                    except Exception as e:
                        print(f"[err] proceed {item['problem_id']}: {e}", flush=True)
                        return None
                    if verdict == "pass":
                        return None  # not a router case — discard
                    return _collect_votes_for_problem(
                        problem_id=item["problem_id"],
                        dataset=item["dataset"],
                        problem_prompt=item["problem_prompt"],
                        code_0=code_0, stderr_0=stderr_0,
                        proceed_verdict=verdict,
                        problem=item["problem_obj"], env=item["env_obj"],
                        n_votes=args.n_votes,
                        base_model=args.base_model, escalate_to=args.escalate_to,
                    )

                unmatched_futures = []
                for ex in unmatched_items:
                    prob_text = _extract_problem_text(ex["input"])
                    key = prob_text[:200]
                    found = text_index.get(key)
                    if found is None:
                        continue
                    problem_obj, env_obj = found
                    pid = problem_obj.task_id
                    if pid in done_ids_now:
                        continue
                    dataset_tag = next(
                        (t for t, e in all_envs.items() if e is env_obj), "unknown"
                    )
                    item = {
                        "problem_id":      pid,
                        "dataset":         dataset_tag,
                        "problem_prompt":  problem_obj.prompt,
                        "problem_obj":     problem_obj,
                        "env_obj":         env_obj,
                    }
                    unmatched_futures.append(pool.submit(_process_unmatched, item))

                print(f"Submitted {len(unmatched_futures)} unmatched jobs", flush=True)
                for fut in as_completed(unmatched_futures):
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = None
                        print(f"[err] {e}", flush=True)
                    _on_done(result)

            except Exception as e:
                print(f"[err] Phase 2 failed: {e}\n{traceback.format_exc()}", flush=True)

    out_file.close()

    elapsed = time.time() - t_start
    print(f"\nDone. {n_done[0]} problems written, {n_err[0]} errors, "
          f"{elapsed / 60:.1f} min elapsed")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
