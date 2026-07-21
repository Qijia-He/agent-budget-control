"""Collect raw 4-action rollouts via sequential short-circuit.

Usage:
    # smoke: 3 problems from TACO MED_HARD
    python scripts/collect_rollouts_opt.py \\
        --dataset taco --taco_difficulties MEDIUM_HARD \\
        --limit 3 \\
        --base_model gpt-5.4-nano --escalate_to gpt-5.4 \\
        --out outputs/rollouts/smoke.jsonl

    # full run on a dataset
    python scripts/collect_rollouts_opt.py \\
        --dataset taco --taco_difficulties HARD \\
        --limit 200 --concurrency 4 \\
        --base_model gpt-5.4-nano --escalate_to gpt-5.4 \\
        --out outputs/rollouts/taco_hard.jsonl

Output: 1 jsonl record per problem (raw schema). See
core/short_circuit_runner.py for record shape.
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_env_and_agent(args):
    from benchmarks.code_env import CodeEnv
    from agents.code_agent import CodeAgent

    env_kwargs = {"limit": args.limit}
    if args.dataset == "livecodebench":
        env_kwargs.update({
            "lcb_version": args.lcb_version,
            "lcb_date_after": args.lcb_date_after,
            "lcb_difficulty": args.lcb_difficulty,
        })
    elif args.dataset == "taco":
        env_kwargs["taco_difficulties"] = args.taco_difficulties
    elif args.dataset in ("apps", "apps_functional"):
        env_kwargs["apps_difficulties"] = args.apps_difficulties
    env = CodeEnv(args.dataset, **env_kwargs)
    agent_factory = lambda: CodeAgent(base_model=args.base_model,
                                      escalate_to=args.escalate_to)
    return env, agent_factory


def _load_done_ids(out_path: Path) -> Set[str]:
    """Resume: collect problem_ids already in out file."""
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["problem_id"])
            except Exception:
                pass
    return seen


def _format_pretty(rec: dict) -> str:
    """Multi-line human-readable formatting of one rollout record."""
    lines = []
    sep = "=" * 78
    lines.append(sep)
    lines.append(f"problem_id: {rec.get('problem_id')}")
    lines.append(f"dataset:    {rec.get('dataset')}")
    s = rec.get("summary", {})
    succ = ", ".join(s.get("successful_actions", []) or []) or "(unsolvable)"
    lines.append(f"successful: {succ}")
    lines.append(f"oracle_unsolvable: {s.get('oracle_unsolvable')}")
    lines.append(f"total: calls={s.get('total_api_calls')}, "
                 f"cost=${s.get('total_cost_usd', 0):.4f}, "
                 f"wall={s.get('total_wall_s', 0):.1f}s")
    lines.append("")
    for i, c in enumerate(rec.get("calls", [])):
        lines.append(f"--- CALL {i+1}: {c.get('action')} [{c.get('model')}] ---")
        lines.append(f"  verdict: {c.get('verdict')}")
        stderr = c.get("stderr") or ""
        if stderr:
            lines.append("  stderr:")
            for ln in stderr.split("\n")[:8]:   # keep first 8 lines of stderr
                lines.append(f"    {ln}")
        lines.append(f"  parse_retries: {c.get('parse_retries')}, "
                     f"wall_call: {c.get('wall_call_s')}s, "
                     f"wall_verify: {c.get('wall_verify_s')}s, "
                     f"cost: ${c.get('cost_usd', 0):.4f}")
        u = c.get("usage") or {}
        lines.append(f"  usage: prompt={u.get('prompt_tokens')}, "
                     f"completion={u.get('completion_tokens')}, "
                     f"reasoning_tokens={u.get('reasoning_tokens')}")
        lines.append("")
    d = rec.get("diagnose")
    if d:
        lines.append(f"--- DIAGNOSE [{d.get('model', '?')}] (parse_ok={d.get('parse_ok')}) ---")
        lines.append("  failure_reason:")
        for ln in (d.get("failure_reason") or "").split("\n"):
            lines.append(f"    {ln}")
        lines.append(f"  recommended_action: {d.get('recommended_action')}")
        u = d.get("usage") or {}
        lines.append(f"  cost: ${d.get('cost_usd', 0):.4f}, wall: {d.get('wall_s', 0):.1f}s; "
                     f"usage: prompt={u.get('prompt_tokens')}, completion={u.get('completion_tokens')}")
        lines.append("")
    if "error" in rec:
        lines.append(f"!! ERROR: {rec['error']}")
        lines.append("")
    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["humaneval", "bigcodebench", "mbpp",
                            "livecodebench", "taco",
                            "apps", "apps_functional", "codecontests", "classeval",
                            "ds1000", "scicode", "humanevalfix", "debugbench"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=1,
                   help="thread-pool worker count (1 = sequential)")
    p.add_argument("--base_model", default="gpt-5.4-nano",
                   choices=["gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano"])
    p.add_argument("--escalate_to", default="gpt-5.4",
                   choices=["gpt-4.1", "gpt-5", "gpt-5.4", "gpt-5.4-mini"])
    # dataset-specific filters
    p.add_argument("--lcb_version", default="release_v6")
    p.add_argument("--lcb_date_after", default=None)
    p.add_argument("--lcb_difficulty", default=None)
    p.add_argument("--taco_difficulties", nargs="+", default=None,
                   choices=["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"])
    p.add_argument("--apps_difficulties", nargs="+", default=None,
                   choices=["introductory", "interview", "competition"])
    p.add_argument("--dataset_tag", default="",
                   help="tag stored inside each record (e.g. taco_medhard)")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(out_path)
    if done_ids:
        print(f"[resume] {len(done_ids)} problems already in {out_path}; skipping.")

    env, agent_factory = _build_env_and_agent(args)
    problems = list(env.problems)
    print(f"[loaded] {len(problems)} problems from dataset={args.dataset}")

    # filter out already-done
    todo = [p for p in problems if p.task_id not in done_ids]
    print(f"[todo]   {len(todo)} problems")

    from core.short_circuit_runner import run_short_circuit_rollout

    if args.dataset_tag:
        dataset_tag = args.dataset_tag
    elif args.dataset == "taco" and args.taco_difficulties:
        dataset_tag = f"taco_{'_'.join(args.taco_difficulties).lower()}"
    elif args.dataset == "apps" and args.apps_difficulties:
        dataset_tag = f"apps_{'_'.join(args.apps_difficulties).lower()}"
    else:
        dataset_tag = args.dataset

    # one agent per worker (last_call_meta is instance attr, not thread-safe shared)
    def _run_one(problem, agent):
        try:
            rec = run_short_circuit_rollout(
                problem_id=problem.task_id,
                problem_prompt=problem.prompt,
                problem=problem,
                agent=agent,
                env=env,
                dataset_tag=dataset_tag,
            )
            return rec, None
        except Exception as e:
            return None, (str(e), traceback.format_exc())

    t0 = time.time()
    n_done = 0
    cumulative_cost = 0.0
    cumulative_calls = 0
    pretty_path = out_path.parent / (out_path.name + ".pretty.txt")  # bcb.jsonl → bcb.jsonl.pretty.txt
    with open(out_path, "a") as fout, open(pretty_path, "a") as fpretty:
        if args.concurrency <= 1:
            agent = agent_factory()
            for prob in todo:
                rec, err = _run_one(prob, agent)
                _emit_progress(fout, fpretty, rec, err, prob, t0, args, len(todo))
                if rec:
                    n_done += 1
                    cumulative_cost += rec["summary"].get("total_cost_usd", 0)
                    cumulative_calls += rec["summary"].get("total_api_calls", 0)
        else:
            # one agent per worker
            agents = [agent_factory() for _ in range(args.concurrency)]
            from itertools import cycle
            agent_pool = cycle(agents)
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(_run_one, prob, next(agent_pool)): prob
                        for prob in todo}
                for fut in as_completed(futs):
                    prob = futs[fut]
                    rec, err = fut.result()
                    _emit_progress(fout, fpretty, rec, err, prob, t0, args, len(todo))
                    if rec:
                        n_done += 1
                        cumulative_cost += rec["summary"].get("total_cost_usd", 0)
                        cumulative_calls += rec["summary"].get("total_api_calls", 0)

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY ===")
    print(f"Done:    {n_done}/{len(todo)} problems in {elapsed:.0f}s "
          f"({elapsed/max(n_done,1):.1f}s/题)")
    print(f"API calls: {cumulative_calls} ({cumulative_calls/max(n_done,1):.1f}/题)")
    print(f"Cost:    ${cumulative_cost:.4f} (${cumulative_cost/max(n_done,1)*1000:.3f}/1000 题)")


def _emit_progress(fout, fpretty, rec, err, prob, t0, args, total):
    """Write rec or error stub to fout (jsonl) + fpretty (human-readable) + print progress."""
    if rec is not None:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        fpretty.write(_format_pretty(rec))
        fpretty.flush()
        s = rec["summary"]
        succ = "+".join(s["successful_actions"]) if s["successful_actions"] else \
               ("unsolv" if s["oracle_unsolvable"] else "?")
        print(f"  [{prob.task_id:38s}] {succ:18s} calls={s['total_api_calls']} "
              f"cost=${s['total_cost_usd']:.4f} wall={s['total_wall_s']:.1f}s")
    else:
        err_msg, tb = err
        stub = {
            "problem_id": prob.task_id,
            "dataset": args.dataset_tag or args.dataset,
            "error": err_msg[:500],
        }
        fout.write(json.dumps(stub) + "\n")
        fout.flush()
        fpretty.write(_format_pretty(stub))
        fpretty.flush()
        print(f"  [{prob.task_id:38s}] ERROR: {err_msg[:80]}")


if __name__ == "__main__":
    main()
