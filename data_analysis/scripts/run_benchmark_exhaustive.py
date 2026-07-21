"""Exhaustive per-action benchmark evaluator.

For each example in datasets/benchmarks/*.json, evaluate ALL of
{reflect, replan, escalate} independently and store the full
successful_actions list.

Short-circuit rollout only tests escalate when reflect+replan both fail.
This script fills the gap: if reflect or replan already passed, it still
runs escalate to know whether escalate also works.

For examples with no rollout data (successful_actions=None), it runs
the full proceed → reflect + replan + escalate sequence.

Usage (from /mnt/bn/ecom-govern-models/qijiahe/cloudide/code):
    python scripts/run_benchmark_exhaustive.py \\
        --concurrency 8 \\
        --out_suffix _exhaustive

Output: updates benchmark files in-place (adds successful_actions).
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_generation"))

BENCH_DIR = Path("/mnt/bn/ecom-govern-models/qijiahe/datasets/benchmarks")

# rollout dataset tag → (CodeEnv dataset name, extra kwargs)
DATASET_MAP = {
    "bcb":               ("bigcodebench", {}),
    "apps":              ("apps", {}),
    "apps_functional":   ("apps_functional", {}),
    "apps_lc":           ("apps", {}),
    "lcb":               ("livecodebench", {}),
    "taco_medhard":      ("taco", {"taco_difficulties": ["MEDIUM_HARD"]}),
    "taco_hard":         ("taco", {"taco_difficulties": ["HARD"]}),
    "taco_veryhard":     ("taco", {"taco_difficulties": ["VERY_HARD"]}),
    "taco_medium":       ("taco", {"taco_difficulties": ["MEDIUM"]}),
    "codecontests_train":("codecontests", {}),
}


def _load_code_env_cls():
    path = ROOT / "data_generation" / "benchmarks" / "code_env.py"
    spec = importlib.util.spec_from_file_location("code_env_direct", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CodeEnv


def _load_code_agent_cls():
    path = ROOT / "data_generation" / "agents" / "code_agent.py"
    spec = importlib.util.spec_from_file_location("code_agent_direct", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CodeAgent


def _load_envs(needed_datasets: Set[str]) -> Dict[str, object]:
    CodeEnv = _load_code_env_cls()
    envs = {}
    for tag in needed_datasets:
        if tag not in DATASET_MAP:
            print(f"[warn] unknown dataset tag: {tag}, skipping")
            continue
        ds_name, kwargs = DATASET_MAP[tag]
        print(f"[env] loading {tag} ({ds_name}) ...")
        try:
            env = CodeEnv(ds_name, **kwargs)
            envs[tag] = env
            print(f"[env] {tag}: {len(list(env.problems))} problems")
        except Exception as e:
            print(f"[env] failed to load {tag}: {e}")
    return envs


def _build_problem_index(envs: Dict[str, object]) -> Dict[str, object]:
    """problem_id -> CodeProblem"""
    idx = {}
    for tag, env in envs.items():
        for p in env.problems:
            idx[p.task_id] = (p, env)
    return idx


def _build_text_index(envs: Dict[str, object]) -> Dict[str, Tuple]:
    """first-200-chars of prompt -> (CodeProblem, env)"""
    idx = {}
    for tag, env in envs.items():
        for p in env.problems:
            key = p.prompt[:200]
            idx[key] = (p, env)
    return idx


def _run_one_action(action: str, agent, env, problem, code_0: str, stderr_0: str) -> bool:
    """Run one action and return True if it passes."""
    from core.short_circuit_runner import _build_call_record
    if action == "reflect":
        fn = lambda: agent.reflect(problem.prompt, code_0, stderr_0)
    elif action == "replan":
        fn = lambda: agent.replan(problem.prompt, [code_0], [stderr_0])
    elif action == "escalate":
        fn = lambda: agent.escalate(problem.prompt)
    elif action == "proceed":
        fn = lambda: agent.propose(problem.prompt)
    else:
        raise ValueError(f"unknown action: {action}")
    rec = _build_call_record(action, agent, env, problem, fn)
    return rec["verdict"] == "pass", rec


def _exhaustive_eval(problem, env, agent) -> Dict:
    """Run proceed, then ALL of reflect/replan/escalate independently.
    Returns dict with per-action results."""
    results = {}

    # 1. proceed
    from core.short_circuit_runner import _build_call_record
    proc_rec = _build_call_record("proceed", agent, env, problem,
                                  lambda: agent.propose(problem.prompt))
    results["proceed"] = proc_rec["verdict"] == "pass"
    code_0 = proc_rec.get("code_output", "")
    stderr_0 = proc_rec.get("stderr", "")

    if results["proceed"]:
        # proceed passed — for archB this is a valid label
        # still run recovery for completeness
        pass

    # 2. reflect, replan, escalate — always run all three
    from concurrent.futures import ThreadPoolExecutor
    def _run(action):
        try:
            passed, _ = _run_one_action(action, agent, env, problem, code_0, stderr_0)
            return action, passed
        except Exception as e:
            print(f"[err] {problem.task_id} {action}: {e}")
            return action, None

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_run, a): a for a in ["reflect", "replan", "escalate"]}
        for fut in as_completed(futs):
            action, passed = fut.result()
            results[action] = passed

    successful = [a for a, ok in results.items() if ok]
    oracle_unsolvable = not any(results.values())
    return {
        "successful_actions": successful,
        "oracle_unsolvable": oracle_unsolvable,
        "per_action": results,
    }


def _escalate_only(problem, env, agent) -> bool:
    """Run just escalate and return pass/fail."""
    code = agent.escalate(problem.prompt)
    meta = agent.last_call_meta or {}
    if not meta.get("parseable", True):
        return False
    verdict, _ = env.step_verdict(problem, code)
    return verdict == "pass"


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

PROPOSE_SYSTEM = """\
You are an expert Python programmer.
You will be given a function signature and docstring. Implement the function.

Output ONLY:
1. All necessary `import` statements at the top (e.g. `import random`, `import re`,
   `from typing import List`, etc.) — include EVERY import the function uses.
2. The complete function definition (signature + body).

Do not include explanations, examples, test code, or markdown fencing.
The output must be a valid Python file that can be saved and imported as-is."""


def _strip_code_fence(text: str) -> str:
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _openai_completion(model: str, messages: list[dict], max_tokens: int = 8192) -> str:
    from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retry_on = (APIConnectionError, APITimeoutError, RateLimitError)
    waits = [0, 5, 10, 20, 40, 60]
    last_exc = None
    for wait in waits:
        if wait:
            time.sleep(wait)
        try:
            kwargs = {"model": model, "messages": messages}
            if "gpt-5" in model.lower() or model.startswith(("o3", "o4")):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["temperature"] = 0.0
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except retry_on as e:
            last_exc = e
    raise last_exc


def _escalate_only_openai(problem, env, model: str) -> bool:
    messages = [
        {"role": "system", "content": PROPOSE_SYSTEM},
        {"role": "user", "content": problem.prompt},
    ]
    raw = _openai_completion(model, messages)
    code = _strip_code_fence(raw)
    if not code or "def " not in code:
        return False
    verdict, _ = env.step_verdict(problem, code)
    return verdict == "pass"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", default=str(BENCH_DIR))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--base_model", default="gpt-5.4-nano")
    p.add_argument("--escalate_to", default="gpt-5.4")
    p.add_argument("--api_backend", default="bytedance",
                   choices=["bytedance", "openai"],
                   help="backend for new escalate-only calls")
    p.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
                   help="OpenAI model used when --api_backend=openai")
    p.add_argument("--dry_run", action="store_true",
                   help="print plan without running API calls")
    p.add_argument("--only_file", default=None,
                   help="process only this benchmark file (stem match)")
    p.add_argument("--only_dataset", default=None,
                   help="process only examples with this _dataset value")
    args = p.parse_args()

    bench_dir = Path(args.bench_dir)
    bench_files = sorted(bench_dir.glob("*.json"))
    if args.only_file:
        bench_files = [f for f in bench_files if args.only_file in f.name]

    # --- Collect all work items ---
    # work_item: {bench_file, idx, example, needs: "escalate_only" | "full"}
    work_items = []
    for bpath in bench_files:
        data = json.load(open(bpath))
        for i, ex in enumerate(data):
            sa = ex.get("successful_actions")
            if sa is None:
                # no rollout data — need full eval
                work_items.append({"file": bpath, "idx": i, "ex": ex, "needs": "full"})
            elif (("reflect" in sa or "replan" in sa)
                  and "escalate" not in sa
                  and not ex.get("_escalate_tested")):
                # Short-circuit stopped after a cheap action passed, so escalate
                # is still unknown. If _escalate_tested is set and escalate is
                # absent, it is a verified escalate failure and should not rerun.
                work_items.append({"file": bpath, "idx": i, "ex": ex, "needs": "escalate_only"})
            # else: already fully evaluated (escalate was tried)

    if args.only_dataset:
        work_items = [w for w in work_items if w["ex"].get("_dataset") == args.only_dataset]

    # Collect unique problem_ids needed
    needed_datasets = set()
    for w in work_items:
        ds = w["ex"].get("_dataset")
        if ds:
            needed_datasets.add(ds)
    # For null examples without _dataset, we'll need to search all datasets
    has_null = any(w["needs"] == "full" for w in work_items)

    print(f"\n=== Benchmark exhaustive eval plan ===")
    print(f"bench files: {len(bench_files)}")
    esc_only = sum(1 for w in work_items if w["needs"] == "escalate_only")
    full = sum(1 for w in work_items if w["needs"] == "full")
    print(f"total work items: {len(work_items)}  (escalate_only={esc_only}, full={full})")
    print(f"datasets needed: {sorted(needed_datasets)}")
    # cost estimate
    esc_cost = esc_only * 0.06   # gpt-5.4 escalate ~$0.06 avg
    full_cost = full * 0.08       # proceed + reflect + replan + escalate
    print(f"cost estimate: ~${esc_cost + full_cost:.1f}  "
          f"(escalate_only=${esc_cost:.1f}, full=${full_cost:.1f})")

    if args.dry_run:
        print("\n[dry_run] exiting without API calls")
        return

    def _text_key(ex):
        inp = ex.get("input", "")
        return inp.split("Problem:\n")[1][:150] if "Problem:\n" in inp else inp[:150]

    def _work_key(w):
        pid = w["ex"].get("_problem_id")
        return pid if pid else _text_key(w["ex"])

    def _apply_results(result_cache):
        from collections import defaultdict
        by_file = defaultdict(list)
        for w in work_items:
            by_file[w["file"]].append(w)

        for bpath, items in sorted(by_file.items()):
            data = json.load(open(bpath))
            print(f"\n[file] {bpath.name}  ({len(items)} items to update)")

            ok, fail = 0, 0
            for w in items:
                k = _work_key(w)
                result = result_cache.get(k)
                ex = data[w["idx"]]
                if result is None:
                    print(f"  [skip] idx={w['idx']} pid={ex.get('_problem_id','?')} key={k[:40]}")
                    fail += 1
                    continue

                if w["needs"] == "escalate_only":
                    if result.get("escalate") and "escalate" not in ex["successful_actions"]:
                        ex["successful_actions"].append("escalate")
                    ex["_escalate_tested"] = True
                    ok += 1
                else:
                    ex["successful_actions"] = result["successful_actions"]
                    ex["oracle_unsolvable"] = result["oracle_unsolvable"]
                    ex["_exhaustive"] = True
                    ok += 1

            print(f"  done: ok={ok} fail={fail}")
            with open(bpath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  saved -> {bpath}")

    # Escalate-only jobs can stream one dataset at a time. This avoids loading
    # all CodeEnv datasets simultaneously, which is too memory-heavy for APPS/TACO.
    if full == 0 and all(w["ex"].get("_dataset") for w in work_items):
        from collections import defaultdict
        CodeAgent = _load_code_agent_cls() if args.api_backend == "bytedance" else None

        by_dataset = defaultdict(list)
        for w in work_items:
            by_dataset[w["ex"]["_dataset"]].append(w)

        result_cache: Dict[str, dict] = {}
        total_ok, total_fail = 0, 0
        t_all = time.time()
        for ds, items in sorted(by_dataset.items()):
            print(f"\n[dataset] {ds}: {len(items)} work items")
            envs = _load_envs({ds})
            if ds not in envs:
                print(f"  [skip dataset] failed to load env for {ds}")
                for w in items:
                    result_cache[_work_key(w)] = None
                    total_fail += 1
                continue
            pid_index = _build_problem_index(envs)

            seen_keys: Set[str] = set()
            deduped = []
            for w in items:
                k = _work_key(w)
                if k not in seen_keys:
                    seen_keys.add(k)
                    deduped.append(w)
            print(f"  unique problems: {len(deduped)}")

            def _process_escalate_only(w):
                key = _work_key(w)
                found = pid_index.get(w["ex"].get("_problem_id"))
                if found is None:
                    return key, None, "no_problem_found"
                problem, env = found
                try:
                    if args.api_backend == "openai":
                        passed = _escalate_only_openai(problem, env, args.openai_model)
                    else:
                        agent = CodeAgent(base_model=args.base_model, escalate_to=args.escalate_to)
                        passed = _escalate_only(problem, env, agent)
                    return key, {"type": "escalate_only", "escalate": passed}, "ok"
                except Exception as e:
                    traceback.print_exc()
                    return key, None, f"error:{e}"

            ok_count, fail_count = 0, 0
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = {pool.submit(_process_escalate_only, w): w for w in deduped}
                for i, fut in enumerate(as_completed(futs)):
                    key, result, status = fut.result()
                    result_cache[key] = result
                    if result is None:
                        fail_count += 1
                        print(f"  [{i+1}/{len(deduped)}] FAIL key={key[:40]} reason={status}")
                    else:
                        ok_count += 1
                        if (i + 1) % 25 == 0 or (i + 1) == len(deduped):
                            elapsed = time.time() - t0
                            print(f"  [{i+1}/{len(deduped)}] ok={ok_count} fail={fail_count} elapsed={elapsed:.0f}s")
            total_ok += ok_count
            total_fail += fail_count
            del envs, pid_index

        print(f"\ncompleted streaming escalate-only: ok={total_ok} fail={total_fail} in {time.time()-t_all:.0f}s")
        _apply_results(result_cache)
        print("\n=== DONE ===")
        return

    # --- Load environments ---
    if has_null:
        # load all known datasets to do text-based search for null examples
        envs = _load_envs(set(DATASET_MAP.keys()))
    else:
        envs = _load_envs(needed_datasets)

    pid_index = _build_problem_index(envs)
    text_index = _build_text_index(envs)

    CodeAgent = _load_code_agent_cls()

    def _find_problem(ex):
        """Return (CodeProblem, env) or (None, None)."""
        pid = ex.get("_problem_id")
        if pid and pid in pid_index:
            return pid_index[pid]
        # text fallback
        inp = ex.get("input", "")
        prob_text = inp.split("Problem:\n")[1][:200] if "Problem:\n" in inp else inp[:200]
        if prob_text in text_index:
            return text_index[prob_text]
        return None, None

    # --- Deduplicate: run each unique problem once, apply results to all files ---
    # key: (problem_id or text_snippet) -> result cache
    result_cache: Dict[str, dict] = {}

    # deduplicate work items: keep one representative per unique problem
    seen_keys: Set[str] = set()
    deduped = []
    for w in work_items:
        k = _work_key(w)
        if k not in seen_keys:
            seen_keys.add(k)
            deduped.append(w)
    print(f"after dedup: {len(deduped)} unique problems to evaluate")

    def _process_deduped(w):
        ex = w["ex"]
        problem, env = _find_problem(ex)
        if problem is None:
            return _work_key(w), None, "no_problem_found"
        try:
            if w["needs"] == "escalate_only":
                if args.api_backend == "openai":
                    passed = _escalate_only_openai(problem, env, args.openai_model)
                else:
                    agent = CodeAgent(base_model=args.base_model, escalate_to=args.escalate_to)
                    passed = _escalate_only(problem, env, agent)
                return _work_key(w), {"type": "escalate_only", "escalate": passed}, "ok"
            else:
                agent = CodeAgent(base_model=args.base_model, escalate_to=args.escalate_to)
                result = _exhaustive_eval(problem, env, agent)
                result["type"] = "full"
                return _work_key(w), result, "ok"
        except Exception as e:
            traceback.print_exc()
            return _work_key(w), None, f"error:{e}"

    print(f"\nrunning {len(deduped)} unique problems (concurrency={args.concurrency})...")
    t0 = time.time()
    ok_count, fail_count = 0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(_process_deduped, w): w for w in deduped}
        for i, fut in enumerate(as_completed(futs)):
            key, result, status = fut.result()
            result_cache[key] = result
            if result is None:
                fail_count += 1
                print(f"  [{i+1}/{len(deduped)}] FAIL key={key[:40]} reason={status}")
            else:
                ok_count += 1
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(deduped)}] ok={ok_count} fail={fail_count} elapsed={elapsed:.0f}s")

    print(f"\ncompleted: ok={ok_count} fail={fail_count} in {time.time()-t0:.0f}s")

    # --- Apply results back to all benchmark files using the cache ---
    from collections import defaultdict
    by_file = defaultdict(list)
    for w in work_items:
        by_file[w["file"]].append(w)

    for bpath, items in sorted(by_file.items()):
        data = json.load(open(bpath))
        print(f"\n[file] {bpath.name}  ({len(items)} items to update)")

        ok, fail = 0, 0
        for w in items:
            k = _work_key(w)
            result = result_cache.get(k)
            ex = data[w["idx"]]
            if result is None:
                print(f"  [skip] idx={w['idx']} pid={ex.get('_problem_id','?')} key={k[:40]}")
                fail += 1
                continue

            if w["needs"] == "escalate_only":
                # result = {"type": "escalate_only", "escalate": bool}
                if result.get("escalate"):
                    ex["successful_actions"].append("escalate")
                ex["_escalate_tested"] = True
                ok += 1
            else:
                # result = {"type": "full", "successful_actions": [...], "oracle_unsolvable": bool, ...}
                ex["successful_actions"] = result["successful_actions"]
                ex["oracle_unsolvable"] = result["oracle_unsolvable"]
                ex["_exhaustive"] = True
                # fill in _problem_id if was null
                pid = ex.get("_problem_id")
                if not pid:
                    prob, _ = _find_problem(ex)
                    if prob:
                        ex["_problem_id"] = prob.task_id
                ok += 1

        print(f"  done: ok={ok} fail={fail}")
        with open(bpath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  saved -> {bpath}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
