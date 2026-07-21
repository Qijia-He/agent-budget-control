"""Optimized 4-action rollout via sequential short-circuit.

Sequential order: proceed → (reflect + replan, parallel-OK) → escalate.
Stops early whenever cheapest-action label can be determined.

Per-problem average ~2.0-3.1 API calls (vs 5.3-6.3 for naive 4-independent rollout),
~50-62% savings depending on dataset difficulty.

Returns a raw record per problem with full per-call details (verdict, stderr,
tokens, cost). SFT label derivation is downstream (build_training_data.py).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from agents.code_agent import CodeAgent
from core.pricing import compute_cost_usd


# Action → model used (depends on agent setup).
# proceed/reflect/replan use base model; escalate uses escalate target.
_BASE_ACTIONS = ("proceed", "reflect", "replan")
_ESC_ACTIONS = ("escalate",)

# 短名 → 带日期 full model id (跟 pricing.py 对齐)
_SHORT_TO_FULL_MODEL = {
    "gpt-5-mini":    "gpt-5-mini-2025-08-07",
    "gpt-5.4-mini":  "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano":  "gpt-5.4-nano-2026-03-17",
    "gpt-4.1":       "gpt-4.1-2025-04-14",
    "gpt-5":         "gpt-5-2025-08-07",
    "gpt-5.4":       "gpt-5.4-2026-03-05",
}


def _truncate(s: Optional[str], n: int = 1500) -> str:
    """Safely truncate string to N chars (for jsonl size hygiene)."""
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + f"... [truncated, +{len(s)-n} chars]"


def _model_for_action(agent: CodeAgent, action: str) -> str:
    """Return the full model id used for a given action."""
    short = agent.base_model if action in _BASE_ACTIONS else agent.escalate_to
    return _SHORT_TO_FULL_MODEL.get(short, short)


def _build_call_record(action: str, agent: CodeAgent, env, problem,
                       call_fn) -> Dict[str, Any]:
    """Run one agent action + env verification, return raw call record."""
    t0 = time.time()
    code = call_fn()
    wall_call = time.time() - t0   # LLM call wall (excl. test runner)
    meta = agent.last_call_meta or {}
    raw_text = meta.get("raw_text", code)
    parse_retries = max((meta.get("n_calls", 1)) - 1, 0)
    parseable = meta.get("parseable", True)
    usage = meta.get("usage")
    model_id = _model_for_action(agent, action)

    # Run env verdict
    t_verify_0 = time.time()
    if not parseable:
        verdict = "parse_error"
        stderr = "ParseError: no valid code block produced after retry"
    else:
        verdict, stderr = env.step_verdict(problem, code)
    wall_verify = time.time() - t_verify_0

    return {
        "action": action,
        "model": model_id,
        "code_output": _truncate(code, 4000),       # parsed code (clean)
        "raw_response": _truncate(raw_text, 2000),  # raw LLM output (with fences)
        "reasoning_content": _truncate(meta.get("reasoning_content"), 8000),  # reasoning trace if exposed
        "verdict": verdict,
        "stderr": _truncate(stderr, 1500),
        "parse_retries": parse_retries,
        "wall_call_s": round(wall_call, 2),
        "wall_verify_s": round(wall_verify, 2),
        "usage": usage,
        "cost_usd": compute_cost_usd(model_id, usage),
    }


def _run_diagnose(agent: CodeAgent, problem_prompt: str, prev_code: str,
                  stderr: str) -> Dict[str, Any]:
    """Auxiliary diagnose call after proceed fail. Captures reasoning for
    later analysis / optional SFT enrichment. Doesn't affect recovery actions.

    Judge model: agent.escalate_to (gpt-5.4) — stronger than base nano.
    """
    t0 = time.time()
    out = agent.diagnose(problem_prompt, prev_code, stderr)
    wall = time.time() - t0
    # diagnose uses escalate (judge) model — full model id needed for pricing
    judge_short = out.get("judge_model") or agent.escalate_to
    model_id = _SHORT_TO_FULL_MODEL.get(judge_short, judge_short)
    return {
        "model": model_id,
        "failure_reason": out["failure_reason"],
        "recommended_action": out["recommended_action"],
        "raw_output": _truncate(out["raw_output"], 1500),
        "reasoning_content": _truncate(out.get("reasoning_content"), 8000),
        "parse_ok": out["parse_ok"],
        "wall_s": round(wall, 2),
        "usage": out["usage"],
        "cost_usd": compute_cost_usd(model_id, out["usage"]),
    }


def run_short_circuit_rollout(problem_id: str, problem_prompt: str,
                              problem, agent: CodeAgent, env,
                              dataset_tag: str = "",
                              collect_diagnose: bool = True) -> Dict[str, Any]:
    """Run one optimized 4-action rollout for a single problem.

    Sequence:
      1. proceed (always)
         if pass → STOP, label candidates = {proceed}
      2. (optional) diagnose — short JSON analysis of failure
      3. proceed failed → run reflect + replan
         if both pass → label candidates = {reflect, replan} (cost tie)
         if only one passes → that one
      4. if both reflect & replan failed → run escalate
         if escalate pass → label = {escalate}, oracle_unsolvable=False
         else → label = {escalate}, oracle_unsolvable=True

    diagnose 不影响 recovery 决策, 纯属 metadata 收集. 由 collect_diagnose
    控制是否跑.

    Returns record with all per-action details + summary.
    """
    t_total_0 = time.time()
    calls: List[Dict[str, Any]] = []
    diagnose_rec: Optional[Dict[str, Any]] = None

    # 1) proceed
    c1 = _build_call_record(
        "proceed", agent, env, problem,
        lambda: agent.propose(problem_prompt),
    )
    calls.append(c1)
    code_0 = c1["code_output"]
    stderr_0 = c1["stderr"]

    if c1["verdict"] == "pass":
        return _finalize(problem_id, dataset_tag, problem_prompt, calls,
                         successful=["proceed"], oracle_unsolvable=False,
                         t_total_0=t_total_0, diagnose=diagnose_rec)

    # 1.5) diagnose (single call, captures reasoning even when API doesn't expose internal)
    if collect_diagnose:
        try:
            diagnose_rec = _run_diagnose(agent, problem_prompt, code_0, stderr_0)
        except Exception as e:
            diagnose_rec = {
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "failure_reason": "",
                "recommended_action": "unknown",
                "parse_ok": False,
            }

    # 2) reflect + replan (cost-tie check)
    c2 = _build_call_record(
        "reflect", agent, env, problem,
        lambda: agent.reflect(problem_prompt, code_0, stderr_0),
    )
    calls.append(c2)
    c3 = _build_call_record(
        "replan", agent, env, problem,
        lambda: agent.replan(problem_prompt, [code_0], [stderr_0]),
    )
    calls.append(c3)

    successful = [c["action"] for c in (c2, c3) if c["verdict"] == "pass"]
    if successful:
        return _finalize(problem_id, dataset_tag, problem_prompt, calls,
                         successful=successful, oracle_unsolvable=False,
                         t_total_0=t_total_0, diagnose=diagnose_rec)

    # 3) all cheap actions failed → escalate to determine oracle_unsolvable
    c4 = _build_call_record(
        "escalate", agent, env, problem,
        lambda: agent.escalate(problem_prompt),
    )
    calls.append(c4)
    if c4["verdict"] == "pass":
        return _finalize(problem_id, dataset_tag, problem_prompt, calls,
                         successful=["escalate"], oracle_unsolvable=False,
                         t_total_0=t_total_0, diagnose=diagnose_rec)
    else:
        return _finalize(problem_id, dataset_tag, problem_prompt, calls,
                         successful=[], oracle_unsolvable=True,
                         t_total_0=t_total_0, diagnose=diagnose_rec)


def _finalize(problem_id, dataset_tag, problem_prompt, calls,
              successful, oracle_unsolvable, t_total_0,
              diagnose: Optional[Dict[str, Any]] = None):
    # Each call record corresponds to ≥1 API call (parse_retries adds extras)
    total_api_calls = sum(1 + c.get("parse_retries", 0) for c in calls)
    total_cost_usd = sum((c.get("cost_usd") or 0.0) for c in calls)
    if diagnose:
        total_api_calls += 1
        total_cost_usd += (diagnose.get("cost_usd") or 0.0)
    total_wall_s = round(time.time() - t_total_0, 2)
    rec = {
        "problem_id": problem_id,
        "dataset": dataset_tag,
        "problem_prompt": _truncate(problem_prompt, 6000),
        "calls": calls,
        "diagnose": diagnose,                        # None when proceed pass
        "summary": {
            "successful_actions": successful,
            "oracle_unsolvable": oracle_unsolvable,
            "total_api_calls": total_api_calls,
            "total_cost_usd": round(total_cost_usd, 6),
            "total_wall_s": total_wall_s,
        },
    }
    return rec
