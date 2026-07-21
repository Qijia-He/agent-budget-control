"""Domain-agnostic cascade runner.

每个 task domain (code / math / judge / ...) 实现三个 protocol:
    - Verifier:  verify(problem, solution) -> (success: bool, error_msg: str)
    - Agent:     propose / reflect / replan / escalate, 各返回 solution string
    - Scorer:    score(problem, solution) -> P(BAD) ∈ [0, 1]

run_problem(condition, problem, agent, verifier, scorer, taus, ...) 是 7-condition
ConfoReAct 实验的通用入口, **跨 domain 复用** —— 新加 domain 不改这里, 只加 3 个
domain-specific plug-in.

Condition:
  A   vanilla: 只用 mini propose, 不 retry
  B1  reflect-only: propose → 失败 → reflect (mini)
  B2  replan-only:  propose → 失败 → replan  (mini)
  B3  escalate-only: propose → 失败 → escalate (full)
  B   full cascade: propose → 失败 → reflect → 失败 → replan → 失败 → escalate
  C   random gate: 随机 X% 走 B 的完整 cascade, 否则同 A
  D   CP-gated:    score 算出来后, 跟 (τ_proceed, τ_reflect, τ_replan) 比, 选 tier
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol


# --------- Cost weights ---------
# 精确 per-call cost (nano-units), 来自 OpenAI 内部 pricing (~2k input + 2k output blended):
#   gpt-5.4-nano : gpt-5.4-mini : gpt-5.4 ≈ 1.0 : 3.6 : 12.1
#   (gpt-4.1 ≈ 5.0 在老 ecosystem; gpt-5 ≈ 12 同 gpt-5.4 tier)
#
# cascade pair → (COST_MINI, COST_FULL):
#   (gpt-5.4-nano, gpt-5.4)       → (1, 12)    ← v55 default, 12x gap
#   (gpt-5.4-nano, gpt-5.4-mini)  → (1, 4)     ← (nano, mini) 紧凑对比
#   (gpt-5.4-mini, gpt-5.4)       → (1, 3)     ← v54 用过
#   (gpt-5-mini,  gpt-4.1)        → (1, 5)     ← 老 v30 setting
#
# 默认设为 v55 (nano, 5.4); 跑其他 pair 时 export COST_FULL=N 覆盖.
import os as _os
COST_MINI = int(_os.environ.get("COST_MINI", "1"))
COST_FULL = int(_os.environ.get("COST_FULL", "12"))


# --------- Protocol interfaces (任何 domain 实现这些就能跑) ---------
class Verifier(Protocol):
    def verify(self, problem: Any, solution: str) -> tuple[bool, str]: ...


class Agent(Protocol):
    def propose(self, problem_prompt: str) -> str: ...
    def reflect(self, problem_prompt: str, prev_solution: str, error: str) -> str: ...
    def replan(self, problem_prompt: str, prev_solutions: List[str], errors: List[str]) -> str: ...
    def escalate(self, problem_prompt: str) -> str: ...


class Scorer(Protocol):
    def score(self, problem_prompt: str, solution: str) -> float: ...


@dataclass
class TaskResult:
    problem_id: str
    condition: str
    score: float | None = None
    success: bool = False
    cost: int = 0           # mini-unit
    n_calls_mini: int = 0
    n_calls_full: int = 0
    trace: List[Dict] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


# --------- Helper: 单步 verify + 记录 ---------
def _try(rec: TaskResult, tier: str, solution: str, problem: Any, verifier: Verifier,
         is_full: bool = False) -> tuple[bool, str]:
    """跑 verifier, 累计 cost, append trace, 返回 (success, error)."""
    succ, err = verifier.verify(problem, solution)
    if is_full:
        rec.n_calls_full += 1
        rec.cost += COST_FULL
    else:
        rec.n_calls_mini += 1
        rec.cost += COST_MINI
    rec.trace.append({
        "tier": tier,
        "solution": solution[:600],   # 截短防 jsonl 太大
        "success": succ,
        "error": (err or "")[:500],
    })
    return succ, err


# --------- 主入口: 7 condition dispatch ---------
def run_problem(
    condition: str,
    problem: Any,
    problem_id: str,
    problem_prompt: str,
    agent: Agent,
    verifier: Verifier,
    scorer: Scorer | None = None,
    taus: Dict[str, float] | None = None,
    random_rate: float = 0.3,
) -> TaskResult:
    """通用 7-condition runner. taus = {tau_proceed, tau_reflect, tau_replan}."""
    rec = TaskResult(problem_id=problem_id, condition=condition)
    taus = taus or {}

    # ------ Stage 1: 一次 propose + 一次 score (always, 给 calibration 用) ------
    code = agent.propose(problem_prompt)
    succ, err = _try(rec, "proceed", code, problem, verifier, is_full=False)

    # 即使 propose 成功也算一次 score (calibration data 需要全 problem 都有 score)
    if scorer is not None:
        rec.score = scorer.score(problem_prompt, code)
        rec.n_calls_mini += 1
        rec.cost += COST_MINI

    if succ:
        rec.success = True
        return rec

    # ------ Stage 2: 失败后按 condition 决定走哪些 tier ------

    if condition == "A":
        return rec   # 不 retry

    if condition == "B1":
        new_sol = agent.reflect(problem_prompt, code, err)
        succ, _ = _try(rec, "reflect", new_sol, problem, verifier, is_full=False)
        rec.success = succ
        return rec

    if condition == "B2":
        new_sol = agent.replan(problem_prompt, [code], [err])
        succ, _ = _try(rec, "replan", new_sol, problem, verifier, is_full=False)
        rec.success = succ
        return rec

    if condition == "B3":
        new_sol = agent.escalate(problem_prompt)
        succ, _ = _try(rec, "escalate", new_sol, problem, verifier, is_full=True)
        rec.success = succ
        return rec

    if condition == "B":
        # full cascade: reflect → replan → escalate
        new_sol = agent.reflect(problem_prompt, code, err)
        succ, err2 = _try(rec, "reflect", new_sol, problem, verifier, is_full=False)
        if succ:
            rec.success = True
            return rec
        new_sol2 = agent.replan(problem_prompt, [code, new_sol], [err, err2])
        succ, err3 = _try(rec, "replan", new_sol2, problem, verifier, is_full=False)
        if succ:
            rec.success = True
            return rec
        new_sol3 = agent.escalate(problem_prompt)
        succ, _ = _try(rec, "escalate", new_sol3, problem, verifier, is_full=True)
        rec.success = succ
        return rec

    if condition == "C":
        # random gate at random_rate, 走 B 的 cascade
        if random.random() < random_rate:
            return run_problem("B", problem, problem_id, problem_prompt,
                               agent, verifier, scorer, taus, random_rate)
        return rec

    if condition == "D":
        # CP-gated: 用 score 跟 taus 比, 选 tier
        if scorer is None or rec.score is None:
            raise ValueError("Condition D 需要 scorer + score")
        s = rec.score
        tau_p = taus.get("tau_proceed", 0.3)
        tau_r = taus.get("tau_reflect", 0.7)
        tau_rp = taus.get("tau_replan", 0.95)

        if s <= tau_p:
            return rec   # CP 说 OK, propose 失败就接受
        elif s <= tau_r:
            new_sol = agent.reflect(problem_prompt, code, err)
            succ, _ = _try(rec, "reflect", new_sol, problem, verifier, is_full=False)
            rec.success = succ
        elif s <= tau_rp:
            new_sol = agent.replan(problem_prompt, [code], [err])
            succ, _ = _try(rec, "replan", new_sol, problem, verifier, is_full=False)
            rec.success = succ
        else:
            new_sol = agent.escalate(problem_prompt)
            succ, _ = _try(rec, "escalate", new_sol, problem, verifier, is_full=True)
            rec.success = succ
        return rec

    raise ValueError(f"unknown condition: {condition}")
