"""Code-writing agent: propose / reflect / replan / escalate, all task-level.

每个方法返回**一段完整 code**, 给 env.step 跑 test. 这跟 ALFWorld 的 step-level
agent 不同——code task 是 single-shot per problem, 4 档 cascade 是 4 种"最后递交
的 code 怎么生成的"。

Parse-retry: 每个方法在内部包了一层 retry — 若 LLM 第一次输出不含可识别的 code
(`def ` 不在 strip 后的文本里), 会以 temperature + 0.2 再调一次. 仍失败则把
最后一次的 raw text 返回, caller 通过 `agent.last_call_meta` 判断是否要标
`parse_error` verdict.

注意: `last_call_meta` 是 instance attribute, 在并行 rollout 时**必须每个
worker 持有独立的 CodeAgent 实例**, 否则会互相覆盖.
"""
import re
from typing import Callable, Dict, Optional, Tuple

from llm_client import (
    chat_mini, chat_full, chat_gpt5,
    chat_gpt54_mini, chat_gpt54, chat_gpt54_nano,
)


# Parse 判断: strip markdown fencing 后必须含 "def " 才算可解析为 Python code.
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Mirror of code_env._strip_code_block, 避免环依赖."""
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _is_parseable_code(text: str) -> bool:
    """LLM 输出是否含可识别的 Python 函数定义."""
    body = _strip_code_fence(text)
    return bool(body) and "def " in body


def _extract_json_block(text: str) -> Optional[str]:
    """从 ```json ... ``` 围栏里抽 JSON."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_first_json_object(text: str) -> Optional[str]:
    """从一团文本里抽第一个 {...} 块 (balanced braces)."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


# 一个 base / escalate call 失败时最多重试几次 (不含原始调用).
_DEFAULT_PARSE_RETRIES = 1
_RETRY_TEMP_BUMP = 0.2


PROPOSE_SYSTEM = """\
You are an expert Python programmer.
You will be given a function signature and docstring. Implement the function.

Output ONLY:
1. All necessary `import` statements at the top (e.g. `import random`, `import re`,
   `from typing import List`, etc.) — include EVERY import the function uses.
2. The complete function definition (signature + body).

Do not include explanations, examples, test code, or markdown fencing.
The output must be a valid Python file that can be saved and imported as-is."""

REFLECT_SYSTEM = """\
You are an expert Python programmer fixing a bug.
You will be given:
  1. The problem (function signature + docstring)
  2. Your previous code attempt
  3. The exact error message from running unit tests

Diagnose the bug and write a CORRECTED version.

Output ONLY:
1. All necessary `import` statements at the top (every import the function uses).
2. The complete corrected function definition.

No explanations, no markdown fencing."""

REPLAN_SYSTEM = """\
You are an expert Python programmer.
A previous attempt and its bug-fix attempt both failed. You need to RESTART with a
fundamentally different approach: decompose the problem into smaller helper
functions if helpful, or pick a different algorithm.

Output ONLY:
1. All necessary `import` statements at the top.
2. Any helper functions needed.
3. The complete main function definition.

No explanations, no markdown fencing."""

ESCALATE_CONTEXT_SYSTEM = """\
You are an expert Python programmer. A weaker model already attempted this
problem and failed; you are being called in as the strong model to solve it
from scratch. You will be given the problem and the weaker model's failure
trace (error message) for context — use it to avoid the same mistake, but do
not assume its approach was on the right track.

Output ONLY:
1. All necessary `import` statements at the top.
2. The complete function definition.

No explanations, no markdown fencing."""


DIAGNOSE_SYSTEM = """\
You are reviewing a failed code attempt. Briefly diagnose why it failed and
recommend a recovery action.

Output ONLY a JSON object with exactly these two fields:
{
  "failure_reason": "<1-2 short sentences explaining the root cause>",
  "recommended_action": "reflect" | "replan" | "escalate"
}

Action semantics:
- reflect: small local fix using the error message (good for AssertionError on
  edge cases, off-by-one, wrong constant, missing strip(), etc.)
- replan: completely different algorithm/approach (good for wrong complexity,
  fundamentally wrong strategy, missing key insight)
- escalate: problem is beyond the small model's capability — needs a stronger
  model (good for hard algorithms / unusual problem types)

Output ONLY the JSON. No prose before or after."""


_BASE_MODELS = ("gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano")
_ESCALATE_TARGETS = ("gpt-4.1", "gpt-5", "gpt-5.4", "gpt-5.4-mini")


class CodeAgent:
    """4 档 cascade 对应 4 种生成方法. 每个 problem 独立调用.

    base_model:  'gpt-5.4-mini' (paper #1 default) 或 'gpt-5-mini' (legacy n=30).
                 用于 propose / reflect / replan.
    escalate_to: 'gpt-5.4' (paper #1 default), 'gpt-4.1', or 'gpt-5'.
                 用于 escalate tier.
    """

    def __init__(self,
                 mini_temperature: float = 0.0,
                 full_temperature: float = 0.0,
                 base_model: str = "gpt-5.4-mini",
                 escalate_to: str = "gpt-5.4",
                 parse_retries: int = _DEFAULT_PARSE_RETRIES):
        self.mini_t = mini_temperature
        self.full_t = full_temperature
        if base_model not in _BASE_MODELS:
            raise ValueError(
                f"base_model must be one of {_BASE_MODELS}, got {base_model!r}"
            )
        if escalate_to not in _ESCALATE_TARGETS:
            raise ValueError(
                f"escalate_to must be one of {_ESCALATE_TARGETS}, got {escalate_to!r}"
            )
        if parse_retries < 0:
            raise ValueError(f"parse_retries must be >= 0, got {parse_retries!r}")
        self.base_model = base_model
        self.escalate_to = escalate_to
        self.parse_retries = parse_retries
        self._chat_base, self._base_max_tokens = self._resolve_base(base_model)
        self._chat_esc, self._esc_max_tokens = self._resolve_escalate(escalate_to)
        # Meta from the most recent agent call.
        # caller (rollout loop) 在 propose/reflect/... 返回后立刻读取这个字段,
        # 用来决定是否需要把 verdict 标成 parse_error 以及计算 cost 时算几次 call.
        self.last_call_meta: Dict[str, object] = {}

    # GPT-5 系列 reasoning model 吃 thinking tokens, max_tokens 必须给足
    # 否则 reasoning 完了没 budget 输出实际代码 (经验: < 2000 经常空输出).
    # GPT-4.1 不带 reasoning, 4k 够.
    _GPT5_FAMILY_MAX_TOKENS = 8192
    _GPT41_MAX_TOKENS = 4096

    @classmethod
    def _resolve_base(cls, name: str):
        if name == "gpt-5-mini":
            return chat_mini, cls._GPT5_FAMILY_MAX_TOKENS
        if name == "gpt-5.4-mini":
            return chat_gpt54_mini, cls._GPT5_FAMILY_MAX_TOKENS
        if name == "gpt-5.4-nano":
            return chat_gpt54_nano, cls._GPT5_FAMILY_MAX_TOKENS
        raise ValueError(f"unsupported base_model: {name}")

    @classmethod
    def _resolve_escalate(cls, name: str):
        if name == "gpt-4.1":
            return chat_full, cls._GPT41_MAX_TOKENS
        if name == "gpt-5":
            return chat_gpt5, cls._GPT5_FAMILY_MAX_TOKENS
        if name == "gpt-5.4":
            return chat_gpt54, cls._GPT5_FAMILY_MAX_TOKENS
        if name == "gpt-5.4-mini":
            return chat_gpt54_mini, cls._GPT5_FAMILY_MAX_TOKENS
        raise ValueError(f"unsupported escalate_to: {name}")

    def _call_with_retry(self, chat_fn: Callable, msgs: list,
                         base_temperature: float, max_tokens: int,
                         action: str) -> str:
        """Call chat_fn; if output isn't parseable code, retry up to
        self.parse_retries times with temperature bumped by +0.2 each retry.

        Records meta into self.last_call_meta:
          - action:        one of {propose, reflect, replan, escalate}
          - n_calls:       total LLM calls made (1 + actual retries)
          - parseable:     whether the final returned text is parseable
          - retry_temps:   list of temperatures tried (for debugging)
          - raw_text:      last raw LLM output (含 markdown 围栏, 用于 logging)
          - usage:         aggregate token usage across all calls (prompt + completion sum)
        """
        text, meta = chat_fn(msgs, temperature=base_temperature,
                             max_tokens=max_tokens, return_usage=True)
        usage = meta.get("usage") if meta else None
        reasoning = meta.get("reasoning_content") if meta else None
        agg_usage = dict(usage) if usage else None
        n_calls = 1
        retry_temps = [base_temperature]
        parseable = _is_parseable_code(text)
        while not parseable and n_calls <= self.parse_retries:
            bumped = base_temperature + _RETRY_TEMP_BUMP * n_calls
            text2, meta2 = chat_fn(msgs, temperature=bumped,
                                   max_tokens=max_tokens, return_usage=True)
            usage2 = meta2.get("usage") if meta2 else None
            n_calls += 1
            retry_temps.append(bumped)
            if usage2:
                if agg_usage is None:
                    agg_usage = dict(usage2)
                else:
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        agg_usage[k] = (agg_usage.get(k) or 0) + (usage2.get(k) or 0)
                    rt1 = agg_usage.get("reasoning_tokens")
                    rt2 = usage2.get("reasoning_tokens")
                    if rt1 is not None or rt2 is not None:
                        agg_usage["reasoning_tokens"] = (rt1 or 0) + (rt2 or 0)
            # Take the latest non-None reasoning_content (final attempt's thinking)
            r2 = meta2.get("reasoning_content") if meta2 else None
            if r2:
                reasoning = r2
            if _is_parseable_code(text2):
                text = text2
                parseable = True
                break
            text = text2   # 保留最后一次, caller 可以看 raw 内容
        self.last_call_meta = {
            "action":            action,
            "n_calls":           n_calls,
            "parseable":         parseable,
            "retry_temps":       retry_temps,
            "raw_text":          text,         # original output (with markdown fencing)
            "usage":             agg_usage,
            "reasoning_content": reasoning,    # may be None if API didn't expose
        }
        return text

    # tier 1: proceed —— base 一次直出
    def propose(self, problem_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": PROPOSE_SYSTEM},
            {"role": "user", "content": problem_prompt},
        ]
        return self._call_with_retry(
            self._chat_base, msgs, self.mini_t, self._base_max_tokens, "propose"
        )

    # tier 2: reflect —— base 看 error message 修
    def reflect(self, problem_prompt: str, prev_code: str, error_msg: str) -> str:
        user = (
            f"## Problem\n{problem_prompt}\n\n"
            f"## Your previous code\n```python\n{prev_code}\n```\n\n"
            f"## Test error\n```\n{error_msg}\n```\n\n"
            "Please write the corrected function with all needed imports."
        )
        msgs = [
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": user},
        ]
        return self._call_with_retry(
            self._chat_base, msgs, self.mini_t, self._base_max_tokens, "reflect"
        )

    # tier 3: replan —— base 重新 decompose
    def replan(self, problem_prompt: str, prev_attempts: list[str], errors: list[str]) -> str:
        attempts_str = "\n\n".join(
            f"### Attempt {i+1}\n```python\n{c}\n```\n### Error\n```\n{e}\n```"
            for i, (c, e) in enumerate(zip(prev_attempts, errors))
        )
        user = (
            f"## Problem\n{problem_prompt}\n\n"
            f"## Failed attempts\n{attempts_str}\n\n"
            "Restart with a fundamentally different approach. "
            "Write the complete function (with helper functions and all imports if useful)."
        )
        msgs = [
            {"role": "system", "content": REPLAN_SYSTEM},
            {"role": "user", "content": user},
        ]
        return self._call_with_retry(
            self._chat_base, msgs, self.mini_t, self._base_max_tokens, "replan"
        )

    # tier 4: escalate —— strong model 直出 (gpt-5.4 / gpt-5 / gpt-4.1)
    # 注意: 不给 error context, 是 cold-restart (跟 propose 同 system prompt).
    def escalate(self, problem_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": PROPOSE_SYSTEM},
            {"role": "user", "content": problem_prompt},
        ]
        return self._call_with_retry(
            self._chat_esc, msgs, self.full_t, self._esc_max_tokens, "escalate"
        )

    # tier 4 variant: escalate_with_context —— strong model 看 weak model 的 verdict/stderr.
    # 实验性: 对比 escalate() (blind cold-restart) 跟这个 (informed restart) 的 pass rate 差异.
    def escalate_with_context(self, problem_prompt: str, prev_code: str, error_msg: str) -> str:
        user = (
            f"## Problem\n{problem_prompt}\n\n"
            f"## Weaker model's previous attempt\n```python\n{prev_code}\n```\n\n"
            f"## Test error\n```\n{error_msg}\n```\n\n"
            "Solve the problem from scratch. Write the complete corrected function "
            "with all needed imports."
        )
        msgs = [
            {"role": "system", "content": ESCALATE_CONTEXT_SYSTEM},
            {"role": "user", "content": user},
        ]
        return self._call_with_retry(
            self._chat_esc, msgs, self.full_t, self._esc_max_tokens, "escalate"
        )

    # auxiliary: diagnose —— 失败后让 escalate model (gpt-5.4) 分析原因 + 推荐 action.
    # 用 gpt-5.4 (强模型) 当 judge 而非 nano, 因为:
    #   1) 推理质量更高, 给 SFT 更可靠的 ground-truth-like signal
    #   2) JSON parse 失败率更低 (复杂 prompt 上 nano 易乱)
    # 不走 parse_retry wrapper (要的是 JSON 而非 code), 但记录 usage / reasoning_content.
    def diagnose(self, problem_prompt: str, prev_code: str, error_msg: str) -> dict:
        """Run a diagnostic call after a failure. Returns:
          {
            "failure_reason": str,
            "recommended_action": "reflect"|"replan"|"escalate",
            "raw_output": str,           # raw LLM text (might be invalid JSON)
            "parse_ok": bool,            # True if JSON parsed correctly
            "usage": dict | None,
            "reasoning_content": str | None,
            "judge_model": str,          # full model id of the judge
          }
        """
        import json as _json
        user = (
            f"## Problem\n{problem_prompt}\n\n"
            f"## Failed code\n```python\n{prev_code}\n```\n\n"
            f"## Test error\n```\n{error_msg}\n```\n\n"
            "Diagnose the failure and recommend an action. Output strict JSON only."
        )
        msgs = [
            {"role": "system", "content": DIAGNOSE_SYSTEM},
            {"role": "user", "content": user},
        ]
        # Use ESCALATE model (gpt-5.4) for judge — stronger reasoning, lower JSON parse fail rate.
        # max_tokens 1024 enough for short JSON output (~100 visible tokens).
        text, meta = self._chat_esc(msgs, temperature=self.full_t,
                                     max_tokens=1024, return_usage=True)
        usage = meta.get("usage") if meta else None
        reasoning = meta.get("reasoning_content") if meta else None
        raw = text or ""
        # Try to extract JSON (model may wrap in ``` or add prose)
        parsed = None
        for candidate in (raw.strip(),
                          _extract_json_block(raw),
                          _extract_first_json_object(raw)):
            if not candidate:
                continue
            try:
                obj = _json.loads(candidate)
                if isinstance(obj, dict) and "recommended_action" in obj:
                    parsed = obj
                    break
            except Exception:
                continue
        if parsed is not None:
            failure_reason = str(parsed.get("failure_reason", ""))[:600]
            rec = str(parsed.get("recommended_action", "")).strip().lower()
            if rec not in ("reflect", "replan", "escalate"):
                rec = "unknown"
            return {
                "failure_reason": failure_reason,
                "recommended_action": rec,
                "raw_output": raw[:1500],
                "parse_ok": True,
                "usage": usage,
                "reasoning_content": reasoning,
                "judge_model": self.escalate_to,
            }
        return {
            "failure_reason": "",
            "recommended_action": "unknown",
            "raw_output": raw[:1500],
            "parse_ok": False,
            "usage": usage,
            "reasoning_content": reasoning,
            "judge_model": self.escalate_to,
        }
