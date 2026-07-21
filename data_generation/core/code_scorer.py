"""Task-level scorer for code generation.

给 (problem, proposed_code) 算 P(BAD) ∈ [0, 1] —— 当前代码会 fail 单元测试的概率.
用 GPT-4.1-mini 当 judge, 单次 logprobs call, 输出连续 score (跟 ALFWorld v0
的 LogprobsJudgeScorer 同构, 只换了 prompt).

CP 用这个 score 做 cascade gating:
    s ≤ τ_proceed         → 直接 submit
    τ_proceed < s ≤ τ_ref → reflect
    τ_ref     < s ≤ τ_rep → replan
    s > τ_replan          → escalate
"""
from __future__ import annotations

import math


CODE_JUDGE_SYSTEM = """\
You are a strict code reviewer.
Given a Python function signature with docstring (the problem) and a candidate
implementation, decide whether the implementation will pass standard unit tests.

Reply with EXACTLY one character:
  0 = the code is correct and will pass tests
  1 = the code has bugs / will fail tests / is incomplete
"""

CODE_JUDGE_USER = """\
## Problem
{problem}

## Candidate code
```python
{code}
```

Verdict (0 or 1):"""


class CodeLogprobsJudgeScorer:
    """单次 deterministic call + token logprobs → P(BAD) ∈ [0, 1]."""

    def __init__(self, max_chars: int = 6000):
        self.max_chars = max_chars

    def score(self, problem_prompt: str, code: str) -> float:
        # 用 full GPT-4.1 因为 GPT-5-mini 不支持 logprobs (BadRequestError),
        # 而 GPT-4.1-mini 自己的 endpoint 当前 502.
        # scorer 单次 max_tokens=1 cost 很低, 用 full 不影响 cascade 经济性.
        from llm_client import _full_client, _FULL_MODEL

        problem = problem_prompt[: self.max_chars]
        code = code[: self.max_chars]
        try:
            resp = _full_client.chat.completions.create(
                model=_FULL_MODEL,
                messages=[
                    {"role": "system", "content": CODE_JUDGE_SYSTEM},
                    {"role": "user",   "content": CODE_JUDGE_USER.format(
                        problem=problem, code=code
                    )},
                ],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=10,
            )
        except Exception:
            return 1.0

        if not resp or not resp.choices:
            return 1.0

        top = resp.choices[0].logprobs.content[0].top_logprobs
        log_p0 = log_p1 = float("-inf")
        for e in top:
            tok = e.token.strip()
            if tok == "0":
                log_p0 = max(log_p0, e.logprob)
            elif tok == "1":
                log_p1 = max(log_p1, e.logprob)

        if log_p0 == float("-inf") and log_p1 == float("-inf"):
            return 0.5
        m = max(log_p0, log_p1)
        p0 = math.exp(log_p0 - m) if log_p0 > float("-inf") else 0.0
        p1 = math.exp(log_p1 - m) if log_p1 > float("-inf") else 0.0
        return p1 / (p0 + p1)
