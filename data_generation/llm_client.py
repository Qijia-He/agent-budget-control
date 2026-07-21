"""共享 LLM client (byteintl Azure).

支持 GPT-4.1 (full) 和 GPT-4.1-mini, 跑 cost-asymmetric cascade.
鉴权配置见 doc/gpt_usage.py.

Retry 策略: _do_chat 在 RateLimitError / APITimeoutError / APIConnectionError 上
做 exponential backoff (5/10/20/40/60s, 最多 5 次). QPM 限流是 byteintl Azure
endpoint 的常态 (~10-30 qpm 默认), 单线程串行也可能踩.
"""
import random
import time
from typing import List

from openai import (
    AzureOpenAI,
    APIConnectionError,
    APITimeoutError,
    NotFoundError,
    RateLimitError,
)


# NotFoundError 加入是因为 byteintl Azure deployment routing 偶发 404
# ("deployment doesn't exist"), 实测 ~5% 抖动率, 但不是真正缺失 — retry 一般通过.
_RETRY_ON = (RateLimitError, APITimeoutError, APIConnectionError, NotFoundError)
_RETRY_BACKOFF_S = [5, 10, 20, 40, 60, 120, 180]   # 7 retries, ~7min total
_RETRY_JITTER_S = 5.0


# ---- GPT-4.1 (full, 用于 escalate) ----
_FULL_ENDPOINT = "https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl"
_FULL_API_KEY = "sGZjVKGidgDee8MzHzfqca4kwMWZ3yOU_GPT_AK"
_FULL_API_VERSION = "2024-03-01-preview"
_FULL_MODEL = "gpt-4.1-2025-04-14"

_full_client = AzureOpenAI(
    azure_endpoint=_FULL_ENDPOINT,
    api_key=_FULL_API_KEY,
    api_version=_FULL_API_VERSION,
    default_headers={"X-TT-LOGID": "${your_logid}"},
)

# ---- mini (base, 用于 proceed/reflect/replan/scorer) ----
# 注意: gpt-4.1-mini 走 search-va modelhub endpoint, 但实测 502;
#       gpt-5-mini 走 search-va gpt/openapi endpoint, 跟 gpt-5 同 endpoint, 稳定.
# 默认用 gpt-5-mini 当 base. 想切回 gpt-4.1-mini 改下面常量.
_MINI_ENDPOINT = "https://search-va.byteintl.net/gpt/openapi/online/v2/crawl"
_MINI_API_KEY = "AVpap6zs6OLYklVuYkkLl2qMDaIOTKsv_GPT_AK"
_MINI_API_VERSION = "2024-02-01"
_MINI_MODEL = "gpt-5-mini-2025-08-07"

_mini_client = AzureOpenAI(
    azure_endpoint=_MINI_ENDPOINT,
    api_key=_MINI_API_KEY,
    api_version=_MINI_API_VERSION,
    default_headers={"X-TT-LOGID": "${your_logid}"},
)


# Backward compat
AZURE_GPT_MODEL = _FULL_MODEL
_client = _full_client  # 保留 v0 默认 = full


def _extract_usage(resp):
    """从 OpenAI response 提取 token usage. 返回 dict 或 None."""
    if not resp or not getattr(resp, "usage", None):
        return None
    u = resp.usage
    out = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
        "reasoning_tokens": None,
    }
    # 部分 reasoning model 把 reasoning tokens 单列出来 (在 completion_tokens_details 里)
    details = getattr(u, "completion_tokens_details", None)
    if details is not None:
        out["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return out


def _extract_reasoning_content(resp):
    """从 message.reasoning_content 提取 reasoning trace (如果 API 暴露).
    Returns None if not present."""
    if not resp or not resp.choices:
        return None
    msg = resp.choices[0].message
    rc = getattr(msg, "reasoning_content", None)
    if rc and isinstance(rc, str) and rc.strip():
        return rc
    return None


def _do_chat(client, model, messages, temperature, max_tokens, n=1,
             return_usage: bool = False, **kw):
    """Internal chat call. 返回 content (str / list[str]).
    若 return_usage=True, 返回 (content, usage_dict).
    """
    # GPT-5 系列只支持 default temperature=1, 显式传 0 会 400 报错
    is_gpt5 = "gpt-5" in model.lower()
    create_kw = dict(messages=messages, n=n, **kw)
    if not is_gpt5:
        create_kw["temperature"] = temperature
    # GPT-5 用 max_completion_tokens 而不是 max_tokens
    if is_gpt5:
        create_kw["max_completion_tokens"] = max_tokens
    else:
        create_kw["max_tokens"] = max_tokens

    # Exponential backoff retry on QPM / transient errors.
    last_exc = None
    resp = None
    for attempt, base_wait in enumerate([0.0] + _RETRY_BACKOFF_S):
        if base_wait > 0:
            time.sleep(base_wait + random.uniform(0, _RETRY_JITTER_S))
        try:
            resp = client.chat.completions.create(model=model, **create_kw)
            break
        except _RETRY_ON as e:
            last_exc = e
            if attempt == len(_RETRY_BACKOFF_S):
                # 用尽 retries, 抛上去, caller 决定怎么处理 (run_pipeline 会捕获并记 cost=0)
                raise
            continue
    else:
        raise last_exc  # pragma: no cover

    # Extract content
    if not resp or not resp.choices:
        content = [""] if n > 1 else ""
        usage = None
        reasoning = None
    else:
        if n == 1:
            content = resp.choices[0].message.content or ""
        else:
            content = [c.message.content or "" for c in resp.choices]
        usage = _extract_usage(resp)
        reasoning = _extract_reasoning_content(resp)

    if return_usage:
        # Returns (content, meta_dict) where meta_dict has usage + reasoning_content.
        return content, {"usage": usage, "reasoning_content": reasoning}
    return content


# v0 兼容: chat / chat_n 默认走 full
def chat(messages, temperature: float = 0.0, max_tokens: int = 512, **kw) -> str:
    return _do_chat(_full_client, _FULL_MODEL, messages, temperature, max_tokens, **kw)


def chat_n(messages, n: int = 5, temperature: float = 0.7, max_tokens: int = 256, **kw) -> List[str]:
    return _do_chat(_full_client, _FULL_MODEL, messages, temperature, max_tokens, n=n, **kw)


# v1+ cost-asymmetric cascade API
def chat_mini(messages, temperature: float = 0.0, max_tokens: int = 1024, **kw) -> str:
    """base agent / scorer / reflect / replan 都走 mini."""
    return _do_chat(_mini_client, _MINI_MODEL, messages, temperature, max_tokens, **kw)


def chat_full(messages, temperature: float = 0.0, max_tokens: int = 1024, **kw) -> str:
    """escalate tier 走 full = GPT-4.1."""
    return _do_chat(_full_client, _FULL_MODEL, messages, temperature, max_tokens, **kw)


# ---- GPT-5 (alternative escalate target, more powerful but slower) ----
_GPT5_ENDPOINT = "https://search-va.byteintl.net/gpt/openapi/online/v2/crawl"
_GPT5_API_KEY = "xwQzVOHpYtue7wpk0jW3Upu4I3ZJJwW8_GPT_AK"
_GPT5_API_VERSION = "2024-02-01"
_GPT5_MODEL = "gpt-5-2025-08-07"

_gpt5_client = AzureOpenAI(
    azure_endpoint=_GPT5_ENDPOINT,
    api_key=_GPT5_API_KEY,
    api_version=_GPT5_API_VERSION,
    default_headers={"X-TT-LOGID": "${your_logid}"},
)


def chat_gpt5(messages, temperature: float = 0.0, max_tokens: int = 4096, **kw) -> str:
    """GPT-5 (full reasoning model, ~5x cost vs mini, similar tier to GPT-4.1)."""
    return _do_chat(_gpt5_client, _GPT5_MODEL, messages, temperature, max_tokens, **kw)


# ---- GPT-5.4 (new escalate target, reasoning model, primary for paper #1) ----
# 鉴权信息抄自 doc/gpt_usage.py 的 "# gpt-5.4" 块.
_GPT54_ENDPOINT = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
_GPT54_API_KEY = "VxmXTg4dzQ6qwnfsgdFHT4OS75nVY9up_GPT_AK"
_GPT54_API_VERSION = "2024-02-01"
_GPT54_MODEL = "gpt-5.4-2026-03-05"

_gpt54_client = AzureOpenAI(
    azure_endpoint=_GPT54_ENDPOINT,
    api_key=_GPT54_API_KEY,
    api_version=_GPT54_API_VERSION,
    default_headers={"X-TT-LOGID": "${your_logid}"},
)


def chat_gpt54(messages, temperature: float = 0.0, max_tokens: int = 8192, **kw) -> str:
    """GPT-5.4 (full reasoning model, paper #1 default escalate target).

    注意: gpt-5.4 是 reasoning model, 内部 thinking tokens 占 budget,
    所以 max_tokens 给 8192. 跟 chat_gpt5 同样的 "gpt-5" 系列特例处理.
    """
    return _do_chat(_gpt54_client, _GPT54_MODEL, messages, temperature, max_tokens, **kw)


# ---- GPT-5.4-mini (mid-tier model in 5.4 family) ----
# doc/gpt_usage.py 里 5.4-mini 只标了 model id, 没给独立 endpoint.
# 假设跟 gpt-5.4 共享 endpoint + api_key (sibling-pair 默认).
# 若实测 401 / 403 再切独立 key.
_GPT54_MINI_ENDPOINT = _GPT54_ENDPOINT
_GPT54_MINI_API_KEY = _GPT54_API_KEY
_GPT54_MINI_API_VERSION = _GPT54_API_VERSION
_GPT54_MINI_MODEL = "gpt-5.4-mini-2026-03-17"

_gpt54_mini_client = AzureOpenAI(
    azure_endpoint=_GPT54_MINI_ENDPOINT,
    api_key=_GPT54_MINI_API_KEY,
    api_version=_GPT54_MINI_API_VERSION,
    default_headers={"X-TT-LOGID": "${your_logid}"},
)


def chat_gpt54_mini(messages, temperature: float = 0.0, max_tokens: int = 8192, **kw) -> str:
    """GPT-5.4-mini (mid-tier in 5.4 family).

    reasoning model, max_tokens 给足.
    """
    return _do_chat(_gpt54_mini_client, _GPT54_MINI_MODEL, messages, temperature, max_tokens, **kw)


# ---- GPT-5.4-nano (cheapest tier, new paper #1 base candidate) ----
# 假设共用 gpt-5.4 endpoint + api_key (跟 mini 同模式). model id TBD —
# 用户提供后填进 _GPT54_NANO_MODEL 即可.
_GPT54_NANO_ENDPOINT = _GPT54_ENDPOINT
_GPT54_NANO_API_KEY = _GPT54_API_KEY
_GPT54_NANO_API_VERSION = _GPT54_API_VERSION
_GPT54_NANO_MODEL = "gpt-5.4-nano-2026-03-17"

_gpt54_nano_client = AzureOpenAI(
    azure_endpoint=_GPT54_NANO_ENDPOINT,
    api_key=_GPT54_NANO_API_KEY,
    api_version=_GPT54_NANO_API_VERSION,
    default_headers={"X-TT-LOGID": "${your_logid}"},
)


def chat_gpt54_nano(messages, temperature: float = 0.0, max_tokens: int = 8192, **kw) -> str:
    """GPT-5.4-nano (cheapest tier, paper #1 base model candidate).

    Reasoning model. 比 gpt-5.4-mini 更便宜更弱, 期望在 BigCodeBench 上
    A vanilla ~20-40% 以恢复 cascade headroom.
    """
    return _do_chat(_gpt54_nano_client, _GPT54_NANO_MODEL, messages, temperature, max_tokens, **kw)
