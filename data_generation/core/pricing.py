"""Model pricing table (USD per 1M tokens) + cost computation helper.

Prices from byteintl Azure pricing (mirrors OpenAI; user-provided table).
For reasoning models, `completion_tokens` field already INCLUDES reasoning
tokens — billing is simply prompt × in_price + completion × out_price.
"""
from __future__ import annotations

from typing import Dict, Optional


PRICING_USD_PER_M: Dict[str, Dict[str, float]] = {
    # gpt-5.4 family (current paper #1 stack)
    "gpt-5.4":              {"input": 2.50,  "output": 15.00},
    "gpt-5.4-2026-03-05":   {"input": 2.50,  "output": 15.00},
    "gpt-5.4-mini":         {"input": 0.75,  "output":  4.50},
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano":         {"input": 0.20,  "output":  1.25},
    "gpt-5.4-nano-2026-03-17": {"input": 0.20, "output": 1.25},
    # gpt-5.5 family (newer, ~2x more expensive than 5.4)
    "gpt-5.5":              {"input": 5.00,  "output": 30.00},
    "gpt-5.5-pro":          {"input": 30.00, "output": 180.00},
    "gpt-5.4-pro":          {"input": 30.00, "output": 180.00},
    # gpt-5 / gpt-4.1 legacy (v30 / v54 / older runs)
    "gpt-5-2025-08-07":     {"input": 2.50,  "output": 15.00},  # estimate; same tier as 5.4
    "gpt-5-mini-2025-08-07": {"input": 0.50, "output":  2.50},  # estimate
    "gpt-4.1-2025-04-14":   {"input": 2.50,  "output": 10.00},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
}


def compute_cost_usd(model: str, usage: Optional[Dict]) -> Optional[float]:
    """Compute USD cost for one API call.

    model: full model id (e.g. "gpt-5.4-nano-2026-03-17")
    usage: dict from _extract_usage with at least prompt_tokens + completion_tokens

    Returns None if model not in price table or usage is None.
    """
    if usage is None or model not in PRICING_USD_PER_M:
        return None
    price = PRICING_USD_PER_M[model]
    p_tokens = usage.get("prompt_tokens", 0) or 0
    c_tokens = usage.get("completion_tokens", 0) or 0
    # completion_tokens already includes reasoning tokens (billing-wise)
    return (p_tokens * price["input"] + c_tokens * price["output"]) / 1_000_000
