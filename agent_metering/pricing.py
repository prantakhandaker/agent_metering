"""Model pricing table and cost calculation."""

from __future__ import annotations

# Prices are USD per 1,000 tokens (approximate public list prices).
PRICING_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-5-mini": {"input": 0.00015, "output": 0.0006},
    "claude-opus-4-8": {"input": 0.015, "output": 0.075},
    "claude-sonnet-5": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash-001": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro-001": {"input": 0.00125, "output": 0.005},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for the given model and token counts.

    Unknown models return 0.0 so metering never crashes the caller's app.
    """
    rates = PRICING_TABLE.get(model)
    if rates is None:
        return 0.0
    return (input_tokens / 1000.0) * rates["input"] + (output_tokens / 1000.0) * rates[
        "output"
    ]
