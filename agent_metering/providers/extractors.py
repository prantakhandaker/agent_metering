"""Extract model and token usage from provider response shapes."""

from __future__ import annotations

import json
from typing import Any, Optional

from agent_metering.providers.registry import ProviderConfig


def _dig(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_usage(
    provider: ProviderConfig,
    response_json: dict[str, Any],
    *,
    request_model: str = "unknown",
) -> tuple[str, int, int]:
    """Return (model, input_tokens, output_tokens) from a response body."""
    extractor = provider.usage_extractor

    if extractor == "openai":
        usage = response_json.get("usage") or {}
        model = str(response_json.get("model") or request_model)
        prompt = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion = _as_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )
        return model, prompt, completion

    if extractor == "anthropic":
        usage = response_json.get("usage") or {}
        model = str(response_json.get("model") or request_model)
        return (
            model,
            _as_int(usage.get("input_tokens")),
            _as_int(usage.get("output_tokens")),
        )

    if extractor == "gemini":
        meta = response_json.get("usageMetadata") or {}
        model = str(response_json.get("model") or request_model)
        prompt = _as_int(meta.get("promptTokenCount"))
        completion = _as_int(meta.get("candidatesTokenCount"))
        return model, prompt, completion

    # generic — YAML-defined dot paths
    model = str(
        _dig(response_json, provider.model_path or "model") or request_model
    )
    prompt = _as_int(
        _dig(response_json, provider.input_tokens_path or "usage.prompt_tokens")
        or _dig(response_json, provider.input_tokens_path or "usage.input_tokens")
    )
    completion = _as_int(
        _dig(
            response_json,
            provider.output_tokens_path or "usage.completion_tokens",
        )
        or _dig(response_json, provider.output_tokens_path or "usage.output_tokens")
    )
    return model, prompt, completion


def parse_stream_event(
    provider: ProviderConfig,
    line: str,
    *,
    request_model: str,
) -> tuple[Optional[str], Optional[dict[str, Any]], int]:
    """Parse one SSE line. Returns (model, usage_dict, content_chars_added)."""
    if provider.stream_mode == "anthropic_sse":
        if not line.startswith("data: "):
            return None, None, 0
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            return None, None, 0
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            return None, None, 0
        model = event.get("model")
        usage = event.get("usage")
        content = ""
        delta = event.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("text") or delta.get("content") or ""
        message = event.get("message") or {}
        if isinstance(message, dict):
            content = content or message.get("content") or ""
        return model, usage, len(content)

    # openai_sse (default)
    if not line.startswith("data: "):
        return None, None, 0
    data_str = line[6:].strip()
    if data_str == "[DONE]":
        return None, None, 0
    try:
        event = json.loads(data_str)
    except json.JSONDecodeError:
        return None, None, 0
    model = event.get("model")
    usage = event.get("usage")
    delta = (event.get("choices") or [{}])[0].get("delta") or {}
    content = delta.get("content") or ""
    return model, usage, len(content)


def usage_from_stream_dict(
    usage: dict[str, Any],
) -> tuple[int, int]:
    prompt = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _as_int(
        usage.get("completion_tokens") or usage.get("output_tokens")
    )
    return prompt, completion
