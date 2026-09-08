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
        message = event.get("message") or {}
        model = event.get("model")
        if not model and isinstance(message, dict):
            model = message.get("model")
        usage = event.get("usage")
        if usage is None and isinstance(message, dict):
            usage = message.get("usage")
        content = ""
        delta = event.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("text") or delta.get("content") or ""
        if isinstance(message, dict):
            msg_content = message.get("content") or ""
            if isinstance(msg_content, str):
                content = content or msg_content
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


def merge_stream_usage(
    existing: Optional[dict[str, Any]],
    incoming: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Merge SSE usage chunks so later partial events do not wipe earlier fields.

    Takes the max of each known token field so Anthropic ``message_start``
    input tokens survive a later ``message_delta`` that only has output tokens.
    """
    if not incoming:
        return existing
    if not existing:
        return dict(incoming)
    merged = dict(existing)
    for key, value in incoming.items():
        if value is None:
            continue
        if key in (
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            merged[key] = max(_as_int(merged.get(key)), _as_int(value))
        else:
            merged[key] = value
    return merged


def usage_from_stream_dict(
    usage: dict[str, Any],
) -> tuple[int, int]:
    prompt = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _as_int(
        usage.get("completion_tokens") or usage.get("output_tokens")
    )
    return prompt, completion
