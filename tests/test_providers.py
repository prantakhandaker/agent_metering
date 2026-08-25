"""Tests for provider usage extractors."""

from __future__ import annotations

from agent_metering.providers.extractors import extract_usage
from agent_metering.providers.registry import ProviderConfig


def test_extract_openai_chat():
    cfg = ProviderConfig(
        name="openai",
        base_url="https://api.openai.com",
        provider_label="openai",
        usage_extractor="openai",
    )
    model, inp, out = extract_usage(
        cfg,
        {
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        },
    )
    assert model == "gpt-4o-mini"
    assert inp == 12
    assert out == 4


def test_extract_openai_responses():
    cfg = ProviderConfig(
        name="azure",
        base_url="https://azure.test",
        provider_label="azure_openai",
        usage_extractor="openai",
    )
    model, inp, out = extract_usage(
        cfg,
        {
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        },
        request_model="gpt-5-mini",
    )
    assert inp == 20
    assert out == 10


def test_extract_anthropic():
    cfg = ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com",
        provider_label="anthropic",
        usage_extractor="anthropic",
    )
    model, inp, out = extract_usage(
        cfg,
        {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 8, "output_tokens": 3},
        },
    )
    assert model == "claude-3-5-sonnet-20241022"
    assert inp == 8
    assert out == 3


def test_extract_gemini():
    cfg = ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com",
        provider_label="google",
        usage_extractor="gemini",
    )
    model, inp, out = extract_usage(
        cfg,
        {
            "model": "gemini-1.5-flash",
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 5,
            },
        },
    )
    assert inp == 15
    assert out == 5


def test_extract_generic_yaml_paths():
    cfg = ProviderConfig(
        name="custom",
        base_url="https://custom.test",
        provider_label="custom",
        usage_extractor="generic",
        model_path="model",
        input_tokens_path="usage.prompt_tokens",
        output_tokens_path="usage.completion_tokens",
    )
    model, inp, out = extract_usage(
        cfg,
        {
            "model": "custom-model",
            "usage": {"prompt_tokens": 6, "completion_tokens": 2},
        },
    )
    assert model == "custom-model"
    assert inp == 6
    assert out == 2
