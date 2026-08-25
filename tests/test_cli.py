"""Tests for zero-code CLI env injection helpers."""

from __future__ import annotations

from agent_metering.cli import build_child_env, normalize_proxy_url


def test_normalize_proxy_url_strips_trailing_slash():
    assert normalize_proxy_url("http://127.0.0.1:8787/") == "http://127.0.0.1:8787"


def test_build_child_env_injects_provider_base_urls():
    env = build_child_env(
        "http://127.0.0.1:8787/",
        environ={"PATH": "/usr/bin", "OPENAI_API_KEY": "sk-test"},
    )
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/proxy/openai/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787/proxy/anthropic"
    assert env["AZURE_OPENAI_ENDPOINT"] == "http://127.0.0.1:8787/proxy/azure"
    assert env["AZURE_OPENAI_BASE_URL"] == "http://127.0.0.1:8787/proxy/azure/v1"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:8787/proxy/gemini"
    assert env["GEMINI_API_BASE"] == "http://127.0.0.1:8787/proxy/gemini"
