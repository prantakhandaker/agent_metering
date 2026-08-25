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


def test_build_child_env_vertex_openai_url(monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config
    import json
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cfg.json"
        path.write_text(
            json.dumps(
                {
                    "providers": {
                        "vertex": {
                            "project_id": "myproj",
                            "location": "us-central1",
                            "credentials_json": "./sa.json",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(ENV_CONFIG, str(path))
        reset_config()
        env = build_child_env("http://127.0.0.1:8787", environ={"PATH": "/usr/bin"})
        assert (
            env["VERTEX_OPENAI_BASE_URL"]
            == "http://127.0.0.1:8787/proxy/vertex/v1/projects/myproj/locations/us-central1/endpoints/openapi"
        )
