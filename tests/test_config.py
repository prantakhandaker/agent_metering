"""Tests for plug-and-play metering config JSON."""

from __future__ import annotations

import json

from agent_metering.config import (
    ENV_CONFIG,
    load_config,
    reset_config,
    resolve_api_key,
    resolve_vertex_settings,
    vertex_openai_compatible_base_url,
)


def test_load_config_missing_file(tmp_path, monkeypatch):
    path = tmp_path / "missing.json"
    monkeypatch.setenv(ENV_CONFIG, str(path))
    reset_config()
    cfg = load_config(path)
    assert cfg.customer_id == "unknown"
    assert cfg.feature == "unknown"
    assert cfg.providers == {}


def test_load_config_api_keys_and_vertex(tmp_path, monkeypatch):
    sa = tmp_path / "sa.json"
    sa.write_text('{"type": "service_account", "client_email": "a@b.c"}', encoding="utf-8")
    path = tmp_path / "agent_metering.config.json"
    path.write_text(
        json.dumps(
            {
                "customer_id": "acme",
                "feature": "support",
                "providers": {
                    "openai": {"api_key": "sk-test"},
                    "vertex": {
                        "project_id": "proj-1",
                        "location": "europe-west1",
                        "credentials_json": str(sa),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(ENV_CONFIG, str(path))
    reset_config()
    cfg = load_config(path)
    assert cfg.customer_id == "acme"
    assert cfg.feature == "support"
    assert cfg.providers["openai"].api_key == "sk-test"
    assert resolve_api_key("openai", cfg) == "sk-test"
    vertex = resolve_vertex_settings(cfg)
    assert vertex is not None
    assert vertex.project_id == "proj-1"
    assert vertex.location == "europe-west1"
    assert vertex.credentials_json == str(sa)


def test_env_overrides_config_api_key(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps({"providers": {"openai": {"api_key": "sk-from-file"}}}),
        encoding="utf-8",
    )
    cfg = load_config(path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert resolve_api_key("openai", cfg) == "sk-from-env"


def test_vertex_openai_compatible_base_url():
    assert (
        vertex_openai_compatible_base_url("p", "us-central1")
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi"
    )
    assert (
        vertex_openai_compatible_base_url("p", "us-central1", proxy_base="http://127.0.0.1:8787")
        == "http://127.0.0.1:8787/proxy/vertex/v1/projects/p/locations/us-central1/endpoints/openapi"
    )
