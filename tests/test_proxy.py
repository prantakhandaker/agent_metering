"""Tests for the universal LLM metering proxy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_metering.providers.registry import ProviderConfig, get_registry, reset_registry
from agent_metering.proxy import create_app, meter


@pytest.fixture
def temp_meter(tmp_path, monkeypatch):
    from agent_metering.storage import SQLiteStorage

    db_path = tmp_path / "proxy_test.db"
    test_meter = type(meter)(storage=SQLiteStorage(db_path=db_path))
    monkeypatch.setattr("agent_metering.proxy.meter", test_meter)
    return test_meter


@pytest_asyncio.fixture
async def client(temp_meter):
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


OPENAI_SUCCESS = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

ANTHROPIC_SUCCESS = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hello!"}],
    "usage": {"input_tokens": 11, "output_tokens": 6},
}


def _mock_request_response(response: httpx.Response):
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.request = AsyncMock(return_value=response)
    return mock_client


@pytest.mark.asyncio
async def test_proxy_openai_route(client, temp_meter):
    mock_response = httpx.Response(
        200,
        json=OPENAI_SUCCESS,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Customer-Id": "cust_123",
                "X-Feature": "datachat_query",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello!"
    assert "cust_123" in temp_meter.cost_by_customer()


@pytest.mark.asyncio
async def test_legacy_openai_alias(client, temp_meter):
    mock_response = httpx.Response(
        200,
        json=OPENAI_SUCCESS,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer sk-test", "X-Customer-Id": "cust_123"},
        )
    assert resp.status_code == 200
    assert temp_meter.cost_by_customer()["cust_123"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_upstream_error_passthrough_no_log(client, temp_meter):
    mock_response = httpx.Response(
        401,
        json={"error": {"message": "Invalid API key"}},
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer sk-bad"},
        )
    assert resp.status_code == 401
    assert temp_meter.cost_by_customer() == {}


@pytest.mark.asyncio
async def test_proxy_anthropic_route(client, temp_meter):
    mock_response = httpx.Response(
        200,
        json=ANTHROPIC_SUCCESS,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/anthropic/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={
                "x-api-key": "sk-ant-test",
                "anthropic-version": "2023-06-01",
                "X-Customer-Id": "cust_ant",
                "X-Feature": "messages",
            },
        )
    assert resp.status_code == 200
    by_customer = temp_meter.cost_by_customer()
    assert "cust_ant" in by_customer
    assert by_customer["cust_ant"]["total_tokens"] == 17


@pytest.mark.asyncio
async def test_proxy_azure_route(client, temp_meter):
    azure_response = {
        "model": "gpt-5-mini",
        "usage": {"input_tokens": 9, "output_tokens": 4},
        "output": [],
    }
    mock_response = httpx.Response(
        200,
        json=azure_response,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/azure/v1/responses",
            json={"model": "gpt-5-mini", "input": "Hi"},
            headers={
                "Authorization": "Bearer azure-key",
                "X-Customer-Id": "cust_az",
                "X-Feature": "responses",
            },
        )
    assert resp.status_code == 200
    assert temp_meter.cost_by_customer()["cust_az"]["total_tokens"] == 13


@pytest.mark.asyncio
async def test_proxy_custom_provider(client, temp_meter, monkeypatch):
    reset_registry()
    registry = get_registry()
    registry["my_custom_llm"] = ProviderConfig(
        name="my_custom_llm",
        base_url="https://llm.mycompany.com",
        provider_label="my_custom_llm",
        usage_extractor="generic",
        model_path="model",
        input_tokens_path="usage.prompt_tokens",
        output_tokens_path="usage.completion_tokens",
    )
    monkeypatch.setattr(
        "agent_metering.providers.registry._REGISTRY",
        registry,
    )

    custom_response = {
        "model": "custom-model",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    mock_response = httpx.Response(
        200,
        json=custom_response,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/my_custom_llm/v1/chat",
            json={"model": "custom-model", "messages": []},
            headers={
                "Authorization": "Bearer custom",
                "X-Customer-Id": "cust_custom",
                "X-Feature": "custom_feat",
            },
        )
    assert resp.status_code == 200
    assert temp_meter.cost_by_customer()["cust_custom"]["total_tokens"] == 10
    reset_registry()


@pytest.mark.asyncio
async def test_streaming_forwards_chunks_and_logs_usage(client, temp_meter):
    stream_body = (
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        'data: {"model":"gpt-4o-mini","usage":{"prompt_tokens":8,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    async def aiter_bytes():
        yield stream_body.encode("utf-8")

    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 200
    mock_upstream_response.headers = {"content-type": "text/event-stream"}
    mock_upstream_response.aiter_bytes = aiter_bytes

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_upstream_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = MagicMock(return_value=mock_stream_ctx)
    mock_client.aclose = AsyncMock()

    with patch("agent_metering.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={
                "X-Customer-Id": "cust_stream",
                "X-Feature": "stream_feat",
            },
        )

    assert resp.status_code == 200
    assert "Hi" in resp.text
    assert temp_meter.cost_by_customer()["cust_stream"]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_proxy_uses_env_attribution_defaults(client, temp_meter, monkeypatch):
    monkeypatch.setenv("AGENT_METERING_CUSTOMER_ID", "env_customer")
    monkeypatch.setenv("AGENT_METERING_FEATURE", "env_feature")
    mock_response = httpx.Response(
        200,
        json=OPENAI_SUCCESS,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer sk-test"},
        )
    assert resp.status_code == 200
    assert "env_customer" in temp_meter.cost_by_customer()
    assert "env_feature" in temp_meter.cost_by_feature()


@pytest.mark.asyncio
async def test_proxy_headers_override_env_defaults(client, temp_meter, monkeypatch):
    monkeypatch.setenv("AGENT_METERING_CUSTOMER_ID", "env_customer")
    monkeypatch.setenv("AGENT_METERING_FEATURE", "env_feature")
    mock_response = httpx.Response(
        200,
        json=OPENAI_SUCCESS,
        headers={"content-type": "application/json"},
    )
    with patch("agent_metering.proxy.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value = _mock_request_response(mock_response)
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Customer-Id": "header_customer",
                "X-Feature": "header_feature",
            },
        )
    assert resp.status_code == 200
    by_customer = temp_meter.cost_by_customer()
    assert "header_customer" in by_customer
    assert "env_customer" not in by_customer
    assert "header_feature" in temp_meter.cost_by_feature()
    assert "env_feature" not in temp_meter.cost_by_feature()


@pytest.mark.asyncio
async def test_proxy_injects_api_key_from_config(client, temp_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    cfg_path = tmp_path / "agent_metering.config.json"
    cfg_path.write_text(
        '{"customer_id":"cfg_cust","feature":"cfg_feat","providers":{"openai":{"api_key":"sk-from-config"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_METERING_CUSTOMER_ID", raising=False)
    monkeypatch.delenv("AGENT_METERING_FEATURE", raising=False)
    reset_config()

    captured = {}

    async def capture_request(method, url, content=None, headers=None):
        captured["headers"] = {k.lower(): v for k, v in dict(headers or {}).items()}
        return httpx.Response(
            200,
            json=OPENAI_SUCCESS,
            headers={"content-type": "application/json"},
        )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.request = AsyncMock(side_effect=capture_request)

    with patch("agent_metering.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 200
    assert captured["headers"].get("authorization") == "Bearer sk-from-config"
    assert "cfg_cust" in temp_meter.cost_by_customer()


@pytest.mark.asyncio
async def test_proxy_client_auth_wins_over_config(client, temp_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    cfg_path = tmp_path / "agent_metering.config.json"
    cfg_path.write_text(
        '{"providers":{"openai":{"api_key":"sk-from-config"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_config()

    captured = {}

    async def capture_request(method, url, content=None, headers=None):
        captured["headers"] = {k.lower(): v for k, v in dict(headers or {}).items()}
        return httpx.Response(
            200,
            json=OPENAI_SUCCESS,
            headers={"content-type": "application/json"},
        )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.request = AsyncMock(side_effect=capture_request)

    with patch("agent_metering.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer sk-from-client"},
        )
    assert resp.status_code == 200
    assert captured["headers"].get("authorization") == "Bearer sk-from-client"


@pytest.mark.asyncio
async def test_proxy_injects_vertex_token(client, temp_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config
    from agent_metering.providers.registry import reset_registry

    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account","client_email":"a@b.c"}', encoding="utf-8")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "providers": {
                    "vertex": {
                        "project_id": "proj",
                        "location": "us-central1",
                        "credentials_json": str(sa),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg_path))
    reset_config()
    reset_registry()

    captured = {}

    async def capture_request(method, url, content=None, headers=None):
        captured["url"] = str(url)
        captured["headers"] = {k.lower(): v for k, v in dict(headers or {}).items()}
        return httpx.Response(
            200,
            json=OPENAI_SUCCESS,
            headers={"content-type": "application/json"},
        )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.request = AsyncMock(side_effect=capture_request)

    with patch(
        "agent_metering.vertex_auth.get_access_token", return_value="ya29.mock"
    ):
        with patch("agent_metering.proxy.httpx.AsyncClient", return_value=mock_client):
            resp = await client.post(
                "/proxy/vertex/v1/projects/proj/locations/us-central1/endpoints/openapi/chat/completions",
                json={"model": "google/gemini-1.5-flash", "messages": [{"role": "user", "content": "Hi"}]},
            )
    assert resp.status_code == 200
    assert captured["headers"].get("authorization") == "Bearer ya29.mock"
    assert "aiplatform.googleapis.com" in captured["url"]
