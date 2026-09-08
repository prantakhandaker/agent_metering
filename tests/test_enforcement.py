"""Tests for per-customer allowance enforcement."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_metering.config import (
    CustomerAllowance,
    EnforcementConfig,
    ENV_CONFIG,
    reset_config,
)
from agent_metering.enforcement import AllowanceEnforcer, AllowanceDenial
from agent_metering.proxy import create_app, meter, reload_enforcer
from agent_metering.storage import SQLiteStorage, UsageRecord
import time


@pytest.fixture
def enf_meter(tmp_path, monkeypatch):
    db_path = tmp_path / "enf.db"
    storage = SQLiteStorage(db_path=db_path, sync_writes=True)
    test_meter = type(meter)(storage=storage)
    monkeypatch.setattr("agent_metering.proxy.meter", test_meter)
    return test_meter


@pytest_asyncio.fixture
async def client(enf_meter):
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def test_enforcer_denies_when_over_allowance(tmp_path):
    storage = SQLiteStorage(db_path=tmp_path / "e.db", sync_writes=True)
    storage.write(
        UsageRecord(
            timestamp=time.time(),
            customer_id="acme",
            feature="f",
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=100,
            cost_usd=10.0,
            latency_ms=1.0,
        )
    )
    enf = AllowanceEnforcer(
        storage,
        EnforcementConfig(
            enabled=True,
            status_code=402,
            allowances={"acme": CustomerAllowance(max_spend_usd=5.0)},
        ),
    )
    result = enf.check("acme")
    assert isinstance(result, AllowanceDenial)
    assert result.status_code == 402
    assert result.current_spend_usd >= 5.0


@pytest.mark.asyncio
async def test_proxy_blocks_when_allowance_exceeded(client, enf_meter, tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "enforcement": {
                    "enabled": True,
                    "status_code": 429,
                    "allowances": {"blocked_cust": {"max_spend_usd": 0.001}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg_path))
    reset_config()

    # Seed spend above allowance
    with enf_meter.track(customer_id="blocked_cust", feature="f") as t:
        t.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=10000,
            output_tokens=10000,
        )
    # Ensure flush + rebuild enforcer against this meter storage
    flush = getattr(enf_meter.storage, "flush", None)
    if callable(flush):
        flush()
    monkeypatch.setattr("agent_metering.proxy.meter", enf_meter)
    reload_enforcer()

    called = {"n": 0}

    async def should_not_run(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("upstream should not be called")

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.request = AsyncMock(side_effect=should_not_run)

    with patch("agent_metering.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Customer-Id": "blocked_cust"},
        )
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"] == "allowance_exceeded"
    assert body["customer_id"] == "blocked_cust"
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_proxy_forwards_when_under_allowance(client, enf_meter, tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "enforcement": {
                    "enabled": True,
                    "status_code": 429,
                    "allowances": {"ok_cust": {"max_spend_usd": 100.0}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg_path))
    reset_config()
    monkeypatch.setattr("agent_metering.proxy.meter", enf_meter)
    reload_enforcer()

    mock_response = httpx.Response(
        200,
        json={
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"content": "ok"}}],
        },
        headers={"content-type": "application/json"},
    )
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.request = AsyncMock(return_value=mock_response)

    with patch("agent_metering.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await client.post(
            "/proxy/openai/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Customer-Id": "ok_cust", "Authorization": "Bearer sk"},
        )
    assert resp.status_code == 200
