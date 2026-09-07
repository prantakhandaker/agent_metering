"""Tests for zero-code user detection helpers and attribution priority."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from agent_metering import context as ctx
from agent_metering.core import Meter
from agent_metering.instrument import (
    _attribution,
    _record_openai_response,
    disable,
    enable,
    set_meter,
)
from agent_metering.storage import SQLiteStorage
from agent_metering.user_detect import (
    user_from_llm_kwargs,
    user_from_object,
    user_from_request,
)


def _jwt_with_claims(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )
    return f"Bearer {header}.{payload}.sig"


@pytest.fixture
def isolated_meter(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_METERING_AUTO", raising=False)
    meter = Meter(storage=SQLiteStorage(db_path=tmp_path / "user_detect.db"))
    set_meter(meter)
    ctx.clear_user()
    ctx.clear_feature()
    yield meter
    ctx.clear_user()
    ctx.clear_feature()
    disable()


def test_user_from_request_state_user_id():
    req = SimpleNamespace(state=SimpleNamespace(user_id="u_state"), headers={})
    assert user_from_request(req) == "u_state"


def test_user_from_request_django_like_user():
    user = SimpleNamespace(is_authenticated=True, id=99, username="alice")
    req = SimpleNamespace(user=user, headers={}, state=None)
    assert user_from_request(req) == "99"


def test_user_from_request_anonymous_skipped():
    user = SimpleNamespace(is_authenticated=False, id=1)
    req = SimpleNamespace(user=user, headers={}, state=None)
    assert user_from_request(req) is None


def test_user_from_request_header():
    req = SimpleNamespace(
        state=None,
        user=None,
        headers={"X-User-Id": "hdr_user"},
    )
    assert user_from_request(req) == "hdr_user"


def test_user_from_request_jwt_sub():
    req = SimpleNamespace(
        state=None,
        user=None,
        headers={"authorization": _jwt_with_claims({"sub": "jwt_sub_1"})},
    )
    assert user_from_request(req) == "jwt_sub_1"


def test_user_from_llm_kwargs_openai_user():
    assert user_from_llm_kwargs({"user": "openai_u"}) == "openai_u"


def test_user_from_llm_kwargs_anthropic_metadata():
    assert user_from_llm_kwargs({"metadata": {"user_id": "ant_u"}}) == "ant_u"


def test_user_from_object_str():
    assert user_from_object("plain") == "plain"


def test_attribution_priority_context_over_kwargs(isolated_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    missing = tmp_path / "missing.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    monkeypatch.delenv("AGENT_METERING_CUSTOMER_ID", raising=False)
    reset_config()

    ctx.set_user("ctx_user")
    assert _attribution(call_user="kw_user")[0] == "ctx_user"
    ctx.clear_user()
    assert _attribution(call_user="kw_user")[0] == "kw_user"
    assert _attribution()[0] == "default"


def test_record_uses_openai_user_kwarg(isolated_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    missing = tmp_path / "missing.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    monkeypatch.delenv("AGENT_METERING_CUSTOMER_ID", raising=False)
    reset_config()
    ctx.clear_user()

    _record_openai_response(
        SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
        ),
        call_user="from_kwargs",
    )
    assert "from_kwargs" in isolated_meter.cost_by_customer()


def test_record_context_wins_over_kwargs(isolated_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    missing = tmp_path / "missing.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    reset_config()
    ctx.set_user("ctx_wins")

    _record_openai_response(
        SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        ),
        call_user="kw_lose",
    )
    assert "ctx_wins" in isolated_meter.cost_by_customer()
    assert "kw_lose" not in isolated_meter.cost_by_customer()


@pytest.mark.asyncio
async def test_starlette_patch_sets_user_context(isolated_meter):
    pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from agent_metering.frameworks import disable_frameworks, enable_frameworks

    disable()
    disable_frameworks()

    async def homepage(request: Request):
        assert ctx.get_user() == "hdr_starlette"
        _record_openai_response(
            SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            )
        )
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", homepage)])
    enable(force=True)

    client = TestClient(app)
    resp = client.get("/", headers={"X-User-Id": "hdr_starlette"})
    assert resp.status_code == 200
    assert "hdr_starlette" in isolated_meter.cost_by_customer()

    disable()


def test_flask_patch_sets_user_context(isolated_meter):
    pytest.importorskip("flask")
    from flask import Flask

    from agent_metering.frameworks import disable_frameworks

    disable()
    disable_frameworks()

    app = Flask(__name__)

    @app.get("/")
    def homepage():
        assert ctx.get_user() == "hdr_flask"
        _record_openai_response(
            SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1),
            )
        )
        return "ok"

    enable(force=True)
    client = app.test_client()
    resp = client.get("/", headers={"X-User-Id": "hdr_flask"})
    assert resp.status_code == 200
    assert "hdr_flask" in isolated_meter.cost_by_customer()
    disable()
