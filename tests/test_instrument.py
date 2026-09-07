"""Tests for import-time auto-instrumentation and user context."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_metering import context as ctx
from agent_metering.core import Meter
from agent_metering.instrument import (
    disable,
    enable,
    get_meter,
    is_enabled,
    set_meter,
    _record_anthropic_response,
    _record_openai_response,
)
from agent_metering.storage import SQLiteStorage


@pytest.fixture
def isolated_meter(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_METERING_AUTO", raising=False)
    meter = Meter(storage=SQLiteStorage(db_path=tmp_path / "inst.db"))
    set_meter(meter)
    ctx.clear_user()
    ctx.clear_feature()
    yield meter
    ctx.clear_user()
    ctx.clear_feature()
    disable()


def test_record_openai_uses_context_user(isolated_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    cfg = tmp_path / "agent_metering.config.json"
    cfg.write_text(
        '{"customer_id":"default_cust","feature":"default_feat"}',
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg))
    reset_config()

    ctx.set_user("user_42")
    ctx.set_feature("checkout")
    _record_openai_response(
        SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
    )
    by_customer = isolated_meter.cost_by_customer()
    assert "user_42" in by_customer
    assert "default_cust" not in by_customer
    assert "checkout" in isolated_meter.cost_by_feature()


def test_record_falls_back_to_config_customer(isolated_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    cfg = tmp_path / "agent_metering.config.json"
    cfg.write_text(
        '{"customer_id":"acme","feature":"bots"}',
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_CONFIG, str(cfg))
    reset_config()
    ctx.clear_user()
    ctx.clear_feature()

    _record_openai_response(
        SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )
    )
    assert "acme" in isolated_meter.cost_by_customer()
    assert "bots" in isolated_meter.cost_by_feature()


def test_record_anthropic(isolated_meter):
    ctx.set_user("ant_user")
    _record_anthropic_response(
        SimpleNamespace(
            model="claude-3-5-sonnet-20241022",
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
        )
    )
    assert isolated_meter.cost_by_customer()["ant_user"]["total_tokens"] == 12


def test_enable_patches_openai_create(isolated_meter):
    pytest.importorskip("openai")
    disable()
    enable(force=True)
    assert is_enabled()

    from openai.resources.chat.completions import Completions

    # Replace the already-wrapped method's underlying call via a spy on recorder path
    fake_response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )
    ctx.set_user("patched_user")

    # Completions.create is bound wrapper; call it with a mock self
    create = Completions.create
    # The wrapper calls orig(self, ...) — patch orig by temporarily swapping
    from agent_metering import instrument as inst

    key = "openai.chat.completions.create"
    assert key in inst._ORIG
    cls, method_name, orig = inst._ORIG[key]

    def fake_orig(self, *args, **kwargs):
        return fake_response

    # Re-wrap with fake orig
    setattr(cls, method_name, inst._wrap_sync(fake_orig, inst._record_openai_response))
    try:
        result = getattr(cls, method_name)(MagicMock())
        assert result is fake_response
        assert "patched_user" in isolated_meter.cost_by_customer()
        assert isolated_meter.cost_by_customer()["patched_user"]["total_tokens"] == 10
    finally:
        disable()
        enable(force=True)


def test_user_context_manager(isolated_meter, tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    cfg = tmp_path / "c.json"
    cfg.write_text('{"customer_id":"fallback"}', encoding="utf-8")
    monkeypatch.setenv(ENV_CONFIG, str(cfg))
    reset_config()

    with ctx.user_context(user_id="scoped", feature="f1"):
        assert ctx.get_user() == "scoped"
        _record_openai_response(
            SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )
        )
    assert ctx.get_user() is None
    assert "scoped" in isolated_meter.cost_by_customer()


def test_maybe_auto_enable_with_config(tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG
    from agent_metering.instrument import maybe_auto_enable

    disable()
    cfg = tmp_path / "agent_metering.config.json"
    cfg.write_text('{"customer_id":"x"}', encoding="utf-8")
    monkeypatch.setenv(ENV_CONFIG, str(cfg))
    monkeypatch.delenv("AGENT_METERING_AUTO", raising=False)
    assert maybe_auto_enable() is True
    assert is_enabled()
    disable()


def test_maybe_auto_enable_without_config(tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG
    from agent_metering.instrument import maybe_auto_enable

    disable()
    missing = tmp_path / "no-such-config.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    monkeypatch.delenv("AGENT_METERING_AUTO", raising=False)
    monkeypatch.delenv("AGENT_METERING_CUSTOMER_ID", raising=False)
    monkeypatch.delenv("AGENT_METERING_FEATURE", raising=False)
    assert maybe_auto_enable() is True
    assert is_enabled()
    disable()


def test_maybe_auto_enable_respects_off(monkeypatch, tmp_path):
    from agent_metering.config import ENV_CONFIG
    from agent_metering.instrument import maybe_auto_enable

    disable()
    cfg = tmp_path / "agent_metering.config.json"
    cfg.write_text('{"customer_id":"x"}', encoding="utf-8")
    monkeypatch.setenv(ENV_CONFIG, str(cfg))
    monkeypatch.setenv("AGENT_METERING_AUTO", "0")
    assert maybe_auto_enable() is False
    assert is_enabled() is False


def test_autoload_module_enables_without_config(tmp_path, monkeypatch):
    """Simulates .pth import of agent_metering.autoload."""
    from agent_metering.config import ENV_CONFIG, reset_config

    disable()
    missing = tmp_path / "no-config.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    monkeypatch.delenv("AGENT_METERING_AUTO", raising=False)
    reset_config()

    import importlib

    import agent_metering.autoload as autoload

    importlib.reload(autoload)
    assert is_enabled()
    disable()


def test_autoload_respects_off(tmp_path, monkeypatch):
    from agent_metering.config import ENV_CONFIG, reset_config

    disable()
    missing = tmp_path / "no-config.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    monkeypatch.setenv("AGENT_METERING_AUTO", "0")
    reset_config()

    import importlib

    import agent_metering.autoload as autoload

    importlib.reload(autoload)
    assert is_enabled() is False


def test_record_uses_default_attribution_without_config(
    isolated_meter, tmp_path, monkeypatch
):
    from agent_metering.config import ENV_CONFIG, reset_config

    missing = tmp_path / "missing.json"
    monkeypatch.setenv(ENV_CONFIG, str(missing))
    monkeypatch.delenv("AGENT_METERING_CUSTOMER_ID", raising=False)
    monkeypatch.delenv("AGENT_METERING_FEATURE", raising=False)
    reset_config()
    ctx.clear_user()
    ctx.clear_feature()

    _record_openai_response(
        SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )
    )
    assert "default" in isolated_meter.cost_by_customer()
    assert "default" in isolated_meter.cost_by_feature()
    assert isolated_meter.cost_by_customer()["default"]["total_tokens"] == 6
