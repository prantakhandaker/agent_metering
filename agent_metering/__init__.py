"""agent_metering — lightweight LLM cost tracking for B2B AI agents.

Install-only (site .pth) or import-once::

    pip install agent-metering
    # or: import agent_metering

    agent_metering.set_user("cust_123")  # optional, once per request

Existing OpenAI / Anthropic calls are metered automatically.
"""

from agent_metering.alerts import check_budgets, slack_notifier
from agent_metering.context import (
    clear_feature,
    clear_user,
    get_feature,
    get_user,
    set_feature,
    set_user,
    user_context,
)
from agent_metering.core import Meter
from agent_metering.enforcement import AllowanceDenial, AllowanceEnforcer, AllowanceOk
from agent_metering.instrument import disable, enable, get_meter, is_enabled, maybe_auto_enable
from agent_metering.storage import BaseStorage, SQLiteStorage, UsageRecord

__all__ = [
    "Meter",
    "SQLiteStorage",
    "BaseStorage",
    "UsageRecord",
    "AllowanceDenial",
    "AllowanceEnforcer",
    "AllowanceOk",
    "check_budgets",
    "slack_notifier",
    "app",
    "enable",
    "disable",
    "is_enabled",
    "get_meter",
    "set_user",
    "get_user",
    "clear_user",
    "set_feature",
    "get_feature",
    "clear_feature",
    "user_context",
]


def __getattr__(name: str):
    if name == "app":
        from agent_metering.proxy import app as proxy_app

        return proxy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Auto-enable when config JSON is present (or AGENT_METERING_AUTO=1).
maybe_auto_enable()
