"""agent_metering — lightweight LLM cost tracking for B2B AI agents."""

from agent_metering.alerts import check_budgets, slack_notifier
from agent_metering.core import Meter
from agent_metering.storage import BaseStorage, SQLiteStorage, UsageRecord

__all__ = [
    "Meter",
    "SQLiteStorage",
    "BaseStorage",
    "UsageRecord",
    "check_budgets",
    "slack_notifier",
    "app",
]


def __getattr__(name: str):
    if name == "app":
        from agent_metering.proxy import app as proxy_app

        return proxy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
