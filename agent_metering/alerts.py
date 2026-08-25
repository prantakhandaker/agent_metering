"""Budget checks and Slack alerting helpers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional


def check_budgets(
    meter: Any,
    customer_limits: Optional[dict[str, float]] = None,
    feature_limits: Optional[dict[str, float]] = None,
    window_seconds: float = 86400,
    on_breach: Optional[Callable[[str, str, float, float], None]] = None,
) -> list[dict[str, Any]]:
    """Check spend within a rolling window against customer/feature limits.

    Returns a list of breach dicts:
    ``{"scope": "customer"|"feature", "id": ..., "spent": ..., "limit": ...}``
    """
    since_ts = time.time() - window_seconds
    breaches: list[dict[str, Any]] = []

    if customer_limits:
        by_customer = meter.cost_by_customer(since_ts=since_ts)
        for customer_id, limit in customer_limits.items():
            spent = float(by_customer.get(customer_id, {}).get("total_cost_usd", 0.0))
            if spent > limit:
                breach = {
                    "scope": "customer",
                    "id": customer_id,
                    "spent": spent,
                    "limit": limit,
                }
                breaches.append(breach)
                if on_breach is not None:
                    on_breach("customer", customer_id, spent, limit)

    if feature_limits:
        by_feature = meter.cost_by_feature(since_ts=since_ts)
        for feature, limit in feature_limits.items():
            spent = float(by_feature.get(feature, {}).get("total_cost_usd", 0.0))
            if spent > limit:
                breach = {
                    "scope": "feature",
                    "id": feature,
                    "spent": spent,
                    "limit": limit,
                }
                breaches.append(breach)
                if on_breach is not None:
                    on_breach("feature", feature, spent, limit)

    return breaches


def slack_notifier(webhook_url: str) -> Callable[[str, str, float, float], None]:
    """Return an ``on_breach``-compatible function that POSTs to Slack."""

    def _notify(scope: str, id_: str, spent: float, limit: float) -> None:
        text = (
            f":warning: Budget breach — {scope} `{id_}` spent "
            f"${spent:.4f} (limit ${limit:.4f})"
        )
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"slack_notifier error: {exc}")

    return _notify
