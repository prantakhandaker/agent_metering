"""Per-customer spend allowance enforcement with an in-memory cache."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent_metering.config import EnforcementConfig, MeteringConfig
    from agent_metering.storage import BaseStorage, UsageRecord


@dataclass(frozen=True)
class AllowanceDenial:
    customer_id: str
    max_spend_usd: float
    current_spend_usd: float
    period: str
    status_code: int


@dataclass(frozen=True)
class AllowanceOk:
    pass


AllowanceResult = AllowanceDenial | AllowanceOk


def calendar_month_start_ts(now: Optional[float] = None) -> float:
    """UTC timestamp of the first second of the current calendar month."""
    dt = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    start = datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)
    return start.timestamp()


class AllowanceEnforcer:
    """Fast pre-request check: in-memory spend cache backed by SQLite totals."""

    def __init__(
        self,
        storage: "BaseStorage",
        enforcement: "EnforcementConfig",
    ) -> None:
        self._storage = storage
        self._enforcement = enforcement
        self._lock = threading.Lock()
        self._spend: dict[str, float] = {}
        self._period_start = calendar_month_start_ts()
        self._seed_from_storage()

    def _seed_from_storage(self) -> None:
        spend_fn = getattr(self._storage, "query_spend_since", None)
        if not callable(spend_fn):
            return
        for customer_id in self._enforcement.allowances:
            self._spend[customer_id] = float(
                spend_fn(customer_id, self._period_start)
            )

    def _maybe_roll_period(self) -> None:
        start = calendar_month_start_ts()
        if start != self._period_start:
            self._period_start = start
            self._spend.clear()
            self._seed_from_storage()

    def check(self, customer_id: str) -> AllowanceResult:
        if not self._enforcement.enabled:
            return AllowanceOk()
        allowance = self._enforcement.allowances.get(customer_id)
        if allowance is None:
            return AllowanceOk()
        with self._lock:
            self._maybe_roll_period()
            current = self._spend.get(customer_id, 0.0)
            if current >= allowance.max_spend_usd:
                return AllowanceDenial(
                    customer_id=customer_id,
                    max_spend_usd=allowance.max_spend_usd,
                    current_spend_usd=current,
                    period=allowance.period,
                    status_code=self._enforcement.status_code,
                )
        return AllowanceOk()

    def on_flush(self, records: list["UsageRecord"]) -> None:
        """Update cache when a batch is committed (or sync-written)."""
        if not records:
            return
        with self._lock:
            self._maybe_roll_period()
            for record in records:
                if not record.customer_id:
                    continue
                if record.customer_id not in self._enforcement.allowances:
                    continue
                if record.timestamp < self._period_start:
                    continue
                self._spend[record.customer_id] = (
                    self._spend.get(record.customer_id, 0.0) + float(record.cost_usd)
                )

    def current_spend(self, customer_id: str) -> float:
        with self._lock:
            self._maybe_roll_period()
            return self._spend.get(customer_id, 0.0)


def build_enforcer_from_config(
    storage: "BaseStorage",
    config: Optional["MeteringConfig"] = None,
) -> Optional[AllowanceEnforcer]:
    from agent_metering.config import get_config

    cfg = config if config is not None else get_config()
    if cfg.enforcement is None or not cfg.enforcement.enabled:
        return None
    return AllowanceEnforcer(storage, cfg.enforcement)
