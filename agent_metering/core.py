"""Meter — main SDK entry point for tracking LLM usage costs."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from agent_metering.pricing import calculate_cost
from agent_metering.storage import BaseStorage, SQLiteStorage, UsageRecord


class _TrackRecorder:
    def __init__(
        self,
        storage: BaseStorage,
        customer_id: Optional[str],
        feature: Optional[str],
        started_at: float,
        call_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        self._storage = storage
        self._customer_id = customer_id
        self._feature = feature
        self._started_at = started_at
        self._call_id = call_id
        self._unit_id = unit_id
        self._correlation_id = correlation_id

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        extra_metadata: Optional[Any] = None,
        *,
        status: Optional[str] = "success",
        call_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        mark_prior_retries: bool = False,
    ) -> UsageRecord:
        latency_ms = (time.time() - self._started_at) * 1000.0
        cost_usd = calculate_cost(model, input_tokens, output_tokens)
        metadata_str: Optional[str] = None
        if extra_metadata is not None:
            if isinstance(extra_metadata, str):
                metadata_str = extra_metadata
            else:
                metadata_str = json.dumps(extra_metadata)

        resolved_correlation = correlation_id if correlation_id is not None else self._correlation_id
        mark_priors = getattr(self._storage, "mark_prior_attempts_as_retry", None)
        if (
            mark_prior_retries
            and resolved_correlation
            and callable(mark_priors)
        ):
            mark_priors(resolved_correlation)

        usage = UsageRecord(
            timestamp=time.time(),
            customer_id=self._customer_id,
            feature=self._feature,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            extra_metadata=metadata_str,
            call_id=call_id if call_id is not None else self._call_id,
            unit_id=unit_id if unit_id is not None else self._unit_id,
            status=status,
            correlation_id=resolved_correlation,
        )
        self._storage.write(usage)
        return usage

    def record_openai_response(
        self, response: Any, extra_metadata: Optional[Any] = None
    ) -> UsageRecord:
        usage = response.usage
        return self.record(
            provider="openai",
            model=response.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            extra_metadata=extra_metadata,
        )

    def record_anthropic_response(
        self, response: Any, extra_metadata: Optional[Any] = None
    ) -> UsageRecord:
        usage = response.usage
        return self.record(
            provider="anthropic",
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            extra_metadata=extra_metadata,
        )


class Meter:
    def __init__(self, storage: Optional[BaseStorage] = None) -> None:
        self.storage = storage if storage is not None else SQLiteStorage()

    @contextmanager
    def track(
        self,
        customer_id: Optional[str] = None,
        feature: Optional[str] = None,
        call_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Generator[_TrackRecorder, None, None]:
        started_at = time.time()
        yield _TrackRecorder(
            self.storage,
            customer_id,
            feature,
            started_at,
            call_id=call_id,
            unit_id=unit_id,
            correlation_id=correlation_id,
        )

    def cost_by_customer(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self.storage.query_cost_by_customer(since_ts=since_ts)

    def cost_by_feature(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self.storage.query_cost_by_feature(since_ts=since_ts)

    def cost_by_call(
        self,
        feature: Optional[str] = None,
        since_ts: Optional[float] = None,
    ) -> dict[str, dict[str, Any]]:
        query = getattr(self.storage, "query_cost_by_call", None)
        if not callable(query):
            return {}
        return query(feature=feature, since_ts=since_ts)

    def cost_by_unit_id(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        query = getattr(self.storage, "query_cost_by_unit_id", None)
        if not callable(query):
            return {}
        return query(since_ts=since_ts)

    def waste_spend(self, since_ts: Optional[float] = None) -> dict[str, Any]:
        query = getattr(self.storage, "query_waste_spend", None)
        if not callable(query):
            return {"total_cost_usd": 0.0, "total_tokens": 0, "call_count": 0}
        return query(since_ts=since_ts)
