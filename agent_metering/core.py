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
    ) -> None:
        self._storage = storage
        self._customer_id = customer_id
        self._feature = feature
        self._started_at = started_at

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        extra_metadata: Optional[Any] = None,
    ) -> UsageRecord:
        latency_ms = (time.time() - self._started_at) * 1000.0
        cost_usd = calculate_cost(model, input_tokens, output_tokens)
        metadata_str: Optional[str] = None
        if extra_metadata is not None:
            if isinstance(extra_metadata, str):
                metadata_str = extra_metadata
            else:
                metadata_str = json.dumps(extra_metadata)

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
    ) -> Generator[_TrackRecorder, None, None]:
        started_at = time.time()
        yield _TrackRecorder(self.storage, customer_id, feature, started_at)

    def cost_by_customer(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self.storage.query_cost_by_customer(since_ts=since_ts)

    def cost_by_feature(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self.storage.query_cost_by_feature(since_ts=since_ts)
