"""Storage interface and SQLite implementation for usage records."""

from __future__ import annotations

import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class UsageRecord:
    timestamp: float
    customer_id: Optional[str]
    feature: Optional[str]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    extra_metadata: Optional[str] = None


class BaseStorage(ABC):
    @abstractmethod
    def write(self, record: UsageRecord) -> None:
        ...

    @abstractmethod
    def query_cost_by_customer(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        ...

    @abstractmethod
    def query_cost_by_feature(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        ...


class SQLiteStorage(BaseStorage):
    """Thread-safe SQLite storage for a single-process MVP."""

    def __init__(self, db_path: str | Path = "agent_metering.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS usage_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        customer_id TEXT,
                        feature TEXT,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        cost_usd REAL NOT NULL,
                        latency_ms REAL NOT NULL,
                        extra_metadata TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def write(self, record: UsageRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_log (
                        timestamp, customer_id, feature, provider, model,
                        input_tokens, output_tokens, cost_usd, latency_ms,
                        extra_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.timestamp,
                        record.customer_id,
                        record.feature,
                        record.provider,
                        record.model,
                        record.input_tokens,
                        record.output_tokens,
                        record.cost_usd,
                        record.latency_ms,
                        record.extra_metadata,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def _aggregate(
        self, group_column: str, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        if since_ts is None:
            since_ts = 0.0
        sql = f"""
            SELECT
                COALESCE({group_column}, 'unknown') AS group_key,
                SUM(cost_usd) AS total_cost_usd,
                SUM(input_tokens + output_tokens) AS total_tokens,
                COUNT(*) AS call_count
            FROM usage_log
            WHERE timestamp >= ?
            GROUP BY group_key
            ORDER BY total_cost_usd DESC
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, (since_ts,)).fetchall()
            finally:
                conn.close()

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[row["group_key"]] = {
                "total_cost_usd": float(row["total_cost_usd"] or 0.0),
                "total_tokens": int(row["total_tokens"] or 0),
                "call_count": int(row["call_count"] or 0),
            }
        return result

    def query_cost_by_customer(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self._aggregate("customer_id", since_ts)

    def query_cost_by_feature(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self._aggregate("feature", since_ts)
