"""Storage interface and SQLite implementation for usage records."""

from __future__ import annotations

import atexit
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Flush when either threshold is hit.
_DEFAULT_FLUSH_INTERVAL_S = 0.2
_DEFAULT_FLUSH_MAX_RECORDS = 50


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
    call_id: Optional[str] = None
    unit_id: Optional[str] = None
    status: Optional[str] = None  # success | error | retry_attempt
    correlation_id: Optional[str] = None


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
    """Thread-safe SQLite storage with WAL mode and batched writes.

    ``write()`` enqueues and returns immediately. A background thread flushes
    when 200ms elapses or 50 records accumulate (whichever first). A crash
    between flushes loses only the in-memory batch; committed WAL data survives.
    """

    def __init__(
        self,
        db_path: str | Path = "agent_metering.db",
        *,
        flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S,
        flush_max_records: int = _DEFAULT_FLUSH_MAX_RECORDS,
        sync_writes: bool = False,
        on_flush: Optional[Callable[[list[UsageRecord]], None]] = None,
    ) -> None:
        self.db_path = str(db_path)
        self._flush_interval_s = flush_interval_s
        self._flush_max_records = flush_max_records
        self._sync_writes = sync_writes
        self._on_flush = on_flush
        self._lock = threading.Lock()
        self._queue: list[UsageRecord] = []
        self._closed = False
        self._conn = self._open_connection()
        self._ensure_schema()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if not sync_writes:
            self._worker = threading.Thread(
                target=self._flush_loop, name="agent-metering-flush", daemon=True
            )
            self._worker.start()
            atexit.register(self.close)

    def set_on_flush(
        self, callback: Optional[Callable[[list[UsageRecord]], None]]
    ) -> None:
        self._on_flush = callback

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
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
                    extra_metadata TEXT,
                    call_id TEXT,
                    unit_id TEXT,
                    status TEXT,
                    correlation_id TEXT
                )
                """
            )
            self._migrate_columns()
            self._conn.commit()

    def _migrate_columns(self) -> None:
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(usage_log)").fetchall()
        }
        for col, decl in (
            ("call_id", "TEXT"),
            ("unit_id", "TEXT"),
            ("status", "TEXT"),
            ("correlation_id", "TEXT"),
        ):
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE usage_log ADD COLUMN {col} {decl}"
                )

    def write(self, record: UsageRecord) -> None:
        if self._sync_writes:
            with self._lock:
                self._insert_many([record])
                self._conn.commit()
            if self._on_flush:
                self._on_flush([record])
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLiteStorage is closed")
            self._queue.append(record)
            should_flush = len(self._queue) >= self._flush_max_records
        if should_flush:
            self.flush()

    def _flush_loop(self) -> None:
        while not self._stop.wait(self._flush_interval_s):
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue
            self._queue = []
            self._insert_many(batch)
            self._conn.commit()
        if self._on_flush:
            self._on_flush(batch)

    def _insert_many(self, records: list[UsageRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO usage_log (
                timestamp, customer_id, feature, provider, model,
                input_tokens, output_tokens, cost_usd, latency_ms,
                extra_metadata, call_id, unit_id, status, correlation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.timestamp,
                    r.customer_id,
                    r.feature,
                    r.provider,
                    r.model,
                    r.input_tokens,
                    r.output_tokens,
                    r.cost_usd,
                    r.latency_ms,
                    r.extra_metadata,
                    r.call_id,
                    r.unit_id,
                    r.status,
                    r.correlation_id,
                )
                for r in records
            ],
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.flush()
        with self._lock:
            self._conn.close()

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def mark_prior_attempts_as_retry(self, correlation_id: str) -> None:
        """Mark previously committed rows for this correlation as retry_attempt."""
        if not correlation_id:
            return
        self.flush()
        with self._lock:
            self._conn.execute(
                """
                UPDATE usage_log
                SET status = 'retry_attempt'
                WHERE correlation_id = ?
                  AND (status IS NULL OR status != 'retry_attempt')
                """,
                (correlation_id,),
            )
            self._conn.commit()

    def query_spend_since(
        self, customer_id: str, since_ts: float
    ) -> float:
        self.flush()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0) AS total
                FROM usage_log
                WHERE customer_id = ? AND timestamp >= ?
                """,
                (customer_id, since_ts),
            ).fetchone()
        return float(row["total"] if row else 0.0)

    def _aggregate(
        self,
        group_column: str,
        since_ts: Optional[float] = None,
        *,
        where_extra: str = "",
        where_params: tuple[Any, ...] = (),
    ) -> dict[str, dict[str, Any]]:
        self.flush()
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
            {where_extra}
            GROUP BY group_key
            ORDER BY total_cost_usd DESC
        """
        with self._lock:
            rows = self._conn.execute(sql, (since_ts, *where_params)).fetchall()

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

    def query_cost_by_call(
        self,
        feature: Optional[str] = None,
        since_ts: Optional[float] = None,
    ) -> dict[str, dict[str, Any]]:
        """Roll up by call_id; optionally filter to one feature."""
        if feature is not None:
            return self._aggregate(
                "call_id",
                since_ts,
                where_extra="AND feature = ? AND call_id IS NOT NULL",
                where_params=(feature,),
            )
        return self._aggregate(
            "call_id",
            since_ts,
            where_extra="AND call_id IS NOT NULL",
        )

    def query_cost_by_unit_id(
        self, since_ts: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        return self._aggregate(
            "unit_id",
            since_ts,
            where_extra="AND unit_id IS NOT NULL",
        )

    def query_waste_spend(self, since_ts: Optional[float] = None) -> dict[str, Any]:
        """Spend from calls that produced no successful output (error / retry)."""
        self.flush()
        if since_ts is None:
            since_ts = 0.0
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                    COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                    COUNT(*) AS call_count
                FROM usage_log
                WHERE timestamp >= ?
                  AND status IN ('error', 'retry_attempt')
                """,
                (since_ts,),
            ).fetchone()
        return {
            "total_cost_usd": float(row["total_cost_usd"] or 0.0),
            "total_tokens": int(row["total_tokens"] or 0),
            "call_count": int(row["call_count"] or 0),
        }
