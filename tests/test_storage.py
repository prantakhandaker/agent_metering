"""Tests for SQLiteStorage round-trip, WAL batching, and rollups."""

import time

from agent_metering.storage import SQLiteStorage, UsageRecord


def _record(**kwargs) -> UsageRecord:
    defaults = dict(
        timestamp=time.time(),
        customer_id="cust_a",
        feature="support_bot",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=0.10,
        latency_ms=12.5,
        extra_metadata=None,
    )
    defaults.update(kwargs)
    return UsageRecord(**defaults)


def test_sqlite_write_and_query_cost_by_customer(tmp_path):
    db = tmp_path / "test_metering.db"
    storage = SQLiteStorage(db_path=db, sync_writes=True)

    storage.write(
        _record(
            customer_id="cust_a",
            feature="support_bot",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.10,
        )
    )
    storage.write(
        _record(
            customer_id="cust_a",
            feature="refund_flow",
            input_tokens=2000,
            output_tokens=500,
            cost_usd=0.20,
            latency_ms=20.0,
        )
    )
    storage.write(
        _record(
            customer_id="cust_b",
            feature="support_bot",
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            latency_ms=5.0,
        )
    )

    by_customer = storage.query_cost_by_customer()
    assert list(by_customer.keys())[0] == "cust_a"
    assert abs(by_customer["cust_a"]["total_cost_usd"] - 0.30) < 1e-9
    assert by_customer["cust_a"]["total_tokens"] == 4000
    assert by_customer["cust_a"]["call_count"] == 2
    assert abs(by_customer["cust_b"]["total_cost_usd"] - 0.01) < 1e-9
    assert by_customer["cust_b"]["call_count"] == 1


def test_batched_write_does_not_block_on_disk(tmp_path):
    db = tmp_path / "batch.db"
    flush_times: list[float] = []

    def on_flush(records):
        flush_times.append(time.time())
        time.sleep(0.05)  # simulate slow disk

    storage = SQLiteStorage(
        db_path=db,
        flush_interval_s=1.0,
        flush_max_records=100,
        on_flush=on_flush,
    )
    start = time.time()
    for i in range(10):
        storage.write(_record(customer_id="c", cost_usd=0.01, input_tokens=1, output_tokens=1))
    elapsed = time.time() - start
    assert storage.pending_count() == 10
    assert elapsed < 0.05  # enqueue must not wait on flush sleep
    assert flush_times == []
    storage.flush()
    assert storage.pending_count() == 0
    assert storage.query_cost_by_customer()["c"]["call_count"] == 10
    storage.close()


def test_wal_mode_enabled(tmp_path):
    db = tmp_path / "wal.db"
    storage = SQLiteStorage(db_path=db, sync_writes=True)
    mode = storage._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    storage.close()


def test_cost_by_call_and_unit(tmp_path):
    db = tmp_path / "tags.db"
    storage = SQLiteStorage(db_path=db, sync_writes=True)
    storage.write(
        _record(feature="agent", call_id="step_1", unit_id="doc_9", cost_usd=0.02)
    )
    storage.write(
        _record(feature="agent", call_id="step_2", unit_id="doc_9", cost_usd=0.03)
    )
    storage.write(
        _record(feature="other", call_id="step_1", unit_id="doc_1", cost_usd=0.01)
    )
    by_call = storage.query_cost_by_call(feature="agent")
    assert by_call["step_1"]["total_cost_usd"] == 0.02
    assert by_call["step_2"]["total_cost_usd"] == 0.03
    by_unit = storage.query_cost_by_unit_id()
    assert abs(by_unit["doc_9"]["total_cost_usd"] - 0.05) < 1e-9
    assert by_unit["doc_9"]["call_count"] == 2


def test_waste_spend_and_retry_marking(tmp_path):
    db = tmp_path / "retry.db"
    storage = SQLiteStorage(db_path=db, sync_writes=True)
    storage.write(
        _record(
            correlation_id="corr-1",
            status="error",
            cost_usd=0.05,
            input_tokens=10,
            output_tokens=0,
        )
    )
    storage.mark_prior_attempts_as_retry("corr-1")
    storage.write(
        _record(
            correlation_id="corr-1",
            status="success",
            cost_usd=0.07,
            input_tokens=10,
            output_tokens=5,
        )
    )
    waste = storage.query_waste_spend()
    assert abs(waste["total_cost_usd"] - 0.05) < 1e-9
    assert waste["call_count"] == 1
