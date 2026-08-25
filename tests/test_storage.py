"""Tests for SQLiteStorage round-trip."""

import time

from agent_metering.storage import SQLiteStorage, UsageRecord


def test_sqlite_write_and_query_cost_by_customer(tmp_path):
    db = tmp_path / "test_metering.db"
    storage = SQLiteStorage(db_path=db)

    storage.write(
        UsageRecord(
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
    )
    storage.write(
        UsageRecord(
            timestamp=time.time(),
            customer_id="cust_a",
            feature="refund_flow",
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=2000,
            output_tokens=500,
            cost_usd=0.20,
            latency_ms=20.0,
            extra_metadata=None,
        )
    )
    storage.write(
        UsageRecord(
            timestamp=time.time(),
            customer_id="cust_b",
            feature="support_bot",
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            latency_ms=5.0,
            extra_metadata=None,
        )
    )

    by_customer = storage.query_cost_by_customer()
    assert list(by_customer.keys())[0] == "cust_a"
    assert abs(by_customer["cust_a"]["total_cost_usd"] - 0.30) < 1e-9
    assert by_customer["cust_a"]["total_tokens"] == 4000
    assert by_customer["cust_a"]["call_count"] == 2
    assert abs(by_customer["cust_b"]["total_cost_usd"] - 0.01) < 1e-9
    assert by_customer["cust_b"]["call_count"] == 1
