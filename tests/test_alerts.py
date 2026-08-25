"""Tests for check_budgets()."""

from agent_metering import Meter, SQLiteStorage, check_budgets


def test_check_budgets_flags_breach_and_calls_on_breach(tmp_path):
    meter = Meter(storage=SQLiteStorage(db_path=tmp_path / "alerts.db"))

    with meter.track(customer_id="cust_hot", feature="support_bot") as t:
        # Enough tokens on an expensive model to exceed $0.05
        t.record(
            provider="openai",
            model="gpt-4-turbo",
            input_tokens=5000,
            output_tokens=2000,
        )

    calls: list[tuple] = []

    def on_breach(scope, id_, spent, limit):
        calls.append((scope, id_, spent, limit))

    breaches = check_budgets(
        meter,
        customer_limits={"cust_hot": 0.05, "cust_cold": 1.0},
        window_seconds=86400,
        on_breach=on_breach,
    )

    assert len(breaches) == 1
    assert breaches[0]["scope"] == "customer"
    assert breaches[0]["id"] == "cust_hot"
    assert breaches[0]["spent"] > 0.05
    assert breaches[0]["limit"] == 0.05

    assert len(calls) == 1
    assert calls[0][0] == "customer"
    assert calls[0][1] == "cust_hot"


def test_check_budgets_no_breach(tmp_path):
    meter = Meter(storage=SQLiteStorage(db_path=tmp_path / "alerts2.db"))
    with meter.track(customer_id="cust_ok", feature="support_bot") as t:
        t.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=10,
        )

    breaches = check_budgets(
        meter,
        customer_limits={"cust_ok": 10.0},
        window_seconds=86400,
    )
    assert breaches == []
