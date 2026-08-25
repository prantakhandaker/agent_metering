"""Tests for calculate_cost()."""

from agent_metering.pricing import calculate_cost


def test_calculate_cost_known_model():
    # gpt-4o: input 0.0025 / 1k, output 0.01 / 1k
    cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
    assert abs(cost - 0.0125) < 1e-9


def test_calculate_cost_partial_tokens():
    cost = calculate_cost("gpt-4o-mini", input_tokens=500, output_tokens=250)
    expected = (500 / 1000.0) * 0.00015 + (250 / 1000.0) * 0.0006
    assert abs(cost - expected) < 1e-12


def test_calculate_cost_unknown_model_returns_zero():
    assert calculate_cost("not-a-real-model", 1000, 1000) == 0.0
