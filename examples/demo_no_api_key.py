"""Simulate multi-customer LLM usage and budget alerts — no API keys required."""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_metering import Meter, SQLiteStorage, check_budgets

CUSTOMERS = ["cust_alpha", "cust_beta", "cust_gamma"]
FEATURES = ["support_bot", "refund_flow"]
MODELS = [
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4o"),
    ("anthropic", "claude-sonnet-5"),
    ("google", "gemini-1.5-flash"),
]

# cust_gamma simulates a runaway agent loop (~15x volume).
CALL_COUNTS = {
    "cust_alpha": 8,
    "cust_beta": 8,
    "cust_gamma": 120,
}


def main() -> None:
    db_path = ROOT / "agent_metering.db"
    if db_path.exists():
        db_path.unlink()

    meter = Meter(storage=SQLiteStorage(db_path=db_path))
    rng = random.Random(42)

    print(f"Writing simulated usage to {db_path}\n")

    for customer_id, n_calls in CALL_COUNTS.items():
        for _ in range(n_calls):
            feature = rng.choice(FEATURES)
            provider, model = rng.choice(MODELS)
            input_tokens = rng.randint(200, 2500)
            output_tokens = rng.randint(50, 800)
            with meter.track(customer_id=customer_id, feature=feature) as t:
                t.record(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    extra_metadata={"sim": True},
                )

    print("=== Cost by customer ===")
    for cid, stats in meter.cost_by_customer().items():
        print(
            f"  {cid}: ${stats['total_cost_usd']:.4f} "
            f"({stats['total_tokens']} tokens, {stats['call_count']} calls)"
        )

    print("\n=== Cost by feature ===")
    for feature, stats in meter.cost_by_feature().items():
        print(
            f"  {feature}: ${stats['total_cost_usd']:.4f} "
            f"({stats['total_tokens']} tokens, {stats['call_count']} calls)"
        )

    def on_breach(scope: str, id_: str, spent: float, limit: float) -> None:
        print(
            f"  ALERT: {scope} '{id_}' spent ${spent:.4f} "
            f"(limit ${limit:.4f})"
        )

    print("\n=== Budget check (customer limit $0.05) ===")
    breaches = check_budgets(
        meter,
        customer_limits={c: 0.05 for c in CUSTOMERS},
        window_seconds=86400,
        on_breach=on_breach,
    )
    if not breaches:
        print("  No breaches.")
    else:
        print(f"\n{len(breaches)} breach(es) flagged.")


if __name__ == "__main__":
    main()
