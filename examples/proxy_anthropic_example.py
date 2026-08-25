"""Anthropic integration via universal proxy — change only base_url."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic

from agent_metering import Meter

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise SystemExit("Set ANTHROPIC_API_KEY before running this example.")

client = Anthropic(
    api_key=API_KEY,
    base_url="http://localhost:8787/proxy/anthropic",
    default_headers={
        "X-Customer-Id": "cust_456",
        "X-Feature": "anthropic_chat",
    },
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print("Response:", response.content[0].text)

meter = Meter()
print("\nCost by customer:")
for customer_id, stats in meter.cost_by_customer().items():
    print(
        f"  {customer_id}: ${stats['total_cost_usd']:.6f} "
        f"({stats['total_tokens']} tokens, {stats['call_count']} calls)"
    )
