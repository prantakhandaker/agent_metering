"""Example: integrate via proxy — change only base_url and optional headers.

Requires:
  1. Proxy running: uvicorn agent_metering.proxy:app --port 8787
  2. OPENAI_API_KEY env var set to a real key
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from agent_metering import Meter

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY before running this example.")

client = OpenAI(
    api_key=API_KEY,
    base_url="http://localhost:8787/proxy/openai/v1",
    default_headers={
        "X-Customer-Id": "cust_123",
        "X-Feature": "datachat_query",
    },
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print("Response:", response.choices[0].message.content)

meter = Meter()
print("\nCost by customer:")
for customer_id, stats in meter.cost_by_customer().items():
    print(
        f"  {customer_id}: ${stats['total_cost_usd']:.6f} "
        f"({stats['total_tokens']} tokens, {stats['call_count']} calls)"
    )
