"""Azure OpenAI integration via universal proxy (responses.create)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from agent_metering import Meter

API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set AZURE_OPENAI_API_KEY before running this example.")

MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5-mini")

client = OpenAI(
    api_key=API_KEY,
    base_url="http://localhost:8787/proxy/azure/v1",
    default_headers={
        "X-Customer-Id": "cust_azure",
        "X-Feature": "azure_responses",
    },
)

response = client.responses.create(
    model=MODEL,
    input="Say hello in one short sentence.",
    max_output_tokens=64,
)

print("Response:", getattr(response, "output_text", response))

meter = Meter()
print("\nCost by customer:")
for customer_id, stats in meter.cost_by_customer().items():
    print(
        f"  {customer_id}: ${stats['total_cost_usd']:.6f} "
        f"({stats['total_tokens']} tokens, {stats['call_count']} calls)"
    )
