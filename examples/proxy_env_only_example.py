"""Zero-code example: OpenAI client with no base_url — uses OPENAI_BASE_URL env.

Requires:
  1. Proxy running with attribution defaults, e.g.:
       set AGENT_METERING_CUSTOMER_ID=acme_corp
       set AGENT_METERING_FEATURE=support_bot
       uvicorn agent_metering.proxy:app --port 8787
  2. Env pointing at the proxy (or use the CLI):
       set OPENAI_BASE_URL=http://127.0.0.1:8787/proxy/openai/v1
       python -m agent_metering run --start-proxy --customer acme_corp --feature support_bot -- python examples/proxy_env_only_example.py
  3. OPENAI_API_KEY set to a real key
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

if not os.environ.get("OPENAI_BASE_URL"):
    raise SystemExit(
        "Set OPENAI_BASE_URL to the metering proxy "
        "(e.g. http://127.0.0.1:8787/proxy/openai/v1), "
        "or run via: python -m agent_metering run --start-proxy -- ..."
    )

# No base_url and no metering headers — SDK + proxy env defaults only.
client = OpenAI(api_key=API_KEY)

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
