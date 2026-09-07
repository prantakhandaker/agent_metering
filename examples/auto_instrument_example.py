"""Import-first plug-and-play demo (no API key required).

Easiest path in a real app::

  pip install "git+https://github.com/prantakhandaker/agent_metering.git"
  python -m agent_metering init
  # then: import agent_metering  at the top of your app

This script shows metering without calling a live API::

  python examples/auto_instrument_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_metering
from agent_metering import Meter, enable, set_feature, set_user
from agent_metering.instrument import _record_openai_response

enable()

set_user("cust_alpha")
set_feature("support_bot")

# Simulate what the OpenAI SDK returns after chat.completions.create
fake = SimpleNamespace(
    model="gpt-4o-mini",
    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
)
_record_openai_response(fake)

# Second user — same app, different request context
set_user("cust_beta")
_record_openai_response(
    SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
)

print("Instrumented:", agent_metering.is_enabled())
print("\nCost by customer (user):")
for customer_id, stats in Meter().cost_by_customer().items():
    print(
        f"  {customer_id}: ${stats['total_cost_usd']:.6f} "
        f"({stats['total_tokens']} tokens, {stats['call_count']} calls)"
    )
