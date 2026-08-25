"""Streamlit dashboard for agent_metering costs.

Run from the repo root::

    streamlit run examples/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_metering import Meter, SQLiteStorage, check_budgets

st.set_page_config(page_title="Agent Metering", layout="wide")
st.title("Agent Cost Metering")
st.caption("Live spend from the local SQLite usage log.")

db_path = st.sidebar.text_input("SQLite DB path", value="agent_metering.db")
meter = Meter(storage=SQLiteStorage(db_path=db_path))

by_customer = meter.cost_by_customer()
by_feature = meter.cost_by_feature()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Cost by customer")
    if by_customer:
        df_c = pd.DataFrame(
            [
                {
                    "customer_id": k,
                    "total_cost_usd": v["total_cost_usd"],
                    "total_tokens": v["total_tokens"],
                    "call_count": v["call_count"],
                }
                for k, v in by_customer.items()
            ]
        )
        st.bar_chart(df_c.set_index("customer_id")["total_cost_usd"])
        st.dataframe(df_c, use_container_width=True)
    else:
        st.info("No customer usage recorded yet. Run examples/demo_no_api_key.py first.")

with col_right:
    st.subheader("Cost by feature")
    if by_feature:
        df_f = pd.DataFrame(
            [
                {
                    "feature": k,
                    "total_cost_usd": v["total_cost_usd"],
                    "total_tokens": v["total_tokens"],
                    "call_count": v["call_count"],
                }
                for k, v in by_feature.items()
            ]
        )
        st.bar_chart(df_f.set_index("feature")["total_cost_usd"])
        st.dataframe(df_f, use_container_width=True)
    else:
        st.info("No feature usage recorded yet.")

st.divider()
st.subheader("On-demand budget check")
with st.form("budget_form"):
    customer_id = st.text_input("Customer ID", value="cust_gamma")
    limit = st.number_input("Limit (USD)", min_value=0.0, value=0.05, step=0.01)
    submitted = st.form_submit_button("Check budget")

if submitted:
    breaches = check_budgets(
        meter,
        customer_limits={customer_id: float(limit)},
        window_seconds=86400 * 365,
    )
    spent = by_customer.get(customer_id, {}).get("total_cost_usd", 0.0)
    st.write(f"Current spend for `{customer_id}`: **${spent:.4f}** (limit ${limit:.4f})")
    if breaches:
        st.error(f"Budget breached for {customer_id}")
    else:
        st.success("Within budget")
