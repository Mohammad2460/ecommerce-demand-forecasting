import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ecom.dashboard.common import SERIES, page, pretty, require_artifacts, style, table  # noqa: E402

page("Geography & Products", "Where revenue comes from, how logistics performs, and what sells together")
require_artifacts()

st_perf = table("state_performance")
a, b = st.columns(2)
top = st_perf.head(15)
fig = go.Figure(
    go.Bar(
        x=top["customer_state"],
        y=top["revenue"],
        marker=dict(color=SERIES[0], cornerradius=4),
        customdata=top[["revenue_share", "orders"]],
        hovertemplate="%{x}<br>R$ %{y:,.0f} (%{customdata[0]:.1%})<br>%{customdata[1]:,} orders<extra></extra>",
    )
)
a.plotly_chart(style(fig, 360, title="Revenue by customer state (top 15, R$)"), width="stretch", theme=None)

big = st_perf[st_perf["orders"] >= 200]
fig = go.Figure(
    go.Scatter(
        x=big["late_rate"],
        y=big["avg_review"],
        mode="markers+text",
        text=big["customer_state"],
        textposition="top center",
        textfont=dict(color="#52514e", size=11),
        marker=dict(
            size=(big["orders"] ** 0.5) / 7 + 8, color=SERIES[0], opacity=0.8, line=dict(color="#fcfcfb", width=2)
        ),
        customdata=big[["orders", "avg_delivery_days"]],
        hovertemplate="%{text}<br>late %{x:.1%} · review %{y:.2f}<br>%{customdata[0]:,} orders · "
        "%{customdata[1]:.1f} days avg<extra></extra>",
    )
)
fig.update_xaxes(tickformat=".0%", title="late delivery rate", showgrid=True, gridcolor="#e1e0d9")
fig.update_yaxes(title="avg review score")
b.plotly_chart(
    style(fig, 360, title="Late deliveries drag reviews down (states with 200+ orders)"),
    width="stretch",
    theme=None,
)

st.dataframe(
    st_perf.style.format(
        {
            "revenue": "R$ {:,.0f}",
            "avg_order_value": "R$ {:,.0f}",
            "avg_freight": "R$ {:,.2f}",
            "avg_delivery_days": "{:.1f}",
            "late_rate": "{:.1%}",
            "avg_review": "{:.2f}",
            "revenue_share": "{:.1%}",
            "freight_ratio": "{:.1%}",
            "orders": "{:,}",
            "customers": "{:,}",
        }
    ),
    hide_index=True,
    width="stretch",
    height=300,
)

st.divider()
st.subheader("Categories")
cat = table("category_performance")
n = st.slider("Show top N categories", 5, min(40, len(cat)), 15)
view = cat.head(n)
a, b = st.columns(2)
fig = go.Figure(
    go.Bar(
        y=[pretty(c) for c in view["category"]][::-1],
        x=view["units"][::-1],
        orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="%{y}: %{x:,} units<extra></extra>",
    )
)
a.plotly_chart(style(fig, 30 * n + 80, title="Units sold"), width="stretch", theme=None)
fig = go.Figure(
    go.Bar(
        y=[pretty(c) for c in view["category"]][::-1],
        x=view["avg_review"][::-1],
        orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    )
)
fig.update_xaxes(range=[3, 5])
b.plotly_chart(style(fig, 30 * n + 80, title="Average review (axis starts at 3)"), width="stretch", theme=None)

st.divider()
st.subheader("Category affinity (market basket)")
level = st.radio(
    "Basket definition",
    ["customer", "order"],
    horizontal=True,
    format_func=lambda x: "Same customer, any order" if x == "customer" else "Same order",
)
aff = table(f"affinity_{level}")
st.caption(
    "Lift > 1 means the pair co-occurs more than chance. Olist baskets are mostly single-item, "
    "so cross-sell signal is thin; rules with few occurrences are filtered out (min 5)."
)
st.dataframe(
    aff.head(25)
    .assign(antecedent=lambda d: d["antecedent"].map(pretty), consequent=lambda d: d["consequent"].map(pretty))
    .style.format({"support": "{:.4%}", "confidence": "{:.2%}", "lift": "{:.2f}"}),
    hide_index=True,
    width="stretch",
)
