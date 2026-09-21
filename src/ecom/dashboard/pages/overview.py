import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ecom.dashboard.common import (  # noqa: E402
    SERIES,
    brl,
    kpis,
    page,
    pct,
    pretty,
    require_artifacts,
    style,
    table,
)

page(
    "E-Commerce Demand & Customer Analytics",
    "Olist Brazilian marketplace · delivered orders · weekly demand forecasts, segments, churn risk and CLV",
)
require_artifacts()
k = kpis()
st.caption(f"Data window: {k['data_start']} → {k['data_end']}")

c = st.columns(4)
c[0].metric("Revenue", f"R$ {k['revenue'] / 1e6:.1f}M")
c[1].metric("Orders", f"{k['orders']:,}")
c[2].metric("Customers", f"{k['customers']:,}")
c[3].metric("Avg order value", brl(k["avg_order_value"]))
c = st.columns(4)
c[0].metric("Repeat customers", pct(k["repeat_customer_rate"]))
c[1].metric("Avg review", f"{k['avg_review_score']:.2f} / 5")
c[2].metric("Late deliveries", pct(k["late_delivery_rate"]))
c[3].metric("Avg delivery", f"{k['avg_delivery_days']:.1f} d")

st.divider()
left, right = st.columns([3, 2])

demand = table("demand_weekly")
fc = table("forecast")
tot = demand[demand["series"] == "__total__"]
f = fc[(fc["series"] == "__total__") & fc["is_best"]].sort_values("week")
fig = go.Figure()
fig.add_scatter(
    x=tot["week"],
    y=tot["y"],
    name="Actual",
    line=dict(color=SERIES[0], width=2),
    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra>Actual</extra>",
)
fig.add_scatter(
    x=list(f["week"]) + list(f["week"][::-1]),
    y=list(f["upper"]) + list(f["lower"][::-1]),
    fill="toself",
    mode="lines",
    fillcolor="rgba(235,104,52,0.15)",
    line=dict(width=0),
    hoverinfo="skip",
    name="80% interval",
)
fig.add_scatter(
    x=f["week"],
    y=f["yhat"],
    name=f"Forecast ({f['model'].iloc[0]})",
    mode="lines",
    line=dict(color=SERIES[1], width=2, dash="dot"),
    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra>Forecast</extra>",
)
left.plotly_chart(
    style(fig, 380, title="Weekly units sold: history and 8-week forecast", hovermode="x unified"),
    use_container_width=True,
    theme=None,
)

nxt = fc[fc["is_best"] & (fc["series"] != "__total__")].groupby("series")["yhat"].sum()
last8 = demand[demand["week"] > demand["week"].max() - pd.Timedelta(weeks=8)]
last8 = last8[last8["series"] != "__total__"].groupby("series")["y"].sum()
growth = ((nxt / last8) - 1).dropna().sort_values()
fig = go.Figure(
    go.Bar(
        y=[pretty(s) for s in growth.index],
        x=growth.values,
        orientation="h",
        marker=dict(color=[SERIES[0] if v >= 0 else SERIES[7] for v in growth.values], cornerradius=4),
        hovertemplate="%{y}<br>%{x:+.1%} vs last 8 weeks<extra></extra>",
    )
)
fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor="#e1e0d9")
fig.update_yaxes(showgrid=False)
right.plotly_chart(
    style(fig, 380, title="Next 8 weeks vs last 8 weeks, by category"), use_container_width=True, theme=None
)

st.divider()
a, b = st.columns(2)
seg = table("segment_summary")
fig = go.Figure()
fig.add_bar(
    y=seg["segment"],
    x=seg["share_customers"],
    orientation="h",
    name="Customers",
    marker=dict(color=SERIES[0], cornerradius=4),
    hovertemplate="%{y}: %{x:.1%} of customers<extra></extra>",
)
fig.add_bar(
    y=seg["segment"],
    x=seg["share_revenue"],
    orientation="h",
    name="Revenue",
    marker=dict(color=SERIES[1], cornerradius=4),
    hovertemplate="%{y}: %{x:.1%} of revenue<extra></extra>",
)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="#e1e0d9")
a.plotly_chart(
    style(fig, 420, title="RFM segments: share of customers vs revenue", barmode="group"),
    use_container_width=True,
    theme=None,
)

cat = table("category_performance").head(12).iloc[::-1]
fig = go.Figure(
    go.Bar(
        y=[pretty(s) for s in cat["category"]],
        x=cat["revenue"],
        orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        customdata=cat[["units", "avg_review"]],
        hovertemplate="%{y}<br>R$ %{x:,.0f}<br>%{customdata[0]:,} units · review %{customdata[1]:.2f}<extra></extra>",
    )
)
fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
fig.update_yaxes(showgrid=False)
b.plotly_chart(style(fig, 420, title="Top categories by revenue (R$)"), use_container_width=True, theme=None)
