import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ecom.dashboard.common import SEQ_BLUE, SERIES, brl, page, require_artifacts, style, table  # noqa: E402

page(
    "Customer Segments",
    "RFM rule-based segments and K-Means behavioural clusters · one row per real customer (customer_unique_id)",
)
require_artifacts()
cust = table("customers")

st.subheader("RFM segments")
seg = table("segment_summary")
a, b = st.columns([3, 2])
a.dataframe(
    seg.style.format(
        {
            "avg_recency": "{:.0f} d",
            "avg_frequency": "{:.2f}",
            "avg_monetary": "R$ {:,.0f}",
            "revenue": "R$ {:,.0f}",
            "share_customers": "{:.1%}",
            "share_revenue": "{:.1%}",
            "customers": "{:,}",
        }
    ),
    hide_index=True,
    width="stretch",
    height=390,
)
# Quintile scores give every R x M cell roughly equal counts, so colour by value instead.
heat = cust.pivot_table(index="R", columns="M", values="clv_12m", aggfunc="mean").sort_index(ascending=False)
counts = cust.pivot_table(index="R", columns="M", values="clv_12m", aggfunc="count").reindex_like(heat)
fig = go.Figure(
    go.Heatmap(
        z=heat.values,
        x=[f"M{m}" for m in heat.columns],
        y=[f"R{r}" for r in heat.index],
        customdata=counts.values,
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        xgap=2,
        ygap=2,
        hovertemplate="%{y} · %{x}<br>avg CLV R$ %{z:,.2f}<br>%{customdata:,} customers<extra></extra>",
        colorbar=dict(title="R$", thickness=10),
    )
)
b.plotly_chart(
    style(fig, 390, title="Average 12-month CLV by Recency × Monetary score (5 = best)"),
    width="stretch",
    theme=None,
)
st.caption(
    "Frequency uses fixed bins (1, 2, 3, 4, 5+ orders) because ~97% of Olist customers purchase once, "
    "which makes frequency quintiles meaningless."
)

st.divider()
st.subheader("Behavioural clusters (K-Means)")
prof = table("cluster_profile")
k_scores = table("cluster_k_scores")
a, b = st.columns([2, 3])
fig = go.Figure(
    go.Bar(
        x=prof["customers"],
        y=prof["cluster_name"],
        orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="%{y}<br>%{x:,} customers<extra></extra>",
    )
)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
a.plotly_chart(style(fig, 330, title="Cluster sizes"), width="stretch", theme=None)
show = prof.drop(columns=["cluster"]).set_index("cluster_name")
fmt = {c: "{:.2f}" for c in show.columns} | {
    "customers": "{:,}",
    "monetary": "R$ {:,.0f}",
    "avg_order_value": "R$ {:,.0f}",
    "late_rate": "{:.0%}",
    "recency_days": "{:.0f}",
}
b.dataframe(
    show.style.format(fmt),
    width="stretch",
)
if len(k_scores):
    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])
    st.caption(
        f"k = {best_k} chosen by silhouette score "
        f"({k_scores['silhouette'].max():.3f}; tried k = {k_scores['k'].min()}–{k_scores['k'].max()})."
    )

st.divider()
st.subheader("Segment × cluster")
xt = cust.pivot_table(index="segment", columns="cluster_name", values="clv_12m", aggfunc="count", fill_value=0)
xt = xt.reindex([s for s in seg["segment"] if s in xt.index])
st.dataframe(xt.style.background_gradient(cmap="Blues", axis=None).format("{:,}"), width="stretch")
st.caption(f"Total 12-month revenue CLV across all customers: {brl(cust['clv_12m'].sum())}")
