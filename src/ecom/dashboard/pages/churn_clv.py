import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ecom.api.main import churn_score  # noqa: E402
from ecom.api.schemas import ChurnRequest  # noqa: E402
from ecom.churn.model import REPEAT_CATEGORIES  # noqa: E402
from ecom.dashboard.common import SERIES, brl, page, pct, pretty, require_artifacts, style, table  # noqa: E402

page(
    "Churn Risk & Customer Lifetime Value",
    "Repeat-purchase propensity (180-day window, out-of-time validated) and 12-month revenue CLV",
)
require_artifacts()
cust = table("customers")
met = table("churn_metrics")
sel = met[met["selected"]].iloc[0]

c = st.columns(5)
c[0].metric("Selected model", sel["model"])
c[1].metric("ROC-AUC (test)", f"{sel['roc_auc']:.3f}")
c[2].metric(
    "PR-AUC vs base rate",
    f"{sel['pr_auc']:.3f}",
    delta=f"{sel['pr_auc'] / sel['base_rate']:.1f}× base",
    delta_color="off",
)
c[3].metric("Top-10% lift", f"{sel['lift_top10']:.2f}×")
c[4].metric("Base repeat rate", pct(sel["base_rate"], 2))
st.info(
    "Olist is dominated by one-time buyers, so repeat purchase is hard to predict. The model is best used "
    "for ranking (who to target first), not as a confident yes/no. Probabilities are calibrated "
    f"(mean predicted {cust['p_repeat'].mean():.2%} vs observed {sel['base_rate']:.2%})."
)

tab1, tab2, tab3 = st.tabs(["Target list", "Customer lookup", "What-if scorer"])

with tab1:
    f = st.columns(4)
    segs = f[0].multiselect("Segment", sorted(cust["segment"].unique()))
    bands = f[1].multiselect("Risk band", ["very high", "high", "medium", "low", "very low"])
    tiers = f[2].multiselect("CLV tier", ["top 5%", "high", "mid", "low"])
    sort = f[3].selectbox("Sort by", ["clv_12m", "p_repeat", "churn_risk", "monetary", "recency_days"])
    view = cust
    if segs:
        view = view[view["segment"].isin(segs)]
    if bands:
        view = view[view["risk_band"].isin(bands)]
    if tiers:
        view = view[view["clv_tier"].isin(tiers)]
    m = st.columns(3)
    m[0].metric("Customers matched", f"{len(view):,}")
    m[1].metric("Expected 12-month CLV", brl(view["clv_12m"].sum()))
    m[2].metric("Expected repeat buyers (180d)", f"{view['p_repeat'].sum():,.0f}")
    cols = [
        "customer_unique_id",
        "customer_state",
        "segment",
        "cluster_name",
        "frequency",
        "monetary",
        "recency_days",
        "p_repeat",
        "risk_band",
        "clv_12m",
        "clv_tier",
    ]
    out = view.sort_values(sort, ascending=sort == "recency_days").head(500)[cols]
    st.dataframe(
        out.style.format({"monetary": "R$ {:,.0f}", "p_repeat": "{:.2%}", "clv_12m": "R$ {:,.2f}"}),
        hide_index=True,
        use_container_width=True,
        height=380,
    )
    st.download_button("Download matched customers (CSV)", view[cols].to_csv(index=False), "target_list.csv")

with tab2:
    cid = st.text_input("customer_unique_id", value=cust.sort_values("clv_12m").iloc[-1]["customer_unique_id"])
    row = cust[cust["customer_unique_id"] == cid.strip()]
    if row.empty:
        st.warning("Customer not found.")
    else:
        r = row.iloc[0]
        c = st.columns(4)
        c[0].metric("Segment", r["segment"])
        c[1].metric("Cluster", r["cluster_name"])
        c[2].metric("P(repeat, 180d)", pct(r["p_repeat"], 2), delta=r["risk_band"] + " risk", delta_color="off")
        c[3].metric("12-month CLV", brl(r["clv_12m"], 2), delta=r["clv_tier"], delta_color="off")
        c = st.columns(4)
        c[0].metric("Orders", int(r["frequency"]))
        c[1].metric("Total spent", brl(r["monetary"], 2))
        c[2].metric("Days since last order", int(r["recency_days"]))
        c[3].metric("Avg review", f"{r['avg_review']:.1f}")

with tab3:
    st.caption("Score a hypothetical customer with the deployed model (same logic as POST /churn/score).")
    c = st.columns(4)
    recency = c[0].number_input("Days since last order", 0, 700, 60)
    freq = c[1].number_input("Orders", 1, 20, 1)
    spent = c[2].number_input("Total spent (R$)", 0.0, 20000.0, 180.0, step=10.0)
    review = c[3].slider("Avg review", 1.0, 5.0, 4.5, 0.5)
    c = st.columns(4)
    late = c[0].slider("Late delivery rate", 0.0, 1.0, 0.0, 0.25)
    inst = c[1].number_input("Avg installments", 1, 24, 2)
    state = c[2].selectbox("State", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "other"])
    catg = c[3].selectbox("Main category", ["(none)", *REPEAT_CATEGORIES], format_func=pretty)
    res = churn_score(
        ChurnRequest(
            recency_days=recency,
            frequency=freq,
            monetary=spent,
            avg_review=review,
            late_rate=late,
            avg_installments=inst,
            customer_state=state,
            main_category=None if catg == "(none)" else catg,
        )
    )
    pct_rank = (cust["p_repeat"] < res.p_repeat).mean()
    c = st.columns(2)
    c[0].metric("P(repeat within 180 days)", pct(res.p_repeat, 2))
    c[1].metric("Percentile among current customers", f"{pct_rank:.0%}")

st.divider()
a, b = st.columns(2)
imp = table("churn_importance").head(12).iloc[::-1]
imp_label = "|standardised coefficient|" if sel["model"] == "logistic" else "gain"
fig = go.Figure(
    go.Bar(
        y=imp["feature"],
        x=imp["importance"],
        orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    )
)
a.plotly_chart(
    style(fig, 380, title=f"What drives repeat purchase ({sel['model']}: {imp_label})"),
    use_container_width=True,
    theme=None,
)
by = cust.groupby("segment")["clv_12m"].mean().sort_values()
fig = go.Figure(
    go.Bar(
        y=by.index,
        x=by.values,
        orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="%{y}: R$ %{x:,.2f}<extra></extra>",
    )
)
b.plotly_chart(style(fig, 380, title="Average 12-month CLV by segment (R$)"), use_container_width=True, theme=None)
