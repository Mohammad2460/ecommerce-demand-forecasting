import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ecom.dashboard.common import SERIES, page, pretty, require_artifacts, style, table  # noqa: E402
from ecom.forecasting.evaluate import leaderboard  # noqa: E402

page("Demand Forecast", "Weekly item demand per category · 8-week horizon · models compared on a 3-fold backtest")
require_artifacts()

demand, fc = table("demand_weekly"), table("forecast")
lb_series = table("forecast_leaderboard_by_series")
series_list = sorted(fc["series"].unique(), key=lambda s: (s != "__total__", s))

c = st.columns([2, 2, 1, 1])
series = c[0].selectbox("Series", series_list, format_func=pretty)
models = fc.loc[fc["series"] == series, "model"].unique()
best = fc.loc[(fc["series"] == series) & fc["is_best"], "model"].iloc[0]
model = c[1].selectbox(
    "Model", ["best", *sorted(models)], format_func=lambda m: f"Best in backtest ({best})" if m == "best" else m
)
horizon = c[2].slider("Horizon (weeks)", 1, 8, 8)
hist_weeks = c[3].slider("History (weeks)", 8, 90, 52)
model = best if model == "best" else model

h = demand[demand["series"] == series].sort_values("week").tail(hist_weeks)
f = fc[(fc["series"] == series) & (fc["model"] == model)].sort_values("week").head(horizon)
wape = lb_series.loc[(lb_series["series"] == series) & (lb_series["model"] == model), "wape"]

m = st.columns(4)
m[0].metric(f"Forecast, next {horizon} weeks", f"{f['yhat'].sum():,.0f} units")
m[1].metric(
    f"Last {horizon} weeks actual",
    f"{h['y'].tail(horizon).sum():,.0f} units",
    delta=f"{f['yhat'].sum() / max(h['y'].tail(horizon).sum(), 1) - 1:+.1%} forecast vs actual",
    delta_color="off",
)
m[2].metric("Backtest WAPE", f"{wape.iloc[0]:.1%}" if len(wape) else "–")
m[3].metric("Model", model)

fig = go.Figure()
fig.add_scatter(
    x=h["week"],
    y=h["y"],
    name="Actual",
    line=dict(color=SERIES[0], width=2),
    hovertemplate="%{y:,.0f} units<extra>Actual</extra>",
)
fig.add_scatter(
    x=list(f["week"]) + list(f["week"][::-1]),
    y=list(f["upper"]) + list(f["lower"][::-1]),
    fill="toself",
    mode="lines",
    fillcolor="rgba(235,104,52,0.15)",
    line=dict(width=0),
    name="80% interval",
    hoverinfo="skip",
)
fig.add_scatter(
    x=f["week"],
    y=f["yhat"],
    name="Forecast",
    mode="lines+markers",
    line=dict(color=SERIES[1], width=2, dash="dot"),
    marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
    customdata=f[["lower", "upper"]],
    hovertemplate="%{y:,.0f} units (80%: %{customdata[0]:,.0f}–%{customdata[1]:,.0f})<extra>Forecast</extra>",
)
st.plotly_chart(
    style(fig, 420, title=f"{pretty(series)} · weekly units", hovermode="x unified"),
    width="stretch",
    theme=None,
)

a, b = st.columns(2)
lb = lb_series[lb_series["series"] == series].sort_values("wape")
a.subheader("Model comparison (this series)")
a.dataframe(
    lb[["model", "wape", "smape", "mae", "rmse", "bias"]]
    .style.format({"wape": "{:.1%}", "smape": "{:.1%}", "mae": "{:.1f}", "rmse": "{:.1f}", "bias": "{:+.1%}"})
    .highlight_min(subset=["wape"], color="#cde2fb"),
    hide_index=True,
    width="stretch",
)

bt = table("forecast_backtest")
by_h = leaderboard(bt[bt["series"] == series], ["h", "model"])
fig = go.Figure()
for i, mdl in enumerate(["naive", "ets_damped", "arima_111", "lightgbm"]):  # <=4 series, fixed slots
    d = by_h[by_h["model"] == mdl]
    fig.add_scatter(
        x=d["h"],
        y=d["wape"],
        name=mdl,
        mode="lines+markers",
        line=dict(color=SERIES[i], width=2),
        marker=dict(size=8),
        hovertemplate="h=%{x}: %{y:.1%}<extra>" + mdl + "</extra>",
    )
fig.update_yaxes(tickformat=".0%")
fig.update_xaxes(title="weeks ahead", dtick=1)
b.subheader("Backtest error by weeks ahead")
b.plotly_chart(style(fig, 320, hovermode="x unified"), width="stretch", theme=None)

with st.expander("Overall leaderboard (all series pooled) and LightGBM feature importance"):
    x, y = st.columns(2)
    x.dataframe(
        table("forecast_leaderboard").style.format(
            {"wape": "{:.1%}", "smape": "{:.1%}", "mae": "{:.1f}", "rmse": "{:.1f}", "bias": "{:+.1%}"}
        ),
        hide_index=True,
        width="stretch",
    )
    fi = table("forecast_feature_importance").head(10).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            y=fi["feature"],
            x=fi["gain"],
            orientation="h",
            marker=dict(color=SERIES[0], cornerradius=4),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    y.plotly_chart(style(fig, 320, title="Feature importance (gain)"), width="stretch", theme=None)

st.download_button("Download all forecasts (CSV)", fc.to_csv(index=False), "forecasts.csv", "text/csv")
