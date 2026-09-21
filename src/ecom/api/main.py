"""FastAPI service over precomputed pipeline artifacts. Run: make api (docs at /docs)."""

from __future__ import annotations

import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ecom.api import store
from ecom.api.schemas import (
    ChurnRequest,
    ChurnResponse,
    Customer,
    ForecastResponse,
    Health,
    Kpis,
)
from ecom.churn.model import FEATURES, REPEAT_CATEGORIES

app = FastAPI(
    title="E-Commerce Demand Forecasting & Customer Analytics API",
    version="1.0.0",
    description="Olist marketplace: weekly category demand forecasts, RFM segments, churn risk and CLV.",
)


@app.exception_handler(store.ArtifactMissing)
async def artifact_missing(_: Request, exc: store.ArtifactMissing):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (NaN -> None, timestamps -> ISO dates)."""
    out = df.copy()
    for c in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in out.to_dict("records")]


@app.get("/health", response_model=Health)
def health():
    try:
        store.table("customers")
        store.table("forecast")
        return Health(status="ok", artifacts_ready=True)
    except store.ArtifactMissing as e:
        return Health(status="degraded", artifacts_ready=False, detail=str(e))


@app.post("/admin/reload", include_in_schema=False)
def reload():
    store.clear()
    return {"reloaded": True}


@app.get("/kpis", response_model=Kpis)
def kpis():
    return store.json_artifact("kpis")


# ---------- forecasting
@app.get("/forecast/series", response_model=list[str])
def forecast_series():
    return sorted(store.table("forecast")["series"].unique())


@app.get("/forecast/leaderboard")
def forecast_leaderboard(series: str | None = None):
    if series:
        lb = store.table("forecast_leaderboard_by_series")
        lb = lb[lb["series"] == series]
        if lb.empty:
            raise HTTPException(404, f"unknown series '{series}'")
        return records(lb)
    return records(store.table("forecast_leaderboard"))


@app.get("/forecast/{series}", response_model=ForecastResponse)
def forecast(
    series: str,
    model: str = Query("best", description="'best' (backtest winner) or a model name"),
    horizon: int = Query(8, ge=1, le=8),
    history_weeks: int = Query(26, ge=0, le=104),
):
    fc = store.table("forecast")
    fc = fc[fc["series"] == series]
    if fc.empty:
        raise HTTPException(404, f"unknown series '{series}'")
    sel = fc[fc["is_best"]] if model == "best" else fc[fc["model"] == model]
    if sel.empty:
        raise HTTPException(404, f"unknown model '{model}'; options: best, {', '.join(sorted(fc['model'].unique()))}")
    name = sel["model"].iloc[0]
    lb = store.table("forecast_leaderboard_by_series")
    wape = lb.loc[(lb["series"] == series) & (lb["model"] == name), "wape"]
    hist = store.table("demand_weekly")
    hist = hist[hist["series"] == series].sort_values("week").tail(history_weeks)
    return ForecastResponse(
        series=series,
        model=name,
        is_backtest_best=bool(sel["is_best"].iloc[0]),
        backtest_wape=float(wape.iloc[0]) if len(wape) else None,
        history=records(hist[["week", "y"]]),
        forecast=records(sel.sort_values("week").head(horizon)[["week", "yhat", "lower", "upper"]]),
    )


# ---------- customers
@app.get("/segments")
def segments():
    return records(store.table("segment_summary"))


@app.get("/clusters")
def clusters():
    return records(store.table("cluster_profile"))


@app.get("/customers/{customer_unique_id}", response_model=Customer)
def customer(customer_unique_id: str):
    c = store.customers()
    if customer_unique_id not in c.index:
        raise HTTPException(404, "customer not found")
    return Customer(customer_unique_id=customer_unique_id, **records(c.loc[[customer_unique_id]])[0])


@app.get("/customers")
def customers(
    segment: str | None = None,
    risk_band: str | None = None,
    clv_tier: str | None = None,
    sort_by: str = Query("clv_12m", pattern="^(clv_12m|churn_risk|p_repeat|monetary|recency_days)$"),
    limit: int = Query(50, ge=1, le=1000),
):
    c = store.customers()
    for col, val in [("segment", segment), ("risk_band", risk_band), ("clv_tier", clv_tier)]:
        if val:
            c = c[c[col] == val]
    cols = ["customer_state", "segment", "cluster_name", "frequency", "monetary", "recency_days", "p_repeat",
            "churn_risk", "risk_band", "clv_12m", "clv_tier"]
    top = c.sort_values(sort_by, ascending=False).head(limit)[cols].reset_index()
    return {"total": int(len(c)), "items": records(top)}


@app.post("/churn/score", response_model=ChurnResponse)
def churn_score(req: ChurnRequest):
    m = store.model("churn_model")
    aov = req.monetary / req.frequency
    # a repeat buyer's first order predates their last; median gap between Olist repeat orders is ~30 days
    tenure = req.tenure_days if req.tenure_days is not None else req.recency_days + 30 * (req.frequency - 1)
    row = {
        "recency_days": req.recency_days,
        "tenure_days": max(tenure, req.recency_days),
        "frequency": req.frequency,
        "monetary": req.monetary,
        "avg_order_value": aov,
        "avg_items": req.avg_items,
        "avg_freight": req.avg_freight,
        "avg_review": req.avg_review,
        "avg_delivery_days": req.avg_delivery_days,
        "late_rate": req.late_rate,
        "avg_installments": req.avg_installments,
        "n_categories": req.n_categories or req.frequency,
        "credit_card_share": req.credit_card_share,
        "freight_ratio": req.avg_freight / aov if aov else 0.0,
        "is_sp": int(req.customer_state == "SP"),
        "is_rj_mg": int(req.customer_state in ("RJ", "MG")),
        **{f"cat_{c}": int(req.main_category == c) for c in REPEAT_CATEGORIES},
    }
    p = float(m.predict_proba(pd.DataFrame([row])[FEATURES])[:, 1][0])
    metrics = store.table("churn_metrics")
    return ChurnResponse(p_repeat=p, churn_risk=1 - p, model=str(metrics.loc[metrics["selected"], "model"].iloc[0]))


# ---------- insights
@app.get("/insights/states")
def states():
    return records(store.table("state_performance"))


@app.get("/insights/categories")
def categories(limit: int = Query(30, ge=1, le=100)):
    return records(store.table("category_performance").head(limit))


@app.get("/insights/affinity")
def affinity(level: str = Query("customer", pattern="^(order|customer)$"), limit: int = Query(20, ge=1, le=200)):
    return records(store.table(f"affinity_{level}").head(limit))
