from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    artifacts_ready: bool
    detail: str | None = None


class Kpis(BaseModel):
    orders: int
    customers: int
    revenue: float
    avg_order_value: float
    repeat_customer_rate: float
    avg_review_score: float
    late_delivery_rate: float
    avg_delivery_days: float
    data_start: str
    data_end: str


class Point(BaseModel):
    week: date
    y: float


class ForecastPoint(BaseModel):
    week: date
    yhat: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    series: str
    model: str
    is_backtest_best: bool
    backtest_wape: float | None
    history: list[Point]
    forecast: list[ForecastPoint]


class Customer(BaseModel):
    customer_unique_id: str
    customer_state: str | None
    frequency: int
    monetary: float
    recency_days: int
    avg_order_value: float
    avg_review: float
    segment: str
    rfm_score: str
    cluster_name: str
    p_repeat: float
    churn_risk: float
    risk_band: str
    clv_12m: float
    clv_tier: str


class ChurnRequest(BaseModel):
    recency_days: int = Field(ge=0, examples=[120])
    frequency: int = Field(ge=1, examples=[1])
    monetary: float = Field(ge=0, examples=[180.0])
    tenure_days: int | None = Field(
        default=None, ge=0,
        description="days since first order; defaults to recency_days + 30 per repeat order",
    )
    avg_items: float = 1.0
    avg_freight: float = 20.0
    avg_review: float = Field(default=4.0, ge=1, le=5)
    avg_delivery_days: float = 12.0
    late_rate: float = Field(default=0.0, ge=0, le=1)
    avg_installments: float = 2.0
    n_categories: int | None = Field(default=None, ge=1, description="defaults to frequency")
    credit_card_share: float = Field(default=1.0, ge=0, le=1)
    customer_state: str = "SP"
    main_category: str | None = Field(default=None, examples=["bed_bath_table"])


class ChurnResponse(BaseModel):
    p_repeat: float
    churn_risk: float
    model: str
