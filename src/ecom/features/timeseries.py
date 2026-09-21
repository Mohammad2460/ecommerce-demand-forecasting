"""Weekly demand panel and supervised features for forecasting."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from ecom.config import FORECAST_FREQ, TOP_N_CATEGORIES

TOTAL = "__total__"
LAGS = [1, 2, 3, 4, 8, 12]
ROLLS = [4, 8]


def week_start(ts: pd.Series) -> pd.Series:
    """Monday of the week containing each timestamp."""
    return pd.to_datetime(ts).dt.to_period("W-SUN").dt.start_time


def weekly_demand(fact_items: pd.DataFrame, top_n: int = TOP_N_CATEGORIES) -> pd.DataFrame:
    """Long panel [series, week, y] of item units per week for the top-N categories plus the total.

    The final week is dropped when incomplete so partial weeks don't look like demand collapses.
    """
    df = fact_items[["order_purchase_timestamp", "category"]].copy()
    df["week"] = week_start(df["order_purchase_timestamp"])
    top = df["category"].value_counts().drop("unknown", errors="ignore").head(top_n).index
    weeks = pd.date_range(df["week"].min(), df["week"].max(), freq=FORECAST_FREQ)
    last_day = df["order_purchase_timestamp"].max().normalize()
    if last_day < weeks[-1] + pd.Timedelta(days=6):
        weeks = weeks[:-1]

    per_cat = (
        df[df["category"].isin(top)].groupby(["category", "week"]).size()
        .unstack(0).reindex(weeks, fill_value=0).fillna(0)
    )
    per_cat[TOTAL] = df.groupby("week").size().reindex(weeks, fill_value=0)
    long = per_cat.rename_axis("week").reset_index().melt("week", var_name="series", value_name="y")
    return long[["series", "week", "y"]].sort_values(["series", "week"]).reset_index(drop=True)


def calendar_features(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    years = range(weeks.min().year, weeks.max().year + 2)
    br = holidays.Brazil(years=years)
    feats = pd.DataFrame(index=weeks)
    feats["weekofyear"] = weeks.isocalendar().week.astype(int).values
    feats["month"] = weeks.month
    feats["n_holidays"] = [sum((w + pd.Timedelta(days=d)).date() in br for d in range(7)) for w in weeks]
    bf = {pd.Timestamp(y, 11, 1) + pd.offsets.WeekOfMonth(week=3, weekday=4) for y in years}
    feats["black_friday"] = [int(any(w <= d < w + pd.Timedelta(days=7) for d in bf)) for w in weeks]
    feats["post_black_friday"] = [int(any(w - pd.Timedelta(days=7) <= d < w for d in bf)) for w in weeks]
    feats["christmas"] = ((weeks.month == 12) & (weeks.day >= 8) & (weeks.day <= 24)).astype(int)
    return feats


def supervised(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling (on log1p(y)) and calendar features. Rows with NaN lags are kept; LightGBM handles them."""
    df = panel.sort_values(["series", "week"]).copy()
    df["ly"] = np.log1p(df["y"])
    g = df.groupby("series")["ly"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLLS:
        df[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
    df["roll_std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())
    cal = calendar_features(pd.DatetimeIndex(sorted(df["week"].unique())))
    df = df.merge(cal, left_on="week", right_index=True, how="left")
    df["series_code"] = df["series"].astype("category")
    return df


FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in ROLLS]
    + ["roll_std_4", "weekofyear", "month", "n_holidays", "black_friday", "post_black_friday", "christmas",
       "series_code"]
)
