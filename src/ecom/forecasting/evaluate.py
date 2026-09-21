"""Rolling-origin backtesting, model leaderboard and final forecasts with prediction intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ecom.config import FORECAST_FREQ, FORECAST_HORIZON
from ecom.forecasting import baselines, stat
from ecom.forecasting.ml import GlobalLGBM

UNIVARIATE: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "naive": baselines.naive,
    "seasonal_naive": baselines.seasonal_naive,
    "moving_avg_4": baselines.moving_average,
    "ets_damped": stat.ets,
    "arima_111": stat.sarima,
}
ENSEMBLE_MEMBERS = ["lightgbm", "arima_111", "ets_damped"]
ALL_MODELS = [*UNIVARIATE, "lightgbm", "ensemble"]


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    denom = np.abs(y) + np.abs(yhat)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / max(np.abs(y).sum(), 1e-9)),
        "smape": float(np.mean(np.where(denom == 0, 0, 2 * np.abs(err) / np.where(denom == 0, 1, denom)))),
        "bias": float(err.sum() / max(np.abs(y).sum(), 1e-9)),
    }


def forecast_all(train: pd.DataFrame, horizon: int, models: list[str] | None = None) -> pd.DataFrame:
    """Forecasts from every model for every series: [series, week, model, yhat]."""
    models = models or ALL_MODELS
    last = train["week"].max()
    step = pd.tseries.frequencies.to_offset(FORECAST_FREQ)
    weeks = [last + step * h for h in range(1, horizon + 1)]
    frames = []
    for series, grp in train.groupby("series"):
        y = grp.sort_values("week")["y"].to_numpy()
        for name in models:
            if name in UNIVARIATE:
                frames.append(pd.DataFrame({"series": series, "week": weeks, "model": name,
                                            "yhat": UNIVARIATE[name](y, horizon)}))
    if "lightgbm" in models:
        frames.append(GlobalLGBM().fit(train).predict(train, horizon).assign(model="lightgbm"))
    fc = pd.concat(frames, ignore_index=True)
    if "ensemble" in models and set(ENSEMBLE_MEMBERS) <= set(fc["model"]):
        ens = fc[fc["model"].isin(ENSEMBLE_MEMBERS)].groupby(["series", "week"], as_index=False)["yhat"].mean()
        fc = pd.concat([fc, ens.assign(model="ensemble")], ignore_index=True)
    return fc


def backtest(panel: pd.DataFrame, horizon: int = FORECAST_HORIZON, n_folds: int = 3,
             models: list[str] | None = None) -> pd.DataFrame:
    """Expanding-window backtest with non-overlapping test windows at the end of the history."""
    weeks = np.sort(panel["week"].unique())
    rows = []
    for fold in range(n_folds, 0, -1):
        cutoff = weeks[len(weeks) - fold * horizon - 1]
        train = panel[panel["week"] <= cutoff]
        test = panel[(panel["week"] > cutoff) & (panel["week"] <= cutoff + np.timedelta64(7 * horizon, "D"))]
        fc = forecast_all(train, horizon, models).merge(test, on=["series", "week"], how="inner")
        rows.append(fc.assign(fold=fold, cutoff=cutoff, h=fc.groupby(["series", "model"]).cumcount() + 1))
    return pd.concat(rows, ignore_index=True)


def leaderboard(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["model"]
    res = bt.groupby(by).apply(lambda g: pd.Series(metrics(g["y"], g["yhat"])), include_groups=False)
    return res.reset_index().sort_values(by[:-1] + ["wape"] if len(by) > 1 else "wape")


def best_models(bt: pd.DataFrame) -> pd.Series:
    """Best model per series by backtest WAPE."""
    lb = leaderboard(bt, ["series", "model"])
    return lb.sort_values("wape").drop_duplicates("series").set_index("series")["model"]


def final_forecast(panel: pd.DataFrame, bt: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Forecast beyond the history with every model; attach 80% intervals from backtest relative errors
    (per series/model/horizon step) and flag each series' backtest winner."""
    fc = forecast_all(panel, horizon)
    fc["h"] = fc.groupby(["series", "model"]).cumcount() + 1
    rel = bt.assign(r=(bt["y"] - bt["yhat"]) / bt["yhat"].clip(lower=1))
    q = rel.groupby(["series", "model", "h"])["r"].quantile([0.1, 0.9]).unstack()
    q.columns = ["q10", "q90"]
    fc = fc.merge(q.reset_index(), on=["series", "model", "h"], how="left").fillna({"q10": -0.2, "q90": 0.2})
    fc["lower"] = np.minimum(fc["yhat"] * (1 + fc["q10"]), fc["yhat"]).clip(lower=0)
    fc["upper"] = np.maximum(fc["yhat"] * (1 + fc["q90"]), fc["yhat"])
    best = best_models(bt)
    fc["is_best"] = fc["model"] == fc["series"].map(best)
    return fc.drop(columns=["q10", "q90"])
