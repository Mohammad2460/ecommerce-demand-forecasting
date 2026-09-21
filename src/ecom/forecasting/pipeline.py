"""Run the forecasting stage end to end and persist outputs."""

from __future__ import annotations

import joblib
import pandas as pd

from ecom.config import models_dir, processed_dir
from ecom.features.timeseries import weekly_demand
from ecom.forecasting.evaluate import backtest, final_forecast, leaderboard
from ecom.forecasting.ml import GlobalLGBM


def run(fact_items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    panel = weekly_demand(fact_items)
    bt = backtest(panel)
    out = {
        "demand_weekly": panel,
        "forecast_backtest": bt,
        "forecast_leaderboard": leaderboard(bt),
        "forecast_leaderboard_by_series": leaderboard(bt, ["series", "model"]),
        "forecast": final_forecast(panel, bt),
    }
    model = GlobalLGBM().fit(panel)
    joblib.dump(model, models_dir() / "lgbm_global.joblib")
    out["forecast_feature_importance"] = model.feature_importance().rename("gain").rename_axis("feature").reset_index()
    for name, df in out.items():
        df.to_parquet(processed_dir() / f"{name}.parquet", index=False)
    return out
