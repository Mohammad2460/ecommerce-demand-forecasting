"""Global LightGBM model across all series with recursive multi-step forecasting."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecom.config import FORECAST_FREQ, RANDOM_STATE
from ecom.features.timeseries import FEATURES, supervised

PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1,
)


class GlobalLGBM:
    def __init__(self, **params):
        self.params = {**PARAMS, **params}
        self.model: lgb.LGBMRegressor | None = None
        self.categories: list[str] = []

    def fit(self, panel: pd.DataFrame) -> GlobalLGBM:
        self.categories = sorted(panel["series"].unique())
        df = self._features(panel).dropna(subset=["lag_4"])
        self.model = lgb.LGBMRegressor(**self.params).fit(df[FEATURES], df["ly"])
        return self

    def _features(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = supervised(panel)
        df["series_code"] = pd.Categorical(df["series"], categories=self.categories)
        return df

    def predict(self, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Recursive forecast: predict one week, append it as history, repeat."""
        hist = panel[["series", "week", "y"]].copy()
        last = hist["week"].max()
        out = []
        for step in range(1, horizon + 1):
            wk = last + pd.tseries.frequencies.to_offset(FORECAST_FREQ) * step
            nxt = pd.DataFrame({"series": self.categories, "week": wk, "y": np.nan})
            feats = self._features(pd.concat([hist, nxt], ignore_index=True))
            row = feats[feats["week"] == wk]
            yhat = np.clip(np.expm1(self.model.predict(row[FEATURES])), 0, None)
            step_df = pd.DataFrame({"series": row["series"].values, "week": wk, "y": yhat})
            hist = pd.concat([hist, step_df], ignore_index=True)
            out.append(step_df)
        return pd.concat(out, ignore_index=True).rename(columns={"y": "yhat"})

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
