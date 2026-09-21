"""Classical statistical models via statsmodels."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ecom.forecasting.baselines import moving_average


def ets(y: np.ndarray, horizon: int) -> np.ndarray:
    """Holt's linear trend with damping, fitted on log1p to keep forecasts positive."""
    if len(y) < 10:
        return moving_average(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(np.log1p(y), trend="add", damped_trend=True).fit()
    return np.clip(np.expm1(fit.forecast(horizon)), 0, None)


def sarima(y: np.ndarray, horizon: int) -> np.ndarray:
    """ARIMA(1,1,1) on log1p. Too few years of data for a 52-week seasonal term."""
    if len(y) < 16:
        return moving_average(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(np.log1p(y), order=(1, 1, 1), enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
    return np.clip(np.expm1(fit.forecast(horizon)), 0, None)
