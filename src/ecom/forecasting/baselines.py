"""Univariate baselines. Each takes a history array and returns `horizon` point forecasts."""

from __future__ import annotations

import numpy as np


def naive(y: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(y[-1], horizon).astype(float)


def seasonal_naive(y: np.ndarray, horizon: int, season: int = 52) -> np.ndarray:
    if len(y) < season:
        return naive(y, horizon)
    return np.array([y[len(y) - season + (h % season)] for h in range(horizon)], dtype=float)


def moving_average(y: np.ndarray, horizon: int, window: int = 4) -> np.ndarray:
    return np.repeat(np.mean(y[-window:]), horizon).astype(float)
