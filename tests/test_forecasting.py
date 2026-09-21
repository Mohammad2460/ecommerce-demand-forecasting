import numpy as np
import pandas as pd
import pytest

from ecom.data import warehouse
from ecom.features.timeseries import TOTAL, calendar_features, weekly_demand
from ecom.forecasting import baselines
from ecom.forecasting.evaluate import ALL_MODELS, backtest, final_forecast, leaderboard, metrics


@pytest.fixture(scope="module")
def panel(tables):
    return weekly_demand(warehouse.build(tables)["fact_order_items"], top_n=5)


def test_panel_complete_and_total_matches(panel):
    counts = panel.groupby("series")["week"].nunique()
    assert counts.nunique() == 1  # every series has every week
    assert TOTAL in counts.index
    total = panel[panel.series == TOTAL].set_index("week")["y"]
    cats = panel[panel.series != TOTAL].groupby("week")["y"].sum()
    assert (total >= cats).all()


def test_calendar_black_friday():
    weeks = pd.date_range("2017-11-13", periods=3, freq="W-MON")
    assert calendar_features(weeks)["black_friday"].tolist() == [0, 1, 0]


def test_metrics_perfect_and_baselines():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics(y, y)["wape"] == 0
    assert baselines.naive(y, 2).tolist() == [3.0, 3.0]
    assert baselines.seasonal_naive(np.arange(60.0), 2).tolist() == [8.0, 9.0]


def test_backtest_and_final_forecast(panel):
    bt = backtest(panel, horizon=4, n_folds=2)
    assert set(bt["model"]) == set(ALL_MODELS)
    lb = leaderboard(bt)
    assert lb["wape"].notna().all()
    fc = final_forecast(panel, bt, horizon=4)
    assert (fc["week"] > panel["week"].max()).all()
    assert (fc["lower"] <= fc["yhat"] + 1e-9).all() and (fc["yhat"] >= 0).all()
    assert fc.groupby("series")["is_best"].sum().eq(4).all()
