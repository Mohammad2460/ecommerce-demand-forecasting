"""Headless render of every dashboard page on sample-data artifacts: no exceptions, key elements present."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ecom.api import store
from ecom.pipeline import run

PAGES = Path(__file__).resolve().parents[1] / "src" / "ecom" / "dashboard" / "pages"


@pytest.fixture(scope="module", autouse=True)
def artifacts():
    run()
    store.clear()


@pytest.mark.parametrize("page", ["overview", "forecast", "customers", "churn_clv", "geo_products"])
def test_page_renders(page):
    at = AppTest.from_file(str(PAGES / f"{page}.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page has a title"


def test_forecast_page_model_switch():
    at = AppTest.from_file(str(PAGES / "forecast.py"), default_timeout=60).run()
    at.selectbox[1].select("naive").run()
    assert not at.exception
    assert any(m.value == "naive" for m in at.metric)


def test_churn_whatif_scores():
    at = AppTest.from_file(str(PAGES / "churn_clv.py"), default_timeout=60).run()
    labels = [m.label for m in at.metric]
    assert "P(repeat within 180 days)" in labels
