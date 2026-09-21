import pytest
from fastapi.testclient import TestClient

from ecom.api import store
from ecom.api.main import app
from ecom.pipeline import run


@pytest.fixture(scope="module")
def client():
    run()
    store.clear()
    return TestClient(app)


def test_health_and_kpis(client):
    assert client.get("/health").json()["artifacts_ready"] is True
    k = client.get("/kpis").json()
    assert k["orders"] > 0


def test_forecast(client):
    series = client.get("/forecast/series").json()
    assert "__total__" in series
    r = client.get("/forecast/__total__", params={"horizon": 4}).json()
    assert len(r["forecast"]) == 4 and r["history"]
    assert all(p["lower"] <= p["yhat"] <= p["upper"] for p in r["forecast"])
    assert client.get("/forecast/__total__", params={"model": "naive"}).json()["model"] == "naive"
    assert client.get("/forecast/nope").status_code == 404
    assert client.get("/forecast/__total__", params={"model": "nope"}).status_code == 404
    assert client.get("/forecast/leaderboard").status_code == 200


def test_customers(client):
    lst = client.get("/customers", params={"limit": 5}).json()
    assert lst["total"] > 0 and len(lst["items"]) == 5
    cid = lst["items"][0]["customer_unique_id"]
    c = client.get(f"/customers/{cid}").json()
    assert c["customer_unique_id"] == cid and 0 <= c["p_repeat"] <= 1
    assert client.get("/customers/does-not-exist").status_code == 404
    assert client.get("/segments").json()
    assert client.get("/clusters").json()


def test_churn_score(client):
    body = {"recency_days": 30, "frequency": 3, "monetary": 400, "main_category": "bed_bath_table"}
    r = client.post("/churn/score", json=body).json()
    assert 0 <= r["p_repeat"] <= 1 and abs(r["p_repeat"] + r["churn_risk"] - 1) < 1e-9
    assert client.post("/churn/score", json={"recency_days": -1, "frequency": 1, "monetary": 1}).status_code == 422


def test_insights(client):
    assert client.get("/insights/states").json()
    assert client.get("/insights/categories").json()
    assert client.get("/insights/affinity", params={"level": "order"}).status_code == 200
