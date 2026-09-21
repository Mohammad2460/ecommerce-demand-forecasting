import pytest

from ecom.data import warehouse


@pytest.fixture(scope="session")
def wh(tables):
    built = warehouse.build(tables)
    warehouse.save(built)
    return built


def test_tables_written(wh):
    for name in warehouse.TABLES:
        assert len(warehouse.load(name)) > 0


def test_grains(wh):
    assert wh["fact_orders"]["order_id"].is_unique
    assert wh["dim_customer"]["customer_unique_id"].is_unique
    assert wh["fact_order_items"]["order_id"].isin(wh["fact_orders"]["order_id"]).all()


def test_customer_totals_consistent(wh):
    assert wh["dim_customer"]["n_orders"].sum() == len(wh["fact_orders"])


def test_black_friday_flag(wh):
    bf = wh["dim_date"].loc[wh["dim_date"]["is_black_friday"], "date"].dt.strftime("%Y-%m-%d")
    assert "2017-11-24" in set(bf)
