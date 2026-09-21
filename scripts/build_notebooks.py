"""Create and execute the analysis notebooks (outputs stay in the .ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = "import sys; sys.path.insert(0, '../src')\nimport pandas as pd\npd.set_option('display.width', 140)"

NOTEBOOKS = {
    "01_eda.ipynb": [
        ("md", "# 01 · Exploratory Data Analysis\nOlist Brazilian E-Commerce, delivered orders Jan 2017 – Aug 2018."),
        ("code", SETUP),
        ("code", "from ecom.data import warehouse\nwh = {n: warehouse.load(n) for n in warehouse.TABLES}\n"
                 "{k: v.shape for k, v in wh.items()}"),
        ("md", "## Headline KPIs"),
        ("code", "from ecom.insights.eda import kpis, make_figures\n"
                 "pd.Series(kpis(wh['fact_orders'], wh['dim_customer'])).round(3)"),
        ("md", "## Figures"),
        ("code", "from IPython.display import Image, display\n"
                 "for p in make_figures(wh):\n    display(Image(filename=str(p)))"),
        ("md", "## Customer repeat behaviour\nMost customers purchase only once — this shapes the churn framing."),
        ("code", "wh['dim_customer']['n_orders'].value_counts().sort_index().head(8)"),
        ("md", "## Delivery performance by state"),
        ("code", "wh['fact_orders'].groupby('customer_state').agg(orders=('order_id','count'), "
                 "avg_days=('delivery_days','mean'), late_rate=('is_late','mean'), "
                 "review=('review_score','mean')).sort_values('orders', ascending=False).round(2).head(12)"),
    ],
    "02_forecasting.ipynb": [
        ("md", "# 02 · Demand Forecasting\nWeekly item demand for the top-15 categories plus the total. "
               "Models: naive, seasonal naive, 4-week moving average, damped ETS, ARIMA(1,1,1), a global "
               "LightGBM on lag/rolling/calendar features, and an ensemble. Evaluation: 3-fold expanding-window "
               "backtest, 8-week horizon."),
        ("code", SETUP + "\nimport matplotlib.pyplot as plt\nP = '../data/processed/'"),
        ("code", "panel = pd.read_parquet(P + 'demand_weekly.parquet')\n"
                 "panel.pivot(index='week', columns='series', values='y').tail()"),
        ("md", "## Overall leaderboard (lower WAPE is better)"),
        ("code", "pd.read_parquet(P + 'forecast_leaderboard.parquet').round(3)"),
        ("md", "## Best model per series"),
        ("code", "lb = pd.read_parquet(P + 'forecast_leaderboard_by_series.parquet')\n"
                 "lb.sort_values('wape').drop_duplicates('series').round(3)"),
        ("md", "## Error by horizon step"),
        ("code", "from ecom.forecasting.evaluate import leaderboard\n"
                 "bt = pd.read_parquet(P + 'forecast_backtest.parquet')\n"
                 "leaderboard(bt, ['h', 'model']).pivot(index='h', columns='model', values='wape').round(3)"),
        ("md", "## LightGBM feature importance (gain)"),
        ("code", "pd.read_parquet(P + 'forecast_feature_importance.parquet').head(10)"),
        ("md", "## Total demand: history and 8-week forecast (best model, 80% interval)"),
        ("code", "fc = pd.read_parquet(P + 'forecast.parquet')\n"
                 "s = '__total__'\nh = panel[panel.series == s]; f = fc[(fc.series == s) & fc.is_best]\n"
                 "fig, ax = plt.subplots(figsize=(10, 3.8))\nax.plot(h.week, h.y, label='actual')\n"
                 "ax.plot(f.week, f.yhat, label=f'forecast ({f.model.iloc[0]})')\n"
                 "ax.fill_between(f.week, f.lower, f.upper, alpha=.25)\n"
                 "ax.legend(); ax.set_title('Total weekly units')\n"
                 "fig.savefig('../reports/figures/06_total_forecast.png', bbox_inches='tight')"),
    ],
    "03_customers.ipynb": [
        ("md", "# 03 · Customer Analytics\\nRFM segmentation, K-Means behavioural clusters, repeat-purchase "
               "(churn) propensity with out-of-time validation, 12-month revenue CLV, and category affinity."),
        ("code", SETUP + "\\nimport matplotlib.pyplot as plt\\nP = '../data/processed/'\\n"
                 "c = pd.read_parquet(P + 'customers.parquet'); c.shape"),
        ("md", "## RFM segments"),
        ("code", "seg = pd.read_parquet(P + 'segment_summary.parquet'); seg.round(3)"),
        ("code", "fig, ax = plt.subplots(figsize=(9, 4))\\n"
                 "ax.barh(seg.segment, seg.share_customers, label='customers', alpha=.7)\\n"
                 "ax.barh(seg.segment, seg.share_revenue, label='revenue', alpha=.5)\\n"
                 "ax.invert_yaxis(); ax.legend(); ax.set_title('Share of customers vs revenue by RFM segment')\\n"
                 "fig.savefig('../reports/figures/07_rfm_segments.png', bbox_inches='tight')"),
        ("md", "## K-Means clusters\\nk chosen by silhouette score on a 10k sample."),
        ("code", "pd.read_parquet(P + 'cluster_k_scores.parquet').round(3)"),
        ("code", "pd.read_parquet(P + 'cluster_profile.parquet').round(2)"),
        ("md", "## Repeat-purchase (churn) model\\nTrain: 4 stacked cutoffs whose 180-day label windows end "
               "before the test cutoff. Test: customers as of the test cutoff. Positive class = repurchase."),
        ("code", "pd.read_parquet(P + 'churn_metrics.parquet').round(4)"),
        ("code", "pd.read_parquet(P + 'churn_importance.parquet').head(10)"),
        ("md", "Only ~1% of customers repurchase within 180 days, so ranking signal is modest: the model "
               "roughly doubles precision over random targeting in the top decile."),
        ("md", "## CLV by segment"),
        ("code", "c.groupby('segment').agg(customers=('customer_unique_id','count'), avg_clv=('clv_12m','mean'), "
                 "total_clv=('clv_12m','sum'), p_repeat=('p_repeat','mean')).sort_values('avg_clv', "
                 "ascending=False).round(3)"),
        ("md", "## Category affinity (customer level)"),
        ("code", "pd.read_parquet(P + 'affinity_customer.parquet').head(10).round(4)"),
        ("md", "## State logistics performance"),
        ("code", "pd.read_parquet(P + 'state_performance.parquet').head(10).round(3)"),
    ],
}


def build(names: list[str] | None = None) -> None:
    for name, cells in NOTEBOOKS.items():
        if names and name not in names:
            continue
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "md" else nbformat.v4.new_code_cell(s) for k, s in cells]
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbformat.write(nb, NB / name)
        print(f"executed {name}")


if __name__ == "__main__":
    build(sys.argv[1:] or None)
