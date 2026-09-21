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
