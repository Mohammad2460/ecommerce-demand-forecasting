"""Streamlit dashboard router. Run: make dashboard"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="Olist Analytics", layout="wide", page_icon="📦")
nav = st.navigation(
    [
        st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(PAGES / "forecast.py", title="Demand forecast", icon=":material/trending_up:"),
        st.Page(PAGES / "customers.py", title="Customer segments", icon=":material/groups:"),
        st.Page(PAGES / "churn_clv.py", title="Churn & CLV", icon=":material/person_search:"),
        st.Page(PAGES / "geo_products.py", title="Geography & products", icon=":material/map:"),
    ]
)
nav.run()
