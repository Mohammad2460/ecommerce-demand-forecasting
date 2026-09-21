"""Shared dashboard helpers: data access, chart theme (validated reference palette), formatting."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

from ecom.api import store  # noqa: E402

# Categorical slots in fixed order (validated palette); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

pio.templates["ecom"] = go.layout.Template(
    layout=dict(
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis=dict(showgrid=False, linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED), zeroline=False),
        yaxis=dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, font=dict(color=INK_2)),
        bargap=0.25,
    )
)
pio.templates.default = "plotly_white+ecom"


def page(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return store.table(name)


@st.cache_data(show_spinner=False)
def kpis() -> dict:
    return store.json_artifact("kpis")


def require_artifacts() -> None:
    try:
        store.table("customers")
        store.table("forecast")
    except store.ArtifactMissing as e:
        st.error(f"{e}. From the project root run: `make pipeline`")
        st.stop()


def brl(x: float, decimals: int = 0) -> str:
    return f"R$ {x:,.{decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


def pretty(name: str) -> str:
    return "All categories (total)" if name == "__total__" else name.replace("_", " ").capitalize()


def style(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    layout.setdefault("legend", dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0))
    fig.update_layout(height=height, **layout)
    return fig
