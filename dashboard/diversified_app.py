"""Streamlit dashboard for the cross-asset diversified portfolio.

Run with:
    streamlit run dashboard/diversified_app.py

Shows the pre-registered best diversified variant (equities + long bonds + gold
+ a capped BTC sleeve) against two honest benchmarks -- 100% BTC and the classic
60/40 -- with:

* portfolio equity vs BTC vs 60/40 (log scale),
* drawdown curves (how deep each strategy fell under water),
* the latest target weights of the diversified book,
* the go/no-go gate verdict and the risk-adjusted metrics table,
* per-year and crisis-window sensitivity (the 2022 stocks-and-bonds-down acid
  test lives here).

Prices are free (yfinance); trading the equity/bond/gold legs needs a broker.
The picked Mexico venue is Interactive Brokers for SPY/TLT/GLD + Bitso for BTC.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ package importable when run via `streamlit run`.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from coinpredictor.config import COSTS, DIVERSIFIED  # noqa: E402
from coinpredictor.diversified import (  # noqa: E402
    annual_sensitivity,
    build_asset_panel,
    build_variant_weights,
    run_diversified,
    stress_windows,
    _variant_from_tag,
)

st.set_page_config(
    page_title="CoinPredictor — Diversified Portfolio", layout="wide"
)

_LABELS = {
    "btc_hold": "100% BTC",
    "sixty_forty": "60/40 (SPY/TLT)",
    "equal_weight_all": "Equal-weight all",
}


def _pretty(name: str) -> str:
    return _LABELS.get(name, f"Diversified ({name})")


# --- Cached compute ----------------------------------------------------------
@st.cache_data(show_spinner="Downloading prices & running the search…")
def _load(refresh: bool) -> dict:
    """Run the full diversified search once and return chart-ready pieces."""
    cfg = DIVERSIFIED
    panel = build_asset_panel(cfg, refresh=refresh)
    outcome = run_diversified(cfg, refresh=False)

    # Net-return series for the best variant + benchmarks -> equity & drawdown.
    best_name = outcome.best.name
    series = {best_name: outcome.best.metrics.returns}
    for b in ("btc_hold", "sixty_forty"):
        if b in outcome.benchmarks:
            series[b] = outcome.benchmarks[b].returns

    equity = {k: (1.0 + v).cumprod() for k, v in series.items()}
    drawdown = {k: (e / e.cummax() - 1.0) for k, e in equity.items()}

    # Latest target weights of the pre-registered best variant.
    scheme, target_vol = _variant_from_tag(cfg, best_name)
    _tag, weights = build_variant_weights(
        panel, cfg, scheme, target_vol,
        fee=COSTS.per_side, periods_per_year=cfg.periods_per_year,
    )
    latest_weights = weights.iloc[-1]

    metrics_rows = [v.metrics.row() | {"kind": "diversified"} for v in outcome.variants]
    metrics_rows += [
        b.row() | {"kind": "benchmark"} for b in outcome.benchmarks.values()
    ]

    annual = annual_sensitivity(outcome, periods_per_year=cfg.periods_per_year)
    stress = stress_windows(outcome, periods_per_year=cfg.periods_per_year)

    return {
        "best_name": best_name,
        "equity": equity,
        "drawdown": drawdown,
        "latest_weights": latest_weights,
        "metrics": pd.DataFrame(metrics_rows),
        "annual": annual,
        "stress": stress,
        "verdict": outcome.verdict,
        "recommendation": outcome.recommendation,
        "gate": outcome.gate.summary(),
    }


# --- UI ----------------------------------------------------------------------
st.title("₿ CoinPredictor — Cross-Asset Diversified Portfolio")
st.caption(
    "Equities (SPY) + long bonds (TLT) + gold (GLD) + a capped BTC sleeve, "
    "vs holding BTC and the classic 60/40. Educational, not financial advice."
)

with st.sidebar:
    st.header("Controls")
    refresh = st.toggle("Force data refresh", value=False)
    st.markdown(
        "**Deployment venue (Mexico)**\n\n"
        "- SPY / TLT / GLD → **Interactive Brokers**\n"
        "- BTC sleeve → **Bitso**"
    )

payload = _load(refresh)

verdict = payload["verdict"]
_color = {"PASS": "green"}.get(verdict, "orange")
st.markdown(f"### Verdict: :{_color}[{verdict}]")
st.write(payload["recommendation"])

# --- Equity + drawdown -------------------------------------------------------
st.subheader("Equity curve & drawdown")
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
    vertical_spacing=0.06,
    subplot_titles=("Growth of $1 (log scale)", "Drawdown"),
)
for name, eq in payload["equity"].items():
    fig.add_trace(
        go.Scatter(x=eq.index, y=eq.values, name=_pretty(name), mode="lines"),
        row=1, col=1,
    )
for name, dd in payload["drawdown"].items():
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values, name=_pretty(name), mode="lines",
            showlegend=False,
        ),
        row=2, col=1,
    )
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(tickformat=".0%", row=2, col=1)
fig.update_layout(height=620, legend_orientation="h", margin=dict(t=40))
st.plotly_chart(fig, use_container_width=True)

# --- Latest weights ----------------------------------------------------------
col_w, col_m = st.columns([1, 2])
with col_w:
    st.subheader("Latest target weights")
    w = payload["latest_weights"].sort_values(ascending=False)
    cash = max(0.0, 1.0 - float(w.sum()))
    labels = list(w.index) + (["cash"] if cash > 1e-6 else [])
    values = list(w.values) + ([cash] if cash > 1e-6 else [])
    pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.45))
    pie.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(pie, use_container_width=True)

with col_m:
    st.subheader("Risk-adjusted metrics (net of costs)")
    metrics = payload["metrics"].copy()
    metrics = metrics.sort_values("sharpe", ascending=False)
    metrics["total_return"] = metrics["total_return"].map(lambda x: f"{x:.1%}")
    metrics["max_drawdown"] = metrics["max_drawdown"].map(lambda x: f"{x:.1%}")
    metrics["sharpe"] = metrics["sharpe"].map(lambda x: f"{x:.3f}")
    metrics["avg_turnover"] = metrics["avg_turnover"].map(lambda x: f"{x:.3f}")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

# --- Sensitivity -------------------------------------------------------------
st.subheader("Sub-period sensitivity")
st.caption(
    "Per-calendar-year Sharpe & max drawdown. 2022 is the acid test: stocks AND "
    "long bonds fell together, so the diversified book has to prove it still "
    "contains its drawdown when the 60/40 hedge fails."
)

annual = payload["annual"].set_index("period")


def _style_annual(df: pd.DataFrame):
    sharpe_cols = [c for c in df.columns if c.endswith("_sharpe")]
    dd_cols = [c for c in df.columns if c.endswith("_maxDD")]
    styler = df.style.format("{:.2f}", subset=sharpe_cols)
    styler = styler.format("{:.1%}", subset=dd_cols)
    styler = styler.background_gradient(cmap="RdYlGn", subset=sharpe_cols)
    styler = styler.background_gradient(cmap="RdYlGn", subset=dd_cols)
    return styler


st.dataframe(_style_annual(annual), use_container_width=True)

st.subheader("Crisis-window stress test")
stress = payload["stress"].set_index("window")
ret_cols = [c for c in stress.columns if c.endswith("_ret")]
dd_cols = [c for c in stress.columns if c.endswith("_maxDD")]
stress_fmt = stress.style.format("{:.1%}").background_gradient(
    cmap="RdYlGn", subset=ret_cols + dd_cols
)
st.dataframe(stress_fmt, use_container_width=True)

with st.expander("Go/no-go gate details"):
    st.code(payload["gate"], language="text")
