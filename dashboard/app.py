"""Streamlit dashboard for the BTC volatility predictor.

Run with:
    streamlit run dashboard/app.py

Features:
* Today's live forward-volatility forecast + regime (calm/elevated), for the
  primary production model.
* Walk-forward backtest of a volatility-targeting strategy vs buy-and-hold.
* Interactive price & realized-volatility charts.
* Model feature-importance (explainability).
* Track record comparing ALL registered models (coinpredictor.registry) side
  by side, from their real daily predictions vs realized outcomes.
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

from coinpredictor.backtest import walk_forward_backtest  # noqa: E402
from coinpredictor.config import MODEL  # noqa: E402
from coinpredictor.data.ohlcv import load_ohlcv  # noqa: E402
from coinpredictor.features import build_default_features  # noqa: E402
from coinpredictor.model import (  # noqa: E402
    build_vol_regressor,
    load_artifact,
    regressor_importances,
    train_and_save,
)
from coinpredictor.predict import predict_next_day  # noqa: E402

st.set_page_config(page_title="CoinPredictor — BTC Volatility", layout="wide")

_PREDICTION_LOG = _ROOT / "data" / "processed" / "prediction_log.csv"


# --- Cached data/model helpers ----------------------------------------------
@st.cache_data(show_spinner="Loading BTC data…")
def _get_features(refresh: bool) -> pd.DataFrame:
    return build_default_features(load_ohlcv(refresh=refresh), refresh=refresh)


@st.cache_data(show_spinner="Running walk-forward backtest…")
def _get_backtest(_feats_key: str, feats: pd.DataFrame):
    return walk_forward_backtest(build_vol_regressor, feats)


@st.cache_data(show_spinner="Loading prediction log…", ttl=300)
def _get_prediction_log(_mtime: float) -> pd.DataFrame:
    """Load the real daily prediction log (long format: one row per day per
    model). _mtime busts the cache when the CSV changes on disk (cron run)."""
    df = pd.read_csv(
        _PREDICTION_LOG,
        parse_dates=["run_at", "as_of_date", "target_date"],
    )
    for col in ("predicted_vol", "actual_vol", "abs_error", "trailing_vol"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "regime_correct" in df.columns:
        df["regime_correct"] = df["regime_correct"].astype(str) == "True"
    return df


def _ensure_model(feats: pd.DataFrame):
    """Load a trained artifact, training one on first run if needed."""
    try:
        return load_artifact()
    except FileNotFoundError:
        with st.spinner("Training model for the first time…"):
            return train_and_save(feats)


# --- Sidebar -----------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
refresh = st.sidebar.button("🔄 Refresh market data")
retrain = st.sidebar.button("🧠 Retrain model")
st.sidebar.caption(
    f"Daily BTC-USD via yfinance. Forecasts forward {MODEL.vol_horizon}-day "
    "annualized volatility."
)

feats = _get_features(refresh)
if retrain:
    with st.spinner("Retraining…"):
        train_and_save(feats)
    st.sidebar.success("Model retrained.")

artifact = _ensure_model(feats)

st.title("₿ CoinPredictor — Volatility Forecast")
st.caption(
    "⚠️ Educational tool, not financial advice. Volatility clusters and is far "
    "more predictable than price direction."
)

# --- Live prediction (primary production model) ------------------------------
pred = predict_next_day(artifact=artifact, refresh=False)
vol_delta = pred.predicted_vol - pred.trailing_vol
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"${pred.last_close:,.0f}", help=str(pred.as_of_date.date()))
c2.metric(
    f"Forecast vol ({MODEL.vol_horizon}d, ann.)",
    f"{pred.predicted_vol:.1%}",
    delta=f"{vol_delta:+.1%} vs recent",
    delta_color="inverse",
)
c3.metric("Recent realized vol", f"{pred.trailing_vol:.1%}")
c4.metric(
    "Regime",
    pred.regime,
    delta=(f"P={pred.regime_proba:.0%}" if pred.regime_proba is not None else None),
    delta_color="off",
)

st.divider()

# --- Tabs --------------------------------------------------------------------
tab_chart, tab_backtest, tab_importance, tab_track = st.tabs(
    ["📈 Charts", "🧪 Backtest", "🔍 Explainability", "📊 Track Record"]
)

with tab_chart:
    lookback = st.slider("Days to display", 60, 720, 180, step=30)
    view = feats.tail(lookback)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("Price & Moving Averages", "Trailing realized volatility (annualized)"),
    )
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="BTC-USD",
        ),
        row=1,
        col=1,
    )
    for ma, color in (("sma_20", "#ff9800"), ("sma_50", "#2196f3")):
        if ma in view:
            fig.add_trace(
                go.Scatter(x=view.index, y=view[ma], name=ma.upper(), line=dict(color=color)),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["realized_vol_trailing"],
            name="Realized vol",
            line=dict(color="#9c27b0"),
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with tab_backtest:
    result = _get_backtest(f"{len(feats)}-{feats.index[-1]}", feats)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy Sharpe", f"{result.strategy_sharpe:.2f}")
    m2.metric("Buy & hold Sharpe", f"{result.bh_sharpe:.2f}")
    m3.metric("Strategy max DD", f"{result.strategy_max_drawdown:.1%}")
    m4.metric("Forecast corr", f"{result.forecast_corr:.2f}")

    st.caption(
        "Volatility-targeting strategy: scale BTC exposure inversely to forecast "
        "volatility. The goal is a higher Sharpe and smaller drawdown than holding."
    )

    eq = result.equity
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["strategy"], name="Vol-targeting strategy"))
    fig_eq.add_trace(
        go.Scatter(x=eq.index, y=eq["buy_and_hold"], name="Buy & hold", line=dict(dash="dash"))
    )
    fig_eq.update_layout(
        height=420,
        title="Out-of-sample equity curve (growth of $1)",
        yaxis_title="Equity",
    )
    st.plotly_chart(fig_eq, use_container_width=True)
    st.code(result.summary())

with tab_importance:
    importances = regressor_importances(artifact.regressor)
    if importances is None:
        st.info("Current model does not expose feature importances.")
    else:
        imp = (
            pd.Series(importances, index=artifact.feature_names)
            .sort_values(ascending=True)
            .tail(20)
        )
        fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
        fig_imp.update_layout(height=600, title="Top 20 feature importances")
        st.plotly_chart(fig_imp, use_container_width=True)

with tab_track:
    if not _PREDICTION_LOG.exists():
        st.info(
            "No predictions logged yet. The daily cron job will populate "
            "data/processed/prediction_log.csv on its next run."
        )
    else:
        log_mtime = _PREDICTION_LOG.stat().st_mtime
        log_df = _get_prediction_log(log_mtime)

        if "model_name" not in log_df.columns:
            st.warning(
                "This log predates the multi-model registry. Run "
                "scripts/migrate_add_model_name.py to upgrade it."
            )
        else:
            all_models = sorted(log_df["model_name"].unique())
            evaluated_all = log_df[log_df["status"] == "evaluated"].copy()
            pending_all = log_df[log_df["status"] == "pending"].copy()

            st.caption(
                f"{len(log_df)} predictions logged across {len(all_models)} model(s) — "
                f"{len(evaluated_all)} evaluated, {len(pending_all)} still pending."
            )

            # --- Leaderboard: one row per model, side by side -------------------
            st.subheader("🏆 Model leaderboard")
            if evaluated_all.empty:
                st.info(
                    "No evaluated predictions yet for any model. Each prediction "
                    "needs its target_date to pass before it can be scored."
                )
            else:
                rows = []
                for name, grp in evaluated_all.groupby("model_name"):
                    mae = grp["abs_error"].mean()
                    rmse = (grp["abs_error"] ** 2).mean() ** 0.5
                    acc = grp["regime_correct"].mean()
                    rows.append(
                        {
                            "Model": name,
                            "Evaluated": len(grp),
                            "MAE": mae,
                            "RMSE": rmse,
                            "Regime accuracy": acc,
                        }
                    )
                board = pd.DataFrame(rows).sort_values("MAE")
                st.dataframe(
                    board.style.format(
                        {"MAE": "{:.2%}", "RMSE": "{:.2%}", "Regime accuracy": "{:.0%}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Lower MAE/RMSE is better. `naive_persistence_v1` (predicts "
                    "tomorrow = today) is the bar every other model should clear — "
                    "if a fancier model doesn't beat it, it isn't earning its "
                    "complexity."
                )

            st.divider()

            # --- Model selector for detail charts --------------------------------
            selected = st.multiselect(
                "Compare models in the charts below",
                options=all_models,
                default=all_models,
            )
            evaluated = evaluated_all[evaluated_all["model_name"].isin(selected)]

            if evaluated.empty:
                st.info("No evaluated predictions yet for the selected model(s).")
            else:
                fig_track = go.Figure()
                colors = [
                    "#2196f3", "#ff9800", "#4caf50", "#e91e63", "#9c27b0", "#795548",
                ]
                for idx, (name, grp) in enumerate(evaluated.groupby("model_name")):
                    grp = grp.sort_values("as_of_date")
                    color = colors[idx % len(colors)]
                    fig_track.add_trace(
                        go.Scatter(
                            x=grp["as_of_date"], y=grp["predicted_vol"],
                            name=f"{name} — predicted", mode="lines+markers",
                            line=dict(color=color),
                        )
                    )
                    fig_track.add_trace(
                        go.Scatter(
                            x=grp["as_of_date"], y=grp["actual_vol"],
                            name=f"{name} — actual", mode="lines+markers",
                            line=dict(color=color, dash="dot"),
                        )
                    )
                fig_track.update_layout(
                    height=440,
                    title="Predicted vs. realized volatility, by model",
                    yaxis_title="Annualized volatility",
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig_track, use_container_width=True)

                fig_mae = go.Figure()
                for idx, (name, grp) in enumerate(evaluated.groupby("model_name")):
                    grp = grp.sort_values("as_of_date").copy()
                    grp["cumulative_mae"] = grp["abs_error"].expanding().mean()
                    fig_mae.add_trace(
                        go.Scatter(
                            x=grp["as_of_date"], y=grp["cumulative_mae"],
                            name=name, line=dict(color=colors[idx % len(colors)]),
                        )
                    )
                fig_mae.update_layout(
                    height=320,
                    title="Cumulative MAE over time, by model",
                    yaxis_title="Mean absolute error",
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            with st.expander("📋 Full prediction log (all models)", expanded=evaluated_all.empty):
                display_cols = [
                    c
                    for c in [
                        "as_of_date", "model_name", "target_type", "target_date", "status",
                        "predicted_vol", "actual_vol", "regime_pred", "actual_regime",
                        "regime_correct", "abs_error",
                    ]
                    if c in log_df.columns
                ]
                st.dataframe(
                    log_df[display_cols].sort_values("as_of_date", ascending=False),
                    use_container_width=True,
                )
