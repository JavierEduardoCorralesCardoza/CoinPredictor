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
from coinpredictor.config import (  # noqa: E402
    ENTRY_LOG,
    JUDGE,
    JUDGE_LOG,
    MODEL,
    RISK_POLICY_RESULTS,
    SENTIMENT_LOG,
    TREND_REGIME_LOG,
    VOLATILITY_LOG,
)
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

# Per-family prediction logs (Phase 1: one csv per family/target_type).
_PREDICTION_LOG = VOLATILITY_LOG  # kept for the volatility track record


# --- Cached data/model helpers ----------------------------------------------
@st.cache_data(show_spinner="Loading BTC data…")
def _get_features(refresh: bool) -> pd.DataFrame:
    return build_default_features(load_ohlcv(refresh=refresh), refresh=refresh)


@st.cache_data(show_spinner="Running walk-forward backtest…")
def _get_backtest(_feats_key: str, feats: pd.DataFrame):
    return walk_forward_backtest(build_vol_regressor, feats)


@st.cache_data(show_spinner="Loading prediction log…", ttl=300)
def _get_prediction_log(_mtime: float, path_str: str) -> pd.DataFrame:
    """Load a per-family prediction log. _mtime busts the cache when the CSV
    changes on disk (cron run). path_str keys the cache per family file."""
    df = pd.read_csv(path_str)
    for col in ("as_of_date", "target_date", "run_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric = (
        "predicted_vol", "actual_vol", "abs_error", "trailing_vol",
        "trend_regime_proba", "entry_proba", "entry_actual",
        "sentiment_score", "sentiment_fwd_return", "sentiment_fwd_vol",
    )
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("regime_correct", "trend_regime_correct", "entry_correct"):
        if col in df.columns:
            df[col] = df[col].astype(str) == "True"
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
tab_chart, tab_importance, tab_track, tab_judges, tab_backtest = st.tabs(
    ["📈 Charts", "🔍 Explainability", "📊 Track Record", "⚖️ LLM Judges", "🧪 Backtest"]
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
    st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig_imp, width="stretch")

with tab_track:
    st.caption(
        "Each model family logs to its OWN file and is scored on its OWN metric. "
        "Pick a model type below, then compare its models one table at a time."
    )

    def _family_log(path):
        if not path.exists():
            return None
        return _get_prediction_log(path.stat().st_mtime, str(path))

    FAMILIES = {
        "📉 Volatility": "volatility",
        "🧭 Trend regime": "trend_regime",
        "🎯 Entry": "entry",
        "📰 Sentiment": "sentiment",
    }
    family_label = st.selectbox("Model type", list(FAMILIES.keys()))
    family = FAMILIES[family_label]

    if family == "volatility":
        st.subheader("📉 Volatility — MAE / RMSE / vol-regime accuracy")
        vol_df = _family_log(VOLATILITY_LOG)
        if vol_df is None or vol_df.empty:
            st.info("No volatility predictions logged yet.")
        else:
            ev = vol_df[vol_df["status"] == "evaluated"]
            if ev.empty:
                st.info("No evaluated volatility predictions yet.")
            else:
                for name, grp in ev.groupby("model_name"):
                    grp = grp.sort_values("as_of_date")
                    st.markdown(f"**{name}**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Evaluated", len(grp))
                    c2.metric("MAE", f"{grp['abs_error'].mean():.2%}")
                    c3.metric(
                        "Vol-regime acc",
                        f"{grp['regime_correct'].mean():.0%}"
                        if grp["regime_correct"].notna().any() else "n/a",
                    )
                    st.dataframe(
                        grp[[
                            "as_of_date", "predicted_vol", "actual_vol",
                            "regime_pred", "actual_regime", "regime_correct",
                            "abs_error",
                        ]].style.format({
                            "predicted_vol": "{:.2%}", "actual_vol": "{:.2%}",
                            "abs_error": "{:.2%}",
                        }),
                        width="stretch", hide_index=True,
                    )
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(
                        x=grp["as_of_date"], y=grp["predicted_vol"], name="predicted"))
                    fig_v.add_trace(go.Scatter(
                        x=grp["as_of_date"], y=grp["actual_vol"], name="actual",
                        line=dict(dash="dot")))
                    fig_v.update_layout(
                        height=280, yaxis_tickformat=".0%",
                        title=f"{name} — predicted vs realized volatility")
                    st.plotly_chart(fig_v, width="stretch")
                    st.divider()
                st.caption(
                    "Baseline `naive_persistence_v1` (tomorrow = today) is the bar "
                    "every model should clear. Lower MAE/RMSE is better."
                )

    elif family == "trend_regime":
        st.subheader("🧭 Trend regime — accuracy + per-class F1 (ALCISTA/BAJISTA/LATERAL)")
        trend_df = _family_log(TREND_REGIME_LOG)
        if trend_df is None or trend_df.empty:
            st.info("No trend-regime predictions logged yet.")
        else:
            ev = trend_df[trend_df["status"] == "evaluated"]
            if ev.empty:
                st.info("No evaluated trend-regime predictions yet.")
            else:
                from sklearn.metrics import f1_score

                labels = ["ALCISTA", "BAJISTA", "LATERAL"]
                for name, grp in ev.groupby("model_name"):
                    grp = grp.sort_values("as_of_date")
                    y_true = grp["trend_regime_actual"].astype(str)
                    y_pred = grp["trend_regime_pred"].astype(str)
                    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                    st.markdown(f"**{name}**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Evaluated", len(grp))
                    c2.metric("Accuracy", f"{(y_true == y_pred).mean():.0%}")
                    c3.metric("F1 ALCISTA", f"{f1s[0]:.2f}")
                    c4.metric("F1 BAJISTA", f"{f1s[1]:.2f}")
                    st.dataframe(
                        grp[[
                            "as_of_date", "trend_regime_pred", "trend_regime_proba",
                            "trend_regime_actual", "trend_regime_correct",
                        ]],
                        width="stretch", hide_index=True,
                    )
                    st.divider()
                st.caption("Baseline: `sma_cross_trend_v1` (rule-based, zero training).")

    elif family == "entry":
        st.subheader("🎯 Entry — precision / recall / calibration (triple-barrier)")
        entry_df = _family_log(ENTRY_LOG)
        if entry_df is None or entry_df.empty:
            st.info("No entry predictions logged yet.")
        else:
            ev = entry_df[entry_df["status"] == "evaluated"]
            if ev.empty:
                st.info("No evaluated entry predictions yet.")
            else:
                for name, grp in ev.groupby("model_name"):
                    grp = grp.sort_values("as_of_date")
                    y_true = grp["entry_actual"].astype(float)
                    pred = (grp["entry_proba"] >= 0.5).astype(int)
                    tp = int(((pred == 1) & (y_true == 1)).sum())
                    fp = int(((pred == 1) & (y_true == 0)).sum())
                    fn = int(((pred == 0) & (y_true == 1)).sum())
                    precision = tp / (tp + fp) if (tp + fp) else float("nan")
                    recall = tp / (tp + fn) if (tp + fn) else float("nan")
                    st.markdown(f"**{name}**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Evaluated", len(grp))
                    c2.metric("Precision", f"{precision:.2f}" if precision == precision else "n/a")
                    c3.metric("Recall", f"{recall:.2f}" if recall == recall else "n/a")
                    c4.metric("Base win-rate", f"{y_true.mean():.0%}")
                    st.dataframe(
                        grp[[
                            "as_of_date", "entry_proba", "tp_pct", "sl_pct",
                            "entry_actual", "entry_correct",
                        ]],
                        width="stretch", hide_index=True,
                    )
                    if name == "lgbm_entry_v1":
                        gg = grp.copy()
                        gg["bin"] = (gg["entry_proba"] * 5).round() / 5
                        calib = gg.groupby("bin").agg(
                            predicted=("entry_proba", "mean"),
                            actual=("entry_actual", lambda s: s.astype(float).mean()),
                            n=("entry_actual", "size"),
                        ).reset_index()
                        fig_c = go.Figure()
                        fig_c.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect",
                                                   line=dict(dash="dash", color="#888")))
                        fig_c.add_trace(go.Scatter(x=calib["predicted"], y=calib["actual"],
                                                   mode="markers+lines", name=name))
                        fig_c.update_layout(height=280, title=f"{name} — calibration",
                                            xaxis_title="Predicted win proba",
                                            yaxis_title="Actual win rate")
                        st.plotly_chart(fig_c, width="stretch")
                    st.divider()
                st.caption("Baseline: `baseline_entry_v1` (flat 0.5).")

    elif family == "sentiment":
        st.subheader("📰 Sentiment — corr(score, forward return)")
        sent_df = _family_log(SENTIMENT_LOG)
        if sent_df is None or sent_df.empty:
            st.info("No sentiment predictions logged yet.")
        else:
            ev = sent_df[sent_df["status"] == "evaluated"]
            if ev.empty:
                st.info(
                    "No evaluated sentiment predictions yet (needs the forward "
                    "window to pass). There is no per-row 'correct' for sentiment."
                )
            else:
                for name, grp in ev.groupby("model_name"):
                    grp = grp.sort_values("as_of_date")
                    score = grp["sentiment_score"]
                    fwd = grp["sentiment_fwd_return"]
                    valid = score.notna() & fwd.notna()
                    if valid.sum() >= 2 and score[valid].nunique() > 1:
                        corr = float(score[valid].corr(fwd[valid]))
                    else:
                        corr = float("nan")
                    st.markdown(f"**{name}**")
                    c1, c2 = st.columns(2)
                    c1.metric("Evaluated", len(grp))
                    c2.metric("corr(score, fwd_return)", f"{corr:.3f}" if corr == corr else "n/a")
                    st.dataframe(
                        grp[[
                            "as_of_date", "sentiment_score", "sentiment_label",
                            "n_headlines", "sentiment_fwd_return", "sentiment_fwd_vol",
                        ]],
                        width="stretch", hide_index=True,
                    )
                    st.divider()
                st.caption(
                    "Baseline: `lexicon_sentiment_v1` (keyword lexicon, deterministic). "
                    "Correlation, not accuracy — sentiment has no realized label."
                )

    st.divider()
    st.subheader("⚖️ Risk policies — Sharpe / max DD / Calmar / total return")
    if not RISK_POLICY_RESULTS.exists():
        st.info(
            "No risk-policy results yet. Run scripts/evaluate_risk_policies.py "
            "to backtest the position-sizing policies."
        )
    else:
        risk_df = pd.read_csv(RISK_POLICY_RESULTS)
        st.dataframe(
            risk_df.sort_values("sharpe", ascending=False).style.format({
                "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
                "calmar": "{:.2f}", "total_return": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Baseline: `buy_and_hold` (FixedWeightPolicy 100%). A sizing policy "
            "earns its keep only by beating it on risk-adjusted return."
        )


with tab_judges:
    st.subheader("⚖️ LLM Judges — decision-quality layer (separate from ML models)")
    st.caption(
        "Non-deterministic agents that read each family's primary model output "
        "and emit a BUY/HOLD/SELL verdict. Judged on hypothetical P&L, hit rate, "
        "cost and consistency — NEVER on MAE/accuracy. Observation only; no "
        "orders are ever placed."
    )

    if not JUDGE.enabled:
        st.info(
            "🟢 Judge layer is DISABLED by default (COINPREDICTOR_JUDGE_ENABLED="
            "false) — it makes zero API calls. Enable the flag and run "
            "scripts/run_judge.py to start logging verdicts."
        )

    if not JUDGE_LOG.exists():
        st.info("No judge verdicts logged yet. This tab populates once "
                "scripts/run_judge.py runs with the flag on.")
    else:
        jdf = pd.read_csv(JUDGE_LOG)
        if jdf.empty:
            st.info("Judge log is empty.")
        else:
            for col in ("as_of_date", "target_date", "run_at"):
                if col in jdf.columns:
                    jdf[col] = pd.to_datetime(jdf[col], errors="coerce")
            for col in ("confidence", "suggested_weight", "estimated_cost_usd",
                        "hypothetical_pnl", "realized_fwd_return"):
                if col in jdf.columns:
                    jdf[col] = pd.to_numeric(jdf[col], errors="coerce")

            total_cost = jdf["estimated_cost_usd"].fillna(0).sum()
            evaluated = jdf[jdf["status"] == "evaluated"].copy()

            k1, k2, k3 = st.columns(3)
            k1.metric("Verdicts logged", len(jdf))
            k2.metric("Evaluated", len(evaluated))
            k3.metric("Cumulative est. cost", f"${total_cost:,.4f}")

            # Hypothetical equity curve (reuse the Backtest chart style).
            if not evaluated.empty:
                for name, grp in evaluated.groupby("model_name"):
                    grp = grp.sort_values("as_of_date")
                    grp = grp.dropna(subset=["hypothetical_pnl"])
                    if grp.empty:
                        continue
                    equity = (1.0 + grp["hypothetical_pnl"]).cumprod()
                    fig_j = go.Figure()
                    fig_j.add_trace(go.Scatter(
                        x=grp["as_of_date"], y=equity, name=f"{name} (hypothetical)"))
                    fig_j.update_layout(
                        height=360,
                        title=f"Hypothetical equity — {name} (growth of $1)",
                        yaxis_title="Equity")
                    st.plotly_chart(fig_j, width="stretch")

                # Consistency: % agreement on action across same-day repeats.
                rows = []
                for name, grp in jdf.groupby("model_name"):
                    shares = []
                    for _d, g in grp.groupby("as_of_date"):
                        if len(g) > 1:
                            shares.append(g["action"].value_counts(normalize=True).iloc[0])
                    hit = (evaluated[evaluated["model_name"] == name]["hypothetical_pnl"] > 0).mean() \
                        if not evaluated.empty else float("nan")
                    rows.append({
                        "Judge": name,
                        "Verdicts": len(grp),
                        "Hit rate": hit,
                        "Consistency (same-day)": (sum(shares) / len(shares)) if shares else float("nan"),
                    })
                st.dataframe(
                    pd.DataFrame(rows).style.format(
                        {"Hit rate": "{:.0%}", "Consistency (same-day)": "{:.0%}"}),
                    width="stretch", hide_index=True,
                )

            with st.expander("📋 Verdict history", expanded=evaluated.empty):
                show = [c for c in [
                    "run_at", "model_name", "as_of_date", "action", "confidence",
                    "suggested_weight", "estimated_cost_usd", "realized_fwd_return",
                    "hypothetical_pnl", "status", "reasoning",
                ] if c in jdf.columns]
                st.dataframe(
                    jdf[show].sort_values("run_at", ascending=False),
                    width="stretch", hide_index=True,
                )

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
    st.plotly_chart(fig_eq, width="stretch")
    st.code(result.summary())

