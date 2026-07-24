"""Backtesting: convert volatility forecasts into a position-sizing strategy
and compare against buy-and-hold.

Strategy: **volatility targeting.** Each day we size BTC exposure so the
portfolio's *expected* volatility stays near a constant target:

    weight_t = clip(target_annual_vol / predicted_vol_t, min_weight, max_weight)

When the model predicts turbulence ahead we shrink exposure; when it predicts
calm we lever up to the cap. The next-day return is realized with that weight,
so there is no look-ahead in the P&L. This is the canonical way a volatility
forecast creates investable value: it typically *raises Sharpe and cuts
drawdowns* versus passively holding the asset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from coinpredictor.config import COSTS, MODEL, STRATEGY
from coinpredictor.features import feature_columns


# Risk profiles map a forecast to a recommended BTC weight. ``power`` controls
# how hard inverse-vol sizing reacts; ``regime_cut`` trims exposure when the
# high-volatility regime is likely (0 = off, 1 = go flat when certain).
STRATEGY_PROFILES = {
    "aggressive": {"power": 0.0, "regime_cut": 0.0},   # always fully invested (~buy & hold)
    "balanced": {"power": 0.5, "regime_cut": 0.0},     # gentle vol-targeting
    "defensive": {"power": 1.0, "regime_cut": 1.0},    # vol-target + regime overlay
}


def recommend_weight(
    predicted_vol: float,
    regime_proba: float | None = None,
    profile: str | None = None,
) -> float:
    """Turn a single volatility forecast into a recommended BTC weight [0, 1].

    Uses the same sizing math as the backtest so live advice matches the
    historically-evaluated behaviour. ``profile`` defaults to
    ``STRATEGY.live_profile``.
    """
    profile = profile or STRATEGY.live_profile
    if profile not in STRATEGY_PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}; choose from {list(STRATEGY_PROFILES)}"
        )
    cfg = STRATEGY_PROFILES[profile]

    if predicted_vol <= 0:
        return STRATEGY.min_weight
    raw = (STRATEGY.target_annual_vol / predicted_vol) ** cfg["power"]
    weight = min(max(raw, STRATEGY.min_weight), STRATEGY.max_weight)

    if regime_proba is not None and cfg["regime_cut"] > 0.0:
        weight *= 1.0 - cfg["regime_cut"] * min(max(regime_proba, 0.0), 1.0)
        weight = min(max(weight, STRATEGY.min_weight), STRATEGY.max_weight)
    return float(weight)


@dataclass
class BacktestResult:
    """Equity curves and headline performance metrics."""

    equity: pd.DataFrame            # columns: strategy, buy_and_hold
    weights: pd.Series              # daily BTC exposure used by the strategy
    forecast_corr: float            # corr(predicted vol, realized next-day |ret|)
    strategy_return: float
    bh_return: float
    strategy_sharpe: float
    bh_sharpe: float
    strategy_max_drawdown: float
    bh_max_drawdown: float

    def summary(self) -> str:
        return (
            f"Vol-forecast vs realized |return| corr: {self.forecast_corr:.4f}\n"
            f"Strategy total return: {self.strategy_return:.2%} "
            f"(Sharpe {self.strategy_sharpe:.2f}, "
            f"max DD {self.strategy_max_drawdown:.2%})\n"
            f"Buy & hold total return: {self.bh_return:.2%} "
            f"(Sharpe {self.bh_sharpe:.2f}, max DD {self.bh_max_drawdown:.2%})"
        )


def _annualized_sharpe(daily_returns: pd.Series, periods: int = 365) -> float:
    """Sharpe ratio assuming ~365 trading days/year (crypto trades daily)."""
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods) * daily_returns.mean() / std)


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of an equity curve (negative number)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def backtest_vol_forecast(
    features_df: pd.DataFrame,
    predicted_vol: np.ndarray,
    fee: float = 0.0,
    *,
    power: float = 1.0,
    regime_proba: np.ndarray | None = None,
    regime_cut: float = 0.0,
) -> BacktestResult:
    """Backtest a volatility-targeting strategy from forecast volatility.

    Parameters
    ----------
    features_df:
        Feature frame (must contain ``close``), aligned row-for-row with
        ``predicted_vol``.
    predicted_vol:
        Predicted forward annualized volatility for each row (the model output).
    fee:
        Per-trade proportional cost charged on the change in exposure.
    power:
        Exponent on the vol-target ratio. ``1.0`` is classic inverse-vol
        sizing; ``>1`` reacts more aggressively to vol changes, ``<1`` more
        gently.
    regime_proba:
        Optional predicted probability of the high-volatility regime, aligned
        row-for-row. When provided, exposure is trimmed in proportion.
    regime_cut:
        How much to cut exposure at full regime certainty (0 = no overlay,
        1 = go flat when the high-vol regime is certain).
    """
    df = features_df.copy()
    df["pred_vol"] = np.asarray(predicted_vol, dtype="float64")
    if regime_proba is not None:
        df["regime_proba"] = np.asarray(regime_proba, dtype="float64")

    # Next-day realized return (what the position is held through).
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1.0
    subset = ["next_return", "pred_vol"]
    if regime_proba is not None:
        subset.append("regime_proba")
    df = df.dropna(subset=subset)

    # Volatility-target weight, clipped to the allowed exposure band.
    raw_weight = (
        STRATEGY.target_annual_vol / df["pred_vol"].replace(0.0, np.nan)
    ) ** power
    weight = raw_weight.clip(STRATEGY.min_weight, STRATEGY.max_weight).fillna(
        STRATEGY.min_weight
    )

    # Regime overlay: shrink exposure when a volatile regime is likely.
    if regime_proba is not None and regime_cut > 0.0:
        weight = weight * (1.0 - regime_cut * df["regime_proba"].clip(0.0, 1.0))
        weight = weight.clip(STRATEGY.min_weight, STRATEGY.max_weight)

    gross = weight * df["next_return"]
    # Cost proportional to how much exposure we rebalanced day-to-day.
    turnover = weight.diff().abs().fillna(weight.abs())
    strat_returns = gross - turnover * fee

    bh_returns = df["next_return"]

    equity = pd.DataFrame(
        {
            "strategy": (1.0 + strat_returns).cumprod(),
            "buy_and_hold": (1.0 + bh_returns).cumprod(),
        },
        index=df.index,
    )

    # Does a higher vol forecast actually precede a bigger next-day move?
    realized_abs = df["next_return"].abs()
    forecast_corr = float(np.corrcoef(df["pred_vol"], realized_abs)[0, 1])

    return BacktestResult(
        equity=equity,
        weights=weight,
        forecast_corr=forecast_corr,
        strategy_return=float(equity["strategy"].iloc[-1] - 1.0),
        bh_return=float(equity["buy_and_hold"].iloc[-1] - 1.0),
        strategy_sharpe=_annualized_sharpe(strat_returns),
        bh_sharpe=_annualized_sharpe(bh_returns),
        strategy_max_drawdown=_max_drawdown(equity["strategy"]),
        bh_max_drawdown=_max_drawdown(equity["buy_and_hold"]),
    )


def walk_forward_backtest(
    model_factory,
    features_df: pd.DataFrame,
    n_splits: int | None = None,
    fee: float = COSTS.per_side,
    *,
    power: float = 1.0,
    clf_factory=None,
    regime_cut: float = 0.0,
) -> BacktestResult:
    """Produce out-of-sample vol forecasts via walk-forward, then backtest them.

    ``model_factory`` is a zero-arg callable returning a fresh, unfitted
    regressor (e.g. ``build_vol_regressor``). Each fold trains on the past and
    predicts the next block, so the equity curve uses only out-of-sample
    forecasts. Pass ``clf_factory`` (+ ``regime_cut`` > 0) to add an
    out-of-sample regime overlay that trims exposure in volatile regimes.
    """
    from sklearn.model_selection import TimeSeriesSplit

    n_splits = n_splits or MODEL.n_splits
    cols = feature_columns(features_df)
    X = features_df[cols]
    y = features_df[MODEL.target_col].astype(float)
    y_regime = features_df[MODEL.regime_col].astype(int)

    preds = pd.Series(index=features_df.index, dtype="float64")
    regime = pd.Series(index=features_df.index, dtype="float64") if clf_factory else None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(X.iloc[test_idx])
        if clf_factory is not None:
            clf = clf_factory()
            clf.fit(X.iloc[train_idx], y_regime.iloc[train_idx])
            regime.iloc[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    mask = preds.notna()
    regime_arr = regime[mask].to_numpy() if regime is not None else None
    return backtest_vol_forecast(
        features_df[mask],
        preds[mask].to_numpy(),
        fee=fee,
        power=power,
        regime_proba=regime_arr,
        regime_cut=regime_cut,
    )


def compare_strategies(features_df: pd.DataFrame, n_splits: int | None = None) -> str:
    """Run several vol-forecast strategy variants and rank them by Sharpe."""
    from coinpredictor.model import build_regime_classifier, build_vol_regressor

    variants = {
        "vol-target (power=1)": dict(power=1.0),
        "vol-target (power=1.5)": dict(power=1.5),
        "vol-target (power=0.5)": dict(power=0.5),
        "regime overlay (cut=0.5)": dict(
            power=1.0, clf_factory=build_regime_classifier, regime_cut=0.5
        ),
        "regime overlay (cut=1.0)": dict(
            power=1.0, clf_factory=build_regime_classifier, regime_cut=1.0
        ),
    }

    lines = ["Strategy variant comparison (out-of-sample):"]
    bh_done = False
    scored = []
    for name, kwargs in variants.items():
        res = walk_forward_backtest(
            build_vol_regressor, features_df, n_splits=n_splits, **kwargs
        )
        scored.append((name, res))
        if not bh_done:
            lines.append(
                f"  buy & hold            : ret {res.bh_return:7.2%}  "
                f"Sharpe {res.bh_sharpe:.2f}  maxDD {res.bh_max_drawdown:7.2%}"
            )
            bh_done = True
    scored.sort(key=lambda nr: nr[1].strategy_sharpe, reverse=True)
    for name, res in scored:
        lines.append(
            f"  {name:22s}: ret {res.strategy_return:7.2%}  "
            f"Sharpe {res.strategy_sharpe:.2f}  maxDD {res.strategy_max_drawdown:7.2%}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Risk family (Phase 1d): pluggable position-sizing policies.
#
# Unlike the other families, ``risk`` is NOT a per-day prediction row. A policy
# is a *rule* mapping today's forecasts to a BTC weight; it is evaluated by
# replaying it over history and looking at portfolio-level outcomes (Sharpe,
# max drawdown, Calmar, total return). One row per policy lands in
# risk_policy_results.csv via scripts/evaluate_risk_policies.py.
# ---------------------------------------------------------------------------
class RiskPolicy(Protocol):
    """A position-sizing rule: forecasts in, a BTC weight in [0, 1] out."""

    name: str

    def size(
        self,
        predicted_vol: float,
        regime_proba: float | None,
        trend_regime: str | None,
    ) -> float:
        ...


@dataclass
class FixedWeightPolicy:
    """Always hold a fixed weight. At 100% this IS the buy-and-hold benchmark
    (the required baseline for this family) — reused explicitly, not reinvented.
    """

    name: str = "fixed_full_weight"
    weight: float = 1.0

    def size(self, predicted_vol, regime_proba=None, trend_regime=None) -> float:
        return float(min(max(self.weight, STRATEGY.min_weight), STRATEGY.max_weight))


@dataclass
class VolTargetPolicy:
    """The existing volatility-targeting logic, refactored into the interface.

    Scales exposure inversely to forecast vol (``power``) and optionally trims
    it when the high-vol regime is likely (``regime_cut``) — identical math to
    ``recommend_weight`` so live advice and this backtest stay consistent.
    """

    name: str = "vol_target"
    power: float = 1.0
    regime_cut: float = 0.0

    def size(self, predicted_vol, regime_proba=None, trend_regime=None) -> float:
        if predicted_vol is None or predicted_vol <= 0:
            return STRATEGY.min_weight
        raw = (STRATEGY.target_annual_vol / predicted_vol) ** self.power
        weight = min(max(raw, STRATEGY.min_weight), STRATEGY.max_weight)
        if regime_proba is not None and self.regime_cut > 0.0:
            weight *= 1.0 - self.regime_cut * min(max(regime_proba, 0.0), 1.0)
            weight = min(max(weight, STRATEGY.min_weight), STRATEGY.max_weight)
        return float(weight)


@dataclass
class KellyFractionPolicy:
    """Fractional-Kelly sizing under a constant-Sharpe assumption.

    Kelly-optimal leverage for a lognormal asset is ``mu/sigma^2 = Sharpe/sigma``.
    We don't forecast ``mu`` per day, so we assume a modest constant Sharpe and
    take a fraction of full Kelly (full Kelly is famously too aggressive), then
    scale the position inversely to the *forecast* volatility.
    """

    name: str = "kelly_fraction"
    assumed_sharpe: float = 0.5
    kelly_fraction: float = 0.5

    def size(self, predicted_vol, regime_proba=None, trend_regime=None) -> float:
        if predicted_vol is None or predicted_vol <= 0:
            return STRATEGY.min_weight
        weight = self.kelly_fraction * self.assumed_sharpe / predicted_vol
        return float(min(max(weight, STRATEGY.min_weight), STRATEGY.max_weight))


def _calmar(total_return: float, n_days: int, max_drawdown: float) -> float:
    """Annualized-return / |max drawdown|. 0 if drawdown is ~0 or no history."""
    if n_days <= 0 or max_drawdown >= -1e-9:
        return 0.0
    annualized = (1.0 + total_return) ** (365.0 / n_days) - 1.0
    return float(annualized / abs(max_drawdown))


def _walk_forward_vol_preds(
    features_df: pd.DataFrame, n_splits: int | None = None
) -> tuple[pd.Series, pd.Series]:
    """Out-of-sample forecast vol + high-vol-regime probability, per row."""
    from sklearn.model_selection import TimeSeriesSplit

    from coinpredictor.model import build_regime_classifier, build_vol_regressor

    n_splits = n_splits or MODEL.n_splits
    cols = feature_columns(features_df)
    X = features_df[cols]
    y = features_df[MODEL.target_col].astype(float)
    y_regime = features_df[MODEL.regime_col].astype(int)

    preds = pd.Series(index=features_df.index, dtype="float64")
    regime = pd.Series(index=features_df.index, dtype="float64")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        reg = build_vol_regressor()
        reg.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = reg.predict(X.iloc[test_idx])
        clf = build_regime_classifier()
        clf.fit(X.iloc[train_idx], y_regime.iloc[train_idx])
        regime.iloc[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
    return preds, regime


def backtest_policy(
    policy: RiskPolicy,
    features_df: pd.DataFrame,
    predicted_vol: pd.Series,
    regime_proba: pd.Series | None = None,
    trend_regime: pd.Series | None = None,
    fee: float = COSTS.per_side,
) -> dict:
    """Replay a sizing policy over pre-computed OOS forecasts; return metrics."""
    df = features_df.copy()
    df["pred_vol"] = predicted_vol.reindex(df.index)
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna(subset=["next_return", "pred_vol"])

    weights = []
    for idx in df.index:
        rp = (
            float(regime_proba[idx])
            if regime_proba is not None and idx in regime_proba.index
            else None
        )
        tr = (
            trend_regime[idx]
            if trend_regime is not None and idx in trend_regime.index
            else None
        )
        weights.append(policy.size(float(df.loc[idx, "pred_vol"]), rp, tr))
    weight = pd.Series(weights, index=df.index).clip(
        STRATEGY.min_weight, STRATEGY.max_weight
    )

    turnover = weight.diff().abs().fillna(weight.abs())
    strat_returns = weight * df["next_return"] - turnover * fee
    equity = (1.0 + strat_returns).cumprod()

    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    max_dd = _max_drawdown(equity) if len(equity) else 0.0
    return {
        "policy": policy.name,
        "sharpe": _annualized_sharpe(strat_returns),
        "max_drawdown": max_dd,
        "calmar": _calmar(total_return, len(equity), max_dd),
        "total_return": total_return,
        "n_days": int(len(equity)),
    }


def evaluate_risk_policies(
    features_df: pd.DataFrame,
    policies: list[RiskPolicy],
    n_splits: int | None = None,
    trend_regime: pd.Series | None = None,
) -> list[dict]:
    """Run every policy through walk-forward OOS forecasts and score each."""
    preds, regime = _walk_forward_vol_preds(features_df, n_splits=n_splits)
    mask = preds.notna()
    fdf = features_df[mask]
    results = []
    for policy in policies:
        results.append(
            backtest_policy(
                policy,
                fdf,
                preds[mask],
                regime_proba=regime[mask],
                trend_regime=trend_regime,
            )
        )
    return results


# Default policy roster evaluated by scripts/evaluate_risk_policies.py. The
# FixedWeightPolicy(1.0) baseline (buy & hold) is listed first, on purpose.
RISK_POLICIES: list[RiskPolicy] = [
    FixedWeightPolicy(name="buy_and_hold", weight=1.0),
    VolTargetPolicy(name="vol_target_p1", power=1.0),
    VolTargetPolicy(name="vol_target_regime", power=1.0, regime_cut=0.5),
    KellyFractionPolicy(name="kelly_half", assumed_sharpe=0.5, kelly_fraction=0.5),
]


if __name__ == "__main__":  # pragma: no cover - manual run
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import build_default_features

    feats = build_default_features(load_ohlcv())
    print(compare_strategies(feats))

