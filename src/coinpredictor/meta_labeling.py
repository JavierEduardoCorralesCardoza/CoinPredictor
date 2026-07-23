"""Phase 3: directional meta-labeling.

Phase 0 showed the honest problem: even a *good* volatility forecast doesn't
beat buy & hold on Sharpe, and naive high-turnover sizing gets eaten by costs.
The edge has to come from **direction + selective execution**, not from sizing
a long-only book by volatility.

Meta-labeling (López de Prado, *Advances in Financial ML*) splits the job:

1. **Primary model** decides the *side* (long / flat / short) with a simple,
   transparent, low-turnover rule (here: a moving-average trend filter).
2. **Meta-labels** ask a narrower question — *given the primary took this side,
   did the trade actually work?* — resolved with a side-aware triple barrier.
3. **Secondary model** (LightGBM) predicts P(the primary is right this time)
   and is used to **filter and size**: trade only when conviction is high.

Because the secondary sits out low-conviction days, turnover (and cost drag)
drops, which is exactly what the Phase 0 diagnostic said was killing the naive
strategy. Everything is evaluated out-of-sample with the purged walk-forward and
discounted with the Deflated Sharpe Ratio, net of realistic costs.

    python -m coinpredictor.meta_labeling            # daily
    python -m coinpredictor.meta_labeling --short    # allow short side
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinpredictor.config import ENTRY, MODEL, PROCESSED_DIR
from coinpredictor.entry import resolve_barrier
from coinpredictor.features import feature_columns
from coinpredictor.model import build_regime_classifier
from coinpredictor.validation import (
    annualized_to_period_sr,
    deflated_sharpe_ratio,
    purged_walk_forward,
)


# --- Primary model: transparent trend/momentum side --------------------------
def primary_trend_side(
    feats: pd.DataFrame,
    *,
    fast: int = 20,
    slow: int = 50,
    allow_short: bool = False,
) -> pd.Series:
    """Primary side from a moving-average trend filter.

    Long (+1) when the fast SMA is above the slow SMA (up-trend). Otherwise flat
    (0), or short (-1) when ``allow_short``. This is deliberately simple and
    low-turnover; the secondary model provides the skill.
    """
    close = feats["close"]
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()
    up = fast_sma > slow_sma
    side = pd.Series(np.where(up, 1.0, -1.0 if allow_short else 0.0), index=feats.index)
    side[fast_sma.isna() | slow_sma.isna()] = np.nan
    return side.rename("primary_side")


# --- Side-aware triple-barrier meta-labels -----------------------------------
def meta_labels(
    ohlcv: pd.DataFrame,
    side: pd.Series,
    *,
    horizon: int | None = None,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
    vol: pd.Series | None = None,
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
) -> pd.Series:
    """Meta-label each bar where the primary took a side: 1 if the trade worked.

    For a long side the take-profit sits above / stop below entry; for a short
    side the barriers flip (implemented by resolving the barrier on a mirrored
    high/low series). Bars where ``side`` is 0/NaN get NaN (no trade to judge).

    Barriers can be **fixed** (``tp_pct``/``sl_pct``) or **volatility-scaled**:
    pass a per-bar ``vol`` fraction and the barriers become
    ``tp_mult·vol`` / ``sl_mult·vol`` at each entry, so they widen in turbulent
    regimes and tighten in calm ones. ``tp_mult`` ≠ ``sl_mult`` gives an
    asymmetric (e.g. let-winners-run) profile.
    """
    horizon = horizon or ENTRY.horizon
    tp_pct = ENTRY.tp_pct if tp_pct is None else tp_pct
    sl_pct = ENTRY.sl_pct if sl_pct is None else sl_pct

    closes = ohlcv["close"].to_numpy(dtype="float64")
    highs = ohlcv["high"].to_numpy(dtype="float64")
    lows = ohlcv["low"].to_numpy(dtype="float64")
    n = len(ohlcv)
    side_arr = side.reindex(ohlcv.index).to_numpy(dtype="float64")
    vol_arr = (
        vol.reindex(ohlcv.index).to_numpy(dtype="float64") if vol is not None else None
    )

    out = np.full(n, np.nan)
    for i in range(n):
        s = side_arr[i]
        if not np.isfinite(s) or s == 0:
            continue
        if vol_arr is not None:
            v = vol_arr[i]
            if not np.isfinite(v) or v <= 0:
                continue
            tpp, slp = tp_mult * v, sl_mult * v
        else:
            tpp, slp = tp_pct, sl_pct
        if s > 0:
            outcome = resolve_barrier(highs, lows, i, closes[i], horizon, tpp, slp, n)
        else:
            # Short: mirror prices so a downward move hits the "take-profit".
            outcome = resolve_barrier(-lows, -highs, i, -closes[i], horizon, tpp, slp, n)
        if outcome is not None:
            out[i] = outcome
    return pd.Series(out, index=ohlcv.index, name="meta_label")


# --- Out-of-sample meta probabilities via purged walk-forward ----------------
def oos_meta_proba(
    X: pd.DataFrame,
    y_meta: pd.Series,
    tradable: pd.Series,
    *,
    horizon: int,
    n_splits: int,
    embargo: int,
    model_factory=build_regime_classifier,
) -> pd.Series:
    """Purged walk-forward P(trade works), predicted only on tradable bars."""
    proba = pd.Series(np.nan, index=X.index, dtype="float64")
    trade_mask = tradable.to_numpy()
    label = y_meta.to_numpy()

    for train_idx, test_idx in purged_walk_forward(len(X), n_splits, horizon, embargo):
        tr = train_idx[trade_mask[train_idx] & np.isfinite(label[train_idx])]
        te = test_idx[trade_mask[test_idx]]
        if len(tr) == 0 or len(te) == 0:
            continue
        y_tr = pd.Series(label[tr], index=X.index[tr]).astype(int)
        if y_tr.nunique() < 2:
            continue
        model = model_factory()
        model.fit(X.iloc[tr], y_tr)
        proba.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
    return proba


# --- Cost-aware backtest -----------------------------------------------------
@dataclass
class StratMetrics:
    name: str
    sharpe: float
    total_return: float
    max_drawdown: float
    avg_turnover: float
    exposure: float          # fraction of bars with a non-zero position
    returns: pd.Series

    def row(self) -> dict:
        return {
            "strategy": self.name,
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "avg_turnover": self.avg_turnover,
            "exposure": self.exposure,
        }


def _max_drawdown(equity: pd.Series) -> float:
    running = equity.cummax()
    return float((equity / running - 1.0).min())


def backtest_positions(
    ohlcv: pd.DataFrame,
    weight: pd.Series,
    *,
    fee: float,
    periods_per_year: int,
    name: str,
) -> StratMetrics:
    """Backtest a daily target-weight series, net of turnover costs."""
    close = ohlcv["close"].reindex(weight.index)
    next_ret = close.shift(-1) / close - 1.0
    w = weight.fillna(0.0)
    df = pd.DataFrame({"w": w, "r": next_ret}).dropna()
    turnover = df["w"].diff().abs().fillna(df["w"].abs())
    strat = df["w"] * df["r"] - turnover * fee
    equity = (1.0 + strat).cumprod()
    std = strat.std()
    sharpe = 0.0 if std == 0 or np.isnan(std) else float(np.sqrt(periods_per_year) * strat.mean() / std)
    return StratMetrics(
        name=name,
        sharpe=sharpe,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        max_drawdown=_max_drawdown(equity) if len(equity) else 0.0,
        avg_turnover=float(turnover.mean()),
        exposure=float((df["w"].abs() > 1e-9).mean()),
        returns=strat,
    )


# --- Orchestration -----------------------------------------------------------
def run_meta_labeling(
    *,
    allow_short: bool = False,
    fast: int | None = None,
    slow: int | None = None,
    fee: float = 0.0015,
    thresholds: tuple[float, ...] = (0.42, 0.45, 0.48, 0.50),
    use_derivatives: bool = False,
    intraday: bool = False,
    vol_scaled: bool = False,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    from coinpredictor.config import INTRADAY
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import (
        build_default_features,
        build_default_intraday_features,
        build_features_full,
    )

    if intraday:
        from coinpredictor.data.exchange_ohlcv import load_exchange_ohlcv

        fast = fast or 24          # ~1-day fast trend on hourly bars
        slow = slow or 72          # ~3-day slow trend
        horizon = INTRADAY.vol_horizon      # 24h barrier window
        tp_pct = sl_pct = 0.03              # intraday moves are smaller than daily
        periods_per_year = INTRADAY.annualization
        ohlcv = load_exchange_ohlcv(timeframe=INTRADAY.interval)  # hourly bars
        base_feats = build_default_intraday_features(ohlcv, drop_na=False)
    else:
        fast = fast or 20
        slow = slow or 50
        horizon = ENTRY.horizon
        tp_pct, sl_pct = ENTRY.tp_pct, ENTRY.sl_pct
        periods_per_year = 365
        ohlcv = load_ohlcv()
        base_feats = build_default_features(ohlcv, drop_na=False)

    embargo = max(2, horizon // 2)
    base_cols = feature_columns(base_feats)

    # Volatility-scaled barriers: per-bar fraction ≈ (per-bar return std) scaled
    # to the barrier horizon. Widens targets in turbulent regimes, tightens them
    # in calm ones; tp_mult ≠ sl_mult makes the profile asymmetric.
    bar_vol = None
    if vol_scaled:
        rets = base_feats["log_return_1d"] if "log_return_1d" in base_feats else np.log(ohlcv["close"]).diff()
        bar_vol = rets.rolling(vol_lookback).std() * np.sqrt(horizon)

    if vol_scaled:
        barrier_desc = f"vol×(tp{tp_mult:g}/sl{sl_mult:g})"
    else:
        barrier_desc = f"tp/sl={tp_pct:.0%}/{sl_pct:.0%}"

    print("=" * 78)
    print(f"PHASE 3 META-LABELING  ({'INTRADAY 1h' if intraday else 'DAILY 1d'}, "
          f"side={'long/short' if allow_short else 'long/flat'}, "
          f"trend {fast}/{slow}, {barrier_desc}, h={horizon}"
          f"{', +derivatives' if use_derivatives else ''})")
    print("=" * 78)

    if use_derivatives and not intraday:
        # Add free derivatives (OKX funding + Deribit DVOL). They only cover the
        # recent ~3y, so we DON'T require them present: LightGBM handles NaN, and
        # the model simply learns to use them once history exists.
        feats = build_features_full(
            ohlcv, use_macro=True, use_onchain=True, use_sentiment=True,
            use_implied_vol=True, use_funding=True, drop_na=False,
        )
        cols = feature_columns(feats)
    else:
        feats, cols = base_feats, base_cols

    # Primary side + meta-labels, aligned to rows with usable *base* features.
    side = primary_trend_side(base_feats, fast=fast, slow=slow, allow_short=allow_short)
    y_meta = meta_labels(
        ohlcv, side, horizon=horizon, tp_pct=tp_pct, sl_pct=sl_pct,
        vol=bar_vol, tp_mult=tp_mult, sl_mult=sl_mult,
    )

    valid = base_feats[base_cols].notna().all(axis=1) & side.notna()
    idx = base_feats.index[valid]
    X = feats.loc[idx, cols]
    side = side.loc[idx]
    y_meta = y_meta.reindex(idx)
    ohlcv_v = ohlcv.loc[idx]
    tradable = side != 0

    print(f"Rows: {len(X)}  tradable bars: {int(tradable.sum())}  "
          f"win-rate on trades: {y_meta[tradable].mean():.3f}\n")

    # Out-of-sample meta probabilities (purged).
    proba = oos_meta_proba(
        X, y_meta, tradable, horizon=horizon, n_splits=MODEL.n_splits, embargo=embargo
    )

    # Baselines: buy & hold, and primary-always (no meta filter).
    bh = backtest_positions(
        ohlcv_v, pd.Series(1.0, index=X.index), fee=fee,
        periods_per_year=periods_per_year, name="buy_and_hold",
    )
    prim = backtest_positions(
        ohlcv_v, side.where(tradable, 0.0), fee=fee,
        periods_per_year=periods_per_year, name="primary_only",
    )
    print(f"  {'buy_and_hold':<22} Sharpe={bh.sharpe:+.2f}  ret={bh.total_return:+.1%}  "
          f"maxDD={bh.max_drawdown:+.1%}  turnover={bh.avg_turnover:.3f}  exp={bh.exposure:.0%}")
    print(f"  {'primary_only':<22} Sharpe={prim.sharpe:+.2f}  ret={prim.total_return:+.1%}  "
          f"maxDD={prim.max_drawdown:+.1%}  turnover={prim.avg_turnover:.3f}  exp={prim.exposure:.0%}")

    # Meta-gated strategies: the meta-label is a binary gate (López de Prado) —
    # take the *full* primary side when conviction clears the threshold, else sit
    # out. This keeps exposure high while dropping low-conviction trades.
    rows = [bh.row(), prim.row()]
    period_srs = []
    best = None
    for thr in thresholds:
        gate = (proba >= thr).astype(float).fillna(0.0)
        weight = side.where(tradable, 0.0) * gate
        m = backtest_positions(
            ohlcv_v, weight, fee=fee, periods_per_year=periods_per_year,
            name=f"meta thr={thr:.2f}",
        )
        rows.append(m.row())
        period_srs.append(annualized_to_period_sr(m.sharpe, periods_per_year))
        if best is None or m.sharpe > best.sharpe:
            best = m
        print(f"  {m.name:<22} Sharpe={m.sharpe:+.2f}  ret={m.total_return:+.1%}  "
              f"maxDD={m.max_drawdown:+.1%}  turnover={m.avg_turnover:.3f}  exp={m.exposure:.0%}")

    sr_var = float(np.var(period_srs, ddof=1)) if len(period_srs) > 1 else 0.0
    dsr = deflated_sharpe_ratio(best.returns, n_trials=len(thresholds), sr_variance=sr_var)
    print(f"\n  -> best meta strategy: {best.name}  (Sharpe {best.sharpe:+.2f} vs B&H {bh.sharpe:+.2f})")
    print(f"  -> beats buy & hold on Sharpe: {'YES' if best.sharpe > bh.sharpe else 'NO'}")
    print(f"  -> Deflated Sharpe (after {len(thresholds)} thresholds): {dsr:.3f}")
    verdict = "credible edge" if dsr > 0.95 else ("weak" if dsr > 0.5 else "NOT credible (likely luck)")
    print(f"  -> verdict: {verdict}\n")

    df = pd.DataFrame(rows)
    df["deflated_sharpe_best"] = dsr
    out = PROCESSED_DIR / f"meta_labeling_results_{'1h' if intraday else '1d'}.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")
    print("=" * 78)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 directional meta-labeling")
    parser.add_argument("--short", action="store_true", help="allow short side")
    parser.add_argument("--intraday", action="store_true", help="run on hourly bars")
    parser.add_argument("--derivatives", action="store_true",
                        help="add free derivatives (funding + DVOL) to the meta model")
    parser.add_argument("--vol-scaled", action="store_true",
                        help="use volatility-scaled (dynamic) barriers")
    parser.add_argument("--tp-mult", type=float, default=1.5,
                        help="take-profit multiple of per-bar vol (vol-scaled mode)")
    parser.add_argument("--sl-mult", type=float, default=1.0,
                        help="stop-loss multiple of per-bar vol (vol-scaled mode)")
    parser.add_argument("--fast", type=int, default=None)
    parser.add_argument("--slow", type=int, default=None)
    args = parser.parse_args()
    run_meta_labeling(
        allow_short=args.short, fast=args.fast, slow=args.slow,
        use_derivatives=args.derivatives, intraday=args.intraday,
        vol_scaled=args.vol_scaled, tp_mult=args.tp_mult, sl_mult=args.sl_mult,
    )


if __name__ == "__main__":
    main()
