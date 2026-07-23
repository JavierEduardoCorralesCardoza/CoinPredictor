"""Phase 0 diagnostic: an honest, leak-free scorecard of the current models.

Run this BEFORE changing any model. It answers the questions the user asked:
"what is the real performance today, and where is the edge (or the leak)?"

What it reports
---------------
1. **Leakage check** — the volatility regressor scored with the current
   ``TimeSeriesSplit`` vs a **purged walk-forward**. A big gap means the
   headline numbers were inflated by overlapping-label leakage.
2. **Baseline gap** — LightGBM vs the naive "tomorrow's vol = trailing vol"
   forecast, both on the SAME purged folds. If LightGBM doesn't beat naive,
   the ML model is not adding value yet.
3. **Strategy edge, net of costs** — the vol-targeting strategy vs buy & hold
   under realistic commission + slippage, plus a **Deflated Sharpe Ratio** that
   discounts the best variant for the number of variants tried.

Usage
-----
    python scripts/diagnose.py            # daily (default)
    python scripts/diagnose.py --intraday # hourly, if you want the comparison

Nothing here is financial advice; it is a measurement tool.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from coinpredictor.backtest import walk_forward_backtest
from coinpredictor.config import MODEL, PROCESSED_DIR, STRATEGY
from coinpredictor.features import build_default_features, feature_columns
from coinpredictor.model import (
    build_regime_classifier,
    build_vol_regressor,
    walk_forward_regress,
)
from coinpredictor.validation import (
    annualized_to_period_sr,
    deflated_sharpe_ratio,
    purged_walk_forward,
    walk_forward_regress_purged,
)

# Realistic round-trip friction per unit of turnover: 10 bps taker commission
# + ~5 bps slippage/spread. Tune to your venue; the backtest charges it on the
# day-to-day change in exposure.
COMMISSION = 0.0010
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + SLIPPAGE


def _naive_purged_metrics(
    y_vol: pd.Series, trailing: pd.Series, horizon: int, n_splits: int, embargo: int
) -> tuple[float, float, float, float, int]:
    """Score the naive 'forecast = trailing realized vol' on the purged folds."""
    test_idx = np.concatenate(
        [te for _, te in purged_walk_forward(len(y_vol), n_splits, horizon, embargo)]
    )
    actual = y_vol.iloc[test_idx].to_numpy()
    pred = trailing.iloc[test_idx].to_numpy()
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    corr = float(np.corrcoef(pred, actual)[0, 1])
    return rmse, mae, r2, corr, int(len(actual))


def _fmt_row(name: str, rmse, mae, r2, corr, n) -> str:
    return f"  {name:<28} rmse={rmse:.4f}  mae={mae:.4f}  r2={r2:+.4f}  corr={corr:+.4f}  n={n}"


def _ann_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized Sharpe from a per-period return series."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def run_diagnostics(intraday: bool = False) -> pd.DataFrame:
    horizon = MODEL.vol_horizon
    embargo = max(2, horizon // 2)

    print("=" * 78)
    print(f"PHASE 0 DIAGNOSTIC  ({'INTRADAY 1h' if intraday else 'DAILY 1d'})")
    print("=" * 78)

    # --- Data + features ----------------------------------------------------
    if intraday:
        from coinpredictor.config import INTRADAY
        from coinpredictor.data.exchange_ohlcv import load_exchange_ohlcv
        from coinpredictor.features import build_default_intraday_features

        horizon = INTRADAY.vol_horizon
        embargo = max(2, horizon // 2)
        ohlcv = load_exchange_ohlcv(timeframe=INTRADAY.interval)
        feats = build_default_intraday_features(ohlcv)
    else:
        from coinpredictor.data.ohlcv import load_ohlcv

        ohlcv = load_ohlcv()
        feats = build_default_features(ohlcv)
    print(f"OHLCV rows: {len(ohlcv)}  ({ohlcv.index.min()} -> {ohlcv.index.max()})")

    cols = feature_columns(feats)
    X = feats[cols]
    y_vol = feats[MODEL.target_col].astype(float)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features\n")

    # --- 1 & 2. Regression: leakage + baseline gap --------------------------
    print("[1/3] Volatility regression — leakage check & baseline gap")
    std = walk_forward_regress(build_vol_regressor(), X, y_vol, model_name="LGBM (TimeSeriesSplit)")
    purged = walk_forward_regress_purged(
        build_vol_regressor, X, y_vol, horizon=horizon, embargo=embargo,
        model_name="LGBM (purged WF)",
    )
    naive = _naive_purged_metrics(
        y_vol, feats["realized_vol_trailing"], horizon, MODEL.n_splits, embargo
    )
    print(_fmt_row("LGBM  TimeSeriesSplit", std.rmse, std.mae, std.r2, std.corr, "-"))
    print(_fmt_row("LGBM  purged WF", purged.rmse, purged.mae, purged.r2, purged.corr, purged.n_test))
    print(_fmt_row("naive persistence  WF", *naive))
    leak = std.r2 - purged.r2
    beats = purged.rmse < naive[0]
    print(f"\n  -> leakage gap (r2 std - purged): {leak:+.4f}"
          f"  {'(LARGE: naive CV was optimistic)' if leak > 0.05 else '(small)'}")
    print(f"  -> LGBM beats naive on RMSE (purged): {'YES' if beats else 'NO — ML adds no edge yet'}\n")

    # --- 3. Strategy edge, net of realistic costs ---------------------------
    periods_per_year = 24 * 365 if intraday else 365
    print(f"[3/3] Vol-targeting strategy vs buy & hold "
          f"(cost/turnover={TOTAL_COST:.4f})")
    variants = {
        "vol-target power=0.5": dict(power=0.5),
        "vol-target power=1.0": dict(power=1.0),
        "vol-target power=1.5": dict(power=1.5),
        "regime overlay cut=0.5": dict(power=1.0, clf_factory=build_regime_classifier, regime_cut=0.5),
        "regime overlay cut=1.0": dict(power=1.0, clf_factory=build_regime_classifier, regime_cut=1.0),
    }
    rows = []
    period_srs = []
    best = None
    for name, kw in variants.items():
        res = walk_forward_backtest(build_vol_regressor, feats, fee=TOTAL_COST, **kw)
        strat_ret = res.equity["strategy"].pct_change().dropna()
        bh_ret = res.equity["buy_and_hold"].pct_change().dropna()
        strat_sharpe = _ann_sharpe(strat_ret, periods_per_year)
        bh_sharpe = _ann_sharpe(bh_ret, periods_per_year)
        rows.append({
            "variant": name,
            "strat_sharpe": strat_sharpe,
            "bh_sharpe": bh_sharpe,
            "strat_return": res.strategy_return,
            "bh_return": res.bh_return,
            "strat_maxdd": res.strategy_max_drawdown,
            "bh_maxdd": res.bh_max_drawdown,
            "forecast_corr": res.forecast_corr,
        })
        period_srs.append(annualized_to_period_sr(strat_sharpe, periods_per_year))
        if best is None or strat_sharpe > best[1]:
            best = (name, strat_sharpe, strat_ret)
        print(f"  {name:<26} Sharpe={strat_sharpe:+.2f} (B&H {bh_sharpe:+.2f})  "
              f"ret={res.strategy_return:+.1%}  maxDD={res.strategy_max_drawdown:+.1%}  "
              f"corr={res.forecast_corr:+.3f}")

    sr_var = float(np.var(period_srs, ddof=1)) if len(period_srs) > 1 else 0.0
    dsr = deflated_sharpe_ratio(best[2], n_trials=len(variants), sr_variance=sr_var)
    print(f"\n  -> best variant: {best[0]}  (Sharpe {best[1]:+.2f})")
    print(f"  -> Deflated Sharpe (P the edge is real, after {len(variants)} trials): {dsr:.3f}")
    verdict = "credible" if dsr > 0.95 else ("weak" if dsr > 0.5 else "NOT credible (likely luck)")
    print(f"  -> verdict: {verdict}\n")

    # --- Persist a machine-readable summary ---------------------------------
    df = pd.DataFrame(rows)
    df["deflated_sharpe"] = dsr
    df["leakage_gap_r2"] = leak
    df["lgbm_purged_rmse"] = purged.rmse
    df["naive_purged_rmse"] = naive[0]
    df["timeframe"] = "1h" if intraday else "1d"
    out = PROCESSED_DIR / f"diagnostics_phase0_{'1h' if intraday else '1d'}.csv"
    df.to_csv(out, index=False)
    print(f"Saved diagnostic summary -> {out}")
    print("=" * 78)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 honest diagnostic report")
    parser.add_argument("--intraday", action="store_true", help="use hourly candles")
    args = parser.parse_args()
    run_diagnostics(intraday=args.intraday)


if __name__ == "__main__":
    main()
