"""Phase 2: classical volatility baselines that LightGBM must beat.

Phase 0 exposed the honest problem on daily BTC: the LightGBM volatility
regressor had a *negative* purged R² (-0.025). Before trusting any ML model we
need strong, well-understood classical baselines. If LightGBM can't beat these,
its complexity isn't earning its keep.

Three baselines, all evaluated on the *same* purged walk-forward as LightGBM and
predicting the *same* target (``MODEL.target_col`` — forward h-day annualized
realized volatility):

1. **Naive persistence** — tomorrow's vol equals today's trailing vol.
2. **HAR-RV** (Corsi 2009) — OLS on daily / weekly / monthly realized-vol
   components, using a **Garman-Klass** OHLC range estimator for the daily
   realized measure (far more efficient than close-to-close on daily data).
3. **GARCH(1,1) / EGARCH** — conditional-variance model on daily returns,
   rolling-origin refit for honest out-of-sample forecasts.

    python -m coinpredictor.vol_baselines            # daily
    python -m coinpredictor.vol_baselines --egarch   # use EGARCH instead
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from coinpredictor.config import MODEL, PROCESSED_DIR
from coinpredictor.features import feature_columns
from coinpredictor.model import build_vol_regressor
from coinpredictor.validation import (
    PurgedRegResult,
    purged_walk_forward,
    walk_forward_regress_purged,
)

try:
    from arch import arch_model

    _HAS_ARCH = True
except Exception:  # pragma: no cover - arch optional at import time
    _HAS_ARCH = False


# --- Range-based daily realized measures -------------------------------------
def garman_klass_variance(ohlcv: pd.DataFrame) -> pd.Series:
    """Garman-Klass daily variance estimate from OHLC (per-day, not annualized).

    σ² = 0.5·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))². Uses the full OHLC bar, so it
    is far more efficient than a close-to-close estimate on daily data.
    """
    o = ohlcv["open"].to_numpy(dtype="float64")
    h = ohlcv["high"].to_numpy(dtype="float64")
    l = ohlcv["low"].to_numpy(dtype="float64")
    c = ohlcv["close"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l)
        co = np.log(c / o)
    var = 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
    var = np.clip(var, 0.0, None)  # tiny negatives from rounding -> 0
    return pd.Series(var, index=ohlcv.index, name="gk_var")


def har_components(ohlcv: pd.DataFrame, *, ann: int) -> pd.DataFrame:
    """Daily / weekly / monthly annualized-vol components for a HAR model.

    Each component is the annualized volatility implied by the average
    Garman-Klass daily *variance* over the trailing window (1 / 5 / 22 days).
    All windows use only past data (no look-ahead).
    """
    gk = garman_klass_variance(ohlcv)
    ann_sqrt = np.sqrt(ann)
    daily = np.sqrt(gk) * ann_sqrt
    weekly = np.sqrt(gk.rolling(5).mean()) * ann_sqrt
    monthly = np.sqrt(gk.rolling(22).mean()) * ann_sqrt
    return pd.DataFrame(
        {"har_daily": daily, "har_weekly": weekly, "har_monthly": monthly}
    )


# --- GARCH family (rolling-origin, honest OOS) -------------------------------
def garch_oos_forecast(
    returns: pd.Series,
    *,
    horizon: int,
    ann: int,
    min_train: int = 500,
    refit_every: int = 10,
    egarch: bool = False,
    dist: str = "t",
) -> pd.Series:
    """Rolling-origin GARCH(1,1)/EGARCH forecast of forward h-day annualized vol.

    At each origin the model is (periodically) refit on an expanding window of
    past returns, then forecasts ``horizon`` steps ahead; the average forecast
    daily variance is converted to an annualized volatility aligned to the same
    target as the other models. Refitting every ``refit_every`` days (carrying
    the forecast forward between refits) keeps it fast while staying leak-free.
    """
    if not _HAS_ARCH:
        raise ImportError("arch is not installed. Run: pip install arch")

    r = returns.dropna() * 100.0  # percent scale for numerical stability
    idx = r.index
    n = len(r)
    out = pd.Series(np.nan, index=idx, dtype="float64")
    vol_spec = "EGARCH" if egarch else "Garch"
    res = None
    last_fit = -(10 ** 9)
    cur = np.nan

    for i in range(min_train, n):
        if res is None or (i - last_fit) >= refit_every:
            am = arch_model(
                r.iloc[: i + 1], mean="Constant", vol=vol_spec, p=1, q=1, dist=dist,
                rescale=False,
            )
            try:
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=horizon, reindex=False)
                daily_var_pct = float(np.mean(fc.variance.values[-1]))
                cur = np.sqrt(daily_var_pct) / 100.0 * np.sqrt(ann)
            except Exception:  # pragma: no cover - a fold that fails to converge
                cur = np.nan
            last_fit = i
        out.iloc[i] = cur
    return out


# --- Metrics on an arbitrary pred/actual pair --------------------------------
def _metrics_on(name: str, pred: pd.Series, actual: pd.Series) -> PurgedRegResult:
    joined = pd.DataFrame({"p": pred, "a": actual}).dropna()
    p, a = joined["p"].to_numpy(), joined["a"].to_numpy()
    err = a - p
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else 0.0
    corr = float(np.corrcoef(p, a)[0, 1]) if len(p) > 1 else float("nan")
    return PurgedRegResult(
        model_name=name,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        r2=r2,
        corr=corr,
        n_test=int(len(p)),
    )


def _test_index_union(n: int, n_splits: int, horizon: int, embargo: int) -> np.ndarray:
    parts = [test for _, test in purged_walk_forward(n, n_splits, horizon, embargo)]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


# --- Latest HAR-RV forecast (for the registry adapter) -----------------------
def har_latest_forecast(ohlcv: pd.DataFrame, *, ann: int, horizon: int) -> tuple:
    """Fit HAR-RV on all known history and forecast the most recent bar.

    Returns ``(as_of_date, predicted_vol, trailing_vol)``. The forecast row may
    have an unknown (future) target — only its past-only HAR features are used.
    """
    from coinpredictor.features import build_default_features

    feats = build_default_features(ohlcv, horizon=horizon, drop_na=False)
    y = feats[MODEL.target_col].astype(float)
    X = har_components(ohlcv, ann=ann).reindex(feats.index)

    x_known = X.notna().all(axis=1)
    train = x_known & y.notna()
    model = LinearRegression().fit(X[train], y[train])

    last_idx = X[x_known].index[-1]
    pred = float(model.predict(X.loc[[last_idx]])[0])
    trailing = float(feats.loc[last_idx, "realized_vol_trailing"])
    return last_idx, max(0.0, pred), trailing


# --- Orchestration -----------------------------------------------------------
def compare_vol_baselines(*, intraday: bool = False, egarch: bool = False) -> pd.DataFrame:
    from coinpredictor.config import INTRADAY
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import build_default_features, build_default_intraday_features

    if intraday:
        from coinpredictor.data.exchange_ohlcv import load_exchange_ohlcv

        ann, horizon, tag = INTRADAY.annualization, INTRADAY.vol_horizon, "intraday"
        ohlcv = load_exchange_ohlcv(timeframe=INTRADAY.interval)  # hourly bars
        feats = build_default_intraday_features(ohlcv, drop_na=True)
    else:
        ann, horizon, tag = MODEL.annualization, MODEL.vol_horizon, "daily"
        ohlcv = load_ohlcv()
        feats = build_default_features(ohlcv, drop_na=True)

    n_splits = MODEL.n_splits
    embargo = max(2, horizon // 2)

    y = feats[MODEL.target_col].astype(float)
    cols = feature_columns(feats)
    X_lgbm = feats[cols]
    ohlcv_v = ohlcv.reindex(feats.index)

    print("=" * 78)
    print(f"PHASE 2 CLASSICAL VOL BASELINES  ({tag}, target={MODEL.target_col}, "
          f"h={horizon}, ann={ann}, n_splits={n_splits})")
    print("=" * 78)
    print(f"Rows: {len(feats)}  test rows: {len(_test_index_union(len(feats), n_splits, horizon, embargo))}\n")

    results: list[PurgedRegResult] = []

    # 1) Naive persistence — trailing realized vol as the forecast.
    test_idx = _test_index_union(len(feats), n_splits, horizon, embargo)
    naive_pred = feats["realized_vol_trailing"].iloc[test_idx]
    results.append(_metrics_on("naive_persistence", naive_pred, y.iloc[test_idx]))

    # 2) HAR-RV (OLS over Garman-Klass daily/weekly/monthly components).
    har_X = har_components(ohlcv_v, ann=ann).reindex(feats.index)
    har_valid = har_X.notna().all(axis=1)
    results.append(
        walk_forward_regress_purged(
            LinearRegression, har_X[har_valid], y[har_valid],
            horizon=horizon, n_splits=n_splits, embargo=embargo, model_name="har_rv",
        )
    )

    # 3) GARCH / EGARCH rolling-origin forecast (daily only; too slow intraday).
    if not intraday and _HAS_ARCH:
        returns = feats["log_return_1d"] if "log_return_1d" in feats else np.log(ohlcv_v["close"]).diff()
        garch_series = garch_oos_forecast(
            returns, horizon=horizon, ann=ann, egarch=egarch,
        ).reindex(feats.index)
        results.append(
            _metrics_on("egarch" if egarch else "garch", garch_series.iloc[test_idx], y.iloc[test_idx])
        )
    elif intraday:
        print("(skipping GARCH for intraday — rolling refit over 70k+ bars is impractical)\n")

    # 4) LightGBM (the incumbent to beat).
    results.append(
        walk_forward_regress_purged(
            build_vol_regressor, X_lgbm, y,
            horizon=horizon, n_splits=n_splits, embargo=embargo, model_name="lgbm_vol",
        )
    )

    # Report, ranked by RMSE (lower is better).
    results.sort(key=lambda r: r.rmse)
    print(f"  {'model':<20} {'rmse':>9} {'mae':>9} {'r2':>8} {'corr':>7}  n")
    for r in results:
        print(f"  {r.model_name:<20} {r.rmse:>9.4f} {r.mae:>9.4f} {r.r2:>8.4f} "
              f"{r.corr:>7.3f}  {r.n_test}")

    best = results[0]
    lgbm = next(r for r in results if r.model_name == "lgbm_vol")
    print(f"\n  -> best by RMSE: {best.model_name}")
    if best.model_name == "lgbm_vol":
        print("  -> LightGBM beats every classical baseline. Its complexity is justified.")
    else:
        print(f"  -> LightGBM does NOT beat {best.model_name}. Prefer the simpler baseline "
              f"(or blend it into the feature set).")

    df = pd.DataFrame([r.__dict__ for r in results]).drop(columns=["fold_r2"], errors="ignore")
    out = PROCESSED_DIR / f"vol_baselines_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("=" * 78)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 classical volatility baselines")
    parser.add_argument("--intraday", action="store_true", help="evaluate on the intraday target")
    parser.add_argument("--egarch", action="store_true", help="use EGARCH instead of GARCH")
    args = parser.parse_args()
    compare_vol_baselines(intraday=args.intraday, egarch=args.egarch)


if __name__ == "__main__":
    main()
