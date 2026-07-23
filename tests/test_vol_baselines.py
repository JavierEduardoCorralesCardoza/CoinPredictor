"""Tests for the Phase 2 classical volatility baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.vol_baselines import (
    garman_klass_variance,
    har_components,
    har_latest_forecast,
)


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    ret = rng.normal(0.0, 0.02, n)
    close = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    span = np.abs(rng.normal(0.0, 0.01, n))
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1e3, 1e4, n)},
        index=idx,
    )


def test_garman_klass_variance_non_negative():
    ohlcv = _synthetic_ohlcv()
    gk = garman_klass_variance(ohlcv)
    assert (gk.dropna() >= 0).all()
    assert len(gk) == len(ohlcv)


def test_har_components_columns_and_monotone_smoothing():
    ohlcv = _synthetic_ohlcv()
    har = har_components(ohlcv, ann=365)
    assert list(har.columns) == ["har_daily", "har_weekly", "har_monthly"]
    # Longer averaging windows require more warm-up (more leading NaNs).
    assert har["har_daily"].notna().sum() >= har["har_monthly"].notna().sum()


def test_har_latest_forecast_returns_positive_vol():
    ohlcv = _synthetic_ohlcv()
    as_of, pred, trailing = har_latest_forecast(ohlcv, ann=365, horizon=5)
    assert as_of == ohlcv.index[-1]
    assert pred >= 0.0
    assert trailing >= 0.0
