"""Tests for the Phase 3 directional meta-labeling module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.meta_labeling import (
    backtest_positions,
    meta_labels,
    primary_trend_side,
)


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # Deterministic uptrend then downtrend so both sides get exercised.
    ramp = np.concatenate([np.linspace(100, 200, n // 2), np.linspace(200, 120, n - n // 2)])
    close = pd.Series(ramp, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
        },
        index=idx,
    )


def test_primary_side_long_only_has_no_shorts():
    ohlcv = _synthetic_ohlcv()
    feats = ohlcv.copy()
    side = primary_trend_side(feats, fast=5, slow=20, allow_short=False)
    non_nan = side.dropna()
    assert set(non_nan.unique()).issubset({0.0, 1.0})
    # The uptrend segment must produce at least some long signals.
    assert (non_nan == 1.0).any()


def test_primary_side_short_enabled_produces_shorts():
    ohlcv = _synthetic_ohlcv()
    side = primary_trend_side(ohlcv, fast=5, slow=20, allow_short=True)
    assert (side.dropna() == -1.0).any()


def test_meta_labels_are_binary_only_where_side_taken():
    ohlcv = _synthetic_ohlcv()
    side = primary_trend_side(ohlcv, fast=5, slow=20, allow_short=False)
    labels = meta_labels(ohlcv, side, horizon=5, tp_pct=0.05, sl_pct=0.05)
    # Flat bars carry no label.
    flat = (side == 0) | side.isna()
    assert labels[flat].isna().all()
    resolved = labels.dropna()
    assert set(resolved.unique()).issubset({0.0, 1.0})


def test_vol_scaled_barriers_produce_binary_labels():
    ohlcv = _synthetic_ohlcv()
    side = primary_trend_side(ohlcv, fast=5, slow=20, allow_short=False)
    vol = pd.Series(0.03, index=ohlcv.index)  # constant per-bar vol fraction
    labels = meta_labels(
        ohlcv, side, horizon=5, vol=vol, tp_mult=1.5, sl_mult=1.0
    )
    resolved = labels.dropna()
    assert set(resolved.unique()).issubset({0.0, 1.0})
    # Bars with non-finite / non-positive vol must stay unlabeled.
    vol_nan = pd.Series(np.nan, index=ohlcv.index)
    labels_nan = meta_labels(ohlcv, side, horizon=5, vol=vol_nan, tp_mult=1.0, sl_mult=1.0)
    assert labels_nan.dropna().empty


def test_backtest_zero_weight_is_flat():
    ohlcv = _synthetic_ohlcv()
    weight = pd.Series(0.0, index=ohlcv.index)
    m = backtest_positions(ohlcv, weight, fee=0.0015, periods_per_year=365, name="flat")
    assert m.exposure == 0.0
    assert m.total_return == 0.0
    assert m.sharpe == 0.0


def test_backtest_buy_and_hold_matches_price_change():
    ohlcv = _synthetic_ohlcv()
    weight = pd.Series(1.0, index=ohlcv.index)
    m = backtest_positions(ohlcv, weight, fee=0.0, periods_per_year=365, name="bh")
    close = ohlcv["close"]
    expected = close.iloc[-1] / close.iloc[0] - 1.0
    # Full-invested, no-fee equity should track close-to-close price change.
    assert m.total_return == np.float64(m.total_return)  # finite
    assert abs(m.total_return - expected) < 0.05
