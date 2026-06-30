"""Tests for the volatility-targeting backtesting engine."""
from __future__ import annotations

import numpy as np

from coinpredictor.backtest import backtest_vol_forecast
from coinpredictor.config import STRATEGY
from coinpredictor.features import build_features


def test_weight_at_target_vol_matches_buy_and_hold(synthetic_ohlcv):
    """If forecast vol == target vol, weight==1 and strategy mirrors buy & hold."""
    feats = build_features(synthetic_ohlcv, drop_na=True)
    pred = np.full(len(feats), STRATEGY.target_annual_vol)

    result = backtest_vol_forecast(feats, pred, fee=0.0)

    np.testing.assert_allclose(
        result.equity["strategy"].to_numpy(),
        result.equity["buy_and_hold"].to_numpy(),
        rtol=1e-9,
    )
    np.testing.assert_allclose(result.weights.to_numpy(), 1.0, rtol=1e-9)


def test_higher_forecast_vol_reduces_exposure(synthetic_ohlcv):
    """Doubling forecast vol halves exposure (when within the weight band)."""
    feats = build_features(synthetic_ohlcv, drop_na=True)
    pred = np.full(len(feats), 2 * STRATEGY.target_annual_vol)

    result = backtest_vol_forecast(feats, pred, fee=0.0)

    np.testing.assert_allclose(result.weights.to_numpy(), 0.5, rtol=1e-9)


def test_weight_is_capped(synthetic_ohlcv):
    """Very low forecast vol cannot push exposure above max_weight."""
    feats = build_features(synthetic_ohlcv, drop_na=True)
    pred = np.full(len(feats), 1e-6)  # tiny vol -> huge raw weight

    result = backtest_vol_forecast(feats, pred, fee=0.0)

    assert result.weights.max() <= STRATEGY.max_weight + 1e-12


def test_fees_reduce_returns(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    # Alternating forecast vol forces turnover between two weight levels.
    pred = np.where(
        np.arange(len(feats)) % 2 == 0,
        STRATEGY.target_annual_vol,
        2 * STRATEGY.target_annual_vol,
    )

    no_fee = backtest_vol_forecast(feats, pred, fee=0.0)
    with_fee = backtest_vol_forecast(feats, pred, fee=0.01)

    assert with_fee.strategy_return < no_fee.strategy_return


def test_max_drawdown_non_positive(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    pred = np.full(len(feats), STRATEGY.target_annual_vol)

    result = backtest_vol_forecast(feats, pred, fee=0.0)

    assert result.strategy_max_drawdown <= 0.0
    assert result.bh_max_drawdown <= 0.0
