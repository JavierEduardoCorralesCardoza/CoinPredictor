"""Tests for the paper-trading broker and live weight recommendation."""
from __future__ import annotations

import math

from coinpredictor.backtest import STRATEGY_PROFILES, recommend_weight
from coinpredictor.trading.paper_broker import PaperBroker


def test_rebalance_reaches_target_weight():
    broker = PaperBroker(cash=10_000.0, fee=0.0)
    broker.rebalance(0.6, price=50_000.0)
    # With no fees, exposure should land exactly on the target.
    assert math.isclose(broker.weight(50_000.0), 0.6, rel_tol=1e-9)


def test_fee_reduces_equity():
    no_fee = PaperBroker(cash=10_000.0, fee=0.0)
    with_fee = PaperBroker(cash=10_000.0, fee=0.01)
    no_fee.rebalance(1.0, price=50_000.0)
    with_fee.rebalance(1.0, price=50_000.0)
    assert with_fee.equity(50_000.0) < no_fee.equity(50_000.0)


def test_buy_and_hold_benchmark_tracks_price():
    broker = PaperBroker(cash=10_000.0, fee=0.0)
    broker.rebalance(0.5, price=50_000.0)  # anchors initial_price at 50k
    # Price doubles -> buy & hold should double the initial capital.
    assert math.isclose(broker.buy_and_hold_equity(100_000.0), 20_000.0, rel_tol=1e-9)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    broker = PaperBroker(cash=10_000.0, fee=0.001)
    broker.rebalance(0.7, price=50_000.0, date="2026-06-30")
    broker.save(path)

    loaded = PaperBroker.load(path)
    assert math.isclose(loaded.cash, broker.cash, rel_tol=1e-12)
    assert math.isclose(loaded.btc, broker.btc, rel_tol=1e-12)
    assert len(loaded.history) == 1


def test_aggressive_profile_is_fully_invested():
    # power=0 -> weight is always the cap regardless of forecast vol.
    w = recommend_weight(predicted_vol=1.5, regime_proba=0.9, profile="aggressive")
    assert math.isclose(w, 1.0, rel_tol=1e-9)


def test_defensive_overlay_cuts_exposure_in_high_regime():
    calm = recommend_weight(0.6, regime_proba=0.0, profile="defensive")
    turbulent = recommend_weight(0.6, regime_proba=1.0, profile="defensive")
    assert turbulent < calm


def test_higher_vol_reduces_balanced_weight():
    low = recommend_weight(0.4, profile="balanced")
    high = recommend_weight(1.2, profile="balanced")
    assert high < low


def test_all_profiles_known():
    for name in STRATEGY_PROFILES:
        w = recommend_weight(0.6, regime_proba=0.5, profile=name)
        assert 0.0 <= w <= 1.0
