"""Phase 1 tests: the cost-aware magnitude filter and edge-search plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.config import COSTS
from coinpredictor.edge_search import (
    SignalSet,
    cost_aware_gate,
    expected_net_edge,
    search_edge,
)


# --- Magnitude filter math ---------------------------------------------------
def test_expected_net_edge_matches_formula():
    edge = expected_net_edge(0.6, 0.05, 0.03, round_trip_cost=0.003)
    # 0.6*0.05 - 0.4*0.03 - 0.003 = 0.030 - 0.012 - 0.003 = 0.015
    assert edge == np.float64(0.015)


def test_high_cost_can_flip_edge_negative():
    # Same win-prob/barriers, but a punishing cost makes the trade unprofitable.
    cheap = expected_net_edge(0.55, 0.04, 0.04, round_trip_cost=0.003)
    dear = expected_net_edge(0.55, 0.04, 0.04, round_trip_cost=0.05)
    assert cheap > 0
    assert dear < 0


# --- Gate abstention behaviour ----------------------------------------------
def _idx(n):
    return pd.date_range("2021-01-01", periods=n, freq="D")


def test_cost_gate_abstains_below_margin_and_on_nan():
    idx = _idx(4)
    proba = pd.Series([0.90, 0.50, 0.30, np.nan], index=idx)   # high, mid, low, unknown
    tradable = pd.Series([True, True, True, True], index=idx)
    gate = cost_aware_gate(
        proba, tradable, tp_pct=0.05, sl_pct=0.03,
        round_trip_cost=COSTS.round_trip, margin=0.0,
    )
    # High conviction trades; low conviction and NaN abstain.
    assert gate.iloc[0] == 1.0
    assert gate.iloc[2] == 0.0     # expected edge below break-even
    assert gate.iloc[3] == 0.0     # NaN proba never trades


def test_cost_gate_respects_non_tradable_bars():
    idx = _idx(3)
    proba = pd.Series([0.95, 0.95, 0.95], index=idx)
    tradable = pd.Series([True, False, True], index=idx)
    gate = cost_aware_gate(
        proba, tradable, tp_pct=0.05, sl_pct=0.03,
        round_trip_cost=COSTS.round_trip,
    )
    assert list(gate) == [1.0, 0.0, 1.0]   # flat bar forced to abstain


def test_higher_margin_trades_less_or_equal():
    idx = _idx(50)
    rng = np.random.default_rng(0)
    proba = pd.Series(rng.uniform(0.3, 0.9, size=50), index=idx)
    tradable = pd.Series(True, index=idx)
    loose = cost_aware_gate(proba, tradable, 0.05, 0.03,
                            round_trip_cost=COSTS.round_trip, margin=0.0)
    strict = cost_aware_gate(proba, tradable, 0.05, 0.03,
                             round_trip_cost=COSTS.round_trip, margin=COSTS.round_trip)
    # A stricter margin can only remove trades, never add them.
    assert strict.sum() <= loose.sum()
    assert ((strict == 1.0) <= (loose == 1.0)).all()


# --- End-to-end search on synthetic signals (no network) --------------------
def _trend_ohlcv(n=260, seed=1):
    idx = _idx(n)
    rng = np.random.default_rng(seed)
    # Persistent drift + noise so the trend primary has something to trade.
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]),
         "high": close * 1.01, "low": close * 0.99, "close": close},
        index=idx,
    )


def test_search_edge_runs_and_gates():
    ohlcv = _trend_ohlcv()
    idx = ohlcv.index
    side = pd.Series(1.0, index=idx)              # always-long primary
    tradable = pd.Series(True, index=idx)
    rng = np.random.default_rng(2)
    proba = pd.Series(rng.uniform(0.3, 0.8, size=len(idx)), index=idx)
    signals = SignalSet(
        label="test", ohlcv=ohlcv, side=side, tradable=tradable,
        proba=proba, tp_pct=0.05, sl_pct=0.03, periods_per_year=365,
    )
    res = search_edge(signals)
    # Search tried every threshold + margin, and produced a gate verdict.
    assert res.n_trials == 6
    assert res.best is not None
    assert isinstance(res.gate.passed, bool)
    assert 0.0 <= res.deflated_sharpe <= 1.0
    # The results table includes the two baselines plus every searched variant.
    assert {"buy_and_hold", "primary_only"}.issubset(set(res.table["strategy"]))
    assert len(res.table) == res.n_trials + 2
