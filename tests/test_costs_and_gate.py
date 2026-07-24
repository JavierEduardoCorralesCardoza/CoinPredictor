"""Phase 0 tests: realistic cost model and the pre-registered go/no-go gate."""
from __future__ import annotations

import numpy as np
import pytest

from coinpredictor.config import COSTS, GATE, CostConfig, GateConfig
from coinpredictor.validation import GateResult, evaluate_strategy_gate


# --- Cost model --------------------------------------------------------------
def test_cost_components_sum_to_per_side():
    c = CostConfig(taker_fee=0.001, half_spread=0.0003, slippage=0.0002)
    assert c.per_side == pytest.approx(0.0015)
    assert c.round_trip == pytest.approx(0.0030)


def test_default_costs_match_published_stress():
    # Round-trip should sit in the papers' realistic 21-51 bps band, near 31 bps.
    assert 0.0020 <= COSTS.round_trip <= 0.0051
    assert COSTS.per_side > 0.0  # backtests must never run cost-free by default


# --- Gate: passing case ------------------------------------------------------
def _strong_strategy(n=750, seed=0):
    """A clearly-profitable, low-vol series that should clear the gate."""
    rng = np.random.default_rng(seed)
    return 0.0015 + rng.normal(0.0, 0.004, size=n)


def _flat_benchmark(n=750, seed=1):
    """Cost-matched buy & hold with ~zero drift and higher vol (weak Sharpe)."""
    rng = np.random.default_rng(seed)
    return 0.0001 + rng.normal(0.0, 0.02, size=n)


def test_gate_passes_strong_strategy():
    strat = _strong_strategy()
    bench = _flat_benchmark()
    result = evaluate_strategy_gate(
        strat,
        bench,
        n_trials=5,
        sr_variance=0.0,
        strategy_net_return=float(np.prod(1.0 + strat) - 1.0),
        max_drawdown=-0.08,
    )
    assert isinstance(result, GateResult)
    assert result.passed, result.summary()
    assert all(result.checks.values())
    assert result.deflated_sharpe_prob >= GATE.min_deflated_sharpe_prob


# --- Gate: failing cases -----------------------------------------------------
def test_gate_fails_on_low_deflated_sharpe():
    rng = np.random.default_rng(42)
    strat = rng.normal(0.0, 0.02, size=400)  # zero-edge noise
    bench = rng.normal(0.0, 0.02, size=400)
    result = evaluate_strategy_gate(
        strat,
        bench,
        n_trials=200,          # many trials -> high deflation bar
        sr_variance=0.5,
        strategy_net_return=0.01,
        max_drawdown=-0.05,
    )
    assert not result.passed
    assert result.deflated_sharpe_prob < GATE.min_deflated_sharpe_prob


def test_gate_fails_on_excessive_drawdown():
    strat = _strong_strategy()
    bench = _flat_benchmark()
    result = evaluate_strategy_gate(
        strat,
        bench,
        n_trials=5,
        sr_variance=0.0,
        strategy_net_return=float(np.prod(1.0 + strat) - 1.0),
        max_drawdown=-0.35,    # breaches the 20% tolerance
    )
    assert not result.passed
    assert any("drawdown" in r.lower() for r in result.reasons)


def test_gate_fails_when_not_beating_benchmark():
    strat = _strong_strategy(seed=7)
    result = evaluate_strategy_gate(
        strat,
        strat * 1.5,           # benchmark strictly dominates on Sharpe
        n_trials=5,
        sr_variance=0.0,
        strategy_net_return=float(np.prod(1.0 + strat) - 1.0),
        max_drawdown=-0.05,
    )
    assert not result.passed
    assert any("buy & hold" in r.lower() for r in result.reasons)


def test_gate_respects_custom_criteria():
    strat = _strong_strategy()
    bench = _flat_benchmark()
    strict = GateConfig(max_drawdown_limit=0.03)  # tighter than default 0.20
    result = evaluate_strategy_gate(
        strat,
        bench,
        n_trials=5,
        sr_variance=0.0,
        strategy_net_return=float(np.prod(1.0 + strat) - 1.0),
        max_drawdown=-0.05,    # fine under default, breaches the custom 3% cap
        criteria=strict,
    )
    assert not result.passed  # custom, stricter drawdown cap blocks it
    assert any("drawdown" in r.lower() for r in result.reasons)
