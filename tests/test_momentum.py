"""Tests for cross-sectional momentum (no network; synthetic price panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coinpredictor.config import MomentumConfig
from coinpredictor.momentum import (
    MomentumOutcome,
    backtest_portfolio,
    build_price_panel,
    equal_weight_hold,
    momentum_scores,
    regime_exposure,
    run_momentum,
    target_weights,
    vol_target_exposure,
    _select_verdict,
)


@pytest.fixture
def trend_panel() -> pd.DataFrame:
    """Deterministic panel where assets have monotonically ordered drifts.

    Asset A drifts up fastest, then B, C, D, E flat/down. Cross-sectional
    momentum should therefore consistently rank A over E.
    """
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    drifts = {"A": 0.003, "B": 0.002, "C": 0.001, "D": 0.0, "E": -0.001}
    rng = np.random.default_rng(7)
    cols = {}
    for sym, mu in drifts.items():
        rets = mu + rng.normal(0, 0.005, n)
        cols[sym] = 100 * np.exp(np.cumsum(rets))
    panel = pd.DataFrame(cols, index=dates)
    panel.index.name = "date"
    return panel


def _fake_loader(panel: pd.DataFrame):
    """Return a loader that serves per-symbol OHLCV frames from a close panel."""

    def loader(symbol: str) -> pd.DataFrame:
        close = panel[symbol].dropna()
        return pd.DataFrame({"close": close}, index=close.index)

    return loader


def test_momentum_scores_use_only_past_prices(trend_panel):
    """The score at day t must not depend on any price at or after t."""
    lookback, skip = 30, 5
    scores = momentum_scores(trend_panel, lookback=lookback, skip=skip)

    # Corrupting a future price must not change the score on an earlier day.
    t = 100
    corrupted = trend_panel.copy()
    corrupted.iloc[t + 1:] *= 10.0
    scores2 = momentum_scores(corrupted, lookback=lookback, skip=skip)
    pd.testing.assert_series_equal(scores.iloc[t], scores2.iloc[t])


def test_ranking_selects_top_performers(trend_panel):
    """Top-2 weights should concentrate on the fastest-drifting assets A/B."""
    scores = momentum_scores(trend_panel, lookback=60, skip=5)
    weights = target_weights(scores, trend_panel, top_n=2, min_history=90)

    # Average weight over the valid tail: A and B should dominate D and E.
    tail = weights.iloc[150:]
    mean_w = tail.mean()
    assert mean_w["A"] > mean_w["E"]
    assert mean_w["B"] > mean_w["D"]
    # Each row of weights is either all-zero (no signal yet) or sums to ~1.
    row_sums = weights.sum(axis=1)
    nonzero = row_sums[row_sums > 0]
    assert np.allclose(nonzero, 1.0)


def test_backtest_portfolio_charges_turnover_cost(trend_panel):
    """A non-zero fee must reduce net return versus a zero fee."""
    scores = momentum_scores(trend_panel, lookback=60, skip=5)
    weights = target_weights(scores, trend_panel, top_n=2, min_history=90)

    free = backtest_portfolio(trend_panel, weights, rebalance_days=7, fee=0.0, name="free")
    costed = backtest_portfolio(
        trend_panel, weights, rebalance_days=7, fee=0.01, name="costed"
    )
    assert costed.total_return < free.total_return
    assert costed.avg_turnover >= 0.0


def test_less_frequent_rebalance_has_no_more_turnover(trend_panel):
    """Monthly rotation should not turn over more often than weekly."""
    scores = momentum_scores(trend_panel, lookback=60, skip=5)
    weights = target_weights(scores, trend_panel, top_n=2, min_history=90)

    weekly = backtest_portfolio(trend_panel, weights, rebalance_days=7, fee=0.0, name="w")
    monthly = backtest_portfolio(trend_panel, weights, rebalance_days=30, fee=0.0, name="m")
    # Fewer rebalances -> total cost drag from turnover is not larger.
    weekly_cost = weekly.avg_turnover * (len(trend_panel) / 7)
    monthly_cost = monthly.avg_turnover * (len(trend_panel) / 30)
    assert monthly_cost <= weekly_cost + 1e-9


def test_equal_weight_hold_is_diversified(trend_panel):
    """The benchmark holds all eligible names, so it is between best and worst."""
    bench = equal_weight_hold(trend_panel, fee=0.0, min_history=90)
    assert bench.returns.notna().all()
    assert -1.0 < bench.total_return < 100.0


def test_build_price_panel_aligns_symbols(trend_panel):
    cfg = MomentumConfig(universe=tuple(trend_panel.columns))
    panel = build_price_panel(cfg, loader=_fake_loader(trend_panel))
    assert list(panel.columns) == list(trend_panel.columns)
    assert len(panel) == len(trend_panel)


def test_vol_target_exposure_bounded_and_causal(trend_panel):
    """Exposure stays in [0, 1] (no leverage) and uses only past volatility."""
    rets = trend_panel["A"].pct_change().dropna()
    exp = vol_target_exposure(rets, target_annual_vol=0.30, lookback=30)
    assert (exp >= 0.0).all() and (exp <= 1.0).all()

    # Corrupting a future return must not change an earlier exposure value.
    t = 100
    corrupted = rets.copy()
    corrupted.iloc[t + 1:] *= 5.0
    exp2 = vol_target_exposure(corrupted, target_annual_vol=0.30, lookback=30)
    assert exp.iloc[t] == exp2.iloc[t]


def test_regime_exposure_binary_and_causal(trend_panel):
    """Regime multiplier is 0/1 and never uses future prices."""
    exp = regime_exposure(trend_panel, ma_days=50)
    assert set(np.unique(exp.values)).issubset({0.0, 1.0})

    t = 120
    corrupted = trend_panel.copy()
    corrupted.iloc[t + 1:] *= 0.1
    exp2 = regime_exposure(corrupted, ma_days=50)
    assert exp.iloc[t] == exp2.iloc[t]


def test_run_momentum_end_to_end(trend_panel):
    cfg = MomentumConfig(
        universe=tuple(trend_panel.columns),
        lookbacks=(30, 60),
        top_n=(2,),
        rebalance_days=(7, 30),
        min_history_days=90,
        overlay_target_vols=(0.20, 0.30),
        regime_ma_days=50,
    )
    outcome = run_momentum(cfg, loader=_fake_loader(trend_panel), fee=0.0015)
    assert isinstance(outcome, MomentumOutcome)
    # 4 base x [2 regime x (1 none + 2 vt)] modes = 4 x 6 = 24
    assert len(outcome.variants) == 24
    # The selected best must fit the drawdown budget when any variant does.
    within = [v for v in outcome.variants if v.metrics.max_drawdown >= -0.20]
    if within:
        assert outcome.best.metrics.max_drawdown >= -0.20
    assert outcome.verdict in {"PASS", "NO-GO", "EDGE-BUT-TOO-RISKY"}
    # The gate compared best vs benchmark on a shared index.
    assert 0.0 <= outcome.gate.deflated_sharpe_prob <= 1.0
    assert isinstance(outcome.summary(), str)


def test_select_verdict_branches():
    pass_v, _ = _select_verdict(gate_passed=True, best_name="mom_x")
    nogo_v, _ = _select_verdict(gate_passed=False, best_name="mom_x")
    risky_v, _ = _select_verdict(
        gate_passed=False,
        best_name="mom_x",
        beats_benchmark=True,
        dsr_ok=True,
        drawdown_ok=False,
    )
    assert pass_v == "PASS"
    assert nogo_v == "NO-GO"
    assert risky_v == "EDGE-BUT-TOO-RISKY"
