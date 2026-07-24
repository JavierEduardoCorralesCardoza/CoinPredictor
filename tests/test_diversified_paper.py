"""Tests for the diversified prospective paper gate (no network; synthetic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coinpredictor.config import DiversifiedConfig
from coinpredictor.trading import diversified_paper as dp


@pytest.fixture
def panel() -> pd.DataFrame:
    """Deterministic 4-asset price panel with a gentle upward drift."""
    n = 90
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(7)
    specs = {"btc": (0.001, 0.03), "equities": (0.0004, 0.01),
             "bonds": (0.0001, 0.005), "gold": (0.0002, 0.007)}
    cols = {a: 100 * np.exp(np.cumsum(rng.normal(mu, sd, n)))
            for a, (mu, sd) in specs.items()}
    df = pd.DataFrame(cols, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def cfg() -> DiversifiedConfig:
    return DiversifiedConfig(
        vol_lookback=5, rebalance_days=1, portfolio_target_vols=(None, 0.1)
    )


def test_paper_book_rebalance_conserves_equity_minus_cost():
    book = dp.PaperBook(cash=1000.0, units={"a": 0.0, "b": 0.0}, initial_capital=1000.0)
    prices = {"a": 10.0, "b": 20.0}
    eq_after = book.rebalance({"a": 0.5, "b": 0.5}, prices, fee=0.001)
    # Cost is 0.1% of the traded notional (1000 total traded here).
    assert eq_after == pytest.approx(1000.0 - 1000.0 * 0.001, rel=1e-9)
    # Target weights are sized on the pre-trade equity (1000), so each leg holds
    # $500 of notional; the fee simply shows up as a small negative cash balance.
    assert book.units["a"] * prices["a"] == pytest.approx(500.0, rel=1e-6)
    assert book.units["b"] * prices["b"] == pytest.approx(500.0, rel=1e-6)


def test_vol_target_leaves_cash_sleeve():
    book = dp.PaperBook(cash=1000.0, units={"a": 0.0}, initial_capital=1000.0)
    # Target only 30% invested -> 70% stays in cash.
    book.rebalance({"a": 0.3}, {"a": 10.0}, fee=0.0)
    assert book.cash == pytest.approx(700.0)
    assert book.units["a"] * 10.0 == pytest.approx(300.0)


def test_run_once_builds_track_record(tmp_path, monkeypatch, panel, cfg):
    monkeypatch.setattr(dp, "STATE_FILE", tmp_path / "state.json")

    # Feed growing slices to simulate one live observation per day.
    for i in range(cfg.vol_lookback + 2, len(panel)):
        rec = dp.run_once(
            cfg, panel=panel.iloc[:i], variant="equal",
            initial_capital=10_000.0, fee=0.0015,
            as_of=str(panel.index[i - 1].date()),
        )
    state = dp.load_state()
    assert state["variant"] == "equal"
    assert len(state["records"]) == len(panel) - (cfg.vol_lookback + 2)
    # Records are unique per date and sorted.
    dates = [r["date"] for r in state["records"]]
    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))
    assert rec["strategy_equity"] > 0


def test_run_once_is_idempotent_per_day(tmp_path, monkeypatch, panel, cfg):
    monkeypatch.setattr(dp, "STATE_FILE", tmp_path / "state.json")
    day = str(panel.index[20].date())
    dp.run_once(cfg, panel=panel.iloc[:21], variant="equal", as_of=day)
    dp.run_once(cfg, panel=panel.iloc[:21], variant="equal", as_of=day)
    state = dp.load_state()
    assert len([r for r in state["records"] if r["date"] == day]) == 1


def test_prospective_gate_pending_then_scored(tmp_path, monkeypatch, panel, cfg):
    monkeypatch.setattr(dp, "STATE_FILE", tmp_path / "state.json")

    # Too few observations -> PENDING.
    dp.run_once(cfg, panel=panel.iloc[:8], variant="equal",
                as_of=str(panel.index[7].date()))
    res = dp.evaluate_prospective_gate(cfg, min_obs=30)
    assert res.verdict == "PENDING" and not res.ready

    # Accumulate enough observations -> a real PASS/FAIL verdict.
    for i in range(9, len(panel)):
        dp.run_once(cfg, panel=panel.iloc[:i], variant="equal",
                    as_of=str(panel.index[i - 1].date()))
    res = dp.evaluate_prospective_gate(cfg, min_obs=10)
    assert res.ready
    assert res.verdict in {"PASS", "FAIL"}
