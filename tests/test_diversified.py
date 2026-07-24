"""Tests for the cross-asset diversified portfolio (no network; synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coinpredictor.config import DiversifiedConfig
from coinpredictor.diversified import (
    DiversifiedOutcome,
    build_asset_panel,
    equal_weights,
    inverse_vol_weights,
    run_diversified,
    _apply_crypto_cap,
    _select_verdict,
)


@pytest.fixture
def multi_asset_panel() -> pd.DataFrame:
    """Deterministic 4-asset panel with very different volatilities.

    btc is wild (high vol, high drift), equities moderate, bonds and gold calm.
    Inverse-vol weighting should therefore lean heavily on bonds/gold, and the
    crypto cap should bind on btc.
    """
    n = 800
    dates = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(11)
    specs = {
        "btc": (0.0010, 0.040),
        "equities": (0.0004, 0.012),
        "bonds": (0.0001, 0.006),
        "gold": (0.0002, 0.008),
    }
    cols = {}
    for name, (mu, sd) in specs.items():
        rets = rng.normal(mu, sd, n)
        cols[name] = 100 * np.exp(np.cumsum(rets))
    panel = pd.DataFrame(cols, index=dates)
    panel.index.name = "date"
    return panel


def _fake_loader(panel: pd.DataFrame):
    # Map any ticker back to a column by matching the config asset order.
    tickers = {"BTC-USD": "btc", "SPY": "equities", "TLT": "bonds", "GLD": "gold"}

    def loader(ticker: str) -> pd.Series:
        return panel[tickers[ticker]].rename(ticker)

    return loader


def test_apply_crypto_cap_binds_and_normalizes(multi_asset_panel):
    w = pd.DataFrame(
        {"btc": [0.7], "equities": [0.1], "bonds": [0.1], "gold": [0.1]}
    )
    capped = _apply_crypto_cap(w, crypto_key="btc", cap=0.25)
    assert capped["btc"].iloc[0] == pytest.approx(0.25)
    assert capped.sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_inverse_vol_downweights_the_wild_asset(multi_asset_panel):
    cfg = DiversifiedConfig()
    w = inverse_vol_weights(multi_asset_panel, cfg)
    tail = w.iloc[200:].mean()
    # Bonds (calmest) should carry more weight than btc on average.
    assert tail["bonds"] > tail["btc"]
    # Crypto cap must never be exceeded.
    assert w["btc"].max() <= cfg.crypto_cap + 1e-9


def test_equal_weights_respect_crypto_cap(multi_asset_panel):
    cfg = DiversifiedConfig()
    w = equal_weights(multi_asset_panel, cfg)
    assert w["btc"].max() <= cfg.crypto_cap + 1e-9
    assert np.allclose(w.sum(axis=1), 1.0)


def test_build_asset_panel_aligns(multi_asset_panel):
    cfg = DiversifiedConfig()
    panel = build_asset_panel(cfg, loader=_fake_loader(multi_asset_panel))
    assert set(panel.columns) == set(cfg.assets.keys())
    assert panel.notna().all().all()


def test_run_diversified_end_to_end(multi_asset_panel):
    cfg = DiversifiedConfig(portfolio_target_vols=(None, 0.08, 0.10))
    outcome = run_diversified(cfg, loader=_fake_loader(multi_asset_panel), fee=0.0015)
    assert isinstance(outcome, DiversifiedOutcome)
    # 2 schemes x 3 vol modes = 6 variants.
    assert len(outcome.variants) == 6
    assert "btc_hold" in outcome.benchmarks
    assert outcome.verdict in {
        "PASS", "NO-GO", "BETTER-THAN-BTC-BUT-OVER-BUDGET"
    }
    # A diversified variant should have a smaller drawdown than 100% BTC.
    btc_dd = outcome.benchmarks["btc_hold"].max_drawdown
    assert outcome.best.metrics.max_drawdown >= btc_dd


def test_select_verdict_branches():
    p, _ = _select_verdict(
        gate_passed=True, best_name="d", beats_btc=True, drawdown_ok=True
    )
    over, _ = _select_verdict(
        gate_passed=False, best_name="d", beats_btc=True, drawdown_ok=False
    )
    no, _ = _select_verdict(
        gate_passed=False, best_name="d", beats_btc=False, drawdown_ok=True
    )
    assert p == "PASS"
    assert over == "BETTER-THAN-BTC-BUT-OVER-BUDGET"
    assert no == "NO-GO"
