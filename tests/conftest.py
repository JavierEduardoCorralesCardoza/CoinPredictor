"""Shared pytest fixtures: a deterministic synthetic OHLCV frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """A 400-day reproducible OHLCV frame (no network)."""
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    returns = rng.normal(0.001, 0.03, n)
    close = 10_000 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    volume = rng.uniform(1e8, 5e8, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df
