"""OHLCV ingestion for Bitcoin daily candles (Phase 1).

Uses yfinance (free, no API key). Data is cached to ``data/raw`` as parquet so
repeated runs and the dashboard don't re-hit the network unnecessarily.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from coinpredictor.config import DATA, RAW_DIR

_CACHE_FILE = RAW_DIR / "btc_ohlcv.parquet"

# Canonical column names used everywhere downstream.
_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to a clean OHLCV frame indexed by date."""
    # yfinance may return MultiIndex columns when one ticker is requested.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df = df[[c for c in _COLUMNS if c in df.columns]].copy()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    return df


def download_ohlcv(
    ticker: str | None = None,
    start: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Download BTC OHLCV from yfinance and return a normalized frame."""
    ticker = ticker or DATA.btc_ticker
    start = start or DATA.start_date
    interval = interval or DATA.interval

    raw = yf.download(
        ticker,
        start=start,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check connectivity or ticker symbol."
        )
    return _normalize(raw)


def load_ohlcv(refresh: bool = False, cache_path: Path | None = None) -> pd.DataFrame:
    """Return OHLCV data, using the on-disk cache when available.

    Parameters
    ----------
    refresh:
        When True, always re-download and overwrite the cache.
    cache_path:
        Override the default cache location (useful for tests).
    """
    cache_path = cache_path or _CACHE_FILE

    if not refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = download_ohlcv()
    df.to_parquet(cache_path)
    return df


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    data = load_ohlcv(refresh=True)
    print(f"Loaded {len(data)} rows: {data.index.min().date()} -> {data.index.max().date()}")
    print(data.tail())
