"""Phase 2: macroeconomic features (S&P 500, Gold, US Dollar index).

All series come from yfinance (free, no key). Because equity/commodity markets
close on weekends while BTC trades daily, series are forward-filled onto the
BTC calendar. Only *backward-looking* transforms are produced so they can be
merged into the feature matrix without leakage.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from coinpredictor.config import DATA, RAW_DIR

_CACHE_FILE = RAW_DIR / "macro.parquet"


def download_macro(refresh: bool = False) -> pd.DataFrame:
    """Download macro close prices, one column per series."""
    if not refresh and _CACHE_FILE.exists():
        return pd.read_parquet(_CACHE_FILE)

    frames = []
    for name, ticker in DATA.macro_tickers.items():
        raw = yf.download(
            ticker,
            start=DATA.start_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].rename(name)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        frames.append(s)

    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"
    macro = macro.sort_index()
    macro.to_parquet(_CACHE_FILE)
    return macro


def macro_features(btc_index: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Return macro-derived features aligned to ``btc_index``.

    Produces daily returns and 5-day momentum for each macro series. All values
    are reindexed/forward-filled to the BTC calendar (past-only).
    """
    macro = download_macro(refresh=refresh)
    macro = macro.reindex(macro.index.union(btc_index)).ffill().reindex(btc_index)

    out = pd.DataFrame(index=btc_index)
    for col in macro.columns:
        out[f"{col}_ret_1d"] = macro[col].pct_change()
        out[f"{col}_ret_5d"] = macro[col].pct_change(5)
    return out
