"""OHLCV ingestion for Bitcoin daily candles (Phase 1).
Uses yfinance (free, no API key). Data is cached to ``data/raw`` as parquet so
repeated runs and the dashboard don't re-hit the network unnecessarily.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
import pandas as pd
import yfinance as yf
from coinpredictor.config import DATA, RAW_DIR

_CACHE_FILE = RAW_DIR / "btc_ohlcv.parquet"
# Canonical column names used everywhere downstream.
_COLUMNS = ["open", "high", "low", "close", "volume"]

_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 5  # 5, 10, 20, 40, 80
_MAX_STALE_CACHE_DAYS = 2  # más viejo que esto -> falla en vez de servir datos viejos

log = logging.getLogger(__name__)


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
    """Download BTC OHLCV from yfinance and return a normalized frame.

    Retries with exponential backoff, since Yahoo Finance connections from
    this server are intermittently flaky on large date ranges (measured
    ~25% failure rate via curl timeout; confirmed intermittent, not constant).
    """
    ticker = ticker or DATA.btc_ticker
    start = start or DATA.start_date
    interval = interval or DATA.interval

    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return _normalize(raw)
            last_err = RuntimeError(f"Empty response for {ticker} (attempt {attempt})")
        except Exception as e:  # network timeouts, curl errors, etc.
            last_err = e

        if attempt < _MAX_RETRIES:
            wait = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "download_ohlcv: attempt %d/%d failed (%s), retrying in %ds",
                attempt, _MAX_RETRIES, last_err, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"No data returned for {ticker} after {_MAX_RETRIES} attempts. "
        f"Last error: {last_err}"
    )


def load_ohlcv(refresh: bool = False, cache_path: Path | None = None) -> pd.DataFrame:
    """Return OHLCV data, preferring clean exchange candles.

    When ``DATA.use_exchange_ohlcv`` is set (default), daily candles are pulled
    from a spot exchange via ccxt (clean, reliable, no API key) and only fall
    back to yfinance if the exchange is unreachable. yfinance keeps longer daily
    history but is intermittently flaky, so it is the safety net, not the
    primary source.

    Parameters
    ----------
    refresh:
        When True, always re-download and overwrite the cache. If the
        download fails even after retries, falls back to the existing
        cache -- but only if it isn't older than _MAX_STALE_CACHE_DAYS,
        to avoid silently serving stale data indefinitely.
    cache_path:
        Override the default yfinance cache location (useful for tests).
    """
    if DATA.use_exchange_ohlcv:
        try:
            from coinpredictor.data.exchange_ohlcv import load_exchange_ohlcv

            return load_exchange_ohlcv(timeframe="1d", refresh=refresh)
        except Exception as e:  # exchange unreachable / ccxt missing
            log.warning(
                "load_ohlcv: exchange source failed (%s); falling back to yfinance",
                e,
            )

    return _load_ohlcv_yfinance(refresh=refresh, cache_path=cache_path)


def _load_ohlcv_yfinance(
    refresh: bool = False, cache_path: Path | None = None
) -> pd.DataFrame:
    """yfinance-backed OHLCV with an on-disk cache and stale-cache guard."""
    cache_path = cache_path or _CACHE_FILE

    if not refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        df = download_ohlcv()
        df.to_parquet(cache_path)
        return df
    except Exception as e:
        if not cache_path.exists():
            raise

        cache_age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if cache_age_days > _MAX_STALE_CACHE_DAYS:
            raise RuntimeError(
                f"download_ohlcv failed ({e}) and cache is {cache_age_days:.1f} "
                f"days old (limit: {_MAX_STALE_CACHE_DAYS}). Refusing to serve "
                f"stale data silently -- check network/Yahoo status."
            ) from e

        log.warning(
            "load_ohlcv: fresh download failed (%s); falling back to cache "
            "at %s (%.1f days old, within %d-day limit)",
            e, cache_path, cache_age_days, _MAX_STALE_CACHE_DAYS,
        )
        return pd.read_parquet(cache_path)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    data = load_ohlcv(refresh=True)
    print(f"Loaded {len(data)} rows: {data.index.min().date()} -> {data.index.max().date()}")
    print(data.tail())
