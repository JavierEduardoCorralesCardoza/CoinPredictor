"""Exchange-native OHLCV ingestion via ccxt (Phase 1, data-quality upgrade).

Yahoo Finance (``ohlcv.py``) is convenient and has long daily history, but its
BTC-USD series is intermittently flaky (~25% failure rate on large ranges) and
only offers ~730 days of intraday candles. For honest intraday research and a
clean daily cross-check we pull candles straight from a spot exchange.

OKX is the default exchange: Binance/Bybit are geo-blocked from some regions
(see ``funding.py``), while OKX responds reliably and needs no API key for
public candles. Any ccxt exchange id works via ``exchange_id``.

Data is cached per ``(exchange, symbol, timeframe)`` as parquet under
``data/raw`` and refreshed incrementally (only the missing tail is fetched).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from coinpredictor.config import DATA, RAW_DIR

try:
    import ccxt

    _HAS_CCXT = True
except Exception:  # pragma: no cover - ccxt optional at import time
    _HAS_CCXT = False

log = logging.getLogger(__name__)

# Canonical column names used everywhere downstream (match ohlcv.py).
_COLUMNS = ["open", "high", "low", "close", "volume"]

_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 2  # 2, 4, 8, 16, 32
_PAGE_LIMIT = 100          # OKX/most exchanges cap candles per request at 100
_MAX_PAGES = 5000          # hard stop so a bug can't loop forever

# Milliseconds per candle for the timeframes we support.
_TIMEFRAME_MS = {
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def _cache_file(exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "").replace(":", "").lower()
    return RAW_DIR / f"ohlcv_{exchange_id}_{safe_symbol}_{timeframe}.parquet"


def _get_exchange(exchange_id: str):
    if not _HAS_CCXT:
        raise ImportError("ccxt is not installed. Run: pip install ccxt")
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown ccxt exchange id: {exchange_id!r}")
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30_000})
    return exchange


def _normalize(rows: list[list]) -> pd.DataFrame:
    """Turn ccxt OHLCV rows into a clean frame indexed by UTC date."""
    df = pd.DataFrame(rows, columns=["ts", *_COLUMNS])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df.index.name = "date"
    df = df[_COLUMNS].astype("float64")
    df = df.dropna(subset=["close"])
    return df


def download_exchange_ohlcv(
    symbol: str | None = None,
    timeframe: str | None = None,
    start: str | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles from a spot exchange with paginated backfill.

    Parameters
    ----------
    symbol:
        ccxt unified symbol, e.g. ``"BTC/USDT"`` (defaults to config).
    timeframe:
        One of ``_TIMEFRAME_MS`` keys, e.g. ``"1d"`` or ``"1h"``.
    start:
        ISO date string for the earliest candle to backfill from.
    exchange_id:
        ccxt exchange id, e.g. ``"okx"`` (defaults to config).
    """
    symbol = symbol or DATA.exchange_symbol
    timeframe = timeframe or DATA.interval
    start = start or DATA.exchange_start
    exchange_id = exchange_id or DATA.exchange_id

    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; choose from {list(_TIMEFRAME_MS)}"
        )

    exchange = _get_exchange(exchange_id)
    step_ms = _TIMEFRAME_MS[timeframe]
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    now_ms = int(time.time() * 1000)

    rows: list[list] = []
    # Some exchanges return an empty page when ``since`` predates their listing
    # (OKX BTC/USDT starts ~2019) instead of clamping. Skip leading empty pages
    # by jumping forward until we reach the present; _MAX_PAGES caps the walk.
    for _page in range(_MAX_PAGES):
        batch = _fetch_page(exchange, symbol, timeframe, since)
        if not batch:
            next_since = since + _PAGE_LIMIT * step_ms
            if next_since >= now_ms:
                break
            since = next_since
            continue
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + step_ms
        if next_since <= since or last_ts >= now_ms - step_ms:
            break
        since = next_since

    if not rows:
        raise RuntimeError(
            f"No candles returned for {symbol} {timeframe} on {exchange_id}."
        )
    return _normalize(rows)


def _fetch_page(exchange, symbol: str, timeframe: str, since: int) -> list[list]:
    """One page of candles with exponential-backoff retries."""
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=_PAGE_LIMIT
            )
        except Exception as e:  # network / rate-limit / exchange errors
            last_err = e
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                log.warning(
                    "fetch_ohlcv attempt %d/%d failed (%s), retrying in %ds",
                    attempt, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
    raise RuntimeError(f"fetch_ohlcv failed after {_MAX_RETRIES} attempts: {last_err}")


def load_exchange_ohlcv(
    symbol: str | None = None,
    timeframe: str | None = None,
    refresh: bool = False,
    cache_path: Path | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Return exchange OHLCV, using an incremental on-disk parquet cache.

    On a cache hit, only the missing tail (from the last cached candle) is
    re-fetched and merged, so daily/hourly refreshes stay cheap.
    """
    symbol = symbol or DATA.exchange_symbol
    timeframe = timeframe or DATA.interval
    exchange_id = exchange_id or DATA.exchange_id
    cache_path = cache_path or _cache_file(exchange_id, symbol, timeframe)

    cached: pd.DataFrame | None = None
    if not refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)

    # Decide the backfill start: full history, or just the tail after the cache.
    start = DATA.exchange_start
    if cached is not None and not cached.empty:
        # Re-fetch the last cached day to catch late-updated candles, then merge.
        start = (cached.index.max() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        fresh = download_exchange_ohlcv(
            symbol=symbol, timeframe=timeframe, start=start, exchange_id=exchange_id
        )
    except Exception as e:
        if cached is not None and not cached.empty:
            log.warning(
                "load_exchange_ohlcv: refresh failed (%s); serving cache at %s",
                e, cache_path,
            )
            return cached
        raise

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    merged.to_parquet(cache_path)
    return merged


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    for tf in ("1d", "1h"):
        data = load_exchange_ohlcv(timeframe=tf, refresh=True)
        print(
            f"[{tf}] {len(data)} rows: "
            f"{data.index.min()} -> {data.index.max()}"
        )
        print(data.tail(3))
