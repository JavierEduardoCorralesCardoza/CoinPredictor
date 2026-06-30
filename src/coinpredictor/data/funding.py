"""Phase 4: perpetual-swap funding-rate features from OKX (free, no key).

The funding rate is what longs pay shorts (or vice-versa) on perpetual futures.
It is a direct read on **leverage and positioning**: persistently positive
funding means crowded longs (fragile, prone to long-squeeze volatility spikes);
negative funding means crowded shorts. Funding extremes therefore often precede
volatility.

OKX ``public/funding-rate-history`` returns the realized rate every 8 hours,
max 100 rows per call, paginated backwards via the ``after`` (timestamp) param.
We aggregate to a daily mean. Binance/Bybit are geo-blocked from some regions,
so OKX is used. Only backward-looking transforms are exposed to avoid leakage.
"""
from __future__ import annotations

import pandas as pd
import requests

from coinpredictor.config import RAW_DIR

_CACHE_FILE = RAW_DIR / "funding.parquet"
_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
_INST = "BTC-USDT-SWAP"
_TIMEOUT = 30
_MAX_PAGES = 120  # ~120 * 100 * 8h ≈ 1100 days of history


def download_funding(refresh: bool = False) -> pd.DataFrame:
    """Download OKX BTC perpetual funding history, aggregated to a daily mean."""
    if not refresh and _CACHE_FILE.exists():
        return pd.read_parquet(_CACHE_FILE)

    rows: list[dict] = []
    after: str | None = None
    try:
        for _ in range(_MAX_PAGES):
            params = {"instId": _INST, "limit": "100"}
            if after is not None:
                params["after"] = after
            resp = requests.get(_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                break
            rows.extend(data)
            after = data[-1]["fundingTime"]  # oldest ts in this page
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download funding history: {exc}") from exc

    if not rows:
        raise RuntimeError("OKX returned no funding data.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms").dt.normalize()
    df["rate"] = pd.to_numeric(df["realizedRate"], errors="coerce")
    daily = df.groupby("date")["rate"].mean().to_frame("funding").sort_index()
    daily.to_parquet(_CACHE_FILE)
    return daily


def funding_features(
    btc_index: pd.DatetimeIndex, refresh: bool = False
) -> pd.DataFrame:
    """Return funding-rate features aligned to ``btc_index`` (past-only)."""
    funding = download_funding(refresh=refresh)
    funding = funding.reindex(funding.index.union(btc_index)).ffill().reindex(btc_index)

    f = funding["funding"]
    out = pd.DataFrame(index=btc_index)
    out["funding_rate"] = f
    out["funding_mean_7d"] = f.rolling(7).mean()
    out["funding_abs_7d"] = f.abs().rolling(7).mean()  # crowding intensity
    out["funding_zscore_30d"] = (f - f.rolling(30).mean()) / f.rolling(30).std()
    return out
