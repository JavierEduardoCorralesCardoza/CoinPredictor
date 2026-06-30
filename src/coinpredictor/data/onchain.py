"""Phase 3: on-chain features from the CoinMetrics community API (free, no key).

Series used (configurable in ``config.DATA.onchain_charts``):
* HashRate    — network hash rate (miner commitment / security)
* TxCnt       — daily transaction count (network usage)
* AdrActCnt   — active addresses (network usage / demand)

The ``timeseries/asset-metrics`` endpoint returns JSON
``{"data": [{"time": ..., "HashRate": ..., ...}], "next_page_url": ...}`` and is
paginated. Only backward-looking transforms are exposed to avoid leakage.

blockchain.info was previously used but is now hard-blocked by Cloudflare (403).
"""
from __future__ import annotations

import pandas as pd
import requests

from coinpredictor.config import DATA, RAW_DIR

_CACHE_FILE = RAW_DIR / "onchain.parquet"
_BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
_TIMEOUT = 30
_MAX_PAGES = 50  # safety cap (~50 * 10k rows covers full daily history)


def _fetch_metrics(metrics: list[str]) -> pd.DataFrame:
    """Fetch CoinMetrics daily metrics for BTC, following pagination."""
    params = {
        "assets": "btc",
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "page_size": "10000",
        "start_time": DATA.start_date,
    }
    url: str | None = _BASE_URL
    rows: list[dict] = []
    for _ in range(_MAX_PAGES):
        resp = requests.get(url, params=params if url == _BASE_URL else None, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")
        if not url:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize().dt.tz_localize(None)
    df = df.drop(columns=["asset", "time"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def download_onchain(refresh: bool = False) -> pd.DataFrame:
    """Download configured on-chain metrics into one frame."""
    if not refresh and _CACHE_FILE.exists():
        return pd.read_parquet(_CACHE_FILE)

    metric_to_name = {v: k for k, v in DATA.onchain_charts.items()}
    try:
        raw = _fetch_metrics(list(metric_to_name))
    except requests.RequestException as exc:  # network / API failure
        raise RuntimeError(f"Failed to download on-chain metrics: {exc}") from exc

    if raw.empty:
        raise RuntimeError("Failed to download any on-chain series.")

    # Rename CoinMetrics columns -> friendly names from config.
    df = raw.rename(columns=metric_to_name).sort_index()
    df.to_parquet(_CACHE_FILE)
    return df


def onchain_features(btc_index: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Return on-chain features aligned to ``btc_index`` (past-only)."""
    onchain = download_onchain(refresh=refresh)
    onchain = onchain.reindex(onchain.index.union(btc_index)).ffill().reindex(btc_index)

    out = pd.DataFrame(index=btc_index)
    for col in onchain.columns:
        out[f"{col}_change_1d"] = onchain[col].pct_change()
        out[f"{col}_zscore_30d"] = (
            (onchain[col] - onchain[col].rolling(30).mean())
            / onchain[col].rolling(30).std()
        )
    return out
