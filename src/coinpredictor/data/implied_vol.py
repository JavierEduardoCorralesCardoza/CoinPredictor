"""Phase 4: implied volatility features from the Deribit DVOL index (free, no key).

DVOL is Deribit's 30-day forward implied-volatility index for BTC options — the
crypto analogue of the VIX. Implied vol is a *market forecast* of future
volatility, so it is one of the strongest possible predictors of realized vol
(and the implied-minus-realized spread is the variance risk premium).

Endpoint: ``public/get_volatility_index_data`` returns daily OHLC of DVOL as
``[timestamp_ms, open, high, low, close]`` (capped at ~1000 rows per call, which
covers the full history of the index). Only backward-looking transforms are
exposed to avoid leakage.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from coinpredictor.config import RAW_DIR

_CACHE_FILE = RAW_DIR / "implied_vol.parquet"
_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_TIMEOUT = 30


def download_dvol(refresh: bool = False) -> pd.DataFrame:
    """Download the Deribit BTC DVOL index (daily) as a date-indexed frame."""
    if not refresh and _CACHE_FILE.exists():
        return pd.read_parquet(_CACHE_FILE)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 1100 * 24 * 60 * 60 * 1000  # ~1100 days back
    try:
        resp = requests.get(
            _URL,
            params={
                "currency": "BTC",
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": "1D",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("result", {}).get("data", [])
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download DVOL: {exc}") from exc

    if not data:
        raise RuntimeError("Deribit returned no DVOL data.")

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    df = df.drop(columns=["ts"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(_CACHE_FILE)
    return df


def implied_vol_features(
    btc_index: pd.DatetimeIndex, refresh: bool = False
) -> pd.DataFrame:
    """Return DVOL-based implied-volatility features aligned to ``btc_index``.

    DVOL is quoted in annualized percentage points (e.g. 38.0 = 38%); we convert
    to a fraction so it is comparable to the model's realized-vol target/feature.
    """
    dvol = download_dvol(refresh=refresh)
    dvol = dvol.reindex(dvol.index.union(btc_index)).ffill().reindex(btc_index)

    level = dvol["close"] / 100.0  # percentage points -> fraction
    out = pd.DataFrame(index=btc_index)
    out["dvol_level"] = level
    out["dvol_change_1d"] = level.diff()
    out["dvol_change_5d"] = level.diff(5)
    # Intraday swing in implied vol (uncertainty about uncertainty).
    out["dvol_range"] = (dvol["high"] - dvol["low"]) / dvol["close"].replace(0, np.nan)
    # Implied vs trailing realized spread (variance risk premium proxy). The
    # realized side is added in the feature merge; here we expose implied level
    # and its z-score so the model can learn the premium itself.
    out["dvol_zscore_30d"] = (
        (level - level.rolling(30).mean()) / level.rolling(30).std()
    )
    return out
