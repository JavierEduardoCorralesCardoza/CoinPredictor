"""Phase 3: market-sentiment features.

Two sources:
* **Crypto Fear & Greed Index** (alternative.me) — free, no key. Daily 0-100
  index of overall crypto market sentiment.
* **NewsAPI** (optional, needs free key in ``.env``) — counts of recent BTC
  headlines as a crude attention/volume proxy.

Only past-only transforms are exposed to avoid leakage.
"""
from __future__ import annotations

import pandas as pd
import requests

from coinpredictor.config import NEWSAPI_KEY, RAW_DIR

_FNG_CACHE = RAW_DIR / "fear_greed.parquet"
_FNG_URL = "https://api.alternative.me/fng/"
_NEWS_URL = "https://newsapi.org/v2/everything"
_TIMEOUT = 30


def download_fear_greed(refresh: bool = False) -> pd.Series:
    """Download the full Crypto Fear & Greed history as a date-indexed series."""
    if not refresh and _FNG_CACHE.exists():
        return pd.read_parquet(_FNG_CACHE)["fear_greed"]

    resp = requests.get(
        _FNG_URL, params={"limit": 0, "format": "json"}, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])

    s = pd.Series(
        {
            pd.to_datetime(int(d["timestamp"]), unit="s").normalize(): float(d["value"])
            for d in data
        },
        name="fear_greed",
    ).sort_index()
    s.index.name = "date"
    s.to_frame().to_parquet(_FNG_CACHE)
    return s


def _newsapi_daily_counts(days: int = 28) -> pd.Series:
    """Daily BTC headline counts from NewsAPI (last ~month on the free tier)."""
    if not NEWSAPI_KEY:
        return pd.Series(dtype="float64", name="news_count")

    resp = requests.get(
        _NEWS_URL,
        params={
            "q": "bitcoin OR BTC",
            "language": "en",
            "pageSize": 100,
            "sortBy": "publishedAt",
            "apiKey": NEWSAPI_KEY,
        },
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        return pd.Series(dtype="float64", name="news_count")

    articles = resp.json().get("articles", [])
    dates = [pd.to_datetime(a["publishedAt"]).normalize() for a in articles]
    counts = pd.Series(1, index=pd.DatetimeIndex(dates)).groupby(level=0).sum()
    counts.name = "news_count"
    counts.index.name = "date"
    return counts.sort_index()


def sentiment_features(btc_index: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Return sentiment features aligned to ``btc_index`` (past-only).

    Always includes Fear & Greed (free). Includes news counts only when a
    NewsAPI key is configured.
    """
    fng = download_fear_greed(refresh=refresh)
    fng = fng.reindex(fng.index.union(btc_index)).ffill().reindex(btc_index)

    out = pd.DataFrame(index=btc_index)
    out["fear_greed"] = fng
    out["fear_greed_change_1d"] = fng.diff()
    out["fear_greed_sma_7"] = fng.rolling(7).mean()

    news = _newsapi_daily_counts()
    if not news.empty:
        news = news.reindex(btc_index).fillna(0.0)
        out["news_count"] = news
        out["news_count_change_1d"] = news.diff()

    return out
