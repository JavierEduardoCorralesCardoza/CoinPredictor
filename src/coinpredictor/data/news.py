"""News headline ingestion for sentiment scoring.

Free by default (Section 2 cost discipline): pulls recent BTC headlines from
public RSS feeds (CoinDesk, Cointelegraph) with no API key. When the dedicated
``COINPREDICTOR_PAID_NEWS`` flag is on it ADDITIONALLY queries a paid provider
(CryptoPanic Pro / NewsAPI) — and if the flag is on but the key is missing it
fails loudly rather than silently skipping.

RSS is parsed with the standard library (``xml.etree``) to avoid a new
dependency. HTTP uses the same retry/backoff discipline as ``ohlcv.py`` so a
flaky feed degrades gracefully instead of crashing the daily cron.
"""
from __future__ import annotations

import logging
import time
from xml.etree import ElementTree

import requests

from coinpredictor.config import (
    CRYPTOPANIC_KEY,
    NEWSAPI_KEY,
    PAID_NEWS_ENABLED,
)

log = logging.getLogger(__name__)

# Public RSS feeds — no key required, free tier, zero cost.
_RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

_TIMEOUT = 20
_MAX_RETRIES = 4
_BACKOFF_BASE_SECONDS = 3  # 3, 6, 12 between attempts


class PaidNewsConfigError(RuntimeError):
    """Raised when a paid news path is requested but not properly configured."""


def _http_get(url: str, params: dict | None = None) -> requests.Response | None:
    """GET with exponential backoff; returns None if all attempts fail.

    A single unreachable feed should not crash the pipeline (mirrors the
    per-source degradation in features._append_source), so failures are logged
    and swallowed here rather than raised.
    """
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001 - varied network failures
            last_err = exc
        if attempt < _MAX_RETRIES:
            wait = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "news._http_get: attempt %d/%d for %s failed (%s), retrying in %ds",
                attempt, _MAX_RETRIES, url, last_err, wait,
            )
            time.sleep(wait)
    log.warning("news._http_get: giving up on %s (%s)", url, last_err)
    return None


def _parse_rss_titles(xml_text: str) -> list[str]:
    """Extract <item><title> texts from an RSS document (best-effort)."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    titles = []
    for item in root.iter("item"):
        title = item.findtext("title")
        if title:
            titles.append(title.strip())
    return titles


def fetch_rss_headlines() -> list[str]:
    """Recent BTC-relevant headlines from the free RSS feeds."""
    headlines: list[str] = []
    for feed in _RSS_FEEDS:
        resp = _http_get(feed)
        if resp is not None:
            headlines.extend(_parse_rss_titles(resp.text))
    return headlines


def fetch_paid_headlines() -> list[str]:
    """Additional headlines from a paid provider. Requires a key.

    Raises PaidNewsConfigError if called without a configured key so a flag-on/
    key-missing misconfiguration fails loudly (never silently no-ops).
    """
    if CRYPTOPANIC_KEY:
        resp = _http_get(
            _CRYPTOPANIC_URL,
            params={"auth_token": CRYPTOPANIC_KEY, "currencies": "BTC", "kind": "news"},
        )
        if resp is None:
            return []
        results = resp.json().get("results", [])
        return [r["title"].strip() for r in results if r.get("title")]

    if NEWSAPI_KEY:
        resp = _http_get(
            _NEWSAPI_URL,
            params={
                "q": "bitcoin OR BTC",
                "language": "en",
                "pageSize": 100,
                "sortBy": "publishedAt",
                "apiKey": NEWSAPI_KEY,
            },
        )
        if resp is None:
            return []
        articles = resp.json().get("articles", [])
        return [a["title"].strip() for a in articles if a.get("title")]

    raise PaidNewsConfigError(
        "COINPREDICTOR_PAID_NEWS is ON but no paid news key is set. Provide "
        "CRYPTOPANIC_KEY or NEWSAPI_KEY in .env, or turn the flag off."
    )


def get_headlines(limit: int | None = None) -> list[str]:
    """Return today's BTC headlines (free by default; paid added if flagged).

    De-duplicates while preserving order and caps to ``limit`` if given.
    """
    headlines = fetch_rss_headlines()
    if PAID_NEWS_ENABLED:
        headlines.extend(fetch_paid_headlines())

    seen: set[str] = set()
    unique = []
    for h in headlines:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique[:limit] if limit else unique
