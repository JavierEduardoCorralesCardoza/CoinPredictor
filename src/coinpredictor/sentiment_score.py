"""Sentiment family (Phase 1e): score a daily aggregate BTC-news sentiment in
``-1..+1`` (plus a categorical label). Three tiers, all free by default except
the third:

* Tier 1 — ``LexiconSentimentAdapter``: a small curated crypto-finance keyword
  lexicon. No download, no API, fully deterministic — the zero-cost floor and
  required baseline.
* Tier 2 — ``FinBERTSentimentAdapter``: HuggingFace ``ProsusAI/finbert``, a
  BERT purpose-built for financial-text sentiment. Runs locally on CPU, zero
  API cost once downloaded. The recommended "real" default.
* Tier 3 — ``LLMSentimentAdapter``: Claude via the Anthropic API. PAID, gated
  behind ``COINPREDICTOR_LLM_SENTIMENT``; fails loudly (never silently falls
  back) if the flag is on but ``ANTHROPIC_API_KEY`` is missing.

There is no clean "actual sentiment" to score against; evaluate_predictions.py
fills forward return/vol later and the leaderboard reports a correlation.
"""
from __future__ import annotations

from coinpredictor.config import ANTHROPIC_API_KEY, LLM_SENTIMENT_ENABLED, SENTIMENT

# --- Tier 1: curated lexicon -------------------------------------------------
# Small, transparent crypto-finance sentiment lexicon. Deterministic and
# dependency-free, so the zero-cost floor always works even offline.
_POSITIVE_WORDS = {
    "surge", "surges", "soar", "soars", "rally", "rallies", "gain", "gains",
    "bullish", "record", "high", "adoption", "breakout", "boom", "jump", "jumps",
    "rise", "rises", "climb", "climbs", "upgrade", "approval", "approved", "inflow",
    "inflows", "buy", "accumulate", "support", "recovery", "rebound", "optimism",
    "milestone", "outperform", "green", "moon",
}
_NEGATIVE_WORDS = {
    "crash", "crashes", "plunge", "plunges", "dump", "dumps", "bearish", "hack",
    "hacked", "ban", "bans", "banned", "selloff", "sell-off", "fear", "fraud",
    "lawsuit", "fall", "falls", "drop", "drops", "slump", "decline", "declines",
    "liquidation", "liquidations", "outflow", "outflows", "collapse", "warning",
    "risk", "scam", "correction", "tumble", "tumbles", "red", "loss", "losses",
}


def _label_from_score(score: float) -> str:
    if score > 0.15:
        return "POSITIVE"
    if score < -0.15:
        return "NEGATIVE"
    return "NEUTRAL"


def score_lexicon(headlines: list[str]) -> float:
    """Average per-headline polarity in ``-1..+1`` using the keyword lexicon."""
    if not headlines:
        return 0.0
    scores = []
    for h in headlines:
        tokens = [t.strip(".,!?:;\"'()").lower() for t in h.split()]
        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        if pos + neg == 0:
            scores.append(0.0)
        else:
            scores.append((pos - neg) / (pos + neg))
    return float(sum(scores) / len(scores))


# --- Tier 2: FinBERT (lazy, cached) -----------------------------------------
_finbert_pipeline = None


def _get_finbert():
    """Lazily build (and memoize) the FinBERT text-classification pipeline.

    Imported lazily so the heavy ``transformers``/``torch`` stack is only loaded
    when Tier 2 is actually used, and the rest of the system stays importable
    without it.
    """
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    from transformers import pipeline  # lazy heavy import

    _finbert_pipeline = pipeline(
        "text-classification",
        model=SENTIMENT.finbert_model,
        top_k=None,
    )
    return _finbert_pipeline


def score_finbert(headlines: list[str]) -> float:
    """Average FinBERT polarity in ``-1..+1`` (positive prob minus negative)."""
    if not headlines:
        return 0.0
    clf = _get_finbert()
    results = clf(headlines)
    per_headline = []
    for scores in results:
        by_label = {s["label"].lower(): s["score"] for s in scores}
        per_headline.append(by_label.get("positive", 0.0) - by_label.get("negative", 0.0))
    return float(sum(per_headline) / len(per_headline))


# --- Tier 3: Claude (paid, flag-gated) --------------------------------------
def score_llm(headlines: list[str]) -> float:
    """Score sentiment with Claude. PAID — only call when the flag is on.

    Raises loudly if the flag is on but the API key is missing (never silently
    falls back to a free tier). All arithmetic stays here in Python; the LLM is
    only asked for a single scalar it can reason about qualitatively.
    """
    if not LLM_SENTIMENT_ENABLED:
        raise RuntimeError(
            "score_llm called while COINPREDICTOR_LLM_SENTIMENT is OFF. This is "
            "a paid path and must never run unless its dedicated flag is on."
        )
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "COINPREDICTOR_LLM_SENTIMENT is ON but ANTHROPIC_API_KEY is missing. "
            "Set the key in .env or turn the flag off — refusing to continue."
        )
    if not headlines:
        return 0.0

    import anthropic  # lazy import — only needed on the paid path

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    joined = "\n".join(f"- {h}" for h in headlines)
    prompt = (
        "You are a financial sentiment analyst. Below are today's Bitcoin news "
        "headlines. Respond with ONLY a single number between -1 and 1 giving the "
        "overall market sentiment (-1 = very bearish, 0 = neutral, 1 = very "
        f"bullish). Do not explain.\n\nHeadlines:\n{joined}"
    )
    msg = client.messages.create(
        model=SENTIMENT.llm_model,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text").strip()
    try:
        return max(-1.0, min(1.0, float(text)))
    except ValueError:
        return 0.0
