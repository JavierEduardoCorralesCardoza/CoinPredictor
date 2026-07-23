"""Model registry: a common interface so multiple predictors (volatility,
direction, or any other target) can run side by side, get logged uniformly,
and be compared fairly.

To add a new model: implement a class with `name`, `target_type`, and a
`predict()` method returning a dict of prediction fields, then add an
instance to MODELS at the bottom of this file. Nothing else needs to change
-- log_prediction.py and evaluate_predictions.py iterate over MODELS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coinpredictor.config import (
    ENTRY_LOG,
    SENTIMENT,
    SENTIMENT_LOG,
    TREND,
    TREND_REGIME_LOG,
    VOLATILITY_LOG,
)
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_default_features
from coinpredictor.predict import predict_next_day


class ModelAdapter(Protocol):
    """Common contract every registered model must satisfy."""

    name: str          # unique id, used as the model_name column in the log
    target_type: str   # "volatility" | "trend_regime" | "entry" | "sentiment"

    def predict(self, refresh: bool = False) -> dict:
        """Return a dict of prediction fields. Required keys depend on
        target_type (see FIELDS_BY_TARGET_TYPE in log_prediction.py)."""
        ...


@dataclass
class LGBMVolatilityAdapter:
    """Wraps the existing trained LightGBM volatility regressor + regime
    classifier. This is your current production model, untouched."""

    name: str = "lgbm_volatility_v1"
    target_type: str = "volatility"

    def predict(self, refresh: bool = False) -> dict:
        p = predict_next_day(refresh=refresh)
        return {
            "as_of_date": p.as_of_date,
            "last_close": p.last_close,
            "predicted_vol": p.predicted_vol,
            "trailing_vol": p.trailing_vol,
            "regime_pred": p.regime,
            "regime_proba": p.regime_proba,
            "profile": p.profile,
            "recommended_weight": p.recommended_weight,
        }


@dataclass
class NaiveVolatilityAdapter:
    """Zero-training baseline: predicts tomorrow's volatility will equal
    today's trailing realized volatility. Any 'real' model should beat this
    consistently, or it isn't earning its complexity. Standard practice in
    forecasting before trusting a fancier model."""

    name: str = "naive_persistence_v1"
    target_type: str = "volatility"

    def predict(self, refresh: bool = False) -> dict:
        ohlcv = load_ohlcv(refresh=refresh)
        feats = build_default_features(ohlcv, refresh=refresh, drop_na=False)
        valid = feats.dropna(subset=["realized_vol_trailing"])
        as_of = valid.index[-1]
        trailing = float(valid.loc[as_of, "realized_vol_trailing"])
        return {
            "as_of_date": as_of,
            "last_close": float(ohlcv.loc[as_of, "close"]),
            "predicted_vol": trailing,   # the naive forecast IS the trailing vol
            "trailing_vol": trailing,
            "regime_pred": "CALM",       # naive model never calls "elevated"
            "regime_proba": None,
            "profile": "n/a",
            "recommended_weight": None,
        }


@dataclass
class HARVolatilityAdapter:
    """HAR-RV volatility model (Phase 2). Corsi's heterogeneous-autoregressive
    OLS over Garman-Klass daily/weekly/monthly components. In purged
    walk-forward it BEATS the LightGBM regressor on both daily and hourly BTC
    (positive R² where LightGBM is ~0), because a parsimonious range-based
    persistence model captures vol level better than a high-dimensional booster
    fed noisy price technicals."""

    name: str = "har_rv_volatility_v1"
    target_type: str = "volatility"

    def predict(self, refresh: bool = False) -> dict:
        # Single source of truth: the same HAR predictor the bot/dashboard use.
        from coinpredictor.predict import predict_next_day_har

        p = predict_next_day_har(refresh=refresh)
        return {
            "as_of_date": p.as_of_date,
            "last_close": p.last_close,
            "predicted_vol": p.predicted_vol,
            "trailing_vol": p.trailing_vol,
            "regime_pred": p.regime,
            "regime_proba": p.regime_proba,
            "profile": p.profile,
            "recommended_weight": p.recommended_weight,
        }


# --- Trend-regime family (Phase 1b) -----------------------------------------
@dataclass
class SmaCrossTrendAdapter:
    """Rule-based trend baseline (required): ALCISTA if sma_20 > sma_50 and
    rising, BAJISTA if the inverse, LATERAL otherwise. Zero training."""

    name: str = "sma_cross_trend_v1"
    target_type: str = "trend_regime"

    def predict(self, refresh: bool = False) -> dict:
        ohlcv = load_ohlcv(refresh=refresh)
        feats = build_default_features(ohlcv, refresh=refresh, drop_na=False)
        valid = feats.dropna(subset=["sma_20", "sma_50"])
        as_of = valid.index[-1]
        sma20, sma50 = valid["sma_20"], valid["sma_50"]
        rising = sma20.iloc[-1] > sma20.iloc[-2] if len(sma20) > 1 else True

        if sma20.iloc[-1] > sma50.iloc[-1] and rising:
            pred = "ALCISTA"
        elif sma20.iloc[-1] < sma50.iloc[-1] and not rising:
            pred = "BAJISTA"
        else:
            pred = "LATERAL"

        return {
            "as_of_date": as_of,
            "last_close": float(ohlcv.loc[as_of, "close"]),
            "trend_regime_pred": pred,
            "trend_regime_proba": None,
            "horizon_days": TREND.horizon,
        }


@dataclass
class LGBMTrendRegimeAdapter:
    """3-class LightGBM trend classifier (ALCISTA / BAJISTA / LATERAL)."""

    name: str = "lgbm_trend_v1"
    target_type: str = "trend_regime"

    def predict(self, refresh: bool = False) -> dict:
        from coinpredictor.trend_regime import load_or_train_trend

        ohlcv = load_ohlcv(refresh=refresh)
        feats = build_default_features(ohlcv, refresh=refresh, drop_na=False)
        art = load_or_train_trend(feats)
        X_all = feats[art.feature_names]
        latest = X_all.dropna().iloc[[-1]]
        as_of = latest.index[-1]

        proba = art.classifier.predict_proba(latest)[0]
        classes = list(art.classifier.classes_)
        pred = classes[int(proba.argmax())]

        return {
            "as_of_date": as_of,
            "last_close": float(ohlcv.loc[as_of, "close"]),
            "trend_regime_pred": pred,
            "trend_regime_proba": float(proba.max()),
            "horizon_days": TREND.horizon,
        }


# --- Entry family (Phase 1c) -------------------------------------------------
@dataclass
class RandomEntryAdapter:
    """Zero-skill entry baseline (required): flat 0.5 probability. An honest
    coin-flip floor -- any real entry model should be better calibrated."""

    name: str = "baseline_entry_v1"
    target_type: str = "entry"

    def predict(self, refresh: bool = False) -> dict:
        from coinpredictor.config import ENTRY

        ohlcv = load_ohlcv(refresh=refresh)
        as_of = ohlcv.index[-1]
        return {
            "as_of_date": as_of,
            "last_close": float(ohlcv.loc[as_of, "close"]),
            "entry_proba": 0.5,
            "tp_pct": ENTRY.tp_pct,
            "sl_pct": ENTRY.sl_pct,
            "horizon_days": ENTRY.horizon,
        }


@dataclass
class LGBMEntryAdapter:
    """Binary LightGBM classifier over triple-barrier win/loss labels."""

    name: str = "lgbm_entry_v1"
    target_type: str = "entry"

    def predict(self, refresh: bool = False) -> dict:
        from coinpredictor.entry import load_or_train_entry

        ohlcv = load_ohlcv(refresh=refresh)
        feats = build_default_features(ohlcv, refresh=refresh, drop_na=False)
        art = load_or_train_entry(feats, ohlcv)
        X_all = feats[art.feature_names]
        latest = X_all.dropna().iloc[[-1]]
        as_of = latest.index[-1]
        proba = float(art.classifier.predict_proba(latest)[:, 1][0])

        return {
            "as_of_date": as_of,
            "last_close": float(ohlcv.loc[as_of, "close"]),
            "entry_proba": proba,
            "tp_pct": art.tp_pct,
            "sl_pct": art.sl_pct,
            "horizon_days": art.horizon,
        }


# --- Sentiment family (Phase 1e) --------------------------------------------
def _sentiment_row(ohlcv, score: float) -> dict:
    from coinpredictor.sentiment_score import _label_from_score

    as_of = ohlcv.index[-1]
    return {
        "as_of_date": as_of,
        "last_close": float(ohlcv.loc[as_of, "close"]),
        "sentiment_score": float(score),
        "sentiment_label": _label_from_score(float(score)),
        "horizon_days": SENTIMENT.horizon,
    }


@dataclass
class LexiconSentimentAdapter:
    """Tier 1 baseline (required): deterministic keyword-lexicon sentiment.
    Zero download, zero API, always available -- the zero-cost floor."""

    name: str = "lexicon_sentiment_v1"
    target_type: str = "sentiment"

    def predict(self, refresh: bool = False) -> dict:
        from coinpredictor.data.news import get_headlines
        from coinpredictor.sentiment_score import score_lexicon

        ohlcv = load_ohlcv(refresh=refresh)
        headlines = get_headlines(limit=SENTIMENT.max_headlines)
        row = _sentiment_row(ohlcv, score_lexicon(headlines))
        row["n_headlines"] = len(headlines)
        return row


@dataclass
class FinBERTSentimentAdapter:
    """Tier 2 (recommended real model): FinBERT, a financial-text BERT run
    locally on CPU. Zero API cost once the model is cached."""

    name: str = "finbert_sentiment_v1"
    target_type: str = "sentiment"

    def predict(self, refresh: bool = False) -> dict:
        from coinpredictor.data.news import get_headlines
        from coinpredictor.sentiment_score import score_finbert

        ohlcv = load_ohlcv(refresh=refresh)
        headlines = get_headlines(limit=SENTIMENT.max_headlines)
        row = _sentiment_row(ohlcv, score_finbert(headlines))
        row["n_headlines"] = len(headlines)
        return row


@dataclass
class LLMSentimentAdapter:
    """Tier 3 (PAID, flag-gated by COINPREDICTOR_LLM_SENTIMENT): Claude scores
    the day's headlines. NOT registered in MODELS by default -- enabling it is
    an explicit, cost-bearing choice. Fails loudly if the flag is on but the
    key is missing (never silently falls back to a free tier)."""

    name: str = "llm_sentiment_v1"
    target_type: str = "sentiment"

    def predict(self, refresh: bool = False) -> dict:
        from coinpredictor.data.news import get_headlines
        from coinpredictor.sentiment_score import score_llm

        ohlcv = load_ohlcv(refresh=refresh)
        headlines = get_headlines(limit=SENTIMENT.max_headlines)
        row = _sentiment_row(ohlcv, score_llm(headlines))  # raises if flag off
        row["n_headlines"] = len(headlines)
        return row


# Register every model you want run + logged each day here. Order doesn't
# matter. To disable a model temporarily, comment out its line -- past rows
# stay in the log untouched.
#
# NOTE: LLMSentimentAdapter is deliberately NOT listed. It is the paid Tier 3
# sentiment model; leaving it out guarantees zero paid API calls with the
# default flags off. Add it here only after setting COINPREDICTOR_LLM_SENTIMENT.
MODELS: list[ModelAdapter] = [
    LGBMVolatilityAdapter(),
    NaiveVolatilityAdapter(),
    HARVolatilityAdapter(),
    SmaCrossTrendAdapter(),
    LGBMTrendRegimeAdapter(),
    RandomEntryAdapter(),
    LGBMEntryAdapter(),
    LexiconSentimentAdapter(),
    FinBERTSentimentAdapter(),
]


# Route each family's rows to its OWN csv file (Phase 1f). log_prediction.py
# and evaluate_predictions.py look up the file here by target_type.
LOG_FILE_BY_TARGET_TYPE: dict[str, Path] = {
    "volatility": VOLATILITY_LOG,
    "trend_regime": TREND_REGIME_LOG,
    "entry": ENTRY_LOG,
    "sentiment": SENTIMENT_LOG,
}


# The single "primary" model per family. Phase 3's judge reads exactly these
# to assemble its context, so defining it here now avoids a later rework.
PRIMARY_MODEL: dict[str, str] = {
    # HAR-RV beats the LightGBM regressor in purged walk-forward on both daily
    # and hourly BTC (positive R² vs ~0), so it is the primary vol forecaster.
    "volatility": "har_rv_volatility_v1",
    "trend_regime": "lgbm_trend_v1",
    "entry": "lgbm_entry_v1",
    "sentiment": "finbert_sentiment_v1",
}