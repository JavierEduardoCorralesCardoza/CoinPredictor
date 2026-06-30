"""Live next-day volatility prediction using a trained model artifact."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from coinpredictor.config import MODEL, STRATEGY
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_default_features
from coinpredictor.model import TrainedArtifact, load_artifact
from coinpredictor.backtest import recommend_weight


@dataclass
class Prediction:
    """A single forward volatility prediction."""

    as_of_date: pd.Timestamp     # last known close date
    predicted_vol: float         # forecast forward annualized volatility
    trailing_vol: float          # recent realized annualized volatility
    regime: str                  # "ELEVATED" or "CALM" vs recent norm
    regime_proba: float | None   # classifier probability of elevated regime
    last_close: float
    profile: str                 # risk profile used for the weight
    recommended_weight: float    # suggested BTC exposure [0, 1]

    def summary(self) -> str:
        horizon = MODEL.vol_horizon
        return (
            f"As of {self.as_of_date.date()} (close ${self.last_close:,.2f}): "
            f"forecast {horizon}-day annualized volatility = "
            f"{self.predicted_vol:.1%} "
            f"(recent {self.trailing_vol:.1%}) -> regime {self.regime}\n"
            f"Recommendation [{self.profile}]: hold {self.recommended_weight:.0%} "
            f"in BTC, {1 - self.recommended_weight:.0%} in cash"
        )


def predict_next_day(
    artifact: TrainedArtifact | None = None,
    refresh: bool = False,
    profile: str | None = None,
) -> Prediction:
    """Predict forward volatility from the latest available data.

    The most recent row has fully-formed features but an unknown future, so we
    build features *without* dropping it, then score that final row. ``profile``
    selects the risk profile for the recommended weight (defaults to
    ``STRATEGY.live_profile``).
    """
    artifact = artifact or load_artifact()
    ohlcv = load_ohlcv(refresh=refresh)

    feats = build_default_features(ohlcv, refresh=refresh, drop_na=False)
    X_all = feats[artifact.feature_names]
    valid = X_all.dropna()
    latest = valid.iloc[[-1]]
    as_of = latest.index[-1]

    predicted_vol = float(artifact.regressor.predict(latest)[0])
    trailing_vol = float(feats.loc[as_of, "realized_vol_trailing"])

    regime_proba = None
    if artifact.classifier is not None:
        regime_proba = float(artifact.classifier.predict_proba(latest)[:, 1][0])
        regime = "ELEVATED" if regime_proba >= 0.5 else "CALM"
    else:
        regime = "ELEVATED" if predicted_vol > trailing_vol else "CALM"

    profile = profile or STRATEGY.live_profile
    weight = recommend_weight(predicted_vol, regime_proba, profile=profile)

    return Prediction(
        as_of_date=as_of,
        predicted_vol=predicted_vol,
        trailing_vol=trailing_vol,
        regime=regime,
        regime_proba=regime_proba,
        last_close=float(ohlcv.loc[as_of, "close"]),
        profile=profile,
        recommended_weight=weight,
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    import sys

    from coinpredictor.backtest import STRATEGY_PROFILES

    # Optional first arg selects the risk profile; default shows all three.
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        print(predict_next_day(refresh=True, profile=arg).summary())
    else:
        pred = predict_next_day(refresh=True)
        print(pred.summary())
        print("\nWeight by risk profile:")
        for name in STRATEGY_PROFILES:
            w = predict_next_day(profile=name).recommended_weight
            print(f"  {name:11s}: {w:.0%} BTC / {1 - w:.0%} cash")
