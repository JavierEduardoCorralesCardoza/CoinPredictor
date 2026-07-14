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
from typing import Protocol

from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_default_features
from coinpredictor.predict import predict_next_day


class ModelAdapter(Protocol):
    """Common contract every registered model must satisfy."""

    name: str          # unique id, used as the model_name column in the log
    target_type: str   # "volatility" | "direction" (extend as needed)

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


# Register every model you want run + logged each day here. Order doesn't
# matter. To disable a model temporarily, comment out its line -- past rows
# stay in the log untouched.
MODELS: list[ModelAdapter] = [
    LGBMVolatilityAdapter(),
    NaiveVolatilityAdapter(),
]
