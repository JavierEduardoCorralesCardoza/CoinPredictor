"""Trend-regime family (Phase 1b): predict market direction over the forward
horizon as one of three classes — ``ALCISTA`` (bullish), ``BAJISTA`` (bearish),
or ``LATERAL`` (sideways).

IMPORTANT NAMING: this is a *completely different* concept from the volatility
regime (ELEVATED / CALM) produced by ``model.py``. It lives in its own file
(``trend_regime_log.csv``) with its own columns (``trend_regime_pred`` etc.) and
must never collide with the ``regime_pred`` column used for volatility regime.

Two models ship in this family (project convention: every family needs a
zero-skill/rule baseline alongside the "real" model):
* ``SmaCrossTrendAdapter`` — rule-based, zero training (the required baseline).
* ``LGBMTrendRegimeAdapter`` — a 3-class LightGBM classifier, validated with the
  same walk-forward ``TimeSeriesSplit`` discipline as the volatility models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from coinpredictor.config import MODEL, MODELS_DIR, TREND
from coinpredictor.features import feature_columns

# Spanish labels per the prompt's naming spec. LATERAL is the "no strong move"
# default so a missing/ambiguous signal never masquerades as directional.
LABELS = ("ALCISTA", "BAJISTA", "LATERAL")

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except Exception:  # pragma: no cover - lightgbm optional at import time
    _HAS_LGBM = False


def _horizon_band(feats: pd.DataFrame, horizon: int, band_vol_mult: float) -> pd.Series:
    """LATERAL band half-width as a forward cumulative-return threshold.

    A fixed percentage band would be too wide in calm regimes (labelling almost
    everything LATERAL) and too tight in turbulent ones. Scaling by the trailing
    daily volatility (annualized ``realized_vol_trailing`` de-annualized, then
    grown by sqrt(horizon)) keeps the three classes roughly balanced across
    regimes and ties the label directly to what "a normal move" means today.
    """
    daily_vol = feats["realized_vol_trailing"] / np.sqrt(MODEL.annualization)
    return band_vol_mult * daily_vol * np.sqrt(horizon)


def build_trend_labels(
    feats: pd.DataFrame,
    *,
    horizon: int | None = None,
    band_vol_mult: float | None = None,
) -> pd.Series:
    """Construct the forward trend label for every row (NaN where unknown).

    Uses the SAME rule at train time and eval time so the leaderboard is honest:
    over the next ``horizon`` days, ALCISTA if the cumulative return exceeds the
    volatility-scaled band, BAJISTA if below its negative, LATERAL otherwise.
    """
    horizon = horizon or TREND.horizon
    band_vol_mult = TREND.band_vol_mult if band_vol_mult is None else band_vol_mult

    close = feats["close"]
    fwd_return = close.shift(-horizon) / close - 1.0
    band = _horizon_band(feats, horizon, band_vol_mult)

    labels = pd.Series("LATERAL", index=feats.index, dtype=object)
    labels[fwd_return > band] = "ALCISTA"
    labels[fwd_return < -band] = "BAJISTA"
    # Rows with no resolvable future (or no trailing-vol band) have no label.
    labels[fwd_return.isna() | band.isna()] = np.nan
    labels.iloc[-horizon:] = np.nan
    return labels


def realized_trend_label(
    close: pd.Series,
    feats: pd.DataFrame,
    as_of: str,
    horizon: int,
    band_vol_mult: float | None = None,
) -> str | None:
    """Realized trend label for a single ``as_of`` date, or None if not scoreable.

    Mirrors ``build_trend_labels`` exactly for evaluate_predictions.py.
    """
    band_vol_mult = TREND.band_vol_mult if band_vol_mult is None else band_vol_mult
    as_of_ts = pd.Timestamp(as_of)

    prior = close.index[close.index <= as_of_ts]
    if len(prior) == 0:
        return None
    as_of_ts = prior[-1]

    future = close.loc[close.index > as_of_ts]
    if len(future) < horizon:
        return None  # target_date hasn't fully arrived yet

    entry = float(close.loc[as_of_ts])
    fwd_return = float(future.iloc[horizon - 1]) / entry - 1.0

    if as_of_ts not in feats.index or pd.isna(feats.loc[as_of_ts, "realized_vol_trailing"]):
        return None
    daily_vol = float(feats.loc[as_of_ts, "realized_vol_trailing"]) / np.sqrt(MODEL.annualization)
    band = band_vol_mult * daily_vol * np.sqrt(horizon)

    if fwd_return > band:
        return "ALCISTA"
    if fwd_return < -band:
        return "BAJISTA"
    return "LATERAL"


# --- Artifact ----------------------------------------------------------------
@dataclass
class TrendArtifact:
    classifier: object
    feature_names: list[str]
    classes: list[str]
    accuracy: float = 0.0
    macro_f1: float = 0.0
    fold_accuracy: list[float] = field(default_factory=list)


def build_trend_classifier() -> "LGBMClassifier":
    """3-class LightGBM classifier for the trend regime."""
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=MODEL.random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _walk_forward_trend(clf, X: pd.DataFrame, y: pd.Series) -> tuple[float, float, list[float]]:
    """Expanding-window CV; returns (accuracy, macro_f1, fold_accuracies)."""
    tscv = TimeSeriesSplit(n_splits=MODEL.n_splits)
    preds, actuals, fold_acc = [], [], []
    for train_idx, test_idx in tscv.split(X):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        preds.append(pd.Series(pred, index=y.iloc[test_idx].index))
        actuals.append(y.iloc[test_idx])
        fold_acc.append(float(accuracy_score(y.iloc[test_idx], pred)))
    all_pred = pd.concat(preds)
    all_actual = pd.concat(actuals)
    acc = float(accuracy_score(all_actual, all_pred))
    macro_f1 = float(f1_score(all_actual, all_pred, average="macro", labels=list(LABELS)))
    return acc, macro_f1, fold_acc


def _artifact_path(path: Path | None = None) -> Path:
    return path or (MODELS_DIR / TREND.model_filename)


def train_and_save_trend(feats: pd.DataFrame, path: Path | None = None) -> TrendArtifact:
    """Validate + fit the trend classifier on labelled rows and persist it."""
    labels = build_trend_labels(feats)
    mask = labels.notna()
    cols = feature_columns(feats)
    X = feats.loc[mask, cols]
    y = labels.loc[mask].astype(str)

    clf = build_trend_classifier()
    acc, macro_f1, fold_acc = _walk_forward_trend(clf, X, y)
    clf.fit(X, y)

    artifact = TrendArtifact(
        classifier=clf,
        feature_names=cols,
        classes=list(clf.classes_),
        accuracy=acc,
        macro_f1=macro_f1,
        fold_accuracy=fold_acc,
    )
    joblib.dump(artifact, _artifact_path(path))
    return artifact


def load_or_train_trend(feats: pd.DataFrame, path: Path | None = None) -> TrendArtifact:
    """Load the persisted trend artifact, training one on first use."""
    p = _artifact_path(path)
    if p.exists():
        return joblib.load(p)
    return train_and_save_trend(feats, path=p)


if __name__ == "__main__":  # pragma: no cover - CLI training entrypoint
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import build_default_features

    feats = build_default_features(load_ohlcv(), drop_na=False)
    art = train_and_save_trend(feats)
    dist = build_trend_labels(feats).value_counts(dropna=True).to_dict()
    print(f"Label distribution: {dist}")
    print(
        f"Trend classifier: accuracy={art.accuracy:.3f} macro_f1={art.macro_f1:.3f} "
        f"(folds acc={[round(a, 3) for a in art.fold_accuracy]})"
    )
    print(f"Saved -> {_artifact_path()}")
