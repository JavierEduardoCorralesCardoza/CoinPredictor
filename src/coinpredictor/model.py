"""Model training, walk-forward validation, and persistence.

The primary model is a **regressor** that predicts forward realized volatility
(annualized). A secondary **classifier** predicts the high-volatility *regime*
(will the coming period be more volatile than the recent norm?).

* Regression baseline: scaled Ridge regression.
* Regression primary: LightGBM regressor.
* Regime classifier: LightGBM classifier.

Validation uses ``TimeSeriesSplit`` (walk-forward) so the model is always
trained on the past and evaluated on the future — never the reverse. This is
essential for honest, leak-free volatility forecasts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from coinpredictor.config import MODEL, MODELS_DIR
from coinpredictor.features import feature_columns

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _HAS_LGBM = True
except Exception:  # pragma: no cover - lightgbm optional at import time
    _HAS_LGBM = False


# --- Regression metrics ------------------------------------------------------
@dataclass
class RegCVResult:
    """Aggregated walk-forward metrics for the volatility regression."""

    model_name: str
    rmse: float
    mae: float
    r2: float
    corr: float                            # Pearson corr(pred, actual)
    fold_rmse: list[float] = field(default_factory=list)
    fold_r2: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.model_name}: rmse={self.rmse:.4f} mae={self.mae:.4f} "
            f"r2={self.r2:.4f} corr={self.corr:.4f} "
            f"(folds r2={[round(a, 3) for a in self.fold_r2]})"
        )


# --- Classification metrics (regime) ----------------------------------------
@dataclass
class CVResult:
    """Aggregated walk-forward metrics for the regime classifier."""

    model_name: str
    accuracy: float
    auc: float
    f1: float
    fold_accuracy: list[float] = field(default_factory=list)
    fold_auc: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.model_name}: acc={self.accuracy:.4f} "
            f"auc={self.auc:.4f} f1={self.f1:.4f} "
            f"(folds acc={[round(a, 3) for a in self.fold_accuracy]})"
        )


def build_baseline() -> Pipeline:
    """Scaled Ridge-regression baseline for volatility."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=MODEL.random_state)),
        ]
    )


def build_vol_regressor(params: dict | None = None) -> "TransformedTargetRegressor":
    """Primary LightGBM regressor for forward realized volatility.

    Trained on ``log1p(vol)`` and inverse-transformed on predict. Volatility is
    strongly right-skewed (rare extreme spikes), so fitting in log space curbs
    the influence of outliers, improves R², and guarantees non-negative
    forecasts. Hyperparameters come from ``MODEL.lgbm_params`` unless overridden.
    """
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    lgbm_params = {**MODEL.lgbm_params, **(params or {})}
    base = LGBMRegressor(
        max_depth=-1,
        random_state=MODEL.random_state,
        n_jobs=-1,
        verbose=-1,
        **lgbm_params,
    )
    return TransformedTargetRegressor(
        regressor=base, func=np.log1p, inverse_func=np.expm1
    )


def regressor_importances(model) -> np.ndarray | None:
    """Return feature importances from a (possibly log-wrapped) regressor."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    # TransformedTargetRegressor exposes the fitted inner model as regressor_.
    inner = getattr(model, "regressor_", None) or getattr(model, "regressor", None)
    return getattr(inner, "feature_importances_", None)


def build_regime_classifier() -> "LGBMClassifier":
    """LightGBM classifier for the high-volatility regime label."""
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    return LGBMClassifier(
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


def walk_forward_regress(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    model_name: str = "model",
) -> RegCVResult:
    """Evaluate a regressor with expanding-window time-series CV."""
    n_splits = n_splits or MODEL.n_splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    preds, actuals = [], []
    fold_rmse, fold_r2 = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        preds.append(pd.Series(pred, index=y_te.index))
        actuals.append(y_te)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        fold_r2.append(float(r2_score(y_te, pred)))

    all_pred = pd.concat(preds)
    all_actual = pd.concat(actuals)

    return RegCVResult(
        model_name=model_name,
        rmse=float(np.sqrt(mean_squared_error(all_actual, all_pred))),
        mae=float(mean_absolute_error(all_actual, all_pred)),
        r2=float(r2_score(all_actual, all_pred)),
        corr=float(np.corrcoef(all_pred, all_actual)[0, 1]),
        fold_rmse=fold_rmse,
        fold_r2=fold_r2,
    )


def walk_forward_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    model_name: str = "model",
) -> CVResult:
    """Evaluate a classifier (regime) with expanding-window time-series CV."""
    n_splits = n_splits or MODEL.n_splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, aucs, f1s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        accs.append(accuracy_score(y_te, pred))
        # AUC is undefined if a fold's test set is single-class.
        aucs.append(roc_auc_score(y_te, proba) if y_te.nunique() > 1 else np.nan)
        f1s.append(f1_score(y_te, pred, zero_division=0))

    return CVResult(
        model_name=model_name,
        accuracy=float(np.nanmean(accs)),
        auc=float(np.nanmean(aucs)),
        f1=float(np.nanmean(f1s)),
        fold_accuracy=accs,
        fold_auc=aucs,
    )


def train_final(model, X: pd.DataFrame, y: pd.Series):
    """Fit the model on the full dataset (for live prediction)."""
    model.fit(X, y)
    return model


@dataclass
class TrainedArtifact:
    """Everything needed to reproduce a live volatility prediction."""

    regressor: object
    feature_names: list[str]
    reg_cv: RegCVResult
    classifier: object | None = None
    clf_cv: CVResult | None = None


def save_artifact(artifact: TrainedArtifact, path: Path | None = None) -> Path:
    path = path or (MODELS_DIR / MODEL.model_filename)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path | None = None) -> TrainedArtifact:
    path = path or (MODELS_DIR / MODEL.model_filename)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first (python -m coinpredictor.model)."
        )
    return joblib.load(path)


def train_and_save(features_df: pd.DataFrame) -> TrainedArtifact:
    """End-to-end: validate & fit the volatility regressor + regime classifier."""
    cols = feature_columns(features_df)
    X = features_df[cols]
    y_vol = features_df[MODEL.target_col].astype(float)
    y_regime = features_df[MODEL.regime_col].astype(int)

    # --- Regression (primary) ---
    regressor = build_vol_regressor() if _HAS_LGBM else build_baseline()
    reg_name = "LightGBM-Regressor" if _HAS_LGBM else "Ridge(baseline)"
    reg_cv = walk_forward_regress(regressor, X, y_vol, model_name=reg_name)
    train_final(regressor, X, y_vol)

    # --- Regime classifier (secondary) ---
    classifier = clf_cv = None
    if _HAS_LGBM:
        classifier = build_regime_classifier()
        clf_cv = walk_forward_validate(classifier, X, y_regime, model_name="Regime-LGBM")
        train_final(classifier, X, y_regime)

    artifact = TrainedArtifact(
        regressor=regressor,
        feature_names=cols,
        reg_cv=reg_cv,
        classifier=classifier,
        clf_cv=clf_cv,
    )
    save_artifact(artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover - CLI training entrypoint
    # Import from the package (not __main__) so pickled dataclasses resolve to
    # ``coinpredictor.model.*`` and can be loaded by other entrypoints.
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import build_default_features, build_features
    from coinpredictor.model import (
        build_baseline as _build_baseline,
        build_vol_regressor as _build_vol_regressor,
        train_and_save as _train_and_save,
        walk_forward_regress as _walk_forward_regress,
    )

    ohlcv = load_ohlcv()

    # --- Comparison: Phase-1 features vs full feature set --------------------
    phase1 = build_features(ohlcv)
    Xp = phase1[feature_columns(phase1)]
    yp = phase1[MODEL.target_col].astype(float)
    print("== Phase 1 (technical only) ==")
    print(_walk_forward_regress(_build_baseline(), Xp, yp, model_name="Ridge baseline").summary())
    print(_walk_forward_regress(_build_vol_regressor(), Xp, yp, model_name="LGBM log-vol").summary())

    full = build_default_features(ohlcv)
    Xf = full[feature_columns(full)]
    yf = full[MODEL.target_col].astype(float)
    print(f"\n== Full feature set ({len(feature_columns(full))} features) ==")
    print(_walk_forward_regress(_build_vol_regressor(), Xf, yf, model_name="LGBM log-vol").summary())

    # --- Train & persist on the full feature set ----------------------------
    art = _train_and_save(full)
    print("\n== Saved artifact ==")
    print(art.reg_cv.summary())
    if art.clf_cv is not None:
        print(art.clf_cv.summary())
    print(f"Saved model -> {MODELS_DIR / MODEL.model_filename}")
