"""Hyperparameter and horizon tuning for the volatility regressor.

Two independent searches, both honest (walk-forward, no look-ahead):

* ``tune_horizon`` — sweep the forward volatility horizon (e.g. 5/10/21 days).
  Longer horizons average more days and are usually *more* predictable, so this
  often lifts R²/correlation at the cost of slower reaction.
* ``tune_hyperparams`` — randomized search over LightGBM settings, scored by the
  out-of-sample correlation between forecast and realized volatility.

Run ``python -m coinpredictor.tune`` to execute both and print recommended
settings to paste into ``config.ModelConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinpredictor.config import MODEL
from coinpredictor.features import build_default_features, feature_columns


# --- Horizon sweep -----------------------------------------------------------
def tune_horizon(
    ohlcv: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 21, 30),
    n_splits: int | None = None,
):
    """Evaluate the production regressor across forward-volatility horizons."""
    from coinpredictor.model import build_vol_regressor, walk_forward_regress

    results = []
    for h in horizons:
        feats = build_default_features(ohlcv, horizon=h)
        cols = feature_columns(feats)
        X = feats[cols]
        y = feats[MODEL.target_col].astype(float)
        res = walk_forward_regress(
            build_vol_regressor(), X, y, n_splits=n_splits, model_name=f"h={h}"
        )
        results.append((h, res))
    return results


# --- Hyperparameter search ---------------------------------------------------
_PARAM_GRID = {
    "n_estimators": [300, 500, 800, 1200],
    "learning_rate": [0.01, 0.02, 0.05],
    "num_leaves": [15, 31, 63],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
    "min_child_samples": [10, 20, 40],
}


@dataclass
class TuneResult:
    """Outcome of a hyperparameter search."""

    best_params: dict
    best_corr: float
    best_rmse: float
    trials: list  # list of (params, corr, rmse), sorted best-first


def _sample_params(rng: np.random.RandomState) -> dict:
    return {k: rng.choice(v).item() for k, v in _PARAM_GRID.items()}


def tune_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 25,
    n_splits: int | None = None,
    random_state: int | None = None,
) -> TuneResult:
    """Randomized walk-forward search; scored by forecast/realized correlation."""
    from coinpredictor.model import build_vol_regressor, walk_forward_regress

    rng = np.random.RandomState(
        MODEL.random_state if random_state is None else random_state
    )
    seen: set[tuple] = set()
    trials = []
    for _ in range(n_iter):
        params = _sample_params(rng)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        res = walk_forward_regress(
            build_vol_regressor(params), X, y, n_splits=n_splits, model_name="trial"
        )
        trials.append((params, res.corr, res.rmse))

    # Rank by correlation (desc), then RMSE (asc).
    trials.sort(key=lambda t: (-t[1], t[2]))
    best_params, best_corr, best_rmse = trials[0]
    return TuneResult(
        best_params=best_params,
        best_corr=best_corr,
        best_rmse=best_rmse,
        trials=trials,
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.tune import tune_horizon as _tune_horizon
    from coinpredictor.tune import tune_hyperparams as _tune_hyperparams

    ohlcv = load_ohlcv()

    print("== Horizon sweep (production regressor) ==")
    horizon_results = _tune_horizon(ohlcv)
    for h, res in horizon_results:
        print(res.summary())
    best_h = max(horizon_results, key=lambda hr: hr[1].corr)[0]
    print(f"-> best horizon by corr: {best_h}")

    print(f"\n== Hyperparameter search (horizon={best_h}) ==")
    feats = build_default_features(ohlcv, horizon=best_h)
    cols = feature_columns(feats)
    X = feats[cols]
    y = feats[MODEL.target_col].astype(float)
    tuned = _tune_hyperparams(X, y, n_iter=25)
    print(f"Best corr={tuned.best_corr:.4f} rmse={tuned.best_rmse:.4f}")
    print("Top 5 trials:")
    for params, corr, rmse in tuned.trials[:5]:
        print(f"  corr={corr:.4f} rmse={rmse:.4f} {params}")

    print("\n== Recommended config.ModelConfig ==")
    print(f"vol_horizon = {best_h}")
    print(f"lgbm_params = {tuned.best_params}")
