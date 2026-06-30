"""Tests for model training & walk-forward validation."""
from __future__ import annotations

import math

from coinpredictor.features import build_features, split_xy
from coinpredictor.model import (
    build_baseline,
    load_artifact,
    save_artifact,
    train_and_save,
    walk_forward_regress,
)


def test_baseline_regression_walk_forward_runs(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    X, y = split_xy(feats)

    result = walk_forward_regress(build_baseline(), X, y, n_splits=3, model_name="base")

    assert len(result.fold_rmse) == 3
    assert result.rmse >= 0.0


def test_train_and_save_roundtrip(synthetic_ohlcv, tmp_path):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    artifact = train_and_save(feats)

    path = tmp_path / "artifact.pkl"
    save_artifact(artifact, path)
    loaded = load_artifact(path)

    assert loaded.feature_names == artifact.feature_names
    X, _ = split_xy(feats)
    pred = loaded.regressor.predict(X[loaded.feature_names].iloc[[-1]])
    assert pred.shape == (1,)
    assert math.isfinite(pred[0])
