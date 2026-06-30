"""Tests for feature engineering, focusing on no look-ahead leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.config import MODEL
from coinpredictor.features import build_features, feature_columns, split_xy


def test_target_matches_forward_realized_vol(synthetic_ohlcv):
    """target_vol[t] must equal forward h-day annualized realized vol on day t."""
    feats = build_features(synthetic_ohlcv, drop_na=True)
    h = MODEL.vol_horizon
    ann = np.sqrt(MODEL.annualization)
    log_ret = np.log(synthetic_ohlcv["close"]).diff()

    for ts in feats.index[:50]:
        pos = synthetic_ohlcv.index.get_loc(ts)
        window = log_ret.iloc[pos + 1 : pos + 1 + h]
        if len(window) < h:
            continue
        expected = window.std(ddof=1) * ann
        assert abs(feats.loc[ts, MODEL.target_col] - expected) < 1e-9


def test_no_future_leakage_in_features(synthetic_ohlcv):
    """Changing only *future* closes must not alter today's feature values."""
    feats_a = build_features(synthetic_ohlcv, drop_na=False)

    tampered = synthetic_ohlcv.copy()
    cut = 300
    tampered.iloc[cut + 1 :, :] = tampered.iloc[cut + 1 :, :] * 1.5
    feats_b = build_features(tampered, drop_na=False)

    cols = feature_columns(feats_a)
    early_a = feats_a.iloc[: cut + 1][cols]
    early_b = feats_b.iloc[: cut + 1][cols]

    pd.testing.assert_frame_equal(early_a, early_b, check_exact=False, atol=1e-9)


def test_features_have_no_nan_after_dropna(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    assert not feats.isna().any().any()
    assert len(feats) > 0


def test_regime_target_is_binary(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    assert set(feats[MODEL.regime_col].unique()).issubset({0, 1})


def test_vol_target_is_positive(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    assert (feats[MODEL.target_col] >= 0).all()


def test_split_xy_excludes_targets_and_ohlcv(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    X, y = split_xy(feats)
    for forbidden in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        MODEL.target_col,
        MODEL.regime_col,
    ]:
        assert forbidden not in X.columns
    assert len(X) == len(y)


def test_split_xy_regime_returns_binary(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    _, y = split_xy(feats, regime=True)
    assert set(y.unique()).issubset({0, 1})


def test_rsi_within_bounds(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    rsi = feats["rsi_14"]
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0
