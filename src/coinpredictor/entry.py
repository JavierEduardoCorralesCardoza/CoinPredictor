"""Entry family (Phase 1c): predict the probability that a long entry taken
today would be profitable, using **triple-barrier labelling**.

For each day we place three barriers relative to the entry close: a take-profit
(+``tp_pct``), a stop-loss (-``sl_pct``), and a time barrier (``horizon`` days).
Walking forward with the daily High/Low (finally using the columns ``ohlcv.py``
has always loaded), whichever barrier is touched first decides the label:
TP first => 1 (win), SL first or timeout => 0 (loss/flat). On a same-day
ambiguity (a candle that spans both barriers) we conservatively assume the stop
was hit first — never optimistically credit a win we can't prove.

Ships with the required zero-skill baseline (``RandomEntryAdapter``) plus the
real model (``LGBMEntryAdapter``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from coinpredictor.config import ENTRY, MODEL, MODELS_DIR
from coinpredictor.features import feature_columns

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except Exception:  # pragma: no cover - lightgbm optional at import time
    _HAS_LGBM = False


def resolve_barrier(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    entry: float,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
    n: int,
) -> float | None:
    """Resolve the triple-barrier outcome for the entry at index ``start``.

    Returns 1.0 (TP touched first), 0.0 (SL first or full-horizon timeout), or
    None when the forward window is incomplete (label not yet knowable).
    """
    tp = entry * (1.0 + tp_pct)
    sl = entry * (1.0 - sl_pct)
    last = min(start + horizon, n - 1)
    for j in range(start + 1, last + 1):
        hit_sl = lows[j] <= sl
        hit_tp = highs[j] >= tp
        if hit_sl:  # conservative: stop resolves first on same-day ambiguity
            return 0.0
        if hit_tp:
            return 1.0
    # Never touched a price barrier within what we observed.
    if (n - 1 - start) >= horizon:
        return 0.0  # full horizon elapsed with no TP -> timeout = loss/flat
    return None  # incomplete window -> unknown


def build_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    *,
    horizon: int | None = None,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
) -> pd.Series:
    """Triple-barrier label (1/0/NaN) for every row of an OHLCV frame."""
    horizon = horizon or ENTRY.horizon
    tp_pct = ENTRY.tp_pct if tp_pct is None else tp_pct
    sl_pct = ENTRY.sl_pct if sl_pct is None else sl_pct

    closes = ohlcv["close"].to_numpy(dtype="float64")
    highs = ohlcv["high"].to_numpy(dtype="float64")
    lows = ohlcv["low"].to_numpy(dtype="float64")
    n = len(ohlcv)

    out = np.full(n, np.nan)
    for i in range(n):
        outcome = resolve_barrier(highs, lows, i, closes[i], horizon, tp_pct, sl_pct, n)
        if outcome is not None:
            out[i] = outcome
    return pd.Series(out, index=ohlcv.index, name="entry_label")


def realized_entry_outcome(
    ohlcv: pd.DataFrame,
    as_of: str,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
) -> float | None:
    """Realized 1/0 barrier outcome for a single ``as_of`` date, else None."""
    as_of_ts = pd.Timestamp(as_of)
    idx = ohlcv.index
    prior = idx[idx <= as_of_ts]
    if len(prior) == 0:
        return None
    pos = idx.get_loc(prior[-1])

    closes = ohlcv["close"].to_numpy(dtype="float64")
    highs = ohlcv["high"].to_numpy(dtype="float64")
    lows = ohlcv["low"].to_numpy(dtype="float64")
    return resolve_barrier(
        highs, lows, int(pos), closes[int(pos)], horizon, tp_pct, sl_pct, len(ohlcv)
    )


# --- Artifact ----------------------------------------------------------------
@dataclass
class EntryArtifact:
    classifier: object
    feature_names: list[str]
    horizon: int
    tp_pct: float
    sl_pct: float
    auc: float = 0.0
    fold_auc: list[float] = field(default_factory=list)


def build_entry_classifier() -> "LGBMClassifier":
    """Binary LightGBM classifier over triple-barrier win/loss labels."""
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


def _walk_forward_auc(clf, X: pd.DataFrame, y: pd.Series) -> tuple[float, list[float]]:
    tscv = TimeSeriesSplit(n_splits=MODEL.n_splits)
    fold_auc = []
    for train_idx, test_idx in tscv.split(X):
        y_tr = y.iloc[train_idx]
        if y_tr.nunique() < 2:
            continue  # a single-class training fold can't fit a proba model
        clf.fit(X.iloc[train_idx], y_tr)
        y_te = y.iloc[test_idx]
        if y_te.nunique() < 2:
            continue
        proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        fold_auc.append(float(roc_auc_score(y_te, proba)))
    auc = float(np.mean(fold_auc)) if fold_auc else float("nan")
    return auc, fold_auc


def _artifact_path(path: Path | None = None) -> Path:
    return path or (MODELS_DIR / ENTRY.model_filename)


def train_and_save_entry(
    feats: pd.DataFrame, ohlcv: pd.DataFrame, path: Path | None = None
) -> EntryArtifact:
    """Validate + fit the entry classifier and persist it.

    ``feats`` supplies the model inputs (technical features); ``ohlcv`` supplies
    the High/Low used to build the triple-barrier labels.
    """
    labels = build_triple_barrier_labels(ohlcv)
    labels = labels.reindex(feats.index)
    mask = labels.notna()
    cols = feature_columns(feats)
    # Feature rows can still contain NaNs (warm-up); require both present.
    valid = mask & feats[cols].notna().all(axis=1)
    X = feats.loc[valid, cols]
    y = labels.loc[valid].astype(int)

    clf = build_entry_classifier()
    auc, fold_auc = _walk_forward_auc(clf, X, y)
    clf.fit(X, y)

    artifact = EntryArtifact(
        classifier=clf,
        feature_names=cols,
        horizon=ENTRY.horizon,
        tp_pct=ENTRY.tp_pct,
        sl_pct=ENTRY.sl_pct,
        auc=auc,
        fold_auc=fold_auc,
    )
    joblib.dump(artifact, _artifact_path(path))
    return artifact


def load_or_train_entry(
    feats: pd.DataFrame, ohlcv: pd.DataFrame, path: Path | None = None
) -> EntryArtifact:
    """Load the persisted entry artifact, training one on first use."""
    p = _artifact_path(path)
    if p.exists():
        return joblib.load(p)
    return train_and_save_entry(feats, ohlcv, path=p)


if __name__ == "__main__":  # pragma: no cover - CLI training entrypoint
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import build_default_features

    ohlcv = load_ohlcv()
    feats = build_default_features(ohlcv, drop_na=False)
    art = train_and_save_entry(feats, ohlcv)
    dist = build_triple_barrier_labels(ohlcv).value_counts(dropna=True).to_dict()
    print(
        f"Triple-barrier (tp={art.tp_pct:.0%}/sl={art.sl_pct:.0%}/"
        f"h={art.horizon}) label distribution: {dist}"
    )
    print(f"Entry classifier: AUC={art.auc:.3f} (folds={[round(a, 3) for a in art.fold_auc]})")
    print(f"Saved -> {_artifact_path()}")
