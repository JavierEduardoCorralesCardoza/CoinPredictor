"""Honest out-of-sample validation primitives (Phase 0 diagnostics).

Two problems make naive cross-validation over-optimistic for this project:

1. **Overlapping labels.** ``target_vol`` on day ``t`` is built from returns over
   the next ``horizon`` days, so consecutive labels share information. A plain
   ``TimeSeriesSplit`` lets the last ``horizon`` training rows leak their
   forward window into the test block, inflating scores. We fix this with a
   **purged walk-forward** that drops (purges) the overlapping tail of each
   training fold, plus an optional embargo.

2. **Multiple testing.** Trying many strategy variants and keeping the best
   inflates the winning Sharpe by luck. The **Deflated Sharpe Ratio** (Bailey &
   López de Prado) discounts an observed Sharpe for the number of trials, the
   sample length, and the return distribution's skew/kurtosis.

These are deliberately dependency-light (numpy + scipy-free) so they run
anywhere the rest of the project runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from coinpredictor.config import MODEL


# --- Purged walk-forward -----------------------------------------------------
def purged_walk_forward(
    n_samples: int,
    n_splits: int,
    horizon: int,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for a purged, expanding walk-forward.

    Each test block sits strictly after its training window (never the reverse).
    The last ``horizon + embargo`` training rows before a test block are purged
    because their forward label window overlaps the test period and would leak.

    Parameters
    ----------
    n_samples:
        Number of rows (assumed chronologically ordered).
    n_splits:
        Number of out-of-sample test blocks.
    horizon:
        Label horizon in rows (e.g. ``MODEL.vol_horizon``). This many training
        rows adjacent to the test block are removed.
    embargo:
        Extra rows purged on top of ``horizon`` (buffer for autocorrelation).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    fold = n_samples // (n_splits + 1)
    if fold == 0:
        raise ValueError("Not enough samples for the requested n_splits")

    gap = max(0, horizon) + max(0, embargo)
    for i in range(1, n_splits + 1):
        test_start = i * fold
        test_end = (i + 1) * fold if i < n_splits else n_samples
        train_end = test_start - gap
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


@dataclass
class PurgedRegResult:
    """Out-of-sample regression metrics from a purged walk-forward."""

    model_name: str
    rmse: float
    mae: float
    r2: float
    corr: float
    n_test: int
    fold_r2: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.model_name}: rmse={self.rmse:.4f} mae={self.mae:.4f} "
            f"r2={self.r2:.4f} corr={self.corr:.4f} n={self.n_test} "
            f"(folds r2={[round(a, 3) for a in self.fold_r2]})"
        )


def walk_forward_regress_purged(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int | None = None,
    n_splits: int | None = None,
    embargo: int = 0,
    model_name: str = "model",
) -> PurgedRegResult:
    """Purged, leak-free walk-forward evaluation of a regressor.

    ``model_factory`` returns a fresh unfitted estimator each fold.
    """
    horizon = horizon if horizon is not None else MODEL.vol_horizon
    n_splits = n_splits or MODEL.n_splits

    preds: list[pd.Series] = []
    actuals: list[pd.Series] = []
    fold_r2: list[float] = []
    for train_idx, test_idx in purged_walk_forward(len(X), n_splits, horizon, embargo):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(X.iloc[test_idx]), dtype="float64")
        y_te = y.iloc[test_idx]
        preds.append(pd.Series(pred, index=y_te.index))
        actuals.append(y_te)
        fold_r2.append(float(_r2(y_te.to_numpy(), pred)))

    if not preds:
        raise RuntimeError("Purged walk-forward produced no usable folds.")

    all_pred = pd.concat(preds)
    all_actual = pd.concat(actuals)
    err = all_actual.to_numpy() - all_pred.to_numpy()
    return PurgedRegResult(
        model_name=model_name,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        r2=float(_r2(all_actual.to_numpy(), all_pred.to_numpy())),
        corr=float(np.corrcoef(all_pred, all_actual)[0, 1]),
        n_test=int(len(all_pred)),
        fold_r2=fold_r2,
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# --- Sharpe ratio statistics -------------------------------------------------
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    """P(true SR > benchmark) given a per-period observed Sharpe.

    ``sr`` and ``sr_benchmark`` are per-observation (NOT annualized).
    ``kurtosis`` is the non-excess (raw) kurtosis of returns (normal = 3).
    """
    if n_obs < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    sr_variance: float,
) -> float:
    """Deflated Sharpe Ratio: P(SR is real) after accounting for selection bias.

    Parameters
    ----------
    returns:
        Per-period strategy returns of the *selected* (best) strategy.
    n_trials:
        Number of independent strategy configurations tried (>= 1).
    sr_variance:
        Variance of the per-period Sharpe estimates across the trials. When
        multiple variants were tested, use ``np.var([sr_i], ddof=1)``.
    """
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    n = r.size
    if n < 3:
        return float("nan")
    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma == 0:
        return float("nan")
    sr = mu / sigma
    skew = float(((r - mu) ** 3).mean() / sigma ** 3)
    kurt = float(((r - mu) ** 4).mean() / sigma ** 4)  # raw kurtosis (normal=3)

    n_trials = max(1, int(n_trials))
    if n_trials == 1 or sr_variance <= 0:
        sr0 = 0.0
    else:
        emc = 0.5772156649015329  # Euler-Mascheroni
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(sr_variance) * ((1.0 - emc) * z1 + emc * z2)

    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)


def annualized_to_period_sr(annual_sr: float, periods_per_year: int = 365) -> float:
    """Convert an annualized Sharpe back to per-period units."""
    return annual_sr / math.sqrt(periods_per_year)