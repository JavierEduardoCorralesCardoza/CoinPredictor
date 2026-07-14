#!/usr/bin/env python
"""Fill in real outcomes for past predictions and report model reliability,
broken down PER REGISTERED MODEL so they can be compared fairly.

For every logged prediction whose `target_date` has already arrived, this
computes what BTC's realized volatility actually was over that window (using
the exact same definition as coinpredictor.features: annualized std of daily
log returns over the next `horizon_days`), compares it to what was predicted,
and updates data/processed/prediction_log.csv in place.

Safe to run as often as you like (e.g. daily alongside log_prediction.py) --
rows already marked "evaluated" are left untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.config import MODEL, PROJECT_ROOT
from coinpredictor.data.ohlcv import load_ohlcv

LOG_FILE = PROJECT_ROOT / "data" / "processed" / "prediction_log.csv"
ANN = np.sqrt(MODEL.annualization)


def realized_vol(close: pd.Series, as_of: str, horizon: int) -> float | None:
    """Annualized realized vol over the `horizon` days following `as_of`.

    Mirrors features.build_features(): uses the std of daily log returns
    computed over the as_of close plus the next `horizon` closes. Returns
    None if not enough future data exists yet (target_date hasn't arrived).
    """
    as_of_ts = pd.Timestamp(as_of)
    prior = close.index[close.index <= as_of_ts]
    if len(prior) == 0:
        return None
    as_of_ts = prior[-1]

    future = close.loc[close.index > as_of_ts]
    if len(future) < horizon:
        return None  # target_date hasn't fully arrived yet

    prices = pd.concat([close.loc[[as_of_ts]], future.iloc[:horizon]])
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return float(log_ret.std() * ANN)


def _evaluate_volatility_row(row: pd.Series, close: pd.Series) -> dict | None:
    """Score a single volatility-target row. Returns None if not scoreable yet."""
    horizon = int(row["horizon_days"])
    vol = realized_vol(close, row["as_of_date"], horizon)
    if vol is None:
        return None

    trailing = float(row["trailing_vol"])
    predicted = float(row["predicted_vol"])
    actual_regime = "ELEVATED" if vol > trailing else "CALM"
    regime_correct = actual_regime == row["regime_pred"]

    return {
        "actual_vol": vol,
        "actual_regime": actual_regime,
        "regime_correct": regime_correct,
        "abs_error": abs(vol - predicted),
    }


# Dispatch table: add an entry here when you register a model with a new
# target_type (e.g. "direction"), implementing the equivalent scoring logic.
_EVALUATORS = {
    "volatility": _evaluate_volatility_row,
}


def main() -> None:
    if not LOG_FILE.exists():
        print("No prediction log found yet. Run log_prediction.py first.")
        return

    df = pd.read_csv(LOG_FILE, dtype=str)
    if df.empty:
        print("Log is empty.")
        return

    if "model_name" not in df.columns:
        print(
            "prediction_log.csv doesn't have a model_name column yet. "
            "Run scripts/migrate_add_model_name.py first."
        )
        return

    # dtype=str above keeps as_of_date/target_date/status/model_name as plain
    # text (avoids pandas mis-inferring them), but the columns we're about to
    # WRITE into during evaluation need their real dtype, or pandas refuses
    # the assignment (a str-typed column can't hold a float/bool value).
    for col in ("actual_vol", "abs_error"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["actual_regime"] = df["actual_regime"].astype(object)
    df["regime_correct"] = df["regime_correct"].astype(object)

    ohlcv = load_ohlcv(refresh=True)
    close = ohlcv["close"]

    updated = 0
    for i, row in df.iterrows():
        if row.get("status") == "evaluated":
            continue

        target_type = row.get("target_type", "volatility")
        evaluator = _EVALUATORS.get(target_type)
        if evaluator is None:
            continue  # no scoring logic registered for this target_type yet

        result = evaluator(row, close)
        if result is None:
            continue  # not scoreable yet (target_date hasn't arrived)

        for col, val in result.items():
            df.at[i, col] = val
        df.at[i, "status"] = "evaluated"
        updated += 1

    if updated:
        df.to_csv(LOG_FILE, index=False)
    print(f"Updated {updated} row(s) with realized outcomes.\n")

    evaluated = df[df["status"] == "evaluated"].copy()
    if evaluated.empty:
        print(
            f"No evaluated predictions yet -- wait until {MODEL.vol_horizon} days "
            "after the first logged run."
        )
        return

    evaluated["abs_error"] = evaluated["abs_error"].astype(float)
    evaluated["regime_correct"] = evaluated["regime_correct"].astype(str) == "True"

    print("=== Model reliability by model (so far) ===\n")
    for model_name, grp in evaluated.groupby("model_name"):
        mae = grp["abs_error"].mean()
        rmse = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        acc = grp["regime_correct"].mean()
        print(f"--- {model_name} ({len(grp)} evaluated) ---")
        print(f"  MAE  (volatility) : {mae:.2%}")
        print(f"  RMSE (volatility) : {rmse:.2%}")
        print(f"  Regime accuracy   : {acc:.1%}")

    print("\nLast 10 evaluated rows (all models):")
    cols = [
        "as_of_date", "model_name", "predicted_vol", "actual_vol",
        "regime_pred", "actual_regime", "regime_correct",
    ]
    print(evaluated[cols].sort_values("as_of_date").tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
