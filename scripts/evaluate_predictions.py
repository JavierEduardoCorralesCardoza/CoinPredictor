#!/usr/bin/env python
"""Fill in real outcomes for past predictions and report model reliability,
broken down PER FAMILY and PER MODEL so they can be compared fairly.

Phase 1 change: predictions now live in one csv PER family
(registry.LOG_FILE_BY_TARGET_TYPE). This is still ONE script / ONE cron step;
it just iterates over every family's file, applying that family's evaluator.

Each evaluator only scores rows whose ``target_date`` has already arrived, then
marks them "evaluated". Safe to run as often as you like -- rows already marked
"evaluated" are left untouched, and a file with no resolvable rows yet stays
pending without raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.config import MODEL
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.entry import realized_entry_outcome
from coinpredictor.features import build_default_features
from coinpredictor.registry import LOG_FILE_BY_TARGET_TYPE
from coinpredictor.trend_regime import LABELS as TREND_LABELS, realized_trend_label

ANN = np.sqrt(MODEL.annualization)


def realized_vol(close: pd.Series, as_of: str, horizon: int) -> float | None:
    """Annualized realized vol over the `horizon` days following `as_of`.

    Mirrors features.build_features(): std of daily log returns over the as_of
    close plus the next `horizon` closes. None if not enough future data yet.
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


def _fwd_return(close: pd.Series, as_of: str, horizon: int) -> float | None:
    """Simple cumulative return over the `horizon` days following `as_of`."""
    as_of_ts = pd.Timestamp(as_of)
    prior = close.index[close.index <= as_of_ts]
    if len(prior) == 0:
        return None
    as_of_ts = prior[-1]
    future = close.loc[close.index > as_of_ts]
    if len(future) < horizon:
        return None
    return float(future.iloc[horizon - 1] / close.loc[as_of_ts] - 1.0)


# --- Per-family evaluators (row -> dict of filled columns, or None) ----------
def _evaluate_volatility_row(row: pd.Series, ctx: dict) -> dict | None:
    horizon = int(row["horizon_days"])
    vol = realized_vol(ctx["close"], row["as_of_date"], horizon)
    if vol is None:
        return None
    trailing = float(row["trailing_vol"])
    predicted = float(row["predicted_vol"])
    actual_regime = "ELEVATED" if vol > trailing else "CALM"
    return {
        "actual_vol": vol,
        "actual_regime": actual_regime,
        "regime_correct": actual_regime == row["regime_pred"],
        "abs_error": abs(vol - predicted),
    }


def _evaluate_trend_regime_row(row: pd.Series, ctx: dict) -> dict | None:
    horizon = int(row["horizon_days"])
    actual = realized_trend_label(ctx["close"], ctx["feats"], row["as_of_date"], horizon)
    if actual is None:
        return None
    return {
        "trend_regime_actual": actual,
        "trend_regime_correct": actual == row["trend_regime_pred"],
    }


def _evaluate_entry_row(row: pd.Series, ctx: dict) -> dict | None:
    horizon = int(row["horizon_days"])
    tp_pct = float(row["tp_pct"])
    sl_pct = float(row["sl_pct"])
    actual = realized_entry_outcome(ctx["ohlcv"], row["as_of_date"], horizon, tp_pct, sl_pct)
    if actual is None:
        return None
    predicted_win = float(row["entry_proba"]) >= 0.5
    return {
        "entry_actual": int(actual),
        "entry_correct": bool(predicted_win) == bool(actual),
    }


def _evaluate_sentiment_row(row: pd.Series, ctx: dict) -> dict | None:
    # No clean "actual sentiment" exists; instead attach the realized forward
    # return/vol so the leaderboard can correlate score vs. outcome later.
    horizon = int(row["horizon_days"])
    fwd_ret = _fwd_return(ctx["close"], row["as_of_date"], horizon)
    fwd_vol = realized_vol(ctx["close"], row["as_of_date"], horizon)
    if fwd_ret is None or fwd_vol is None:
        return None
    return {"sentiment_fwd_return": fwd_ret, "sentiment_fwd_vol": fwd_vol}


_EVALUATORS = {
    "volatility": _evaluate_volatility_row,
    "trend_regime": _evaluate_trend_regime_row,
    "entry": _evaluate_entry_row,
    "sentiment": _evaluate_sentiment_row,
}

# Columns each evaluator writes into (coerced to object before assignment so a
# str-typed CSV column can accept float/bool values).
_WRITABLE = {
    "volatility": ["actual_vol", "actual_regime", "regime_correct", "abs_error"],
    "trend_regime": ["trend_regime_actual", "trend_regime_correct"],
    "entry": ["entry_actual", "entry_correct"],
    "sentiment": ["sentiment_fwd_return", "sentiment_fwd_vol"],
}


# --- Per-family reporting ----------------------------------------------------
def _report_volatility(ev: pd.DataFrame) -> None:
    ev = ev.copy()
    ev["abs_error"] = pd.to_numeric(ev["abs_error"], errors="coerce")
    ev["regime_correct"] = ev["regime_correct"].astype(str) == "True"
    for name, grp in ev.groupby("model_name"):
        mae = grp["abs_error"].mean()
        rmse = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        acc = grp["regime_correct"].mean()
        print(f"--- {name} ({len(grp)} evaluated) ---")
        print(f"  MAE  : {mae:.2%}   RMSE: {rmse:.2%}   Vol-regime acc: {acc:.1%}")


def _report_trend_regime(ev: pd.DataFrame) -> None:
    from sklearn.metrics import f1_score

    for name, grp in ev.groupby("model_name"):
        y_true = grp["trend_regime_actual"].astype(str)
        y_pred = grp["trend_regime_pred"].astype(str)
        acc = (y_true == y_pred).mean()
        f1s = f1_score(y_true, y_pred, labels=list(TREND_LABELS), average=None, zero_division=0)
        per_class = ", ".join(f"{lab}={f:.2f}" for lab, f in zip(TREND_LABELS, f1s))
        print(f"--- {name} ({len(grp)} evaluated) ---")
        print(f"  Accuracy: {acc:.1%}   per-class F1: {per_class}")


def _report_entry(ev: pd.DataFrame) -> None:
    for name, grp in ev.groupby("model_name"):
        y_true = pd.to_numeric(grp["entry_actual"], errors="coerce")
        proba = pd.to_numeric(grp["entry_proba"], errors="coerce")
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        win_rate = y_true.mean()
        print(f"--- {name} ({len(grp)} evaluated) ---")
        print(f"  Precision: {precision:.2f}   Recall: {recall:.2f}   "
              f"Base win-rate: {win_rate:.1%}")


def _report_sentiment(ev: pd.DataFrame) -> None:
    for name, grp in ev.groupby("model_name"):
        score = pd.to_numeric(grp["sentiment_score"], errors="coerce")
        fwd_ret = pd.to_numeric(grp["sentiment_fwd_return"], errors="coerce")
        valid = score.notna() & fwd_ret.notna()
        if valid.sum() >= 2 and score[valid].nunique() > 1:
            corr = float(np.corrcoef(score[valid], fwd_ret[valid])[0, 1])
        else:
            corr = float("nan")
        print(f"--- {name} ({len(grp)} evaluated) ---")
        print(f"  corr(sentiment_score, fwd_return): {corr:.3f}")


_REPORTERS = {
    "volatility": _report_volatility,
    "trend_regime": _report_trend_regime,
    "entry": _report_entry,
    "sentiment": _report_sentiment,
}


def _evaluate_file(target_type: str, log_file, ctx: dict) -> None:
    if not log_file.exists():
        return

    df = pd.read_csv(log_file, dtype=str)
    if df.empty or "model_name" not in df.columns:
        return

    evaluator = _EVALUATORS[target_type]
    for col in _WRITABLE[target_type]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    updated = 0
    for i, row in df.iterrows():
        if row.get("status") == "evaluated":
            continue
        result = evaluator(row, ctx)
        if result is None:
            continue  # not scoreable yet
        for col, val in result.items():
            df.at[i, col] = val
        df.at[i, "status"] = "evaluated"
        updated += 1

    if updated:
        df.to_csv(log_file, index=False)

    evaluated = df[df["status"] == "evaluated"]
    print(f"\n=== {target_type} ({log_file.name}): "
          f"{updated} newly scored, {len(evaluated)} evaluated total ===")
    if evaluated.empty:
        print("  No evaluated rows yet -- waiting for target_date(s) to pass.")
        return
    _REPORTERS[target_type](evaluated)


def main() -> None:
    ohlcv = load_ohlcv(refresh=True)
    ctx = {
        "ohlcv": ohlcv,
        "close": ohlcv["close"],
        "feats": build_default_features(ohlcv, drop_na=False),
    }

    any_file = False
    for target_type, log_file in LOG_FILE_BY_TARGET_TYPE.items():
        if log_file.exists():
            any_file = True
        _evaluate_file(target_type, log_file, ctx)

    if not any_file:
        print("No prediction logs found yet. Run log_prediction.py first.")


if __name__ == "__main__":
    main()
