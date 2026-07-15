#!/usr/bin/env python
"""Evaluate logged LLM-Judge verdicts once their horizon has passed.

For each pending verdict whose ``target_date`` has arrived, compute the realized
forward return and the hypothetical P&L had ``suggested_weight`` been followed,
then aggregate per judge: hypothetical total return, hypothetical Sharpe, hit
rate, confidence-vs-hit-rate calibration, and a CONSISTENCY metric (% agreement
across same-day repeated runs, when any exist).

Handles zero evaluated rows gracefully (no crash, no exceptions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from coinpredictor.config import JUDGE, JUDGE_LOG
from coinpredictor.data.ohlcv import load_ohlcv


def _fwd_return(close: pd.Series, as_of: str, horizon: int) -> float | None:
    as_of_ts = pd.Timestamp(as_of)
    prior = close.index[close.index <= as_of_ts]
    if len(prior) == 0:
        return None
    as_of_ts = prior[-1]
    future = close.loc[close.index > as_of_ts]
    if len(future) < horizon:
        return None
    return float(future.iloc[horizon - 1] / close.loc[as_of_ts] - 1.0)


def _consistency(df: pd.DataFrame) -> float | None:
    """% agreement on action across same-day repeated runs (None if no repeats)."""
    shares = []
    for _as_of, grp in df.groupby("as_of_date"):
        if len(grp) < 2:
            continue
        top = grp["action"].value_counts(normalize=True).iloc[0]
        shares.append(float(top))
    return float(np.mean(shares)) if shares else None


def main() -> None:
    if not JUDGE_LOG.exists():
        print("No judge log found yet. Nothing to evaluate.")
        return

    df = pd.read_csv(JUDGE_LOG, dtype=str)
    if df.empty or "model_name" not in df.columns:
        print("Judge log is empty. Nothing to evaluate.")
        return

    for col in ("realized_fwd_return", "hypothetical_pnl"):
        if col in df.columns:
            df[col] = df[col].astype(object)

    close = load_ohlcv(refresh=True)["close"]
    horizon = JUDGE.horizon

    updated = 0
    for i, row in df.iterrows():
        if row.get("status") == "evaluated":
            continue
        fwd = _fwd_return(close, row["as_of_date"], horizon)
        if fwd is None:
            continue  # target_date hasn't fully arrived
        try:
            weight = float(row["suggested_weight"])
        except (TypeError, ValueError):
            weight = 0.0
        df.at[i, "realized_fwd_return"] = fwd
        df.at[i, "hypothetical_pnl"] = weight * fwd
        df.at[i, "status"] = "evaluated"
        updated += 1

    if updated:
        df.to_csv(JUDGE_LOG, index=False)

    evaluated = df[df["status"] == "evaluated"].copy()
    print(f"Scored {updated} newly-resolved verdict(s); "
          f"{len(evaluated)} evaluated total.\n")
    if evaluated.empty:
        print("No evaluated verdicts yet -- waiting for horizon(s) to pass.")
        return

    evaluated["hypothetical_pnl"] = pd.to_numeric(evaluated["hypothetical_pnl"], errors="coerce")
    evaluated["confidence"] = pd.to_numeric(evaluated["confidence"], errors="coerce")

    periods_per_year = 365.0 / horizon
    print("=== Judge decision quality (hypothetical, observation only) ===\n")
    for name, grp in evaluated.groupby("model_name"):
        pnl = grp["hypothetical_pnl"].dropna()
        total = float(np.prod(1.0 + pnl.to_numpy()) - 1.0) if len(pnl) else 0.0
        if len(pnl) > 1 and pnl.std() > 0:
            sharpe = float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))
        else:
            sharpe = 0.0
        hit_rate = float((pnl > 0).mean()) if len(pnl) else float("nan")

        wins = (pnl > 0).astype(int)
        conf = grp.loc[pnl.index, "confidence"]
        if conf.notna().sum() >= 2 and conf.nunique() > 1:
            calib = float(np.corrcoef(conf.fillna(0.5), wins)[0, 1])
        else:
            calib = float("nan")

        consistency = _consistency(grp)
        cons_str = f"{consistency:.0%}" if consistency is not None else "n/a (no repeats)"

        print(f"--- {name} ({len(grp)} evaluated) ---")
        print(f"  Hypothetical total return : {total:.2%}")
        print(f"  Hypothetical Sharpe       : {sharpe:.2f}")
        print(f"  Hit rate                  : {hit_rate:.1%}")
        print(f"  Confidence calibration    : {calib:.2f}")
        print(f"  Consistency (same-day)    : {cons_str}")


if __name__ == "__main__":
    main()
