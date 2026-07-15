#!/usr/bin/env python
"""Evaluate position-sizing risk policies (Phase 1d) and write one row per
policy to data/processed/risk_policy_results.csv.

Unlike the other families this is NOT a daily per-row prediction: a policy is a
rule replayed over history. Each policy is run through walk-forward out-of-sample
volatility/regime forecasts and scored on portfolio-level outcomes (Sharpe,
max drawdown, Calmar, total return). Overwrites the results file each run.

Zero cost: uses only local OHLCV + the free technical feature set.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from coinpredictor.backtest import RISK_POLICIES, evaluate_risk_policies
from coinpredictor.config import RISK_POLICY_RESULTS
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_default_features


def main() -> None:
    feats = build_default_features(load_ohlcv(refresh=True))
    results = evaluate_risk_policies(feats, RISK_POLICIES)

    evaluated_at = datetime.now().isoformat(timespec="seconds")
    for r in results:
        r["evaluated_at"] = evaluated_at

    df = pd.DataFrame(
        results,
        columns=[
            "evaluated_at",
            "policy",
            "sharpe",
            "max_drawdown",
            "calmar",
            "total_return",
            "n_days",
        ],
    )
    RISK_POLICY_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RISK_POLICY_RESULTS, index=False)

    print(f"Wrote {len(df)} policy result(s) -> {RISK_POLICY_RESULTS.name}\n")
    print(
        df[["policy", "sharpe", "max_drawdown", "calmar", "total_return"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
