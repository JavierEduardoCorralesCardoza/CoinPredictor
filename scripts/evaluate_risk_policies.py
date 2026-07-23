#!/usr/bin/env python
"""Evaluate position-sizing risk policies (Phase 1d) and write one row per
policy to data/processed/risk_policy_results.csv.

Unlike the other families this is NOT a daily per-row prediction: a policy is a
rule replayed over history. Each policy is run through walk-forward out-of-sample
volatility/regime forecasts and scored on portfolio-level outcomes (Sharpe,
max drawdown, Calmar, total return). Overwrites the results file each run.

With ``--with-meta`` it *also* refreshes the Phase 3 directional meta-labeling
defensive strategy (vol-scaled barriers) into meta_labeling_results_1d.csv, so a
single command produces both the long-only sizing policies and the directional
drawdown-control strategy for side-by-side comparison in the dashboard.

Zero cost: uses only local OHLCV + the free technical feature set.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from coinpredictor.backtest import RISK_POLICIES, evaluate_risk_policies
from coinpredictor.config import RISK_POLICY_RESULTS
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_default_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate position-sizing risk policies")
    parser.add_argument(
        "--with-meta",
        action="store_true",
        help="also refresh the Phase 3 directional meta-labeling defensive "
        "strategy (vol-scaled barriers) into meta_labeling_results_1d.csv",
    )
    args = parser.parse_args()

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

    if args.with_meta:
        # Directional meta-labeling is a DIFFERENT backtest methodology (purged
        # walk-forward, side + meta-gate) so it keeps its own CSV rather than
        # being merged into the sizing-policy table. Its strength is drawdown
        # control, which is exactly the risk-policy question, so it belongs in
        # the same operator workflow.
        from coinpredictor.meta_labeling import run_meta_labeling

        print("\n" + "=" * 78)
        print("Refreshing directional meta-labeling (defensive, vol-scaled barriers)")
        run_meta_labeling(vol_scaled=True)


if __name__ == "__main__":
    main()
