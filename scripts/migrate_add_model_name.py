#!/usr/bin/env python
"""One-off migration: adds model_name/target_type columns to an existing
prediction_log.csv written before the multi-model registry existed.

All pre-existing rows are tagged as "lgbm_volatility_v1" / "volatility",
since that's the only model that was logging before this change. Safe to
run more than once (no-ops if already migrated).
"""
from __future__ import annotations

import pandas as pd

from coinpredictor.config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "data" / "processed" / "prediction_log.csv"


def main() -> None:
    if not LOG_FILE.exists():
        print("No prediction log found -- nothing to migrate.")
        return

    df = pd.read_csv(LOG_FILE, dtype=str)
    if "model_name" in df.columns:
        print("Already migrated (model_name column present). No changes made.")
        return

    df.insert(1, "model_name", "lgbm_volatility_v1")
    df.insert(2, "target_type", "volatility")
    df.to_csv(LOG_FILE, index=False)
    print(f"Migrated {len(df)} row(s) -> tagged as lgbm_volatility_v1 / volatility.")


if __name__ == "__main__":
    main()
