#!/usr/bin/env python
"""One-off migration: split the legacy single ``prediction_log.csv`` into the
new per-family CSV layout (Phase 1a).

The old design kept every model in ONE long-format file discriminated by a
``target_type`` column. Phase 1 gives each family its own file (the filename
now encodes the target_type), so this script:

    1. Copies rows where target_type == "volatility" to volatility_log.csv.
    2. Drops the now-redundant ``target_type`` column (keeps ``model_name``).
    3. Leaves the original prediction_log.csv untouched so it can be verified
       before deletion (the prompt: "do not delete the original until the new
       one is verified").

Safe to run more than once: if volatility_log.csv already exists it no-ops,
mirroring the existing migrate_add_model_name.py convention.
"""
from __future__ import annotations

import pandas as pd

from coinpredictor.config import PROJECT_ROOT, VOLATILITY_LOG

LEGACY_LOG = PROJECT_ROOT / "data" / "processed" / "prediction_log.csv"


def main() -> None:
    if VOLATILITY_LOG.exists():
        print(
            f"Already migrated ({VOLATILITY_LOG.name} present). No changes made."
        )
        return

    if not LEGACY_LOG.exists():
        print(
            "No legacy prediction_log.csv found -- nothing to migrate. New "
            "per-family files will be created on the next log_prediction.py run."
        )
        return

    df = pd.read_csv(LEGACY_LOG, dtype=str)
    if df.empty:
        print("Legacy log is empty -- nothing to migrate.")
        return

    if "model_name" not in df.columns:
        print(
            "Legacy log has no model_name column. Run "
            "scripts/migrate_add_model_name.py first, then re-run this."
        )
        return

    # Only volatility rows existed before Phase 1; be defensive anyway and keep
    # exactly the volatility slice (default to volatility when the column is
    # missing, matching the old evaluate_predictions.py fallback).
    if "target_type" in df.columns:
        vol = df[df["target_type"].fillna("volatility") == "volatility"].copy()
        vol = vol.drop(columns=["target_type"])
    else:
        vol = df.copy()

    VOLATILITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    vol.to_csv(VOLATILITY_LOG, index=False)

    print(
        f"Migrated {len(vol)} volatility row(s) -> {VOLATILITY_LOG.name} "
        f"(dropped redundant target_type column, kept model_name).\n"
        f"Original {LEGACY_LOG.name} left in place for verification -- delete "
        f"it manually once you've confirmed the new file looks correct."
    )


if __name__ == "__main__":
    main()
