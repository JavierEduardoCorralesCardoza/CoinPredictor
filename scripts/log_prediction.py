#!/usr/bin/env python
"""Run one live prediction per registered model and append each to the
tracking log. Meant to run once a day (via cron / run_daily.sh) so each
day's forecast -- per model -- gets a permanent record that can later be
compared against what BTC actually did.

To add/remove a model, edit coinpredictor.registry.MODELS -- nothing here
needs to change.

Log file: data/processed/prediction_log.csv (long format: one row per
day PER MODEL, gitignored dir).
"""
from __future__ import annotations
import csv
import sys
from datetime import datetime, timedelta

import pandas as pd

from coinpredictor.config import MODEL, PROJECT_ROOT
from coinpredictor.registry import MODELS

LOG_DIR = PROJECT_ROOT / "data" / "processed"
LOG_FILE = LOG_DIR / "prediction_log.csv"

FIELDS = [
    "run_at",           # wall-clock timestamp when this row was written
    "model_name",        # which registered model produced this row
    "target_type",       # "volatility" | "direction" | ...
    "as_of_date",        # last known BTC data date used for the forecast
    "target_date",        # as_of_date + horizon_days -> when we can score it
    "horizon_days",
    "last_close",
    "predicted_vol",      # forecast forward annualized volatility (volatility models)
    "trailing_vol",       # recent realized annualized volatility (the "norm")
    "regime_pred",         # "ELEVATED" or "CALM"
    "regime_proba",
    "profile",
    "recommended_weight",
    "actual_vol",         # filled in later by evaluate_predictions.py
    "actual_regime",
    "regime_correct",
    "abs_error",
    "status",              # "pending" -> "evaluated"
]


def _already_logged(as_of_iso: str, model_name: str) -> bool:
    if not LOG_FILE.exists():
        return False
    existing = pd.read_csv(LOG_FILE, dtype=str)
    if existing.empty:
        return False
    if "model_name" not in existing.columns:
        # Old single-model log, not yet migrated -- treat as not logged so
        # migrate_add_model_name.py can be run separately. Avoid crashing.
        return False
    mask = (existing["as_of_date"] == as_of_iso) & (existing["model_name"] == model_name)
    return bool(mask.any())


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now().isoformat(timespec="seconds")
    file_exists = LOG_FILE.exists()

    rows_to_write = []
    for adapter in MODELS:
        try:
            out = adapter.predict(refresh=True)
        except Exception as e:  # one bad model shouldn't kill the others
            print(f"[{run_at}] {adapter.name}: FAILED to predict ({e})")
            continue

        as_of = pd.Timestamp(out["as_of_date"]).date()
        as_of_iso = as_of.isoformat()

        if _already_logged(as_of_iso, adapter.name):
            print(f"[{run_at}] {adapter.name}: prediction for {as_of} already logged, skipping.")
            continue

        target_date = as_of + timedelta(days=MODEL.vol_horizon)
        row = {
            "run_at": run_at,
            "model_name": adapter.name,
            "target_type": adapter.target_type,
            "as_of_date": as_of_iso,
            "target_date": target_date.isoformat(),
            "horizon_days": MODEL.vol_horizon,
            "last_close": out.get("last_close"),
            "predicted_vol": out.get("predicted_vol"),
            "trailing_vol": out.get("trailing_vol"),
            "regime_pred": out.get("regime_pred"),
            "regime_proba": out.get("regime_proba"),
            "profile": out.get("profile"),
            "recommended_weight": out.get("recommended_weight"),
            "actual_vol": "",
            "actual_regime": "",
            "regime_correct": "",
            "abs_error": "",
            "status": "pending",
        }
        rows_to_write.append(row)
        print(
            f"[{run_at}] {adapter.name}: logged prediction for {as_of} "
            f"-> scoreable on {target_date} "
            f"(predicted_vol={out.get('predicted_vol')})"
        )

    if rows_to_write:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_write)


if __name__ == "__main__":
    main()
