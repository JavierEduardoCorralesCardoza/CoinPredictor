#!/usr/bin/env python
"""Run one live prediction per registered model and append each to the correct
per-family tracking log. Meant to run once a day (via cron / run_daily_docker.sh)
so each day's forecast -- per model -- gets a permanent record that can later be
compared against what BTC actually did.

Phase 1 change: instead of one long-format prediction_log.csv, each model
family now writes to its OWN csv (registry.LOG_FILE_BY_TARGET_TYPE), because
the families have genuinely different row shapes. This is still ONE script /
ONE cron step -- it just routes each adapter's row to the right file.

To add/remove a model, edit coinpredictor.registry.MODELS -- nothing here
needs to change.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta

import pandas as pd

from coinpredictor.config import MODEL
from coinpredictor.registry import LOG_FILE_BY_TARGET_TYPE, MODELS

# Common columns present in every family's file.
_COMMON_FIELDS = [
    "run_at",        # wall-clock timestamp when this row was written
    "model_name",    # which registered model produced this row
    "as_of_date",    # last known BTC data date used for the forecast
    "target_date",   # as_of_date + horizon_days -> when we can score it
    "horizon_days",
    "last_close",
]

# Full column layout per family. The filename encodes the target_type, so it is
# no longer stored as a column. Eval-filled columns start blank ("").
FIELDS_BY_TARGET_TYPE: dict[str, list[str]] = {
    "volatility": _COMMON_FIELDS
    + [
        "predicted_vol",
        "trailing_vol",
        "regime_pred",
        "regime_proba",
        "profile",
        "recommended_weight",
        "actual_vol",
        "actual_regime",
        "regime_correct",
        "abs_error",
        "status",
    ],
    "trend_regime": _COMMON_FIELDS
    + [
        "trend_regime_pred",
        "trend_regime_proba",
        "trend_regime_actual",
        "trend_regime_correct",
        "status",
    ],
    "entry": _COMMON_FIELDS
    + [
        "entry_proba",
        "tp_pct",
        "sl_pct",
        "entry_actual",
        "entry_correct",
        "status",
    ],
    "sentiment": _COMMON_FIELDS
    + [
        "sentiment_score",
        "sentiment_label",
        "n_headlines",
        "sentiment_fwd_return",
        "sentiment_fwd_vol",
        "status",
    ],
}


def _already_logged(log_file, as_of_iso: str, model_name: str) -> bool:
    if not log_file.exists():
        return False
    existing = pd.read_csv(log_file, dtype=str)
    if existing.empty or "model_name" not in existing.columns:
        return False
    mask = (existing["as_of_date"] == as_of_iso) & (existing["model_name"] == model_name)
    return bool(mask.any())


def main() -> None:
    run_at = datetime.now().isoformat(timespec="seconds")

    # Collect rows per destination file so we open each file once.
    rows_by_file: dict = {}
    for adapter in MODELS:
        target_type = adapter.target_type
        log_file = LOG_FILE_BY_TARGET_TYPE.get(target_type)
        if log_file is None:
            print(f"[{run_at}] {adapter.name}: no log file for target_type "
                  f"'{target_type}', skipping.")
            continue

        try:
            out = adapter.predict(refresh=True)
        except Exception as e:  # one bad model shouldn't kill the others
            print(f"[{run_at}] {adapter.name}: FAILED to predict ({e})")
            continue

        as_of = pd.Timestamp(out["as_of_date"]).date()
        as_of_iso = as_of.isoformat()

        if _already_logged(log_file, as_of_iso, adapter.name):
            print(f"[{run_at}] {adapter.name}: prediction for {as_of} already "
                  f"logged, skipping.")
            continue

        horizon = int(out.get("horizon_days") or MODEL.vol_horizon)
        target_date = as_of + timedelta(days=horizon)

        fields = FIELDS_BY_TARGET_TYPE[target_type]
        row = {f: "" for f in fields}
        row.update(
            {
                "run_at": run_at,
                "model_name": adapter.name,
                "as_of_date": as_of_iso,
                "target_date": target_date.isoformat(),
                "horizon_days": horizon,
                "last_close": out.get("last_close"),
                "status": "pending",
            }
        )
        # Fill any family-specific prediction fields the adapter provided.
        for key, val in out.items():
            if key in fields and key != "as_of_date":
                row[key] = val

        rows_by_file.setdefault(log_file, (fields, []))[1].append(row)
        print(f"[{run_at}] {adapter.name}: logged prediction for {as_of} "
              f"-> scoreable on {target_date} ({target_type})")

    for log_file, (fields, rows) in rows_by_file.items():
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = log_file.exists()
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
