#!/usr/bin/env bash
# Invoked once a day by the HOST's cron (Ubuntu Server), not by anything
# inside Docker. Runs the log + evaluate scripts inside the coinpredictor
# image so results are written straight to the bind-mounted ./data and
# ./logs folders.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
docker compose run --rm coinpredictor python scripts/log_prediction.py
docker compose run --rm coinpredictor python scripts/evaluate_predictions.py
