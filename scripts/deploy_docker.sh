#!/usr/bin/env bash
# One-shot setup for your Ubuntu Server (Docker-based, consistent with your
# Nextcloud/Terraria containers).
#
# Usage:
#   ./scripts/deploy_docker.sh [HOUR] [MINUTE]
#
# Example:
#   ./scripts/deploy_docker.sh 09 30
set -euo pipefail

HOUR="${1:-09}"
MINUTE="${2:-30}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Building coinpredictor image"
docker compose build

mkdir -p data/raw data/processed models logs

# Train the model once, if no artifact exists yet.
if [ ! -f "models/btc_vol_lgbm.pkl" ]; then
  echo "==> No trained model found — training now (downloads data, ~1-3 min)"
  docker compose run --rm coinpredictor python -m coinpredictor.model
else
  echo "==> Trained model already present at models/btc_vol_lgbm.pkl, skipping"
fi

echo "==> Starting the dashboard container (restart: unless-stopped)"
docker compose up -d coinpredictor-dashboard

chmod +x scripts/run_daily_docker.sh

# Register the daily cron job on the HOST (idempotent).
MARKER="# coinpredictor-daily-predict"
CRON_CMD="$MINUTE $HOUR * * * $ROOT/scripts/run_daily_docker.sh >> $ROOT/logs/cron.log 2>&1 $MARKER"
( crontab -l 2>/dev/null | grep -v "$MARKER" ; echo "$CRON_CMD" ) | crontab -

echo ""
echo "==> Done."
echo "==> Dashboard running at http://localhost:8501 (localhost-only)."
echo "==> Point your Cloudflare Tunnel's public hostname at http://localhost:8501"
echo "    (same way you exposed Nextcloud/Terraria)."
echo "==> Daily cron installed: $HOUR:$MINUTE -> scripts/run_daily_docker.sh"
echo "==> Verify: crontab -l"
echo "==> Test the daily job right now: ./scripts/run_daily_docker.sh"
echo "==> Prediction log: data/processed/prediction_log.csv"
