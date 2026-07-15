#!/usr/bin/env bash
# ============================================================================
# SEPARATE, COST-BEARING, ONCE-DAILY judge run — DO NOT merge into
# run_daily_docker.sh (which is free and runs twice daily).
#
# This invokes the LLM Judge layer, which calls a PAID API. It only does
# anything when COINPREDICTOR_JUDGE_ENABLED=true (checked inside run_judge.py)
# AND stays under the hard daily spend cap (JudgeConfig.max_daily_cost_usd).
# With the flag off (the shipped default) it exits immediately, zero cost.
#
# Suggested HOST crontab entry (once a day; note it is deliberately its own
# line, distinct from the twice-daily free job):
#   30 22 * * *  /path/to/CoinPredictor/scripts/run_judge_docker.sh >> /path/to/CoinPredictor/logs/judge.log 2>&1
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "===== JUDGE $(date '+%Y-%m-%d %H:%M:%S') ====="
docker compose run --rm coinpredictor python scripts/run_judge.py
docker compose run --rm coinpredictor python scripts/evaluate_judges.py
