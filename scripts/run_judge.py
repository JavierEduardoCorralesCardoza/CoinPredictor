#!/usr/bin/env python
"""Run the LLM Judge layer once and append each verdict to judge_log.csv.

SEPARATE, COST-BEARING, ONCE-DAILY cron entry — deliberately NOT merged into the
free twice-daily run_daily_docker.sh. It only does anything when
``COINPREDICTOR_JUDGE_ENABLED`` is on, and it respects a hard daily spend cap.

Unlike the ML log, this file is deliberately NOT deduped: multiple runs per day
are allowed and expected, specifically so evaluate_judges.py can measure run-to-
run consistency of a non-deterministic agent.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta

import pandas as pd

from coinpredictor.config import ANTHROPIC_API_KEY, JUDGE, JUDGE_LOG
from coinpredictor.judges import JUDGES, assemble_judge_context

FIELDS = [
    "run_at",
    "model_name",
    "as_of_date",
    "target_date",
    "action",
    "confidence",
    "suggested_weight",
    "reasoning",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "realized_fwd_return",   # filled later by evaluate_judges.py
    "hypothetical_pnl",
    "status",
]


def _spent_today() -> float:
    """Sum of estimated_cost_usd already logged today (for the spend cap)."""
    if not JUDGE_LOG.exists():
        return 0.0
    df = pd.read_csv(JUDGE_LOG, dtype=str)
    if df.empty or "run_at" not in df.columns:
        return 0.0
    today = datetime.now().date().isoformat()
    same_day = df[df["run_at"].astype(str).str.startswith(today)]
    if same_day.empty:
        return 0.0
    return float(pd.to_numeric(same_day["estimated_cost_usd"], errors="coerce").fillna(0).sum())


def main() -> int:
    if not JUDGE.enabled:
        print(
            "COINPREDICTOR_JUDGE_ENABLED is OFF (default). The Judge layer makes "
            "zero API calls. Set the flag to enable it — exiting."
        )
        return 0

    # Flag ON but no key: fail loudly, do NOT silently no-op.
    if not ANTHROPIC_API_KEY:
        print(
            "ERROR: COINPREDICTOR_JUDGE_ENABLED is ON but ANTHROPIC_API_KEY is "
            "missing. Set the key in .env or turn the flag off."
        )
        return 1

    # Hard daily spend cap on top of the master flag. Mirrors the "refuse
    # silently-stale-data" guard in ohlcv.py: over budget => skip + warn.
    spent = _spent_today()
    if spent >= JUDGE.max_daily_cost_usd:
        print(
            f"WARNING: daily judge spend cap reached (${spent:.4f} >= "
            f"${JUDGE.max_daily_cost_usd:.2f}). Skipping this run."
        )
        return 0

    run_at = datetime.now().isoformat(timespec="seconds")
    context = assemble_judge_context(refresh=True)
    as_of = context["as_of_date"]
    target_date = (
        datetime.fromisoformat(as_of).date() + timedelta(days=JUDGE.horizon)
    ).isoformat()

    rows = []
    for judge in JUDGES:
        # Re-check the cap before EACH call so a batch of judges can't blow past it.
        if spent + rows_cost(rows) >= JUDGE.max_daily_cost_usd:
            print(f"WARNING: spend cap would be exceeded; stopping before {judge.name}.")
            break
        try:
            v = judge.verdict(context)
        except Exception as e:  # one bad judge shouldn't corrupt the log
            print(f"[{run_at}] {judge.name}: FAILED ({e})")
            continue

        rows.append(
            {
                "run_at": run_at,
                "model_name": judge.name,
                "as_of_date": as_of,
                "target_date": target_date,
                "action": v.action,
                "confidence": v.confidence,
                "suggested_weight": v.suggested_weight,
                "reasoning": v.reasoning,
                "input_tokens": getattr(judge, "last_input_tokens", ""),
                "output_tokens": getattr(judge, "last_output_tokens", ""),
                "estimated_cost_usd": getattr(judge, "last_cost_usd", ""),
                "realized_fwd_return": "",
                "hypothetical_pnl": "",
                "status": "pending",
            }
        )
        print(
            f"[{run_at}] {judge.name}: {v.action} "
            f"(conf={v.confidence:.2f}, weight={v.suggested_weight:.2f}, "
            f"cost=${getattr(judge, 'last_cost_usd', 0.0):.4f})"
        )

    if rows:
        JUDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        file_exists = JUDGE_LOG.exists()
        with open(JUDGE_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
    return 0


def rows_cost(rows: list[dict]) -> float:
    total = 0.0
    for r in rows:
        try:
            total += float(r.get("estimated_cost_usd") or 0.0)
        except (TypeError, ValueError):
            pass
    return total


if __name__ == "__main__":
    import sys

    sys.exit(main())
