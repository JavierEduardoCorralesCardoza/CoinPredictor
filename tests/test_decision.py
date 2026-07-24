"""Phase 2 tests: the final decision verdict logic and the vol-target harness."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression, LogisticRegression

from coinpredictor.decision import (
    VOL_TARGET_VARIANTS,
    DecisionOutcome,
    _select_verdict,
    evaluate_decision,
)
from coinpredictor.features import build_features
from coinpredictor.validation import GateResult


# --- Pure verdict branches ---------------------------------------------------
def test_verdict_pass():
    verdict, rec = _select_verdict(
        gate_passed=True, best_name="vol_target_p1.0",
        benchmark_dd=-0.60, dd_cutter_name=None, dd_cutter_dd=None,
    )
    assert verdict == "PASS"
    assert "prospective paper" in rec


def test_verdict_risk_reduction():
    verdict, rec = _select_verdict(
        gate_passed=False, best_name="vol_target_p1.0",
        benchmark_dd=-0.60, dd_cutter_name="defensive_cut1.0", dd_cutter_dd=-0.25,
    )
    assert verdict == "NO-EDGE / RISK-REDUCTION ONLY"
    assert "defensive_cut1.0" in rec
    assert "-60%" in rec and "-25%" in rec


def test_verdict_no_go():
    verdict, rec = _select_verdict(
        gate_passed=False, best_name="vol_target_p1.0",
        benchmark_dd=-0.60, dd_cutter_name=None, dd_cutter_dd=None,
    )
    assert verdict == "NO-GO"
    assert "NOT to run an active strategy" in rec


# --- Lightweight end-to-end harness (no network, no LightGBM) ----------------
def test_evaluate_decision_runs_with_injected_models(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv, drop_na=True)
    outcome = evaluate_decision(
        feats,
        n_splits=3,
        vol_factory=lambda: LinearRegression(),
        regime_factory=lambda: LogisticRegression(max_iter=200),
    )
    assert isinstance(outcome, DecisionOutcome)
    assert outcome.n_trials == len(VOL_TARGET_VARIANTS)
    # Table = every variant plus the buy & hold benchmark row.
    assert len(outcome.table) == len(VOL_TARGET_VARIANTS) + 1
    assert "buy_and_hold" in set(outcome.table["strategy"])
    assert outcome.best.name in VOL_TARGET_VARIANTS
    assert isinstance(outcome.gate, GateResult)
    assert outcome.verdict in {"PASS", "NO-EDGE / RISK-REDUCTION ONLY", "NO-GO"}
    assert outcome.recommendation
