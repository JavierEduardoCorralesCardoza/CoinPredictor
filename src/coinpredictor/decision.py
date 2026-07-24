"""Phase 2: the final go/no-go decision, net of realistic costs.

Phase 1 searched for a *directional* edge and the pre-registered gate rejected
every variant on both daily and hourly bars: after ~30 bps round-trip, none of
them beat a cost-matched buy & hold. That leaves exactly one mechanism the
evidence still supports — **volatility targeting**, whose value was never higher
directional accuracy but *lower drawdown for a given amount of return*.

This module makes the honest final call. It replays the volatility-targeting
family (plain inverse-vol at several strengths, plus the defensive regime
overlay) out-of-sample with the real cost model, then runs each candidate
through the SAME pre-registered ``evaluate_strategy_gate`` used everywhere else.
It prints one of three verdicts:

* a vol-target variant PASSES the gate -> it earns the prospective paper stage;
* nothing passes but a defensive variant cuts drawdown a lot for a small Sharpe
  give-up -> a *risk-reduction* recommendation (trade it only if you value the
  drawdown cut and accept it is not a Sharpe improvement);
* nothing meaningfully beats holding -> the honest recommendation is buy & hold
  / DCA, i.e. do NOT run an active strategy.

    python -m coinpredictor.decision
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinpredictor.backtest import BacktestResult, walk_forward_backtest
from coinpredictor.config import GATE, MODEL
from coinpredictor.validation import (
    GateResult,
    annualized_to_period_sr,
    evaluate_strategy_gate,
)


# The volatility-targeting family evaluated for deployment. These mirror
# ``backtest.compare_strategies`` so the decision uses the exact sizing logic
# that has been validated elsewhere — no new, unvetted knobs at the last step.
VOL_TARGET_VARIANTS: dict[str, dict] = {
    "vol_target_p1.0": dict(power=1.0),
    "vol_target_p1.5": dict(power=1.5),
    "vol_target_p0.5": dict(power=0.5),
    "defensive_cut0.5": dict(power=1.0, regime_cut=0.5, use_regime=True),
    "defensive_cut1.0": dict(power=1.0, regime_cut=1.0, use_regime=True),
}


@dataclass
class Candidate:
    name: str
    result: BacktestResult
    returns: pd.Series          # per-period strategy returns (net of cost)


@dataclass
class DecisionOutcome:
    table: pd.DataFrame
    best: Candidate
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    n_trials: int
    gate: GateResult
    verdict: str
    recommendation: str


def _strategy_returns(result: BacktestResult) -> pd.Series:
    """Per-period net strategy returns implied by the equity curve."""
    return result.equity["strategy"].pct_change().dropna()


def _benchmark_returns(result: BacktestResult) -> pd.Series:
    return result.equity["buy_and_hold"].pct_change().dropna()


def _select_verdict(
    *,
    gate_passed: bool,
    best_name: str,
    benchmark_dd: float,
    dd_cutter_name: str | None,
    dd_cutter_dd: float | None,
) -> tuple[str, str]:
    """Map the evaluated facts to a verdict + plain-language recommendation.

    Pure so the three branches (pass / risk-reduction / no-go) are testable
    without training any models.
    """
    if gate_passed:
        return "PASS", (
            f"'{best_name}' clears the pre-registered gate net of costs. "
            "Promote it to the 6-week prospective paper stage (Phase 4) before "
            "risking any real money."
        )
    if dd_cutter_name is not None and dd_cutter_dd is not None:
        return "NO-EDGE / RISK-REDUCTION ONLY", (
            f"No variant beats buy & hold on Sharpe after costs, so there is no "
            f"profit edge. However '{dd_cutter_name}' cuts max drawdown from "
            f"{benchmark_dd:.0%} to {dd_cutter_dd:.0%}. "
            "Only run it if you explicitly value the smaller drawdown and accept a "
            "lower return; otherwise buy & hold / DCA is the honest choice."
        )
    return "NO-GO", (
        "Nothing beats buy & hold on Sharpe or drawdown after costs. The "
        "evidence-based recommendation is NOT to run an active strategy: hold "
        "BTC (DCA) with your risk-capital only. Do not build live execution."
    )


def evaluate_decision(
    features_df: pd.DataFrame,
    n_splits: int | None = None,
    *,
    vol_factory=None,
    regime_factory=None,
) -> DecisionOutcome:
    """Run the vol-target family net of costs and apply the pre-registered gate.

    ``vol_factory`` / ``regime_factory`` default to the project's real models;
    they are injectable so the decision logic can be exercised with lightweight
    estimators in tests.
    """
    if vol_factory is None or regime_factory is None:
        from coinpredictor.model import build_regime_classifier, build_vol_regressor

        vol_factory = vol_factory or build_vol_regressor
        regime_factory = regime_factory or build_regime_classifier

    candidates: list[Candidate] = []
    for name, cfg in VOL_TARGET_VARIANTS.items():
        kwargs = dict(power=cfg["power"])
        if cfg.get("use_regime"):
            kwargs["clf_factory"] = regime_factory
            kwargs["regime_cut"] = cfg["regime_cut"]
        res = walk_forward_backtest(
            vol_factory, features_df, n_splits=n_splits, **kwargs
        )
        candidates.append(Candidate(name, res, _strategy_returns(res)))

    n_trials = len(candidates)
    best = max(candidates, key=lambda c: c.result.strategy_sharpe)
    benchmark_sharpe = best.result.bh_sharpe
    benchmark_dd = best.result.bh_max_drawdown

    period_srs = [
        annualized_to_period_sr(c.result.strategy_sharpe, 365) for c in candidates
    ]
    sr_var = float(np.var(period_srs, ddof=1)) if len(period_srs) > 1 else 0.0

    gate = evaluate_strategy_gate(
        best.returns,
        _benchmark_returns(best.result),
        n_trials=n_trials,
        sr_variance=sr_var,
        strategy_net_return=best.result.strategy_return,
        max_drawdown=best.result.strategy_max_drawdown,
        criteria=GATE,
        periods_per_year=365,
    )

    # Is there at least a defensive variant that trades Sharpe for a big DD cut?
    dd_cutters = [
        c for c in candidates
        if c.result.strategy_max_drawdown > benchmark_dd + 0.10   # >=10pp shallower
    ]
    best_dd_cutter = (
        max(dd_cutters, key=lambda c: c.result.strategy_max_drawdown)
        if dd_cutters else None
    )

    verdict, recommendation = _select_verdict(
        gate_passed=gate.passed,
        best_name=best.name,
        benchmark_dd=benchmark_dd,
        dd_cutter_name=best_dd_cutter.name if best_dd_cutter else None,
        dd_cutter_dd=(
            best_dd_cutter.result.strategy_max_drawdown if best_dd_cutter else None
        ),
    )

    rows = []
    for c in candidates:
        r = c.result
        rows.append({
            "strategy": c.name,
            "sharpe": r.strategy_sharpe,
            "total_return": r.strategy_return,
            "max_drawdown": r.strategy_max_drawdown,
        })
    rows.append({
        "strategy": "buy_and_hold",
        "sharpe": benchmark_sharpe,
        "total_return": best.result.bh_return,
        "max_drawdown": benchmark_dd,
    })
    table = pd.DataFrame(rows)

    return DecisionOutcome(
        table=table,
        best=best,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_max_drawdown=benchmark_dd,
        n_trials=n_trials,
        gate=gate,
        verdict=verdict,
        recommendation=recommendation,
    )


def _print_outcome(outcome: DecisionOutcome) -> None:
    print("=" * 78)
    print("PHASE 2 FINAL DECISION  (volatility targeting vs buy & hold, net of costs)")
    print("=" * 78)
    for _, r in outcome.table.iterrows():
        print(f"  {r['strategy']:<20} Sharpe={r['sharpe']:+.2f}  "
              f"ret={r['total_return']:+.1%}  maxDD={r['max_drawdown']:+.1%}")
    print(f"\n  -> best by Sharpe: {outcome.best.name} "
          f"(Sharpe {outcome.best.result.strategy_sharpe:+.2f} vs "
          f"B&H {outcome.benchmark_sharpe:+.2f})")
    print()
    print(outcome.gate.summary())
    print()
    print(f"VERDICT: {outcome.verdict}")
    print(f"RECOMMENDATION: {outcome.recommendation}")
    print("=" * 78)


def run_decision(n_splits: int | None = None) -> DecisionOutcome:
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import build_default_features

    ohlcv = load_ohlcv()
    feats = build_default_features(ohlcv, drop_na=True)
    outcome = evaluate_decision(feats, n_splits=n_splits)
    _print_outcome(outcome)
    return outcome


def main() -> None:
    run_decision()


if __name__ == "__main__":
    main()
