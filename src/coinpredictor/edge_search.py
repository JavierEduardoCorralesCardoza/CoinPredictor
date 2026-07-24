"""Phase 1: cost-aware directional edge search with abstention.

Phase 0 froze the rules of the game: a realistic cost (``COSTS``) and a
pre-registered go/no-go gate (``GATE``). This module runs the actual *search*
for a directional edge that survives those rules, on daily and hourly bars.

The literature is blunt about why naive direction models fail live:

* Bare sign signals look great gross but go negative once costs (~10-31 bps
  round-trip) are charged (2407.18334; 2607.19453).
* What restores profitability is a **cost-aware magnitude filter with
  abstention** (Bysik & Ślepaczuk 2606.00060): take a trade only when the
  *expected favorable move is large enough to pay for its own round-trip cost*,
  and otherwise sit out (NO_TRADE).

We reuse the meta-labeling machinery as the signal generator — primary trend
side, side-aware triple-barrier meta-labels, and purged walk-forward
out-of-sample P(trade works) — then compare three gating families under the
*same* cost harness:

1. ``primary_only``   -- always take the primary side (no filter, the thing that
   historically loses to costs).
2. ``meta_thr``       -- take the side only when P(win) clears a probability
   threshold (López de Prado meta gate).
3. ``cost_filter``    -- take the side only when the **expected net edge**
   ``p·tp − (1−p)·sl − round_trip_cost`` clears a safety margin (the cost-aware
   magnitude filter).

Every searched configuration counts as a trial, so the winner's Sharpe is
discounted by the Deflated Sharpe Ratio and then run through the pre-registered
``evaluate_strategy_gate``. Nothing here is deployed; this only decides which
candidate (if any) earns the right to the prospective paper stage.

    python -m coinpredictor.edge_search             # daily
    python -m coinpredictor.edge_search --intraday  # hourly
    python -m coinpredictor.edge_search --both      # both, side by side
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinpredictor.config import COSTS, ENTRY, GATE, MODEL, PROCESSED_DIR
from coinpredictor.features import feature_columns
from coinpredictor.meta_labeling import (
    StratMetrics,
    backtest_positions,
    meta_labels,
    oos_meta_proba,
    primary_trend_side,
)
from coinpredictor.validation import (
    GateResult,
    annualized_to_period_sr,
    deflated_sharpe_ratio,
    evaluate_strategy_gate,
)


# --- Cost-aware magnitude filter --------------------------------------------
def expected_net_edge(
    proba: pd.Series | np.ndarray,
    tp_pct: float,
    sl_pct: float,
    *,
    round_trip_cost: float,
) -> pd.Series | np.ndarray:
    """Expected per-trade net edge for a meta-labeled trade.

    With ``proba`` = P(the trade hits its take-profit before its stop) and
    barriers ``tp_pct`` / ``sl_pct``, the expected favorable move is
    ``proba·tp − (1−proba)·sl``. Subtracting ``round_trip_cost`` gives the edge
    the trade actually keeps after paying to enter and exit. Only positive-edge
    trades can pay for themselves; everything else should abstain.
    """
    return proba * tp_pct - (1.0 - proba) * sl_pct - round_trip_cost


def cost_aware_gate(
    proba: pd.Series,
    tradable: pd.Series,
    tp_pct: float,
    sl_pct: float,
    *,
    round_trip_cost: float,
    margin: float = 0.0,
) -> pd.Series:
    """Binary trade gate (1 = trade, 0 = abstain) from the magnitude filter.

    Trades only where the primary took a side (``tradable``) AND the expected
    net edge clears ``margin`` (a safety buffer above break-even). NaN
    probabilities abstain. This is the operational NO_TRADE band.
    """
    edge = expected_net_edge(proba, tp_pct, sl_pct, round_trip_cost=round_trip_cost)
    take = edge.to_numpy() > margin
    take = take & tradable.reindex(proba.index).fillna(False).to_numpy().astype(bool)
    return pd.Series(np.where(take, 1.0, 0.0), index=proba.index, name="cost_gate")


# --- Prepared signals for one timeframe -------------------------------------
@dataclass
class SignalSet:
    """Everything the variant search needs for a single timeframe."""

    label: str                 # "1d" or "1h"
    ohlcv: pd.DataFrame
    side: pd.Series            # primary long/flat side, aligned to X
    tradable: pd.Series        # side != 0
    proba: pd.Series           # out-of-sample P(trade works)
    tp_pct: float
    sl_pct: float
    periods_per_year: int


def prepare_signals(*, intraday: bool, allow_short: bool = False) -> SignalSet:
    """Load data and compute primary side + purged OOS meta probabilities.

    Mirrors the daily/hourly setup used by ``meta_labeling.run_meta_labeling``
    so the two modules judge the same underlying signal, differing only in how
    the trade is gated.
    """
    from coinpredictor.config import INTRADAY
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.features import (
        build_default_features,
        build_default_intraday_features,
    )

    if intraday:
        from coinpredictor.data.exchange_ohlcv import load_exchange_ohlcv

        fast, slow = 24, 72
        horizon = INTRADAY.vol_horizon
        tp_pct = sl_pct = 0.03
        periods_per_year = INTRADAY.annualization
        ohlcv = load_exchange_ohlcv(timeframe=INTRADAY.interval)
        base_feats = build_default_intraday_features(ohlcv, drop_na=False)
        label = "1h"
    else:
        fast, slow = 20, 50
        horizon = ENTRY.horizon
        tp_pct, sl_pct = ENTRY.tp_pct, ENTRY.sl_pct
        periods_per_year = 365
        ohlcv = load_ohlcv()
        base_feats = build_default_features(ohlcv, drop_na=False)
        label = "1d"

    embargo = max(2, horizon // 2)
    base_cols = feature_columns(base_feats)

    side = primary_trend_side(base_feats, fast=fast, slow=slow, allow_short=allow_short)
    y_meta = meta_labels(ohlcv, side, horizon=horizon, tp_pct=tp_pct, sl_pct=sl_pct)

    valid = base_feats[base_cols].notna().all(axis=1) & side.notna()
    idx = base_feats.index[valid]
    X = base_feats.loc[idx, base_cols]
    side = side.loc[idx]
    y_meta = y_meta.reindex(idx)
    ohlcv_v = ohlcv.loc[idx]
    tradable = side != 0

    proba = oos_meta_proba(
        X, y_meta, tradable, horizon=horizon, n_splits=MODEL.n_splits, embargo=embargo
    )
    return SignalSet(
        label=label,
        ohlcv=ohlcv_v,
        side=side,
        tradable=tradable,
        proba=proba,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        periods_per_year=periods_per_year,
    )


# --- Variant search + pre-registered gate -----------------------------------
@dataclass
class EdgeSearchResult:
    label: str
    table: pd.DataFrame
    benchmark: StratMetrics
    best: StratMetrics
    n_trials: int
    deflated_sharpe: float
    gate: GateResult


def search_edge(
    signals: SignalSet,
    *,
    prob_thresholds: tuple[float, ...] = (0.45, 0.50, 0.55),
    cost_margins: tuple[float, ...] = (0.0, 0.5, 1.0),
    fee: float = COSTS.per_side,
) -> EdgeSearchResult:
    """Compare gating families for one timeframe under the shared cost harness.

    ``cost_margins`` are expressed in multiples of the round-trip cost, so a
    margin of 1.0 requires the expected net edge to clear a *second* round-trip
    of buffer on top of break-even.
    """
    s = signals
    side_pos = s.side.where(s.tradable, 0.0)
    round_trip = COSTS.round_trip

    def _bt(weight: pd.Series, name: str) -> StratMetrics:
        return backtest_positions(
            s.ohlcv, weight, fee=fee,
            periods_per_year=s.periods_per_year, name=name,
        )

    benchmark = _bt(pd.Series(1.0, index=s.side.index), "buy_and_hold")
    primary = _bt(side_pos, "primary_only")

    searched: list[StratMetrics] = []
    for thr in prob_thresholds:
        gate = (s.proba >= thr).astype(float).fillna(0.0)
        searched.append(_bt(side_pos * gate, f"meta_thr={thr:.2f}"))
    for mult in cost_margins:
        gate = cost_aware_gate(
            s.proba, s.tradable, s.tp_pct, s.sl_pct,
            round_trip_cost=round_trip, margin=mult * round_trip,
        )
        searched.append(_bt(side_pos * gate, f"cost_filter={mult:g}xRT"))

    n_trials = len(searched)
    best = max(searched, key=lambda m: m.sharpe)

    period_srs = [annualized_to_period_sr(m.sharpe, s.periods_per_year) for m in searched]
    sr_var = float(np.var(period_srs, ddof=1)) if len(period_srs) > 1 else 0.0
    dsr = deflated_sharpe_ratio(best.returns, n_trials=n_trials, sr_variance=sr_var)

    gate_result = evaluate_strategy_gate(
        best.returns,
        benchmark.returns,
        n_trials=n_trials,
        sr_variance=sr_var,
        strategy_net_return=best.total_return,
        max_drawdown=best.max_drawdown,
        criteria=GATE,
        periods_per_year=s.periods_per_year,
    )

    rows = [benchmark.row(), primary.row(), *(m.row() for m in searched)]
    table = pd.DataFrame(rows)
    table["deflated_sharpe_best"] = np.nan
    table.loc[table["strategy"] == best.name, "deflated_sharpe_best"] = dsr

    return EdgeSearchResult(
        label=s.label,
        table=table,
        benchmark=benchmark,
        best=best,
        n_trials=n_trials,
        deflated_sharpe=float(dsr),
        gate=gate_result,
    )


def _print_result(res: EdgeSearchResult) -> None:
    print("=" * 78)
    print(f"PHASE 1 EDGE SEARCH  ({res.label})  "
          f"round-trip cost = {COSTS.round_trip * 1e4:.0f} bps")
    print("=" * 78)
    for _, r in res.table.iterrows():
        print(f"  {r['strategy']:<20} Sharpe={r['sharpe']:+.2f}  "
              f"ret={r['total_return']:+.1%}  maxDD={r['max_drawdown']:+.1%}  "
              f"turnover={r['avg_turnover']:.3f}  exp={r['exposure']:.0%}")
    print(f"\n  -> best: {res.best.name}  (Sharpe {res.best.sharpe:+.2f} "
          f"vs B&H {res.benchmark.sharpe:+.2f})")
    print(f"  -> Deflated Sharpe (after {res.n_trials} trials): {res.deflated_sharpe:.3f}")
    print()
    print(res.gate.summary())
    print("=" * 78)


def run_edge_search(*, intraday: bool, allow_short: bool = False) -> EdgeSearchResult:
    signals = prepare_signals(intraday=intraday, allow_short=allow_short)
    res = search_edge(signals)
    _print_result(res)
    out = PROCESSED_DIR / f"edge_search_results_{res.label}.csv"
    res.table.to_csv(out, index=False)
    print(f"Saved -> {out}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 cost-aware edge search")
    parser.add_argument("--intraday", action="store_true", help="run on hourly bars")
    parser.add_argument("--both", action="store_true", help="run daily and hourly")
    parser.add_argument("--short", action="store_true", help="allow short side")
    args = parser.parse_args()

    if args.both:
        run_edge_search(intraday=False, allow_short=args.short)
        run_edge_search(intraday=True, allow_short=args.short)
    else:
        run_edge_search(intraday=args.intraday, allow_short=args.short)


if __name__ == "__main__":
    main()
