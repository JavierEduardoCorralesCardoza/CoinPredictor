"""Cross-sectional (relative-strength) momentum over a diversified crypto basket.

Phases 0-2 established the honest verdict for *single-asset* BTC: neither
direction models nor volatility targeting beat buy & hold after realistic costs.
This module tests the one predictive signal with genuine academic support --
**cross-sectional momentum**: instead of asking "will BTC go up?", it ranks a
basket of coins by recent relative strength each rebalance and holds the top
performers, equal-weighted, long-only (spot).

The honest question is not "does momentum make money gross?" (survivorship makes
almost anything look good) but "does a momentum *tilt* beat simply holding the
same basket equal-weighted, net of costs, after discounting for the number of
configurations tried?". So every variant is scored with the shared cost model
(``COSTS``) and the pre-registered gate (``GATE``) against an equal-weight
buy & hold of the identical universe.

    python -m coinpredictor.momentum            # full search on the real basket
    python -m coinpredictor.momentum --refresh  # force a data refresh first

SURVIVORSHIP WARNING: the universe is coins liquid *today*; coins that died are
absent, which inflates backtested momentum. A pass here is necessary, not
sufficient -- it must still clear the 6-week prospective paper stage before any
real money.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinpredictor.config import COSTS, GATE, MOMENTUM, PROCESSED_DIR, MomentumConfig
from coinpredictor.validation import (
    GateResult,
    annualized_to_period_sr,
    evaluate_strategy_gate,
)


# --- Price panel -------------------------------------------------------------
def build_price_panel(
    cfg: MomentumConfig = MOMENTUM,
    *,
    refresh: bool = False,
    loader=None,
) -> pd.DataFrame:
    """Aligned daily close-price panel (columns = symbols, index = dates).

    Each symbol is loaded independently and outer-joined on the date index, so a
    coin that listed late simply carries NaN before its first candle (and is
    excluded from ranking on those dates). ``loader`` is injectable for tests.
    """
    if loader is None:
        from coinpredictor.data.exchange_ohlcv import load_exchange_ohlcv

        def loader(symbol: str) -> pd.DataFrame:
            return load_exchange_ohlcv(
                symbol=symbol, timeframe=cfg.timeframe,
                refresh=refresh, exchange_id=cfg.exchange_id,
            )

    closes: dict[str, pd.Series] = {}
    for symbol in cfg.universe:
        try:
            df = loader(symbol)
        except Exception as e:  # a single bad symbol must not kill the panel
            print(f"  ! skipping {symbol}: {e}")
            continue
        closes[symbol] = df["close"].rename(symbol)

    if not closes:
        raise RuntimeError("No symbols could be loaded for the momentum panel.")
    panel = pd.concat(closes.values(), axis=1).sort_index()
    return panel


# --- Signal ------------------------------------------------------------------
def momentum_scores(
    panel: pd.DataFrame,
    *,
    lookback: int,
    skip: int,
) -> pd.DataFrame:
    """Relative-strength score per (date, asset): return over the formation
    window ``[t-lookback-skip, t-skip]``.

    Skipping the most recent ``skip`` days sidesteps short-term reversal, the
    classic "12-1" construction adapted to crypto's faster clock. All inputs are
    strictly past prices, so the score on day ``t`` is knowable at ``t``.
    """
    past = panel.shift(skip)
    formation = past / past.shift(lookback) - 1.0
    return formation


def target_weights(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    top_n: int,
    min_history: int,
) -> pd.DataFrame:
    """Equal-weight long-only target weights: 1/top_n on the top-``top_n`` names
    with a valid score and enough history, 0 elsewhere.
    """
    # An asset is eligible only once it has ``min_history`` days of prices.
    eligible = panel.notna().rolling(min_history, min_periods=min_history).count()
    eligible = eligible.ge(min_history)
    valid = scores.where(eligible & scores.notna())

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    ranks = valid.rank(axis=1, ascending=False, method="first")
    winners = ranks.le(top_n) & valid.notna()
    count = winners.sum(axis=1).replace(0, np.nan)
    weights = winners.div(count, axis=0).fillna(0.0)
    return weights


# --- Portfolio backtest ------------------------------------------------------
@dataclass
class PortfolioMetrics:
    name: str
    sharpe: float
    total_return: float
    max_drawdown: float
    avg_turnover: float
    returns: pd.Series

    def row(self) -> dict:
        return {
            "strategy": self.name,
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "avg_turnover": self.avg_turnover,
        }


def _max_drawdown(equity: pd.Series) -> float:
    running = equity.cummax()
    return float((equity / running - 1.0).min()) if len(equity) else 0.0


def backtest_portfolio(
    panel: pd.DataFrame,
    target: pd.DataFrame,
    *,
    rebalance_days: int,
    fee: float,
    name: str,
    periods_per_year: int = 365,
) -> PortfolioMetrics:
    """Backtest an equal-weight long-only portfolio from target weights.

    Target weights are only *acted on* every ``rebalance_days`` (held constant
    in between, so turnover -- and cost -- stays low). Each day's portfolio
    return is last-decided weights times each asset's next-day return; cost is
    charged on the L1 change of weights at each rebalance.
    """
    rets = panel.pct_change().shift(-1)          # next-day return per asset
    dates = panel.index

    held = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    current = pd.Series(0.0, index=panel.columns)
    turnovers: list[float] = []
    last_rebalance = -10**9
    for i, dt in enumerate(dates):
        if i - last_rebalance >= rebalance_days:
            new = target.loc[dt].fillna(0.0)
            if new.abs().sum() > 0:               # only rebalance on a valid signal
                turnovers.append(float((new - current).abs().sum()))
                current = new
                last_rebalance = i
        held.loc[dt] = current.values

    gross = (held * rets).sum(axis=1)
    # Cost hits on rebalance days, proportional to weight change (L1).
    turnover_series = held.diff().abs().sum(axis=1).fillna(held.iloc[0].abs().sum())
    net = (gross - turnover_series * fee).dropna()

    equity = (1.0 + net).cumprod()
    std = net.std()
    sharpe = 0.0 if std == 0 or np.isnan(std) else float(
        np.sqrt(periods_per_year) * net.mean() / std
    )
    return PortfolioMetrics(
        name=name,
        sharpe=sharpe,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        max_drawdown=_max_drawdown(equity),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        returns=net,
    )


def equal_weight_hold(
    panel: pd.DataFrame, *, fee: float, min_history: int, periods_per_year: int = 365
) -> PortfolioMetrics:
    """Benchmark: hold every eligible asset equal-weighted, rebalanced daily.

    This is the honest bar the momentum tilt must clear -- the diversified
    'just hold the basket' portfolio, not BTC alone.
    """
    eligible = panel.notna()
    counts = eligible.sum(axis=1).replace(0, np.nan)
    weights = eligible.div(counts, axis=0).fillna(0.0)
    # Freeze rebalancing to daily via a 1-day cadence on the eligibility weights.
    return backtest_portfolio(
        panel, weights, rebalance_days=1, fee=fee,
        name="equal_weight_hold", periods_per_year=periods_per_year,
    )


# --- Risk overlay: volatility targeting + cash sleeve ------------------------
def vol_target_exposure(
    raw_returns: pd.Series,
    *,
    target_annual_vol: float,
    lookback: int,
    periods_per_year: int = 365,
) -> pd.Series:
    """Daily gross-exposure multiplier in [0, 1] that scales the portfolio
    toward ``target_annual_vol`` using only *past* realized volatility.

    Spot, long-only, no leverage: exposure is capped at 1.0, so on calm days we
    are fully invested and on wild days we shift the excess into cash (which
    earns nothing here). The realized vol is shifted one day so the exposure on
    day ``t`` depends only on returns up to ``t-1`` -- no look-ahead.
    """
    realized = raw_returns.rolling(lookback, min_periods=lookback).std()
    realized = realized * np.sqrt(periods_per_year)
    exposure = (target_annual_vol / realized).clip(upper=1.0)
    # Only act once we have a full vol estimate; flat (0) until then.
    return exposure.shift(1).reindex(raw_returns.index).fillna(0.0)


def regime_exposure(panel: pd.DataFrame, *, ma_days: int) -> pd.Series:
    """Binary risk-on/off multiplier (1.0 or 0.0) from a basket-level trend.

    The 'market' here is the equal-weight basket index (mean price across the
    universe). When it closes below its ``ma_days`` moving average the whole
    portfolio goes to cash, sitting out the deepest bear legs -- the one thing
    vol targeting cannot do, because in a crypto crash every coin falls together.
    The comparison is shifted one day so the decision on day ``t`` uses only the
    trend known at ``t-1`` (no look-ahead).
    """
    index_level = panel.mean(axis=1)
    ma = index_level.rolling(ma_days, min_periods=ma_days).mean()
    risk_on = (index_level > ma).astype("float64")
    return risk_on.shift(1).reindex(panel.index).fillna(0.0)


def _build_exposure(
    raw_returns: pd.Series,
    regime: pd.Series,
    *,
    target_vol: float | None,
    use_regime: bool,
    vol_lookback: int,
) -> pd.Series:
    """Compose the gross-exposure multiplier from the optional vol target and
    the optional regime filter (both in [0, 1], multiplied together)."""
    if target_vol is None:
        exp = pd.Series(1.0, index=raw_returns.index)
    else:
        exp = vol_target_exposure(
            raw_returns, target_annual_vol=target_vol, lookback=vol_lookback
        )
    if use_regime:
        exp = exp * regime.reindex(exp.index).fillna(0.0)
    return exp



# --- Search + gate + verdict -------------------------------------------------
@dataclass
class MomentumVariant:
    name: str
    lookback: int
    top_n: int
    rebalance_days: int
    metrics: PortfolioMetrics
    exposure_mode: str = "raw"


@dataclass
class MomentumOutcome:
    variants: list[MomentumVariant]
    benchmark: PortfolioMetrics
    best: MomentumVariant
    gate: GateResult
    verdict: str
    recommendation: str

    def summary(self) -> str:
        lines = ["Cross-sectional momentum search", "=" * 60]
        header = f"{'variant':28} {'Sharpe':>8} {'return':>9} {'maxDD':>8} {'turn':>7}"
        lines.append(header)
        for v in sorted(self.variants, key=lambda x: x.metrics.sharpe, reverse=True):
            m = v.metrics
            lines.append(
                f"{v.name:28} {m.sharpe:8.3f} {m.total_return:9.2%} "
                f"{m.max_drawdown:8.2%} {m.avg_turnover:7.3f}"
            )
        b = self.benchmark
        lines.append("-" * 60)
        lines.append(
            f"{b.name:28} {b.sharpe:8.3f} {b.total_return:9.2%} "
            f"{b.max_drawdown:8.2%} {b.avg_turnover:7.3f}"
        )
        lines.append("=" * 60)
        lines.append(f"Best variant: {self.best.name}")
        lines.append(self.gate.summary())
        lines.append("")
        lines.append(f"VERDICT: {self.verdict}")
        lines.append(f"RECOMMENDATION: {self.recommendation}")
        return "\n".join(lines)


def _select_verdict(
    *,
    gate_passed: bool,
    best_name: str,
    beats_benchmark: bool = False,
    dsr_ok: bool = False,
    drawdown_ok: bool = True,
) -> tuple[str, str]:
    """Pure go/no-go mapping, mirroring decision._select_verdict.

    Three honest outcomes:
      * PASS -- clears every gate check; proceed to prospective paper.
      * EDGE-BUT-TOO-RISKY -- the tilt genuinely beats equal-weight hold on
        Sharpe (even after a volatility-targeting overlay), yet its drawdown
        still blows through the risk tolerance. The relative-strength signal is
        real, but long-only crypto is too violent to fit the drawdown budget
        with the tools here (``dsr_ok`` additionally flags whether the edge also
        fails multiple-testing deflation once every knob tried is counted).
      * NO-GO -- no tilt beats simply holding the basket after costs.
    """
    if gate_passed:
        return (
            "PASS",
            f"'{best_name}' beats equal-weight hold net of costs and clears the "
            "gate. Proceed to a 6-week prospective PAPER run before any real "
            "money (survivorship still unproven live).",
        )
    if beats_benchmark and not drawdown_ok:
        fragility = (
            ""
            if dsr_ok
            else " On top of that the Deflated Sharpe no longer clears 0.95 once "
            "every overlay/parameter tried is counted, so the edge is also "
            "fragile to multiple testing."
        )
        return (
            "EDGE-BUT-TOO-RISKY",
            f"'{best_name}' genuinely beats equal-weight hold on Sharpe -- the "
            "relative-strength signal is real and survives a volatility-targeting "
            "overlay -- but even the tightest vol target cannot pull its drawdown "
            "inside the risk tolerance (crypto crashes are correlated, so the "
            "cash sleeve throttles too late)." + fragility + " Do NOT deploy it; "
            "fitting the drawdown budget would need a trend/regime exit or a much "
            "larger, non-survivorship universe, neither proven here.",
        )
    return (
        "NO-GO",
        "No momentum tilt beats simply holding the basket equal-weighted after "
        "costs and multiple-testing deflation. Do not trade it; an equal-weight "
        "rebalanced hold is the honest baseline.",
    )


def run_momentum(
    cfg: MomentumConfig = MOMENTUM,
    *,
    refresh: bool = False,
    loader=None,
    fee: float | None = None,
) -> MomentumOutcome:
    """Run the full cross-sectional momentum search, gate and verdict."""
    fee = COSTS.per_side if fee is None else fee
    panel = build_price_panel(cfg, refresh=refresh, loader=loader)

    benchmark = equal_weight_hold(panel, fee=fee, min_history=cfg.min_history_days)

    regime = regime_exposure(panel, ma_days=cfg.regime_ma_days)

    variants: list[MomentumVariant] = []
    for lookback in cfg.lookbacks:
        scores = momentum_scores(panel, lookback=lookback, skip=cfg.skip_days)
        for top_n in cfg.top_n:
            weights = target_weights(
                scores, panel, top_n=top_n, min_history=cfg.min_history_days
            )
            for reb in cfg.rebalance_days:
                base = f"mom_L{lookback}_top{top_n}_reb{reb}"
                raw = backtest_portfolio(
                    panel, weights, rebalance_days=reb, fee=fee, name=base
                )
                # Exposure modes: raw, vol-target(s), regime, and their combos.
                for use_regime in (False, True):
                    for target_vol in (None, *cfg.overlay_target_vols):
                        tag_parts = []
                        if target_vol is not None:
                            tag_parts.append(f"vt{target_vol:g}")
                        if use_regime:
                            tag_parts.append("reg")
                        mode = "_".join(tag_parts) if tag_parts else "raw"
                        if mode == "raw":
                            metrics = raw
                        else:
                            exposure = _build_exposure(
                                raw.returns, regime,
                                target_vol=target_vol, use_regime=use_regime,
                                vol_lookback=cfg.vol_lookback,
                            )
                            scaled = weights.mul(exposure, axis=0).fillna(0.0)
                            metrics = backtest_portfolio(
                                panel, scaled, rebalance_days=reb, fee=fee,
                                name=f"{base}_{mode}",
                            )
                        variants.append(
                            MomentumVariant(
                                metrics.name, lookback, top_n, reb, metrics, mode
                            )
                        )

    best = max(variants, key=lambda v: v.metrics.sharpe)

    # Pre-registered selection rule: among variants that FIT the drawdown budget
    # pick the highest Sharpe (that is the whole point of the risk overlay); only
    # if none fit do we fall back to the raw highest-Sharpe variant so the honest
    # EDGE-BUT-TOO-RISKY verdict can still fire.
    dd_limit = -abs(GATE.max_drawdown_limit)
    within_budget = [v for v in variants if v.metrics.max_drawdown >= dd_limit]
    if within_budget:
        best = max(within_budget, key=lambda v: v.metrics.sharpe)

    # Deflate for the whole grid tried (multiple-testing honesty).
    n_trials = len(variants)
    period_sharpes = [
        annualized_to_period_sr(v.metrics.sharpe) for v in variants
    ]
    sr_variance = float(np.var(period_sharpes, ddof=1)) if len(period_sharpes) > 1 else 0.0

    # Align best-vs-benchmark returns on their common index for a fair gate.
    strat_ret = best.metrics.returns
    bench_ret = benchmark.returns
    common = strat_ret.index.intersection(bench_ret.index)
    gate = evaluate_strategy_gate(
        strat_ret.loc[common],
        bench_ret.loc[common],
        n_trials=n_trials,
        sr_variance=sr_variance,
        strategy_net_return=best.metrics.total_return,
        max_drawdown=best.metrics.max_drawdown,
        criteria=GATE,
    )

    verdict, recommendation = _select_verdict(
        gate_passed=gate.passed,
        best_name=best.name,
        beats_benchmark=gate.strategy_sharpe > gate.benchmark_sharpe,
        dsr_ok=gate.deflated_sharpe_prob >= GATE.min_deflated_sharpe_prob,
        drawdown_ok=best.metrics.max_drawdown >= -abs(GATE.max_drawdown_limit),
    )
    return MomentumOutcome(
        variants=variants,
        benchmark=benchmark,
        best=best,
        gate=gate,
        verdict=verdict,
        recommendation=recommendation,
    )


def _save_results(outcome: MomentumOutcome) -> None:
    rows = [v.metrics.row() | {"kind": "momentum"} for v in outcome.variants]
    rows.append(outcome.benchmark.row() | {"kind": "benchmark"})
    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "momentum_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum search")
    parser.add_argument(
        "--refresh", action="store_true", help="force a data refresh before running"
    )
    args = parser.parse_args(argv)

    outcome = run_momentum(refresh=args.refresh)
    print(outcome.summary())
    _save_results(outcome)


if __name__ == "__main__":
    main()
