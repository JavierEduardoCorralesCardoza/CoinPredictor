"""Cross-asset diversified portfolio (equities + bonds + gold + a crypto sleeve).

Every crypto-only experiment hit the same wall: crypto is one correlated risk
factor, so no risk management *inside* crypto pulls its drawdown under a 20%
budget. The evidence-based fix is diversification across *uncorrelated factors*.
This module builds a diversified portfolio from free yfinance data --

    equities (SPY)  +  long bonds (TLT)  +  gold (GLD)  +  a capped BTC sleeve

-- weights it either equally or by inverse volatility (a risk-parity proxy that
naturally down-weights the wildest asset), optionally scales the whole book
toward a portfolio volatility target (cash sleeve for the rest), and scores each
variant with the SAME cost model (``COSTS``) and pre-registered gate (``GATE``)
against simply holding BTC.

    python -m coinpredictor.diversified            # full search on real data
    python -m coinpredictor.diversified --refresh  # force a data refresh first

DATA vs DEPLOYMENT: prices are free, so this validates offline now. Trading the
equity/bond/gold legs needs a broker. For a Mexico-based deployment the picked
venue is **Interactive Brokers** for the SPY/TLT/GLD legs (real US-listed ETFs,
fractional shares, ~5 bps commissions -- comfortably under the 15 bps modelled
in ``COSTS``) plus **Bitso** for the BTC sleeve (MXN on/off-ramp). A PASS still
means "the diversified thesis earns a broker", i.e. run a prospective PAPER gate
first, then micro real capital.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinpredictor.config import COSTS, DIVERSIFIED, GATE, PROCESSED_DIR, DiversifiedConfig
from coinpredictor.momentum import (
    PortfolioMetrics,
    backtest_portfolio,
    vol_target_exposure,
)
from coinpredictor.validation import (
    GateResult,
    annualized_to_period_sr,
    evaluate_strategy_gate,
)


# --- Price panel -------------------------------------------------------------
def build_asset_panel(
    cfg: DiversifiedConfig = DIVERSIFIED,
    *,
    refresh: bool = False,
    loader=None,
) -> pd.DataFrame:
    """Aligned daily close-price panel on the equity trading calendar.

    Each asset is loaded independently; the equity leg defines the master
    calendar (~252 days/yr) and the others are reindexed onto it and
    forward-filled over their own holidays, then the head is trimmed to where
    every asset has data. ``loader`` is injectable for tests (no network).
    """
    if loader is None:
        import yfinance as yf

        def loader(ticker: str) -> pd.Series:
            raw = yf.download(
                ticker, start=cfg.start_date, interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError(f"no data for {ticker}")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"]
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s

    series: dict[str, pd.Series] = {}
    for name, ticker in cfg.assets.items():
        s = loader(ticker)
        series[name] = pd.Series(s).rename(name)

    # Use the equity leg as the master calendar when present, else the first.
    equity_name = "equities" if "equities" in series else next(iter(series))
    master_index = series[equity_name].index

    panel = pd.concat(series.values(), axis=1).sort_index()
    panel = panel.reindex(panel.index.union(master_index)).ffill()
    panel = panel.reindex(master_index).dropna()
    return panel


# --- Weight schemes ----------------------------------------------------------
def _apply_crypto_cap(
    weights: pd.DataFrame, *, crypto_key: str, cap: float
) -> pd.DataFrame:
    """Cap the crypto weight at ``cap`` and redistribute the excess pro-rata
    across the other assets (keeps each row summing to 1)."""
    if crypto_key not in weights.columns or cap is None:
        return weights
    others = [c for c in weights.columns if c != crypto_key]
    capped = weights[crypto_key].clip(upper=cap)
    remainder = 1.0 - capped
    others_sum = weights[others].sum(axis=1).replace(0, np.nan)
    scaled_others = weights[others].div(others_sum, axis=0).mul(remainder, axis=0)
    out = scaled_others.copy()
    out[crypto_key] = capped
    return out[weights.columns].fillna(0.0)


def equal_weights(panel: pd.DataFrame, cfg: DiversifiedConfig) -> pd.DataFrame:
    """Constant equal weights (then crypto-capped)."""
    n = panel.shape[1]
    w = pd.DataFrame(1.0 / n, index=panel.index, columns=panel.columns)
    return _apply_crypto_cap(w, crypto_key=cfg.crypto_key, cap=cfg.crypto_cap)


def inverse_vol_weights(panel: pd.DataFrame, cfg: DiversifiedConfig) -> pd.DataFrame:
    """Risk-parity proxy: weight each asset by 1/volatility (past-only), then
    cap the crypto sleeve. Naturally down-weights the wildest assets."""
    rets = panel.pct_change()
    vol = rets.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std()
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    w = _apply_crypto_cap(w, crypto_key=cfg.crypto_key, cap=cfg.crypto_cap)
    return w.fillna(0.0)


_WEIGHT_BUILDERS = {
    "equal": equal_weights,
    "inverse_vol": inverse_vol_weights,
}


def build_variant_weights(
    panel: pd.DataFrame,
    cfg: DiversifiedConfig,
    scheme: str,
    target_vol: float | None,
    *,
    fee: float,
    periods_per_year: int | None = None,
) -> tuple[str, pd.DataFrame]:
    """Build the daily weight frame for one (scheme, target_vol) variant.

    Returns ``(tag, weights)`` where ``tag`` is the variant name used across the
    search/sensitivity outputs. When ``target_vol`` is set, the base scheme is
    first measured un-scaled and then throttled toward that portfolio vol using
    only *past* realized volatility (a cash sleeve absorbs the rest).
    """
    ppy = periods_per_year or cfg.periods_per_year
    base_weights = _WEIGHT_BUILDERS[scheme](panel, cfg)
    if target_vol is None:
        return scheme, base_weights
    tag = f"{scheme}_vt{target_vol:g}"
    raw = backtest_portfolio(
        panel, base_weights, rebalance_days=cfg.rebalance_days,
        fee=fee, name=tag, periods_per_year=ppy,
    )
    exposure = vol_target_exposure(
        raw.returns, target_annual_vol=target_vol,
        lookback=cfg.vol_lookback, periods_per_year=ppy,
    )
    weights = base_weights.mul(exposure, axis=0).fillna(0.0)
    return tag, weights


def _variant_from_tag(cfg: DiversifiedConfig, tag: str) -> tuple[str, float | None]:
    """Reverse a variant ``tag`` back into its ``(scheme, target_vol)``."""
    for scheme in cfg.weight_schemes:
        for tv in cfg.portfolio_target_vols:
            t = scheme if tv is None else f"{scheme}_vt{tv:g}"
            if t == tag:
                return scheme, tv
    raise KeyError(f"unknown variant tag: {tag!r}")


# --- Search + gate + verdict -------------------------------------------------
@dataclass
class DiversifiedVariant:
    name: str
    scheme: str
    target_vol: float | None
    metrics: PortfolioMetrics


@dataclass
class DiversifiedOutcome:
    variants: list[DiversifiedVariant]
    benchmarks: dict[str, PortfolioMetrics]
    best: DiversifiedVariant
    gate: GateResult
    verdict: str
    recommendation: str

    def summary(self) -> str:
        lines = ["Cross-asset diversified portfolio search", "=" * 64]
        header = f"{'variant':30} {'Sharpe':>8} {'return':>10} {'maxDD':>8} {'turn':>7}"
        lines.append(header)
        for v in sorted(self.variants, key=lambda x: x.metrics.sharpe, reverse=True):
            m = v.metrics
            lines.append(
                f"{v.name:30} {m.sharpe:8.3f} {m.total_return:10.2%} "
                f"{m.max_drawdown:8.2%} {m.avg_turnover:7.3f}"
            )
        lines.append("-" * 64)
        for b in self.benchmarks.values():
            lines.append(
                f"{b.name:30} {b.sharpe:8.3f} {b.total_return:10.2%} "
                f"{b.max_drawdown:8.2%} {b.avg_turnover:7.3f}"
            )
        lines.append("=" * 64)
        lines.append(f"Best variant: {self.best.name}")
        lines.append(self.gate.summary())
        lines.append("")
        lines.append(f"VERDICT: {self.verdict}")
        lines.append(f"RECOMMENDATION: {self.recommendation}")
        return "\n".join(lines)


def _select_verdict(
    *, gate_passed: bool, best_name: str, beats_btc: bool, drawdown_ok: bool
) -> tuple[str, str]:
    """Pure go/no-go mapping for the diversified portfolio.

      * PASS -- beats plain BTC on Sharpe net of costs AND fits the drawdown
        budget AND survives deflation: the diversified thesis is worth a broker;
        proceed to a prospective paper run.
      * BETTER-THAN-BTC-BUT-OVER-BUDGET -- beats BTC risk-adjusted but its
        drawdown still exceeds tolerance (usually too much crypto/equity beta).
      * NO-GO -- does not even beat holding BTC risk-adjusted.
    """
    if gate_passed:
        return (
            "PASS",
            f"'{best_name}' beats simply holding BTC on Sharpe net of costs, fits "
            "the drawdown budget and survives deflation. The cross-factor "
            "diversification thesis is validated offline -- it earns a broker. "
            "Next: a 6-week prospective PAPER run, then micro real capital.",
        )
    if beats_btc and not drawdown_ok:
        return (
            "BETTER-THAN-BTC-BUT-OVER-BUDGET",
            f"'{best_name}' beats BTC risk-adjusted but its drawdown still exceeds "
            "the tolerance -- trim the crypto/equity sleeve or tighten the "
            "portfolio vol target, then re-gate.",
        )
    return (
        "NO-GO",
        "No diversified variant beats simply holding BTC risk-adjusted after "
        "costs. Holding BTC (or DCA) remains the honest baseline.",
    )


def _benchmarks(
    panel: pd.DataFrame, cfg: DiversifiedConfig, *, fee: float
) -> dict[str, PortfolioMetrics]:
    """Reference portfolios: 100% BTC, classic 60/40, and equal-weight all."""
    ppy = cfg.periods_per_year
    out: dict[str, PortfolioMetrics] = {}

    btc = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    btc[cfg.crypto_key] = 1.0
    out["btc_hold"] = backtest_portfolio(
        panel, btc, rebalance_days=cfg.rebalance_days, fee=fee,
        name="btc_hold", periods_per_year=ppy,
    )

    if {"equities", "bonds"}.issubset(panel.columns):
        sf = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        sf["equities"], sf["bonds"] = 0.6, 0.4
        out["sixty_forty"] = backtest_portfolio(
            panel, sf, rebalance_days=cfg.rebalance_days, fee=fee,
            name="sixty_forty", periods_per_year=ppy,
        )

    n = panel.shape[1]
    ew = pd.DataFrame(1.0 / n, index=panel.index, columns=panel.columns)
    out["equal_weight_all"] = backtest_portfolio(
        panel, ew, rebalance_days=cfg.rebalance_days, fee=fee,
        name="equal_weight_all", periods_per_year=ppy,
    )
    return out


def run_diversified(
    cfg: DiversifiedConfig = DIVERSIFIED,
    *,
    refresh: bool = False,
    loader=None,
    fee: float | None = None,
    panel: pd.DataFrame | None = None,
) -> DiversifiedOutcome:
    """Run the full diversified-portfolio search, gate and verdict."""
    fee = COSTS.per_side if fee is None else fee
    ppy = cfg.periods_per_year
    if panel is None:
        panel = build_asset_panel(cfg, refresh=refresh, loader=loader)

    benchmarks = _benchmarks(panel, cfg, fee=fee)

    variants: list[DiversifiedVariant] = []
    for scheme in cfg.weight_schemes:
        for target_vol in cfg.portfolio_target_vols:
            tag, weights = build_variant_weights(
                panel, cfg, scheme, target_vol, fee=fee, periods_per_year=ppy,
            )
            metrics = backtest_portfolio(
                panel, weights, rebalance_days=cfg.rebalance_days,
                fee=fee, name=tag, periods_per_year=ppy,
            )
            variants.append(DiversifiedVariant(tag, scheme, target_vol, metrics))

    # Pre-registered selection: among variants that FIT the drawdown budget pick
    # the highest Sharpe; only if none fit fall back to the global best Sharpe.
    dd_limit = -abs(GATE.max_drawdown_limit)
    within = [v for v in variants if v.metrics.max_drawdown >= dd_limit]
    pool = within if within else variants
    best = max(pool, key=lambda v: v.metrics.sharpe)

    n_trials = len(variants)
    period_sharpes = [annualized_to_period_sr(v.metrics.sharpe, ppy) for v in variants]
    sr_variance = float(np.var(period_sharpes, ddof=1)) if len(period_sharpes) > 1 else 0.0

    btc_bench = benchmarks["btc_hold"]
    strat_ret = best.metrics.returns
    common = strat_ret.index.intersection(btc_bench.returns.index)
    gate = evaluate_strategy_gate(
        strat_ret.loc[common],
        btc_bench.returns.loc[common],
        n_trials=n_trials,
        sr_variance=sr_variance,
        strategy_net_return=best.metrics.total_return,
        max_drawdown=best.metrics.max_drawdown,
        criteria=GATE,
        periods_per_year=ppy,
    )

    verdict, recommendation = _select_verdict(
        gate_passed=gate.passed,
        best_name=best.name,
        beats_btc=gate.strategy_sharpe > gate.benchmark_sharpe,
        drawdown_ok=best.metrics.max_drawdown >= dd_limit,
    )
    return DiversifiedOutcome(
        variants=variants,
        benchmarks=benchmarks,
        best=best,
        gate=gate,
        verdict=verdict,
        recommendation=recommendation,
    )


def _save_results(outcome: DiversifiedOutcome) -> None:
    rows = [v.metrics.row() | {"kind": "diversified"} for v in outcome.variants]
    rows += [b.row() | {"kind": "benchmark"} for b in outcome.benchmarks.values()]
    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "diversified_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")


# --- Sub-period & stress sensitivity ----------------------------------------
# Pre-registered crisis windows: the whole point of the diversified thesis is
# surviving regimes where a single risk factor blows up. 2022 is the acid test
# -- stocks AND long bonds fell together (the classic 60/40 hedge failed), so a
# portfolio leaning on bonds for ballast has to prove it still holds up.
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2018_crypto_winter": ("2018-01-01", "2018-12-31"),
    "2020_covid_crash": ("2020-02-01", "2020-04-30"),
    "2021_22_crypto_bust": ("2021-11-01", "2022-12-31"),
    "2022_stocks_bonds_down": ("2022-01-01", "2022-12-31"),
    "2023_24_recovery": ("2023-01-01", "2024-12-31"),
}


def _slice_stats(returns: pd.Series, periods_per_year: int) -> tuple[float, float, float]:
    """``(sharpe, total_return, max_drawdown)`` for one slice of net returns."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan"), float("nan"), float("nan")
    std = r.std()
    sharpe = 0.0 if std == 0 or np.isnan(std) else float(
        np.sqrt(periods_per_year) * r.mean() / std
    )
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return sharpe, total, max_dd


def _returns_by_name(outcome: DiversifiedOutcome, name: str) -> pd.Series:
    for v in outcome.variants:
        if v.name == name:
            return v.metrics.returns
    if name in outcome.benchmarks:
        return outcome.benchmarks[name].returns
    raise KeyError(f"no variant/benchmark named {name!r}")


def _sensitivity_columns(outcome: DiversifiedOutcome) -> list[str]:
    """The best diversified variant vs the two honest benchmarks."""
    cols = [outcome.best.name]
    for b in ("btc_hold", "sixty_forty"):
        if b in outcome.benchmarks:
            cols.append(b)
    return cols


def annual_sensitivity(
    outcome: DiversifiedOutcome,
    *,
    periods_per_year: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Per-calendar-year Sharpe and max drawdown for the best variant vs BTC/60-40.

    Splits each strategy's net-return series by year so a single strong bull run
    can't hide a bad regime. The ``period`` column is the calendar year.
    """
    columns = columns or _sensitivity_columns(outcome)
    series = {c: _returns_by_name(outcome, c) for c in columns}
    years = sorted({int(y) for s in series.values() for y in s.index.year})
    rows = []
    for year in years:
        row: dict[str, object] = {"period": year}
        for name, s in series.items():
            sh, _tot, dd = _slice_stats(s[s.index.year == year], periods_per_year)
            row[f"{name}_sharpe"] = sh
            row[f"{name}_maxDD"] = dd
        rows.append(row)
    return pd.DataFrame(rows)


def stress_windows(
    outcome: DiversifiedOutcome,
    *,
    periods_per_year: int,
    columns: list[str] | None = None,
    windows: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Total return and max drawdown over each pre-registered crisis window."""
    columns = columns or _sensitivity_columns(outcome)
    windows = windows or STRESS_WINDOWS
    series = {c: _returns_by_name(outcome, c) for c in columns}
    rows = []
    for label, (start, end) in windows.items():
        row: dict[str, object] = {"window": label}
        for name, s in series.items():
            _sh, total, dd = _slice_stats(s.loc[start:end], periods_per_year)
            row[f"{name}_ret"] = total
            row[f"{name}_maxDD"] = dd
        rows.append(row)
    return pd.DataFrame(rows)


def run_sensitivity(
    cfg: DiversifiedConfig = DIVERSIFIED,
    *,
    refresh: bool = False,
    loader=None,
    fee: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DiversifiedOutcome]:
    """Compute annual + stress sensitivity for the pre-registered best variant."""
    outcome = run_diversified(cfg, refresh=refresh, loader=loader, fee=fee)
    annual = annual_sensitivity(outcome, periods_per_year=cfg.periods_per_year)
    stress = stress_windows(outcome, periods_per_year=cfg.periods_per_year)
    return annual, stress, outcome


def _save_sensitivity(annual: pd.DataFrame, stress: pd.DataFrame) -> None:
    annual_out = PROCESSED_DIR / "diversified_sensitivity_annual.csv"
    stress_out = PROCESSED_DIR / "diversified_sensitivity_stress.csv"
    annual.to_csv(annual_out, index=False)
    stress.to_csv(stress_out, index=False)
    print(f"Saved annual sensitivity  -> {annual_out}")
    print(f"Saved stress sensitivity  -> {stress_out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-asset diversified search")
    parser.add_argument(
        "--refresh", action="store_true", help="force a data refresh before running"
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="also compute per-year + crisis-window sensitivity and save CSVs",
    )
    args = parser.parse_args(argv)

    if args.sensitivity:
        annual, stress, outcome = run_sensitivity(refresh=args.refresh)
        print(outcome.summary())
        print("\nPer-year sensitivity (Sharpe / maxDD):")
        print(annual.to_string(index=False))
        print("\nCrisis-window sensitivity (return / maxDD):")
        print(stress.to_string(index=False))
        _save_results(outcome)
        _save_sensitivity(annual, stress)
        return

    outcome = run_diversified(refresh=args.refresh)
    print(outcome.summary())
    _save_results(outcome)


if __name__ == "__main__":
    main()
