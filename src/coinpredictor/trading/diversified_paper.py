"""Prospective PAPER gate for the cross-asset diversified portfolio.

The offline search (``coinpredictor.diversified``) already PASSes the
pre-registered gate, but an offline PASS is in-sample by construction. Before
risking real money the honest next step is a *prospective* paper run: freeze the
winning variant, then -- one rebalance per invocation on live prices -- track a
simulated multi-asset book against two benchmarks (100% BTC and 60/40) and, once
enough live observations accumulate, re-score the SAME pre-registered gate on the
purely out-of-sample track record.

Nothing here touches real funds. Three simulated books (strategy, BTC, 60/40)
are rebalanced with the same cost model (``COSTS``) and persisted to JSON so
running daily builds a genuine forward record.

    python -m coinpredictor.trading.diversified_paper          # one live rebalance
    python -m coinpredictor.trading.diversified_paper --gate   # score the record
    python -m coinpredictor.trading.diversified_paper --status # print the record
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from coinpredictor.config import COSTS, DIVERSIFIED, GATE, PROCESSED_DIR, DiversifiedConfig
from coinpredictor.diversified import (
    build_asset_panel,
    build_variant_weights,
    run_diversified,
    _variant_from_tag,
)
from coinpredictor.validation import evaluate_strategy_gate

STATE_FILE = PROCESSED_DIR / "diversified_paper_state.json"

# Minimum live observations before the prospective gate is meaningful. ~6 weeks
# of daily rebalances (the plan in the diversified verdict) is ~30 trading days.
MIN_PROSPECTIVE_OBS = 30


# --- One simulated book (cash + fractional units of each asset) --------------
@dataclass
class PaperBook:
    """A simulated multi-asset book: cash plus fractional units per asset."""

    cash: float
    units: dict[str, float] = field(default_factory=dict)
    initial_capital: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.units.get(a, 0.0) * p for a, p in prices.items())

    def rebalance(
        self, weights: dict[str, float], prices: dict[str, float], fee: float
    ) -> float:
        """Trade toward ``weights`` at ``prices``; charge ``fee`` on L1 turnover.

        Returns the post-trade equity. The un-invested remainder (when target
        weights sum to < 1, e.g. a vol-target cash sleeve) simply stays in cash.
        """
        eq = self.equity(prices)
        for asset, price in prices.items():
            if price <= 0:
                continue
            target_value = weights.get(asset, 0.0) * eq
            desired_units = target_value / price
            trade_units = desired_units - self.units.get(asset, 0.0)
            notional = abs(trade_units * price)
            cost = notional * fee
            self.cash -= trade_units * price + cost
            self.units[asset] = desired_units
        return self.equity(prices)

    def to_dict(self) -> dict:
        return {"cash": self.cash, "units": self.units, "initial_capital": self.initial_capital}

    @classmethod
    def from_dict(cls, d: dict) -> "PaperBook":
        return cls(cash=d["cash"], units=dict(d["units"]), initial_capital=d["initial_capital"])


# --- Persisted state ---------------------------------------------------------
def _fresh_state(
    assets: list[str], *, variant: str, fee: float, initial_capital: float
) -> dict:
    book = lambda: PaperBook(cash=initial_capital, units={a: 0.0 for a in assets},
                             initial_capital=initial_capital).to_dict()
    return {
        "variant": variant,
        "fee": fee,
        "initial_capital": initial_capital,
        "start_date": None,
        "assets": assets,
        "strategy": book(),
        "btc": book(),
        "sixty_forty": book(),
        "records": [],
    }


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text())


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# --- Benchmarks' fixed weights ----------------------------------------------
def _btc_weights(cfg: DiversifiedConfig) -> dict[str, float]:
    return {cfg.crypto_key: 1.0}


def _sixty_forty_weights(assets: list[str]) -> dict[str, float]:
    if {"equities", "bonds"}.issubset(assets):
        return {"equities": 0.6, "bonds": 0.4}
    return {}


# --- One live rebalance ------------------------------------------------------
def run_once(
    cfg: DiversifiedConfig = DIVERSIFIED,
    *,
    refresh: bool = True,
    loader=None,
    fee: float | None = None,
    initial_capital: float = 10_000.0,
    variant: str | None = None,
    panel: pd.DataFrame | None = None,
    as_of: str | None = None,
) -> dict:
    """Run one live rebalance of all three books and append a record."""
    fee = COSTS.per_side if fee is None else fee
    if panel is None:
        panel = build_asset_panel(cfg, refresh=refresh, loader=loader)
    assets = list(panel.columns)

    state = load_state()
    if state is None:
        # Freeze the pre-registered winning variant on the very first run.
        if variant is None:
            variant = run_diversified(cfg, loader=loader, fee=fee, panel=panel).best.name
        state = _fresh_state(assets, variant=variant, fee=fee, initial_capital=initial_capital)

    variant = state["variant"]
    fee = state["fee"]

    # Target weights: latest row of the FROZEN variant + fixed benchmark weights.
    scheme, target_vol = _variant_from_tag(cfg, variant)
    _tag, weight_frame = build_variant_weights(
        panel, cfg, scheme, target_vol, fee=fee, periods_per_year=cfg.periods_per_year
    )
    strat_w = weight_frame.iloc[-1].to_dict()
    prices = panel.iloc[-1].to_dict()
    as_of = as_of or str(panel.index[-1].date())

    strat = PaperBook.from_dict(state["strategy"])
    btc = PaperBook.from_dict(state["btc"])
    sf = PaperBook.from_dict(state["sixty_forty"])

    strat_eq = strat.rebalance(strat_w, prices, fee)
    btc_eq = btc.rebalance(_btc_weights(cfg), prices, fee)
    sf_w = _sixty_forty_weights(assets)
    sf_eq = sf.rebalance(sf_w, prices, fee) if sf_w else float("nan")

    state["strategy"] = strat.to_dict()
    state["btc"] = btc.to_dict()
    state["sixty_forty"] = sf.to_dict()
    if state["start_date"] is None:
        state["start_date"] = as_of

    record = {
        "date": as_of,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "weights": {k: round(v, 6) for k, v in strat_w.items()},
        "prices": {k: round(v, 6) for k, v in prices.items()},
        "strategy_equity": strat_eq,
        "btc_equity": btc_eq,
        "sixty_forty_equity": sf_eq,
    }
    # Idempotent per day: replace an existing record for the same date.
    state["records"] = [r for r in state["records"] if r["date"] != as_of]
    state["records"].append(record)
    state["records"].sort(key=lambda r: r["date"])
    save_state(state)
    return record


# --- Prospective gate --------------------------------------------------------
@dataclass
class ProspectiveResult:
    n_obs: int
    ready: bool
    verdict: str
    detail: str

    def summary(self) -> str:
        return f"Prospective paper gate ({self.n_obs} obs): {self.verdict}\n{self.detail}"


def evaluate_prospective_gate(
    cfg: DiversifiedConfig = DIVERSIFIED,
    *,
    state: dict | None = None,
    min_obs: int = MIN_PROSPECTIVE_OBS,
) -> ProspectiveResult:
    """Score the SAME pre-registered gate on the live paper track record.

    This is prospective and single-shot: the variant was frozen at registration,
    so ``n_trials=1`` (no in-sample search to deflate away).
    """
    state = state or load_state()
    if state is None or len(state.get("records", [])) < 2:
        return ProspectiveResult(0, False, "PENDING", "No paper track record yet.")

    df = pd.DataFrame(state["records"]).sort_values("date")
    strat_eq = pd.to_numeric(df["strategy_equity"], errors="coerce")
    btc_eq = pd.to_numeric(df["btc_equity"], errors="coerce")
    strat_ret = strat_eq.pct_change().dropna()
    btc_ret = btc_eq.pct_change().dropna()
    n_obs = int(len(strat_ret))

    if n_obs < min_obs:
        return ProspectiveResult(
            n_obs, False, "PENDING",
            f"Need >= {min_obs} live observations; have {n_obs}. Keep the daily "
            "paper run going before trusting the prospective verdict.",
        )

    equity_curve = strat_eq / strat_eq.iloc[0]
    max_dd = float((equity_curve / equity_curve.cummax() - 1.0).min())
    net_return = float(strat_eq.iloc[-1] / strat_eq.iloc[0] - 1.0)

    gate = evaluate_strategy_gate(
        strat_ret.to_numpy(),
        btc_ret.to_numpy(),
        n_trials=1,                      # frozen variant: nothing to deflate
        sr_variance=0.0,
        strategy_net_return=net_return,
        max_drawdown=max_dd,
        criteria=GATE,
        periods_per_year=cfg.periods_per_year,
    )
    verdict = "PASS" if gate.passed else "FAIL"
    return ProspectiveResult(n_obs, True, verdict, gate.summary())


# --- CLI ---------------------------------------------------------------------
def _print_status(state: dict | None) -> None:
    if state is None or not state.get("records"):
        print("No paper track record yet. Run without flags to start one.")
        return
    df = pd.DataFrame(state["records"]).sort_values("date")
    print(f"Frozen variant: {state['variant']}  |  start: {state['start_date']}")
    print(f"Observations:   {len(df)}")
    last = df.iloc[-1]
    print(
        f"Latest ({last['date']}): strategy ${last['strategy_equity']:,.2f}  "
        f"BTC ${last['btc_equity']:,.2f}  60/40 ${last['sixty_forty_equity']:,.2f}"
    )
    strat0 = float(df.iloc[0]["strategy_equity"])
    btc0 = float(df.iloc[0]["btc_equity"])
    print(
        f"Since start:    strategy {last['strategy_equity']/strat0-1:+.2%}  "
        f"BTC {last['btc_equity']/btc0-1:+.2%}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diversified prospective paper gate")
    parser.add_argument("--gate", action="store_true", help="score the paper record")
    parser.add_argument("--status", action="store_true", help="print the paper record")
    parser.add_argument("--no-refresh", action="store_true", help="skip data refresh")
    parser.add_argument("--capital", type=float, default=10_000.0)
    args = parser.parse_args(argv)

    if args.status:
        _print_status(load_state())
        return
    if args.gate:
        print(evaluate_prospective_gate().summary())
        return

    record = run_once(refresh=not args.no_refresh, initial_capital=args.capital)
    print(f"Rebalanced {record['date']}:")
    print(f"  strategy equity:   ${record['strategy_equity']:,.2f}")
    print(f"  BTC equity:        ${record['btc_equity']:,.2f}")
    print(f"  60/40 equity:      ${record['sixty_forty_equity']:,.2f}")
    print(f"  target weights:    {record['weights']}")
    print(evaluate_prospective_gate().summary())


if __name__ == "__main__":
    main()
