"""Paper-trading bot: predict volatility -> recommended weight -> simulated trade.

Runs one rebalance per invocation (designed to be scheduled daily). It never
touches real money: it drives a :class:`PaperBroker` and keeps a JSON track
record so you can compare the strategy against buy-and-hold over time before
risking any capital.

Live price: tries a public exchange ticker via ``ccxt`` (no API key needed); if
``ccxt`` is unavailable or the exchange is unreachable, it falls back to the
latest daily close from the cached OHLCV data.

Usage::

    python -m coinpredictor.trading.bot                 # uses default profile
    python -m coinpredictor.trading.bot --profile defensive --capital 10000
"""
from __future__ import annotations

import argparse

from coinpredictor.config import STRATEGY
from coinpredictor.predict import predict_next_day
from coinpredictor.trading.paper_broker import PaperBroker


def live_price() -> tuple[float, str]:
    """Best-effort live BTC/USD price; falls back to the latest cached close."""
    try:
        import ccxt  # optional dependency

        for name in ("kraken", "coinbase", "bitstamp"):
            try:
                exchange = getattr(ccxt, name)()
                ticker = exchange.fetch_ticker("BTC/USD")
                price = ticker.get("last") or ticker.get("close")
                if price:
                    return float(price), f"ccxt:{name}"
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: latest daily close from cached data.
    from coinpredictor.data.ohlcv import load_ohlcv

    ohlcv = load_ohlcv(refresh=True)
    return float(ohlcv["close"].iloc[-1]), "ohlcv-close"


def run_once(
    profile: str | None = None,
    initial_capital: float = 10_000.0,
    fee: float = 0.001,
) -> dict:
    """Execute a single predict -> rebalance cycle against the paper portfolio."""
    profile = profile or STRATEGY.live_profile

    pred = predict_next_day(profile=profile)
    price, source = live_price()

    broker = PaperBroker.load_or_create(initial_capital)
    broker.fee = fee
    record = broker.rebalance(
        pred.recommended_weight, price, date=str(pred.as_of_date.date())
    )
    broker.save()

    eq = record["equity"]
    bh = record["buy_and_hold_equity"]
    edge = eq / bh - 1.0 if bh > 0 else 0.0

    print(pred.summary())
    print(f"Live price: ${price:,.2f} (source: {source})")
    print(
        f"Rebalanced to {record['target_weight']:.0%} BTC "
        f"(traded ${record['trade_value']:+,.2f}, fee ${record['fee']:,.2f})"
    )
    print(
        f"Portfolio equity: ${eq:,.2f}  |  buy & hold: ${bh:,.2f}  "
        f"|  strategy edge: {edge:+.2%}  ({len(broker.history)} trades logged)"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="CoinPredictor paper-trading bot")
    parser.add_argument(
        "--profile",
        default=None,
        help="Risk profile: aggressive | balanced | defensive",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial capital for a fresh portfolio (ignored once state exists)",
    )
    parser.add_argument("--fee", type=float, default=0.001, help="Per-trade fee fraction")
    args = parser.parse_args()
    run_once(profile=args.profile, initial_capital=args.capital, fee=args.fee)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
