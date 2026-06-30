"""Paper-trading broker: a simulated portfolio with no real money at risk.

Holds cash + BTC and rebalances toward a target weight, charging a proportional
fee on each trade. State is persisted to JSON so running the bot once per day
accumulates a real track record. A buy-and-hold benchmark is tracked from the
first run for an honest comparison.

NOTHING here touches a real exchange or real funds. Swapping in a live broker
later means replacing ``rebalance`` with real order calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from coinpredictor.config import PROCESSED_DIR

_STATE_FILE = PROCESSED_DIR / "paper_state.json"


@dataclass
class PaperBroker:
    """A simulated cash + BTC portfolio."""

    cash: float
    btc: float = 0.0
    fee: float = 0.001
    initial_capital: float = 0.0
    initial_price: float = 0.0          # BTC price at first funding (for benchmark)
    history: list = field(default_factory=list)

    # --- Portfolio maths -----------------------------------------------------
    def equity(self, price: float) -> float:
        """Total portfolio value (cash + BTC) at ``price``."""
        return self.cash + self.btc * price

    def weight(self, price: float) -> float:
        """Current BTC exposure as a fraction of equity."""
        eq = self.equity(price)
        return (self.btc * price) / eq if eq > 0 else 0.0

    def buy_and_hold_equity(self, price: float) -> float:
        """Benchmark: value if all initial capital was held in BTC from day one."""
        if self.initial_price <= 0:
            return self.initial_capital
        return self.initial_capital * (price / self.initial_price)

    # --- Trading -------------------------------------------------------------
    def rebalance(self, target_weight: float, price: float, date: str | None = None) -> dict:
        """Trade toward ``target_weight`` BTC exposure at ``price``.

        Returns a record of the trade. The fee is charged on the traded notional.
        """
        target_weight = min(max(target_weight, 0.0), 1.0)
        if self.initial_price <= 0:  # first rebalance sets the benchmark anchor
            self.initial_price = price
            if self.initial_capital <= 0:
                self.initial_capital = self.equity(price)

        eq = self.equity(price)
        target_btc_value = target_weight * eq
        delta_value = target_btc_value - self.btc * price  # +buy / -sell
        fee_cost = abs(delta_value) * self.fee

        self.btc += delta_value / price
        self.cash -= delta_value + fee_cost

        record = {
            "date": date,
            "price": price,
            "target_weight": target_weight,
            "trade_value": delta_value,
            "fee": fee_cost,
            "cash": self.cash,
            "btc": self.btc,
            "equity": self.equity(price),
            "buy_and_hold_equity": self.buy_and_hold_equity(price),
        }
        self.history.append(record)
        return record

    # --- Persistence ---------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or _STATE_FILE
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "PaperBroker":
        path = path or _STATE_FILE
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def load_or_create(cls, initial_capital: float, path: Path | None = None) -> "PaperBroker":
        path = path or _STATE_FILE
        if Path(path).exists():
            return cls.load(path)
        return cls(cash=initial_capital, initial_capital=initial_capital)
