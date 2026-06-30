"""Central configuration: paths, symbols, and runtime settings.

All other modules import from here so paths and parameters live in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env (if present) so API keys become available via os.getenv.
load_dotenv()

# --- Filesystem layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Data parameters ---------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Symbols and date range for data ingestion."""

    btc_ticker: str = "BTC-USD"          # yfinance ticker
    start_date: str = "2015-01-01"        # earliest reliable daily BTC data
    interval: str = "1d"                   # daily candles -> next-day horizon

    # Phase 2 macro tickers (yfinance, free, no key)
    macro_tickers: dict = field(
        default_factory=lambda: {
            "sp500": "^GSPC",     # S&P 500 index
            "gold": "GC=F",       # Gold futures
            "dxy": "DX-Y.NYB",    # US Dollar index
        }
    )

    # Phase 3 on-chain metrics (CoinMetrics community API, free, no key)
    onchain_charts: dict = field(
        default_factory=lambda: {
            "hash_rate": "HashRate",        # network hash rate
            "n_transactions": "TxCnt",      # daily transaction count
            "active_addresses": "AdrActCnt",  # active addresses (network usage)
        }
    )


# --- Model parameters --------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Modeling and validation settings for volatility prediction."""

    # Regression target: forward realized volatility (annualized).
    target_col: str = "target_vol"
    # Classification target: will the coming period be more volatile than recent norm?
    regime_col: str = "target_high_vol"

    # Forward horizon (trading days) over which realized volatility is measured.
    vol_horizon: int = 5
    # Trailing window (days) defining the "recent norm" for the regime label.
    regime_lookback: int = 30
    # Annualization factor (crypto trades ~365 days/year).
    annualization: int = 365

    n_splits: int = 5                      # TimeSeriesSplit folds (walk-forward)
    test_size: float = 0.2                 # final hold-out fraction
    random_state: int = 42
    model_filename: str = "btc_vol_lgbm.pkl"

    # LightGBM hyperparameters for the volatility regressor. Tuned via
    # ``python -m coinpredictor.tune``; override here to apply the best found.
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.01,
            "num_leaves": 15,
            "subsample": 1.0,
            "colsample_bytree": 0.7,
            "reg_lambda": 0.0,
            "min_child_samples": 40,
        }
    )


# --- Strategy parameters -----------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    """Volatility-targeting backtest settings."""

    target_annual_vol: float = 0.60       # desired annualized portfolio vol
    max_weight: float = 1.0               # cap exposure (1.0 = no leverage)
    min_weight: float = 0.0               # long/flat only

    # Risk profile used to turn a live forecast into a recommended BTC weight.
    # One of: "aggressive", "balanced", "defensive" (see backtest.STRATEGY_PROFILES).
    live_profile: str = "balanced"


# --- Feature-phase selection -------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Which external feature phases to include when training/serving.

    Phase 1 (technical) is always on. Enabling these adds network fetches that
    are cached to ``data/raw`` after the first run.
    """

    use_macro: bool = True        # Phase 2: S&P500 / Gold / DXY
    use_sentiment: bool = True    # Phase 3: Fear & Greed (+ optional NewsAPI)
    use_onchain: bool = True      # Phase 3: hash rate / tx count / active addresses

    # Phase 4 features. OFF by default: their history is short (DVOL ~2023-10+,
    # OKX funding ~3 months), so enabling them shrinks the trainable window from
    # ~4000 days. Use ``coinpredictor.evaluate`` to test them on the recent
    # subset before turning them on for production.
    use_implied_vol: bool = False  # Phase 4: Deribit DVOL (implied volatility)
    use_funding: bool = False      # Phase 4: OKX perpetual funding rate


# --- API keys (optional, Phase 3) -------------------------------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY", "")

DATA = DataConfig()
MODEL = ModelConfig()
STRATEGY = StrategyConfig()
FEATURES = FeatureConfig()
