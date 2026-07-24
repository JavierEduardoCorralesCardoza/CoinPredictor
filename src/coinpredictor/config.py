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


# --- Per-family prediction log files -----------------------------------------
# One CSV per model family (see prompt Section 3). Each family has a genuinely
# different row shape, so cramming them into one wide table with a target_type
# discriminator leaves most columns empty most of the time. The filename encodes
# the target_type; registry.LOG_FILE_BY_TARGET_TYPE maps adapters -> file.
VOLATILITY_LOG = PROCESSED_DIR / "volatility_log.csv"
TREND_REGIME_LOG = PROCESSED_DIR / "trend_regime_log.csv"
ENTRY_LOG = PROCESSED_DIR / "entry_log.csv"
SENTIMENT_LOG = PROCESSED_DIR / "sentiment_log.csv"
RISK_POLICY_RESULTS = PROCESSED_DIR / "risk_policy_results.csv"
# Phase 3 directional meta-labeling backtest (one row per strategy). Written by
# scripts/evaluate_risk_policies.py --with-meta / coinpredictor.meta_labeling.
META_LABELING_RESULTS_DAILY = PROCESSED_DIR / "meta_labeling_results_1d.csv"
JUDGE_LOG = PROCESSED_DIR / "judge_log.csv"


# --- Cost discipline: one dedicated flag PER paid capability ------------------
# Section 2 of the prompt: everything must run at zero cost by default. Each
# capability that costs real money ships DISABLED behind its OWN boolean flag
# (never one master switch), so the user can turn on exactly one paid thing.
def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable ("1"/"true"/"yes"/"on" => True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PAID_NEWS_ENABLED = _env_bool("COINPREDICTOR_PAID_NEWS", False)      # Phase 2
LLM_SENTIMENT_ENABLED = _env_bool("COINPREDICTOR_LLM_SENTIMENT", False)  # Phase 2
JUDGE_ENABLED = _env_bool("COINPREDICTOR_JUDGE_ENABLED", False)     # Phase 3


# --- Data parameters ---------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Symbols and date range for data ingestion."""

    btc_ticker: str = "BTC-USD"          # yfinance ticker
    start_date: str = "2015-01-01"        # earliest reliable daily BTC data
    interval: str = "1d"                   # daily candles -> next-day horizon

    # --- Exchange-native OHLCV (ccxt, data-quality upgrade) ------------------
    # When True, load_ohlcv() prefers clean exchange candles and only falls back
    # to yfinance if the exchange is unreachable. OKX needs no API key and is not
    # geo-blocked (Binance/Bybit are, in some regions -- see funding.py).
    use_exchange_ohlcv: bool = True
    exchange_id: str = "okx"             # any ccxt exchange id
    exchange_symbol: str = "BTC/USDT"    # ccxt unified spot symbol
    exchange_start: str = "2017-01-01"    # OKX BTC/USDT history depth
    # Intraday timeframe used when comparing daily vs intraday models.
    intraday_interval: str = "1h"

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


# --- Intraday parameters -----------------------------------------------------
@dataclass(frozen=True)
class IntradayConfig:
    """Timeframe-aware settings for the hourly (intraday) pipeline.

    The daily config measures vol in *days*; on hourly candles the same rolling
    windows and annualization would be wrong. These values let the same feature
    builder produce a comparable intraday model: forward realized vol over the
    next ``vol_horizon`` hours, annualized with ~8760 hours/year.
    """

    interval: str = "1h"
    annualization: int = 24 * 365          # ~8760 hourly periods per year
    vol_horizon: int = 24                   # forward window: next 24 hours (~1 day)
    regime_lookback: int = 24 * 7           # trailing norm: past 7 days


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


# --- Trading-cost model ------------------------------------------------------
@dataclass(frozen=True)
class CostConfig:
    """Realistic proportional trading costs charged in every backtest.

    Backtests must NEVER run cost-free by default: fees, half the bid/ask spread
    and slippage are all charged whenever exposure changes, so a strategy has to
    clear a real hurdle before it can PASS the gate.
    """

    taker_fee: float = 0.001       # exchange taker fee (10 bps)
    half_spread: float = 0.0003    # half the bid/ask spread (3 bps)
    slippage: float = 0.0002       # market-impact slippage (2 bps)
    min_notional_usd: float = 10.0  # ignore trades smaller than this

    @property
    def per_side(self) -> float:
        """All-in proportional cost charged on one side of a trade."""
        return self.taker_fee + self.half_spread + self.slippage

    @property
    def round_trip(self) -> float:
        """All-in proportional cost for a full round-trip (enter + exit)."""
        return self.per_side * 2.0


# --- Pre-registered strategy gate --------------------------------------------
@dataclass(frozen=True)
class GateConfig:
    """Pre-registered go/no-go criteria a strategy must clear to be trusted.

    Registering these thresholds BEFORE searching guards against p-hacking: a
    variant only PASSes if it beats the benchmark Sharpe net of costs, survives
    deflation (Deflated Sharpe probability >= threshold, which discounts the
    number of trials), fits the drawdown budget and earns a positive net return.
    """

    min_deflated_sharpe_prob: float = 0.95      # deflated-Sharpe confidence
    require_beat_benchmark_sharpe: bool = True  # must beat the benchmark Sharpe
    max_drawdown_limit: float = 0.20            # drawdown budget (20%)
    min_net_total_return: float = 0.0           # net return must be positive


# --- Cross-sectional crypto momentum -----------------------------------------
@dataclass(frozen=True)
class MomentumConfig:
    """Cross-sectional momentum over a liquid crypto universe."""

    universe: tuple[str, ...] = (
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        "LTC/USDT", "TRX/USDT", "ATOM/USDT", "ETC/USDT", "XLM/USDT",
        "BCH/USDT", "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
    )
    exchange_id: str = "okx"
    timeframe: str = "1d"
    lookbacks: tuple[int, ...] = (30, 60, 90)   # momentum formation windows (days)
    skip_days: int = 7                          # skip most-recent days (reversal)
    top_n: tuple[int, ...] = (3, 5)             # long the top-N momentum names
    rebalance_days: tuple[int, ...] = (7, 30)   # rebalance cadence variants
    min_history_days: int = 120                 # require this much history
    vol_lookback: int = 30                      # inverse-vol / vol-target window
    overlay_target_vols: tuple[float, ...] = (0.1, 0.15, 0.2, 0.3)
    regime_ma_days: int = 100                   # BTC trend filter (SMA length)


# --- Cross-asset diversified portfolio ---------------------------------------
@dataclass(frozen=True)
class DiversifiedConfig:
    """Cross-asset portfolio: equities + bonds + gold + a capped crypto sleeve."""

    assets: dict = field(
        default_factory=lambda: {
            "btc": "BTC-USD",     # crypto sleeve
            "equities": "SPY",    # US equities
            "bonds": "TLT",       # long US Treasuries
            "gold": "GLD",        # gold
        }
    )
    crypto_key: str = "btc"
    start_date: str = "2015-01-01"
    periods_per_year: int = 252            # equity trading calendar
    vol_lookback: int = 60                 # inverse-vol / vol-target window
    rebalance_days: int = 21               # ~monthly rebalance
    crypto_cap: float = 0.25               # max crypto weight
    weight_schemes: tuple[str, ...] = ("equal", "inverse_vol")
    portfolio_target_vols: tuple[float | None, ...] = (None, 0.08, 0.10, 0.12)


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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# --- Trend-regime family (Phase 1b) -----------------------------------------
@dataclass(frozen=True)
class TrendConfig:
    """Trend-regime classifier settings (ALCISTA / BAJISTA / LATERAL).

    NOTE: this is a DIFFERENT concept from the volatility regime (ELEVATED /
    CALM). Never reuse the ``regime_pred`` column here -- this family owns
    ``trend_regime_pred`` in its own ``trend_regime_log.csv``.
    """

    # Forward horizon (days) over which the trend label is measured. Kept equal
    # to the volatility horizon so all families score on the same target_date.
    horizon: int = 5

    # The LATERAL band is defined dynamically as ``band_vol_mult`` times the
    # trailing daily volatility scaled to the horizon, rather than a fixed % --
    # a static "+/-3%" band would call every move ALCISTA in a calm regime and
    # every move LATERAL in a turbulent one. Scaling by realized vol keeps the
    # three classes roughly balanced across regimes.
    band_vol_mult: float = 1.0

    model_filename: str = "btc_trend_lgbm.pkl"


# --- Entry family (Phase 1c) -------------------------------------------------
@dataclass(frozen=True)
class EntryConfig:
    """Triple-barrier entry classifier settings.

    A long entry taken today is labelled 1 if price touches the take-profit
    barrier before the stop-loss barrier (or timeout) using daily High/Low.
    """

    horizon: int = 10          # max holding period (time barrier), trading days
    tp_pct: float = 0.05       # take-profit barrier: +5% from entry close
    sl_pct: float = 0.05       # stop-loss barrier: -5% from entry close
    model_filename: str = "btc_entry_lgbm.pkl"


# --- Sentiment models (Phase 1e / Phase 2) ----------------------------------
@dataclass(frozen=True)
class SentimentConfig:
    """Daily aggregate news-sentiment scoring settings (score in -1..+1)."""

    horizon: int = 5                       # forward window for eval correlation
    finbert_model: str = "ProsusAI/finbert"  # HF financial-sentiment BERT
    llm_model: str = "claude-sonnet-4-20250514"  # paid tier (flag-gated)
    max_headlines: int = 40                # cap headlines scored per day
    finbert_cache_dir: str = str(MODELS_DIR / "finbert")


# --- LLM Judge layer (Phase 3) ----------------------------------------------
@dataclass(frozen=True)
class JudgeConfig:
    """LLM Judge layer settings. Disabled by default via JUDGE_ENABLED."""

    enabled: bool = JUDGE_ENABLED
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    horizon: int = 5                       # forward window for hypothetical P&L

    # Hard daily spend cap (on TOP of the master flag). run_judge.py checks the
    # cumulative estimated_cost_usd already logged today and skips if over cap,
    # mirroring the "refuse silently-stale-data" guard in ohlcv.py.
    max_daily_cost_usd: float = 0.50

    # Anthropic list prices (USD per million tokens). Used only to ESTIMATE and
    # cap spend locally -- no arithmetic is ever delegated to the LLM itself.
    input_cost_per_mtok: float = 3.0
    output_cost_per_mtok: float = 15.0


DATA = DataConfig()
MODEL = ModelConfig()
INTRADAY = IntradayConfig()
STRATEGY = StrategyConfig()
COSTS = CostConfig()
GATE = GateConfig()
MOMENTUM = MomentumConfig()
DIVERSIFIED = DiversifiedConfig()
FEATURES = FeatureConfig()
TREND = TrendConfig()
ENTRY = EntryConfig()
SENTIMENT = SentimentConfig()
JUDGE = JudgeConfig()
