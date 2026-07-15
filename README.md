# ₿ CoinPredictor — Bitcoin Volatility Predictor

Forecasts **Bitcoin's forward realized volatility** (how turbulent the next few
days will be) and the **volatility regime** (calm vs. elevated vs. the recent
norm), using free market data, technical indicators, and LightGBM. Ships with
research notebooks and an interactive Streamlit dashboard.

> ⚠️ **Educational project, not financial advice.** Unlike price *direction*
> (which is close to a coin flip), Bitcoin *volatility clusters* and is far more
> predictable. The model turns that signal into a **volatility-targeting
> strategy** designed to improve risk-adjusted return and cut drawdowns.

## What it predicts

- **`target_vol`** — forward `vol_horizon`-day annualized realized volatility
  (regression).
- **`target_high_vol`** — whether the coming period is more volatile than the
  recent trailing norm (regime classification).

## Features

- **Phase 1** — OHLCV + technical indicators (RSI, MACD, moving averages,
  Bollinger Bands, ATR, lagged returns, rolling/realized volatility, volume).
- **Phase 2** — Macro signals (S&P 500, Gold, US Dollar index) via yfinance.
- **Phase 3** — Sentiment (Crypto Fear & Greed, optional NewsAPI) and on-chain
  metrics (hash rate, transaction count, miner revenue) via blockchain.info.
- **Walk-forward validation** (`TimeSeriesSplit`) and explicit no-leakage design.
- **Volatility-targeting backtest** vs buy-and-hold (Sharpe + max drawdown).
- **Streamlit dashboard**: live forecast, regime, equity curve, charts, importance.

## Project structure

```
CoinPredictor/
├── data/{raw,processed}/        # cached datasets (gitignored)
├── models/                      # trained artifacts (gitignored)
├── notebooks/                   # research workflow
├── src/coinpredictor/
│   ├── config.py                # paths, symbols, vol & strategy settings
│   ├── data/                    # ohlcv, macro, onchain, sentiment loaders
│   ├── features.py              # indicators + forward-vol targets (no leakage)
│   ├── model.py                 # vol regressor + regime classifier, walk-forward CV
│   ├── backtest.py              # volatility-targeting strategy vs buy-and-hold
│   └── predict.py               # live forward-volatility forecast
├── dashboard/app.py             # Streamlit UI
└── tests/                       # leakage / feature / model / backtest tests
```

## Setup

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (optional) enable news sentiment
Copy-Item .env.example .env   # then add NEWSAPI_KEY
```

## Deploy on a new machine

A one-shot script creates the virtual environment, installs everything, and
registers the daily paper-trading task:

```powershell
git clone https://github.com/<your-user>/CoinPredictor.git
cd CoinPredictor
powershell -ExecutionPolicy Bypass -File scripts\setup_machine.ps1
```

Customize the schedule, risk profile, or starting capital:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_machine.ps1 -Time 09:30 -Profile defensive -Capital 5000
```

> 📝 **100% paper trading** — the bot simulates a portfolio and never touches
> real funds. State is kept in `data/processed/paper_state.json` (gitignored).

Manage the scheduled task afterwards:

```powershell
Get-ScheduledTask -TaskName CoinPredictorBot                       # status
powershell -ExecutionPolicy Bypass -File scripts\run_bot.ps1       # run now
Get-Content logs\bot.log -Tail 20                                  # history
Unregister-ScheduledTask -TaskName CoinPredictorBot -Confirm:$false # remove
```

## Usage

```powershell
# 1. Train (downloads data, validates, saves vol regressor + regime classifier)
python -m coinpredictor.model

# 2. Get the forward-volatility forecast & regime
python -m coinpredictor.predict

# 3. Backtest the volatility-targeting strategy
python -m coinpredictor.backtest

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

### Using extra feature phases

```python
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_features_full

feats = build_features_full(
    load_ohlcv(),
    use_macro=True,       # Phase 2
    use_onchain=True,     # Phase 3
    use_sentiment=True,   # Phase 3
)
```

## Testing

```powershell
pytest
```

Tests run fully offline on synthetic data and include a leakage guard that
verifies future prices never influence past feature values.

## How leakage is avoided

- Every feature on day `t` uses only data up to and including `t`.
- The targets are the **only** forward-looking columns (forward realized
  volatility over the next `vol_horizon` days).
- Validation always trains on the past and tests on the future
  (`TimeSeriesSplit`); data is never shuffled.

## Data sources (all free)

| Source | Used for | Key required |
| --- | --- | --- |
| yfinance | BTC OHLCV, macro | No |
| blockchain.info | On-chain metrics | No |
| alternative.me | Fear & Greed index | No |
| NewsAPI | Headline counts | Free key (optional) |
| CoinDesk / Cointelegraph RSS | News headlines (sentiment) | No |

## Hierarchical multi-model architecture

Beyond the original volatility model, CoinPredictor runs several **model
families** side by side. Each family has its OWN csv log, its OWN required
naive/free baseline, and its OWN leaderboard metric (one leaderboard per family
in the dashboard's Track Record tab).

| Family | File | Baseline | Real model | Metric |
| --- | --- | --- | --- | --- |
| volatility | `volatility_log.csv` | `naive_persistence_v1` | `lgbm_volatility_v1` | MAE / RMSE |
| trend regime | `trend_regime_log.csv` | `sma_cross_trend_v1` | `lgbm_trend_v1` | accuracy + per-class F1 |
| entry | `entry_log.csv` | `baseline_entry_v1` | `lgbm_entry_v1` | precision / recall / calibration |
| sentiment | `sentiment_log.csv` | `lexicon_sentiment_v1` | `finbert_sentiment_v1` | corr(score, fwd return) |
| risk | `risk_policy_results.csv` | `buy_and_hold` | vol-target / Kelly | Sharpe / maxDD / Calmar |

- **Trend regime** (`ALCISTA` / `BAJISTA` / `LATERAL`) is a *different* concept
  from the volatility regime (`ELEVATED` / `CALM`) — separate columns, separate
  file, no naming collision.
- **Entry** uses **triple-barrier** labelling (take-profit / stop-loss / time,
  using daily High/Low) to predict whether a long taken today would win.
- **Risk** is not a daily row: `scripts/evaluate_risk_policies.py` replays each
  position-sizing policy over history and writes one portfolio-level row each.

Daily cron (unchanged, twice-daily, free):

```bash
python scripts/log_prediction.py       # one row per model into its family file
python scripts/evaluate_predictions.py # scores rows whose target_date has passed
```

One-off migration from the legacy single log:

```bash
python scripts/migrate_split_by_target_type.py   # prediction_log.csv -> volatility_log.csv
```

## Cost discipline & paid capabilities (all OFF by default)

Everything runs at **zero external API cost** by default. Each paid capability
has its OWN dedicated flag (never one master switch) — see `.env.example`:

| Flag | Enables | Requires |
| --- | --- | --- |
| `COINPREDICTOR_PAID_NEWS` | paid news provider on top of free RSS | `CRYPTOPANIC_KEY` or `NEWSAPI_KEY` |
| `COINPREDICTOR_LLM_SENTIMENT` | Claude sentiment (Tier 3) | `ANTHROPIC_API_KEY` |
| `COINPREDICTOR_JUDGE_ENABLED` | the LLM Judge layer (Phase 3) | `ANTHROPIC_API_KEY` |

If a flag is ON but its key is missing, the code **fails loudly** rather than
silently skipping.

### Sentiment tiers (news → score)

News headlines are pulled free from CoinDesk / Cointelegraph RSS (no key). Three
scoring tiers:

1. **Lexicon** (`lexicon_sentiment_v1`) — curated keyword lexicon. Deterministic,
   dependency-free, always available. The zero-cost floor.
2. **FinBERT** (`finbert_sentiment_v1`) — HuggingFace `ProsusAI/finbert`, runs
   locally on CPU. **Recommended real model.** One-time download ≈ **440 MB**;
   first run without a cached model takes a few minutes. In Docker it is
   pre-downloaded at build time (`HF_HOME=/app/.cache/huggingface`) so the daily
   job never fetches it. For local dev, `pip install torch` to enable it —
   otherwise it degrades gracefully to the lexicon tier.
3. **Claude** (`llm_sentiment_v1`) — paid, gated by `COINPREDICTOR_LLM_SENTIMENT`.
   Not registered in `MODELS` by default.

## LLM Judge layer (Phase 3, disabled by default)

A **separate, non-deterministic** decision layer (`src/coinpredictor/judges.py`)
that consumes each family's *primary* model output plus raw indicators and
sentiment, and emits a `BUY` / `HOLD` / `SELL` verdict with reasoning. It is
architecturally separate from `MODELS` (judged on decision quality, never on
MAE/accuracy) and **never** appears in the ML leaderboards.

- Disabled by default (`COINPREDICTOR_JUDGE_ENABLED=false`) — makes zero API
  calls until you flip the flag.
- Its own once-daily, cost-bearing cron entry (`scripts/run_judge.py`), **not**
  merged into the twice-daily free job.
- Hard daily spend cap (`JudgeConfig.max_daily_cost_usd`) on top of the flag.
- **Never** asked to do arithmetic — it only reasons qualitatively over
  already-computed numbers (all math stays in Python / the ML models).

```bash
python scripts/run_judge.py        # only runs if the flag is on; respects cost cap
python scripts/evaluate_judges.py  # hypothetical P&L, hit rate, consistency
```

In Docker, use the dedicated wrapper (its own once-daily crontab line, separate
from the free twice-daily `run_daily_docker.sh`):

```bash
scripts/run_judge_docker.sh
```

Verdicts appear in the dashboard's **LLM Judges** tab (hypothetical equity,
cumulative cost, hit rate, same-day consistency) — kept fully separate from the
ML model leaderboards.

