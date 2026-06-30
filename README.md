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
