# ₿ CoinPredictor — Bitcoin Volatility Predictor

Forecasts **Bitcoin's forward realized volatility** (how turbulent the next few
days will be) and the **volatility regime** (calm vs. elevated vs. the recent
norm), using free market data, technical indicators, a parsimonious **HAR-RV**
model (the primary forecaster) and LightGBM. Ships with research notebooks and
an interactive Streamlit dashboard.

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
  Bollinger Bands, ATR, lagged returns, rolling/realized volatility, volume),
  plus **Garman-Klass** range-based volatility and its daily/weekly/monthly
  **HAR** components.
- **Phase 2** — Macro signals (S&P 500, Gold, US Dollar index) and **classical
  volatility baselines** (naive persistence, HAR-RV, GARCH/EGARCH) that every ML
  model must beat.
- **Phase 3** — Sentiment (Crypto Fear & Greed, optional NewsAPI), on-chain
  metrics (hash rate, transaction count, miner revenue), **free derivatives**
  (OKX funding, Deribit DVOL), and **directional meta-labeling**.
- **Robust exchange OHLCV** via ccxt/OKX (daily + hourly), replacing flaky
  yfinance for BTC, with an incremental parquet cache.
- **Honest out-of-sample validation** — purged walk-forward, embargo, and the
  **Deflated Sharpe Ratio** to discount strategies found by trial-and-error.
- **Volatility-targeting backtest** vs buy-and-hold (Sharpe + max drawdown),
  net of realistic commissions and slippage.
- **HAR-RV is the production forecaster**: `predict.py`, the paper-trading bot,
  and the dashboard headline all size exposure from the HAR-RV forecast (the
  purged-walk-forward winner), not the LightGBM artifact.
- **Streamlit dashboard**: live forecast, regime, equity curve, charts,
  importance, risk-policy leaderboard, and a directional meta-labeling panel.
- **Cross-asset diversified portfolio** (equities + long bonds + gold + a capped
  BTC sleeve) that PASSes the pre-registered gate net of costs, with per-year /
  crisis-window sensitivity, its own dashboard, and a prospective paper gate.

## Project structure

```
CoinPredictor/
├── data/{raw,processed}/        # cached datasets (gitignored)
├── models/                      # trained artifacts (gitignored)
├── notebooks/                   # research workflow
├── src/coinpredictor/
│   ├── config.py                # paths, symbols, vol/intraday & strategy settings
│   ├── data/                    # ohlcv, exchange_ohlcv, macro, onchain,
│   │                            #   sentiment, funding, implied_vol loaders
│   ├── features.py              # indicators + GK/HAR + forward-vol targets (no leakage)
│   ├── model.py                 # vol regressor + regime classifier, walk-forward CV
│   ├── vol_baselines.py         # classical HAR-RV / GARCH baselines (Phase 2)
│   ├── meta_labeling.py         # directional trend + meta-label filter (Phase 3)
│   ├── diversified.py           # cross-asset portfolio search + sensitivity + gate
│   ├── validation.py            # purged walk-forward + Deflated Sharpe
│   ├── registry.py              # model families run + logged side by side
│   ├── backtest.py              # volatility-targeting strategy vs buy-and-hold
│   └── predict.py               # live forward-volatility forecast
├── scripts/diagnose.py          # Phase 0 honest diagnostic (leakage / baseline / edge)
├── dashboard/app.py             # Streamlit UI
├── dashboard/diversified_app.py # Streamlit UI for the diversified portfolio
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

### Research & evaluation entry points

```powershell
# Phase 0 — honest diagnostic: leakage check, baseline gap, strategy vs
# buy-and-hold net of costs + Deflated Sharpe (add --intraday for hourly)
python scripts/diagnose.py
python scripts/diagnose.py --intraday

# Phase 2 — classical volatility baselines vs LightGBM (purged walk-forward)
python -m coinpredictor.vol_baselines            # daily
python -m coinpredictor.vol_baselines --intraday # hourly
python -m coinpredictor.vol_baselines --egarch   # EGARCH instead of GARCH

# Phase 3 — directional meta-labeling (trend primary + meta-label filter)
python -m coinpredictor.meta_labeling                 # daily, fixed barriers
python -m coinpredictor.meta_labeling --vol-scaled    # volatility-scaled barriers
python -m coinpredictor.meta_labeling --intraday      # hourly
python -m coinpredictor.meta_labeling --derivatives   # add free funding + DVOL

# Risk policies (sizing) + optionally the defensive meta-labeling strategy
python scripts/evaluate_risk_policies.py              # long-only sizing policies
python scripts/evaluate_risk_policies.py --with-meta  # also refresh meta-labeling
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

## Honest evaluation & findings

The project is validated the hard way: a **purged** walk-forward (with an
embargo around each test fold so the forward-looking target can't leak into
training) plus the **Deflated Sharpe Ratio**, which discounts a strategy's
Sharpe by how many variants were tried before finding it. Backtests are net of
realistic commissions and slippage. Run `python scripts/diagnose.py` to
reproduce the numbers below.

What the evidence actually says (BTC, out-of-sample):

- **Volatility level is predictable — and a simple model wins.** A parsimonious
  **HAR-RV** (Corsi) over Garman-Klass daily/weekly/monthly components beats the
  LightGBM regressor on both daily (R² ≈ +0.13 vs ≈ 0) and hourly (R² ≈ +0.36
  vs +0.32) data. HAR-RV is therefore the **primary** volatility model; GARCH(1,1)
  underperforms even naive persistence on daily data.
- **Direction is hard.** A daily trend filter (SMA 20/50) roughly matches
  buy-and-hold on Sharpe while cutting max drawdown from ≈ −77 % to ≈ −59 %.
  Adding a meta-label filter with volatility-scaled barriers cuts drawdown
  further (to ≈ −20 %) but does **not** beat buy-and-hold on Sharpe, and the
  intraday directional signal fails outright.
- **Bottom line:** the robust wins so far are a *better volatility model*
  (HAR-RV) and *drawdown reduction*, not a directional edge. Free derivatives
  (funding, DVOL) did not unlock one — so paying for premium data is **not yet
  justified** by the evidence.

HAR-RV is wired end-to-end: the live prediction, the paper-trading bot, and the
dashboard headline all use it via `predict_primary_vol`, and the defensive
directional meta-labeling strategy is refreshed with
`python scripts/evaluate_risk_policies.py --with-meta` and shown in the
dashboard's Track Record tab.

## Data sources (all free)

| Source | Used for | Key required |
| --- | --- | --- |
| ccxt / OKX | BTC OHLCV (daily + hourly), funding | No |
| yfinance | Macro (S&P 500, Gold, DXY), OHLCV fallback | No |
| Deribit | BTC implied volatility (DVOL) | No |
| blockchain.info | On-chain metrics | No |
| alternative.me | Fear & Greed index | No |
| NewsAPI | Headline counts | Free key (optional) |
| CoinDesk / Cointelegraph RSS | News headlines (sentiment) | No |

## Hierarchical multi-model architecture

Beyond the original volatility model, CoinPredictor runs several **model
families** side by side. Each family has its OWN csv log, its OWN required
naive/free baseline, and its OWN leaderboard metric (one leaderboard per family
in the dashboard's Track Record tab).

| Family | File | Baseline | Primary model | Metric |
| --- | --- | --- | --- | --- |
| volatility | `volatility_log.csv` | `naive_persistence_v1` | `har_rv_volatility_v1` | MAE / RMSE |
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

## Cross-asset diversified portfolio (the drawdown fix)

Every crypto-*only* experiment hit the same wall: crypto is a single correlated
risk factor, so no risk management *inside* crypto pulls its drawdown under a
20 % budget. The evidence-based fix is diversification across *uncorrelated*
factors — equities (SPY) + long bonds (TLT) + gold (GLD) + a **capped** BTC
sleeve — weighted equally or by inverse volatility, optionally throttled toward a
portfolio volatility target, and scored with the SAME cost model and
pre-registered gate against simply holding BTC.

```powershell
python -m coinpredictor.diversified                # full offline search on real data
python -m coinpredictor.diversified --sensitivity  # + per-year & crisis-window stress
streamlit run dashboard/diversified_app.py         # equity vs BTC vs 60/40, drawdown, weights
```

**Result (2015→today, net of costs): PASS.** The pre-registered winner
`equal_vt0.08` (equal-weight, throttled to 8 % portfolio vol) beats holding BTC
on Sharpe (**1.21 vs 1.03**) while cutting max drawdown from **≈ −83 % to
≈ −18 %**, and survives deflation. A PASS means *the diversified thesis earns a
broker* — not "trade it tomorrow".

### Recommended venue/broker (Mexico)

Trading the equity/bond/gold legs needs a broker. For a Mexico-based deployment:

| Sleeve | Venue | Why |
| --- | --- | --- |
| SPY / TLT / GLD | **Interactive Brokers** | Real US-listed ETFs (deep liquidity, tight spreads), **fractional shares** for a ~monthly rebalanced small book, commissions ≈ **5 bps** — comfortably under the **15 bps** modelled in `COSTS`. MXN funding via wire. |
| BTC sleeve | **Bitso** | Regulated Mexican exchange, MXN on/off-ramp. Mind the wider spread vs the 15 bps model. |

> Alternative single-domicile route: **GBM+** (US ETFs via the SIC) + **Bitso**
> for BTC. Simpler onboarding, but SIC tickers differ and spreads/fees are wider,
> so verify real fills stay under the modelled 15 bps per side.

### Sub-period & crisis-window sensitivity

The whole point of diversifying is surviving regimes where one factor blows up.
`--sensitivity` splits the record by calendar year and by pre-registered crisis
windows (saved to `data/processed/diversified_sensitivity_{annual,stress}.csv`):

- **2022 — the acid test** (stocks *and* long bonds fell together, so the classic
  60/40 hedge failed): the diversified book still contained its drawdown to
  **≈ −14 %** with a **−12 %** return, versus **−67 % / −64 %** for BTC and
  **≈ −26 % / −22 %** for 60/40. The vol-target throttle plus the crypto cap keep
  it well inside the 20 % budget even in the worst bond regime in decades.
- Across every year, the diversified variant's drawdown stays shallow while its
  risk-adjusted return tracks or beats both benchmarks outside raging bull runs.

### Prospective paper gate (live, forward-only)

An offline PASS is in-sample by construction. The honest next step is a
*prospective* paper run: freeze the winning variant, then track a **simulated**
multi-asset book (cash + fractional units, same `COSTS`) against 100 % BTC and
60/40 on live prices — one rebalance per invocation — and re-score the SAME gate
on the purely out-of-sample record once ≥ 30 live observations accumulate
(`n_trials=1`: the variant is frozen, so there is nothing to deflate).

```powershell
python -m coinpredictor.trading.diversified_paper            # one live rebalance
python -m coinpredictor.trading.diversified_paper --status   # print the track record
python -m coinpredictor.trading.diversified_paper --gate     # score the prospective gate
```

> 📝 **100 % paper trading** — state lives in
> `data/processed/diversified_paper_state.json` (gitignored) and never touches
> real funds. Delete it to restart the 6-week forward run from scratch.


