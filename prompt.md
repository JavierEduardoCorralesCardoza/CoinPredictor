# CoinPredictor — Hierarchical Multi-Model Architecture + LLM Judge Layer
---

## 0. CONTEXT: WHAT ALREADY EXISTS (read this, then verify against the repo)

CoinPredictor is a BTC volatility forecasting system running in Docker on a
personal server, deployed via cron. It already has:

- `src/coinpredictor/data/ohlcv.py` — OHLCV ingestion via yfinance, with
  retry/backoff (5 attempts, exponential) and a stale-cache fallback
  (refuses to serve cached data older than 2 days).
- `src/coinpredictor/predict.py`, `model.py` — a LightGBM volatility
  regressor + regime (ELEVATED/CALM) classifier, trained via walk-forward
  `TimeSeriesSplit`.
- `src/coinpredictor/registry.py` — a `ModelAdapter` pattern: any model
  implements `.name`, `.target_type`, `.predict(refresh: bool) -> dict`,
  and gets added to a `MODELS: list[ModelAdapter]` registry. Currently
  contains `LGBMVolatilityAdapter` (target_type="volatility") and
  `NaiveVolatilityAdapter` (persistence baseline — every new model family
  MUST ship with a naive/rule-based baseline in the same family, this is a
  hard project convention, not optional).
- `scripts/log_prediction.py` — iterates `MODELS`, logs one row per model
  per day, with duplicate protection keyed on `(as_of_date, model_name)`.
- `scripts/evaluate_predictions.py` — once `target_date` has passed, scores
  each pending row via a `_EVALUATORS: dict[str, Callable]` dispatch table
  keyed by `target_type`. Currently only `"volatility"` is implemented.
- `dashboard/app.py` — Streamlit, tabs: Charts / Backtest / Explainability /
  Track Record. Track Record has a leaderboard and comparison charts.
- Deployment: Docker Compose (`coinpredictor` = on-demand CLI container run
  by host cron; `coinpredictor-dashboard` = always-on Streamlit), cron runs
  `run_daily_docker.sh` twice daily (09:30 and 21:30 UTC — the second run
  is a safety net for Yahoo Finance data-publication lag).

**IMPORTANT — CSV architecture is changing in this prompt (see Section 3).**
The current single `data/processed/prediction_log.csv` (long format, one
row per `(as_of_date, model_name)`, with a `target_type` column) becomes
`data/processed/volatility_log.csv` after the Phase 1 migration — new
families each get their OWN csv file. Do not assume the old single-file
schema still applies once Phase 1 starts.

**Before writing any code**, inspect `config.py`, `backtest.py`, and
`features.py` in full — they are referenced throughout this prompt but
their exact contents are assumed, not guaranteed. Adapt field/function
names to match reality; note any assumption you had to override.

---

## 1. NAMING COLLISION — FIX THIS FIRST, IT'S NOT OPTIONAL

The existing `regime_pred` column refers to **volatility regime**
(ELEVATED/CALM). The new "trend regime" model below (bullish/bearish/
sideways) is a **completely different concept** and MUST NOT reuse that
column or any name that could collide with it.

- Existing volatility regime → refer to it as **`vol_regime`** in all new
  code, comments, and dashboard labels (the underlying CSV column can stay
  `regime_pred`/`actual_regime`/`regime_correct` inside `volatility_log.csv`
  for backward compat).
- New trend regime (its own file, `trend_regime_log.csv`) → columns
  **`trend_regime_pred`**, **`trend_regime_actual`**,
  **`trend_regime_correct`** (values: `"ALCISTA"`, `"BAJISTA"`, `"LATERAL"`).

---

## 2. COST DISCIPLINE — GLOBAL PRINCIPLE, APPLIES TO EVERYTHING BELOW

**Everything must run at zero cost by default.** Every capability that
costs real money (a paid data API, an LLM API call of any kind) ships
DISABLED, gated behind its OWN dedicated boolean flag — not one master
switch, one flag PER paid capability, so the user can turn on exactly the
one thing they want to pay for and nothing else. Proposed flags (adapt
names to whatever config pattern the repo already uses — env vars or a
`config.py` dataclass, match existing convention):

```python
COINPREDICTOR_PAID_NEWS = False      # paid news API for sentiment (Phase 2)
COINPREDICTOR_LLM_SENTIMENT = False  # use Claude for sentiment instead of
                                      # the free FinBERT tier (Phase 2)
COINPREDICTOR_JUDGE_ENABLED = False  # run the LLM Judge layer at all (Phase 3)
```

With all three flags OFF (the shipped default), the entire system —
including everything built in this prompt — must run with **zero external
API calls that cost money**. Every phase's Definition of Done includes
verifying this explicitly.

---

## 3. TARGET ARCHITECTURE

```
                    Data (OHLCV + news, existing + new)
                                    |
                                    v
                     Feature engineering (existing, extend as needed)
                                    |
        +--------------+-----------+-----------+--------------+
        v              v           v           v              v
   volatility    trend_regime   entry        risk         sentiment
   (existing)      (NEW)       (NEW)        (NEW)          (NEW)
        |              |           |           |              |
   volatility_    trend_regime  entry_log   risk_policy   sentiment_
   log.csv         _log.csv      .csv       _results.csv   log.csv
        |              |           |           |              |
        +--------------+-----------+-----------+--------------+
                     MODELS registry (existing pattern, extended)
                     -- each family: its OWN csv file, its OWN
                        naive/rule baseline, its OWN metric
                     -- risk is NOT a daily row (policy backtest,
                        see 4c) -- already its own file/script
                                    |
                                    v
                 +---------------------------------------+
                 |  JUDGES registry -- SEPARATE module,   |
                 |  SEPARATE file, SEPARATE csv           |
                 |  (judge_log.csv). Disabled by default   |
                 |  (COINPREDICTOR_JUDGE_ENABLED=False).   |
                 |  Consumes the PRIMARY model's output    |
                 |  from each family above + raw           |
                 |  technical indicators + sentiment, and  |
                 |  produces a final BUY/HOLD/SELL         |
                 |  verdict with reasoning.                 |
                 |  NEVER appears in the ML leaderboards -- |
                 |  evaluated on decision quality, not on   |
                 |  MAE/accuracy, and it is NON-            |
                 |  DETERMINISTIC.                          |
                 +---------------------------------------+
                                    |
                                    v
                    Dashboard: one leaderboard per csv file/family
                    + a separate "LLM Judges" tab
```

**Why one CSV per family (changed from the original single-file design):**
Each family has a genuinely different row shape (volatility needs
predicted/actual vol; entry needs TP/SL fields; sentiment has no
per-row "correct" at all, only a correlation computed later). Cramming
all of them into one wide table with a `target_type` discriminator means
most columns are empty most of the time. `risk` and `judges` already
needed their own files for the same reason — applying that consistently
to all families keeps every file's schema legible on its own.

**Why judges stay architecturally separate from `MODELS`:** ML models
here are deterministic functions scored against a single realized
numeric/categorical outcome. Judges are non-deterministic agents scored
on decision quality (hypothetical P&L, hit rate, run-to-run consistency,
cost, latency) — mixing them would corrupt the leaderboard math.

---

## PHASE 1 — New ML model families + registry/evaluator/dashboard extension

### 1a. Migration (do this first, before adding anything new)
New script `scripts/migrate_split_by_target_type.py` (follow the same
"safe to run more than once, no-op if already migrated" pattern as the
existing `migrate_add_model_name.py`): copy the current
`prediction_log.csv` to `data/processed/volatility_log.csv`, drop the
now-redundant `target_type` column (the filename encodes it), keep
`model_name`. Do not delete the original file until the new one is
verified.

### 1b. `trend_regime` family — file: `data/processed/trend_regime_log.csv`
Predicts market trend over the same horizon as `MODEL.vol_horizon` (or a
separate configurable horizon if that reads more naturally from
`config.py` — your call, document it).

- **Baseline (required):** `SmaCrossTrendAdapter` — rule-based, zero
  training: `ALCISTA` if `sma_20 > sma_50` and rising, `BAJISTA` if
  inverse, `LATERAL` otherwise (reuse existing `sma_20`/`sma_50` features).
- **Real model:** `LGBMTrendRegimeAdapter` — LightGBM classifier
  (3-class), same walk-forward `TimeSeriesSplit` discipline as the
  existing volatility classifier. Label construction: over the forward
  horizon, `ALCISTA` if cumulative return > +X%, `BAJISTA` if < -X%,
  `LATERAL` otherwise (pick X empirically, e.g. via trailing volatility;
  justify the choice in a comment, don't hardcode a magic number blindly).
- **Evaluator** (`_evaluate_trend_regime_row`): compute the realized
  label with the same rule used for training labels, fill
  `trend_regime_actual` / `trend_regime_correct`.
- **Leaderboard metric:** accuracy + per-class F1 (3-class confusion is
  more informative than accuracy alone — surface it in the dashboard).

### 1c. `entry` family — file: `data/processed/entry_log.csv`
Predicts the probability that a long entry taken today would be
profitable, using **triple-barrier labeling** (take-profit / stop-loss /
time-horizon — whichever is touched first, using daily High/Low, not just
Close — `ohlcv.py` already loads High/Low, this is where they finally
get used).

- **Baseline (required):** `RandomEntryAdapter` (flat `entry_proba = 0.5`)
  or another honestly zero-skill rule — pick whichever is the fairer floor.
- **Real model:** `LGBMEntryAdapter` — binary classifier trained on
  triple-barrier labels (TP hit first = 1, SL hit first or timeout = 0).
  Expose `entry_proba` (float 0-1) plus the TP%/SL%/horizon used, so the
  dashboard can show concretely what "entry" means.
- **Evaluator:** walk forward from `as_of_date` using High/Low, resolve
  which barrier is touched first (or timeout), fill `entry_actual` (0/1)
  and `entry_correct`.
- **Leaderboard metric:** precision/recall/AUC, plus calibration (does
  "70% proba" actually win ~70% of the time?).

### 1d. `risk` family — file: `data/processed/risk_policy_results.csv`
**Different evaluation shape — not a per-day row.** A position-sizing/
stop policy, evaluated by replaying it over historical data and looking
at portfolio-level outcomes.

- Extend the existing `backtest.py` (`recommend_weight` already exists —
  study it first) so sizing policies are pluggable:
  ```python
  class RiskPolicy(Protocol):
      name: str
      def size(self, predicted_vol: float, regime_proba: float | None,
                trend_regime: str | None) -> float: ...  # -> [0,1] BTC weight
  ```
- **Baseline (required):** `FixedWeightPolicy` (always 100% — this IS
  your existing buy-and-hold benchmark, reuse it explicitly).
- **Candidates:** the existing volatility-targeting policy (refactored
  into this interface), a Kelly-fraction policy, maybe an ATR-based stop.
- **New script:** `scripts/evaluate_risk_policies.py` — runs each policy
  through `walk_forward_backtest` (or equivalent), writes one row per
  policy (sharpe, max_dd, calmar, total_return) to
  `risk_policy_results.csv`.

### 1e. `sentiment` family — file: `data/processed/sentiment_log.csv`
Predicts a daily aggregate sentiment score for BTC news (`sentiment_score`
float -1..+1, plus `sentiment_label`). **Three tiers, all free by default
except the third:**

- **Tier 1 — baseline (required):** `LexiconSentimentAdapter` — VADER or
  a small curated crypto-finance keyword lexicon. No model download, no
  API, deterministic. Zero-cost floor.
- **Tier 2 — specialized free (required, this is the new recommended
  default "real" model):** `FinBERTSentimentAdapter` — FinBERT
  (Hugging Face `ProsusAI/finbert` or equivalent), a small BERT-based
  model purpose-built for financial-text sentiment. Runs locally on CPU
  (no GPU needed, light enough for this server), zero API cost once
  downloaded. This is the standard baseline the financial-NLP literature
  itself compares newer approaches against — it's a legitimate "real"
  model, not a toy.
- **Tier 3 — paid, flag-gated (`COINPREDICTOR_LLM_SENTIMENT`):**
  `LLMSentimentAdapter` — Claude via the Anthropic API, given the day's
  headlines, asked to score sentiment. Only runs if the flag is on and
  `ANTHROPIC_API_KEY` is present; fails loudly and clearly if the flag is
  on but the key is missing, never silently falls back.
  **Do not build or recommend a separate "financial-specialized paid API"
  for this** — evaluate this decision as already settled: general
  frontier LLMs with a well-designed prompt outperform finance-tuned
  models on this exact task in the literature, so Claude (already the
  project's paid-tier choice for judges) is also the right paid tier here.
  No new vendor needed.
- **Evaluator** (important nuance): there's no clean "actual sentiment"
  to compare against. Instead, fill `sentiment_fwd_return` and/or
  `sentiment_fwd_vol` once the horizon passes (reuse `realized_vol`, add
  a realized-return equivalent). Don't compute "correct/incorrect" per row.
- **Leaderboard metric:** computed at report time (like MAE is today) as
  a **correlation coefficient** between `sentiment_score` and realized
  forward return/vol across all evaluated rows for that model.

### 1f. Registry/evaluator/dashboard wiring
- Extend `registry.py`'s `MODELS` list with every adapter above except
  `risk` (handled separately per 1d). Add a
  `LOG_FILE_BY_TARGET_TYPE: dict[str, Path]` mapping so
  `log_prediction.py` routes each adapter's row to the correct csv file.
- Extend `_EVALUATORS` in `evaluate_predictions.py` with
  `"trend_regime"`, `"entry"`, `"sentiment"` handlers. Restructure the
  script to loop over `LOG_FILE_BY_TARGET_TYPE.items()`, reading/writing
  each family's own file with its own evaluator — keep this as ONE script/
  ONE cron step, just internally iterating multiple files now.
- Also define `PRIMARY_MODEL: dict[str, str]` in `registry.py` (e.g.
  `{"volatility": "lgbm_volatility_v1", "trend_regime": "lgbm_trend_v1",
  "entry": "lgbm_entry_v1", "sentiment": "finbert_sentiment_v1"}`) — this
  is what Phase 3's judge will read from later; defining it now avoids a
  rework.
- Dashboard (`dashboard/app.py`, Track Record tab): one leaderboard table
  per family/file, each with its own appropriate metric columns as
  specified above, each showing its naive/free baseline for comparison.

### Phase 1 — Definition of Done
- [ ] `python3 -m py_compile` passes on every touched/new file.
- [ ] `docker compose build` succeeds for both services.
- [ ] Migration script runs cleanly, `volatility_log.csv` has the same
      row count/content as the old `prediction_log.csv` minus the
      redundant column.
- [ ] Running `log_prediction.py` once logs into the correct per-family
      file for every registered model (minus `risk`), dedup still works.
- [ ] Running `evaluate_predictions.py` doesn't crash on any file with no
      resolvable rows yet (pending stays pending, no exceptions).
- [ ] Dashboard loads (`curl localhost:8501` -> 200), shows 4 separate
      leaderboards, each with a naive/free baseline visible.
- [ ] **With all three cost flags OFF, zero paid API calls occur anywhere
      in a full `run_daily_docker.sh` run** — verify this explicitly,
      e.g. by running with no `ANTHROPIC_API_KEY`/news key set at all and
      confirming nothing errors due to a missing paid credential.
- [ ] No existing `lgbm_volatility_v1` / `naive_persistence_v1` behavior
      changed or broke.

---

## PHASE 2 — Sentiment data sources (free by default, paid via flag)

- **Default (`COINPREDICTOR_PAID_NEWS=False`):** free sources only — RSS
  feeds (CoinDesk, Cointelegraph) and/or CryptoPanic's free tier. No key
  required, or a free-tier key in `.env` (already gitignored).
- **Flag on:** additionally query a paid provider (NewsAPI.org,
  CryptoPanic Pro). Read the key from `.env`; if the flag is on but the
  key is missing, fail loudly and clearly, don't silently skip.
- Both paths feed the same three sentiment adapters from 1e — the flag
  only changes how many/which headlines get collected.
- Add retry/backoff for whichever HTTP client is used, mirroring the
  discipline already in `ohlcv.py` — a flaky news source shouldn't crash
  the daily cron.
- FinBERT (Tier 2) needs a one-time model download (`transformers`
  library, `ProsusAI/finbert` or similar) — document the disk space and
  first-run download time in `README.md`, and make sure it's cached in
  the Docker image build (not re-downloaded every run).

### Phase 2 — Definition of Done
- [ ] Works end-to-end with both flags OFF, zero paid credentials present,
      FinBERT running locally with no network call beyond the RSS pull.
- [ ] With a flag ON and a placeholder/missing key, fails loudly in logs,
      does not crash the rest of `run_daily_docker.sh`.
- [ ] Document exact env vars and FinBERT setup in `README.md` or
      `docs/sentiment_setup.md`.

---

## PHASE 3 — LLM Judge layer (Claude only, disabled by default, pluggable)

**Entirely separate from `registry.py`/`MODELS`.** New file:
`src/coinpredictor/judges.py`. **Ships with `COINPREDICTOR_JUDGE_ENABLED
= False`** — this phase, once merged, must not start costing money until
the human explicitly flips the flag.

```python
@dataclass
class JudgeVerdict:
    action: str              # "BUY" | "HOLD" | "SELL"
    confidence: float        # 0-1
    suggested_weight: float  # 0-1, consistent with recommended_weight
    reasoning: str           # short, logged for audit
    raw_response: str        # full LLM output, for debugging

class JudgeAdapter(Protocol):
    name: str          # e.g. "claude_hierarchical_v1"
    provider: str      # "anthropic" -- kept as a field so adding another
                        # provider later is a new adapter, not a refactor
    architecture: str  # "hierarchical" (v1) -- reserve "debate" for later
    def verdict(self, context: dict) -> JudgeVerdict: ...
```

Since only Claude is in scope, the Judges leaderboard compares
**architecture, not provider**: start with one `ClaudeHierarchicalJudge`
(single call, structured output via the Anthropic API's tool-use/
structured-output pattern). Design `judges.py` so a second entry (a
multi-step "gather -> reason -> decide" judge, or eventually a debate-
style judge, or another provider) can be added later without touching
anything outside this file.

**Numeric-reasoning guardrail (important, grounded in the literature):**
LLMs — including strong general models — are unreliable at doing math
correctly inside free-form generation (benchmarks show native numeric
calculation accuracy dropping sharply even when document extraction is
accurate). **Never ask the judge to calculate anything.** It must only
receive already-computed numbers (predicted_vol, trend_regime_pred,
entry_proba, sentiment_score, raw ADX/volume values) and reason
qualitatively over them — all arithmetic stays in the deterministic ML
models and Python code, never inside the LLM call.

**Context assembly:** new function `assemble_judge_context() -> dict`
pulling the *primary* model's latest output from each family (via the
`PRIMARY_MODEL` mapping from 1f) plus raw ADX/volume indicators (compute
ADX if not already a feature) plus the latest sentiment score/headlines.

**Logging — new file, new script, deliberately NOT deduped:**
`data/processed/judge_log.csv` — `scripts/run_judge.py`, its OWN separate
cron entry (once a day, NOT twice — this costs real API money, unlike the
free ML models, and only runs at all if `COINPREDICTOR_JUDGE_ENABLED` is
true). Do **not** apply the "already logged, skip duplicate" logic here —
multiple runs per day are allowed and expected, specifically to measure
consistency. Columns: `run_at, model_name, as_of_date, target_date,
action, confidence, suggested_weight, reasoning, input_tokens,
output_tokens, estimated_cost_usd, realized_fwd_return, hypothetical_pnl,
status`.

**Cost guardrail (non-negotiable, on top of the master flag):** a hard
daily spend cap (`JUDGE.max_daily_cost_usd`). Before calling the API,
check cumulative `estimated_cost_usd` already logged today; if over cap,
skip and log a warning — mirror the "refuse silently-stale-data" pattern
already used in `ohlcv.py`.

**Evaluation (`scripts/evaluate_judges.py`, new):** once `target_date`
passes, compute `realized_fwd_return` and hypothetical P&L had
`suggested_weight` been followed. Aggregate per `model_name`:
hypothetical total return, hypothetical Sharpe, hit rate, confidence-vs-
hit-rate calibration, and a **consistency metric** (% agreement across
same-day repeated runs, when any exist).

**Dashboard:** new top-level tab, "LLM Judges" — separate from Track
Record. Show: verdict history table, hypothetical equity curve (reuse
the Backtest tab's chart style), cumulative `estimated_cost_usd` spent,
consistency check results. Tab must render an empty state (not crash)
when the flag is off / zero rows logged.

### Phase 3 — Definition of Done
- [ ] With `COINPREDICTOR_JUDGE_ENABLED=False` (default), `run_judge.py`
      exits immediately with a clear message, makes zero API calls.
- [ ] With the flag on, `run_judge.py` produces one well-formed
      `JudgeVerdict`, logs it, respects the daily cost cap.
- [ ] Missing/invalid `ANTHROPIC_API_KEY` (flag on) fails loudly, does
      not silently no-op.
- [ ] `evaluate_judges.py` handles zero evaluated rows gracefully.
- [ ] Dashboard tab renders in the empty/disabled state without crashing.
- [ ] Cron entry added, clearly commented as separate, cost-bearing,
      once-daily, and gated on the flag — not merged into the existing
      twice-daily `run_daily_docker.sh`.

---

## HARD NON-GOALS (do not implement, even if it seems like a natural next
step — flag it back to the human instead)

- **No live order execution, no exchange API keys, no real-money trading
  wiring of any kind.** Every judge verdict is logged for observation
  only. Non-negotiable for this phase of the project.
- **No paid capability may ever be silently enabled.** If you find
  yourself writing code that calls a paid API without checking its
  dedicated flag first, stop — that's a bug against Section 2.
- Do not remove or weaken the existing stale-cache guard, retry/backoff,
  or duplicate-protection logic while extending these files.
- Do not commit `.env`, API keys, or anything under `data/`/`models/`/
  `logs/` — the existing `.gitignore` already handles this; extend it
  only for genuinely new generated paths (the new csv files, FinBERT's
  downloaded model cache dir if it lands outside the usual model path).
- If a design decision isn't covered above and materially changes cost,
  data access, or risk posture, stop and ask rather than assuming.