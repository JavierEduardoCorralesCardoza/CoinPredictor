"""Feature engineering for Bitcoin volatility prediction.

Design principle: **no look-ahead bias.** Every feature on row ``t`` is computed
exclusively from information available up to and including the close of day
``t``. The targets on row ``t`` describe the *forward* realized volatility over
the next ``vol_horizon`` days, so they are the only columns that reference the
future (via a forward-shifted rolling window) and are what the model learns.

Two targets are produced:
* ``target_vol`` — forward annualized realized volatility (regression).
* ``target_high_vol`` — 1 if the coming period is more volatile than the recent
  trailing norm (classification / regime detection).

Indicators are implemented with plain pandas to avoid third-party dependency and
numpy-compatibility issues.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from coinpredictor.config import MODEL


# --- Individual indicators ---------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    width = (upper - lower) / sma
    # Position of price within the band: 0 = lower, 1 = upper.
    pct_b = (close - lower) / (upper - lower)
    return width, pct_b


# --- Public API --------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    *,
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10),
    sma_windows: tuple[int, ...] = (5, 10, 20, 50),
    horizon: int | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Construct the model-ready feature matrix plus the target column.

    Parameters
    ----------
    df:
        Normalized OHLCV frame indexed by date (from ``data.ohlcv``).
    return_lags:
        Lags (in days) of daily return to include as features.
    sma_windows:
        Window sizes for simple moving averages / distance-to-MA features.
    drop_na:
        Drop warm-up rows that contain NaNs from rolling windows.

    Returns
    -------
    DataFrame containing OHLCV, engineered features, and ``target``.
    """
    out = df.copy()
    close = out["close"]

    # Returns & momentum (all backward-looking).
    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = np.log(close).diff()
    for lag in return_lags:
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag)

    # Rolling volatility of daily returns.
    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()

    # Moving averages & distance of price from them.
    for w in sma_windows:
        sma = close.rolling(w).mean()
        out[f"sma_{w}"] = sma
        out[f"close_to_sma_{w}"] = close / sma - 1.0
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # Momentum oscillators.
    out["rsi_14"] = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # Volatility / range.
    out["atr_14"] = _atr(out, 14)
    bb_width, bb_pct = _bollinger(close, 20)
    out["bb_width"] = bb_width
    out["bb_pct"] = bb_pct

    # Volume behaviour.
    out["volume_change"] = out["volume"].pct_change()
    out["volume_sma_ratio"] = out["volume"] / out["volume"].rolling(20).mean()

    # High-low range relative to close.
    out["hl_range"] = (out["high"] - out["low"]) / close

    # Trailing annualized realized volatility (past-only feature & regime base).
    ann = np.sqrt(MODEL.annualization)
    out["realized_vol_trailing"] = (
        out["log_return_1d"].rolling(MODEL.regime_lookback).std() * ann
    )

    # --- Targets: forward realized volatility --------------------------------
    # Forward h-day realized volatility (annualized). rolling(h).std() at index
    # t+h covers returns r[t+1..t+h]; shifting by -h aligns it to day t. This is
    # the ONLY forward-looking computation (it defines the label).
    h = horizon or MODEL.vol_horizon
    fwd_vol = out["log_return_1d"].rolling(h).std().shift(-h) * ann
    out[MODEL.target_col] = fwd_vol

    # Regime label: will the coming period be more volatile than the recent norm?
    # Threshold (trailing vol) uses only past data; comparison vs forward vol.
    out[MODEL.regime_col] = (fwd_vol > out["realized_vol_trailing"]).astype(float)
    # Rows whose forward window runs off the end have no known label.
    out.loc[out.index[-h:], [MODEL.target_col, MODEL.regime_col]] = np.nan

    if drop_na:
        out = out.dropna()
        out[MODEL.regime_col] = out[MODEL.regime_col].astype(int)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model input columns (everything except raw OHLCV and targets)."""
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        MODEL.target_col,
        MODEL.regime_col,
    }
    return [c for c in df.columns if c not in exclude]


def split_xy(df: pd.DataFrame, *, regime: bool = False):
    """Split a feature frame into (X, y).

    Parameters
    ----------
    regime:
        When True, return the binary high-volatility regime label; otherwise
        return the continuous forward-realized-volatility regression target.
    """
    cols = feature_columns(df)
    if regime:
        return df[cols], df[MODEL.regime_col].astype(int)
    return df[cols], df[MODEL.target_col].astype(float)


def _append_source(extras, name, fetch_fn, index, refresh):
    """Fetch one external feature source, skipping it (with a warning) on error.

    External APIs can be unreachable or rate-limited; a single failed source
    should degrade the feature set rather than crash training/serving.
    """
    try:
        extras.append(fetch_fn(index, refresh=refresh))
    except Exception as exc:  # noqa: BLE001 - network/parse failures are varied
        warnings.warn(
            f"Skipping {name} features: {exc}", RuntimeWarning, stacklevel=2
        )


def build_features_full(
    df: pd.DataFrame,
    *,
    use_macro: bool = False,
    use_onchain: bool = False,
    use_sentiment: bool = False,
    use_implied_vol: bool = False,
    use_funding: bool = False,
    horizon: int | None = None,
    refresh: bool = False,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Phase 1 technical features plus optional Phase 2/3/4 external sources.

    External features are merged on the BTC date index *before* the target is
    finalized. Each source is fetched lazily so missing optional dependencies or
    API keys only disable that source rather than failing the whole pipeline.
    """
    # Build technical features without dropping rows yet so external columns can
    # be joined on the full index before a single combined dropna().
    base = build_features(df, horizon=horizon, drop_na=False)

    extras: list[pd.DataFrame] = []
    if use_macro:
        from coinpredictor.data.macro import macro_features

        _append_source(extras, "macro", macro_features, base.index, refresh)
    if use_onchain:
        from coinpredictor.data.onchain import onchain_features

        _append_source(extras, "on-chain", onchain_features, base.index, refresh)
    if use_sentiment:
        from coinpredictor.data.sentiment import sentiment_features

        _append_source(extras, "sentiment", sentiment_features, base.index, refresh)
    if use_implied_vol:
        from coinpredictor.data.implied_vol import implied_vol_features

        _append_source(extras, "implied-vol", implied_vol_features, base.index, refresh)
    if use_funding:
        from coinpredictor.data.funding import funding_features

        _append_source(extras, "funding", funding_features, base.index, refresh)

    combined = pd.concat([base, *extras], axis=1) if extras else base

    if drop_na:
        combined = combined.dropna()
        combined[MODEL.regime_col] = combined[MODEL.regime_col].astype(int)

    return combined


def build_default_features(
    df: pd.DataFrame, *, horizon: int | None = None, refresh: bool = False, drop_na: bool = True
) -> pd.DataFrame:
    """Build the feature set selected by ``config.FEATURES``.

    This is the single source of truth used by training, prediction, and the
    dashboard so the columns always line up. Phase 1 is always included; the
    macro / sentiment / on-chain phases are toggled in config.
    """
    from coinpredictor.config import FEATURES

    return build_features_full(
        df,
        use_macro=FEATURES.use_macro,
        use_onchain=FEATURES.use_onchain,
        use_sentiment=FEATURES.use_sentiment,
        use_implied_vol=FEATURES.use_implied_vol,
        use_funding=FEATURES.use_funding,
        horizon=horizon,
        refresh=refresh,
        drop_na=drop_na,
    )
