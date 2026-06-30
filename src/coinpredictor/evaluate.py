"""Evaluate whether Phase-4 features (implied vol / funding) improve the model.

These features have short history, so a fair test trains the model on the
*same recent date range* with and without them and compares walk-forward
metrics. This avoids the confound of simply having less data when the feature
is enabled.

Run ``python -m coinpredictor.evaluate``.
"""
from __future__ import annotations

import pandas as pd

from coinpredictor.config import FEATURES, MODEL
from coinpredictor.features import build_features_full, feature_columns
from coinpredictor.model import build_vol_regressor, walk_forward_regress


def _evaluate(feats: pd.DataFrame, name: str):
    cols = feature_columns(feats)
    X = feats[cols]
    y = feats[MODEL.target_col].astype(float)
    return walk_forward_regress(build_vol_regressor(), X, y, model_name=name)


def compare_phase4(ohlcv: pd.DataFrame) -> str:
    """Compare baseline vs +implied-vol over the DVOL window (the useful test).

    Funding history from OKX is only ~3 months, too short for a meaningful
    walk-forward, so it is reported separately rather than mixed in (which would
    collapse the common window to a few dozen days).
    """
    base_kwargs = dict(
        use_macro=FEATURES.use_macro,
        use_onchain=FEATURES.use_onchain,
        use_sentiment=FEATURES.use_sentiment,
    )

    # Build with DVOL; its window (~2023-10+) defines the comparison range.
    with_iv = build_features_full(ohlcv, **base_kwargs, use_implied_vol=True)
    common = with_iv.index
    base = build_features_full(ohlcv, **base_kwargs).reindex(common).dropna()
    common = base.index
    with_iv = with_iv.reindex(common)

    lines = [
        f"DVOL window: {common.min().date()} -> {common.max().date()} "
        f"({len(common)} days)",
        "Walk-forward regression (higher corr / lower rmse = better):",
    ]
    for name, feats in {
        "baseline (no DVOL)": base,
        "+ implied vol (DVOL)": with_iv,
    }.items():
        res = _evaluate(feats, name)
        lines.append(
            f"  {name:22s}: rmse={res.rmse:.4f} r2={res.r2:+.4f} corr={res.corr:.4f}"
        )

    # Funding: report coverage only.
    try:
        from coinpredictor.data.funding import download_funding

        fund = download_funding()
        lines.append(
            f"\nFunding coverage: {fund.index.min().date()} -> "
            f"{fund.index.max().date()} ({len(fund)} days) "
            "— too short for training; available for the live bot as a signal."
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"\nFunding unavailable: {exc}")

    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    from coinpredictor.data.ohlcv import load_ohlcv
    from coinpredictor.evaluate import compare_phase4 as _compare_phase4

    print(_compare_phase4(load_ohlcv()))
