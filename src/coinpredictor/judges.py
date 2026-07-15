"""LLM Judge layer (Phase 3) — ENTIRELY SEPARATE from registry.MODELS.

The ML models are deterministic functions scored against a single realized
outcome (MAE/accuracy). Judges are **non-deterministic agents** scored on
*decision quality* (hypothetical P&L, hit rate, run-to-run consistency, cost,
latency). Mixing them would corrupt the leaderboard math, so they live here in
their own module, log to their own file (``judge_log.csv``), and NEVER appear
in the ML leaderboards.

Ships with ``COINPREDICTOR_JUDGE_ENABLED = False``: nothing here costs money
until the human explicitly flips the flag.

NUMERIC-REASONING GUARDRAIL (grounded in the literature): LLMs are unreliable at
doing arithmetic inside free-form generation. The judge is therefore NEVER asked
to calculate anything — it only receives already-computed numbers (predicted
vol, trend regime, entry proba, sentiment score, raw ADX/volume) and reasons
qualitatively over them. All math stays in the deterministic models / Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coinpredictor.config import ANTHROPIC_API_KEY, JUDGE
from coinpredictor.data.ohlcv import load_ohlcv
from coinpredictor.features import build_default_features
from coinpredictor.registry import MODELS, PRIMARY_MODEL


@dataclass
class JudgeVerdict:
    action: str              # "BUY" | "HOLD" | "SELL"
    confidence: float        # 0-1
    suggested_weight: float  # 0-1, consistent with recommended_weight
    reasoning: str           # short, logged for audit
    raw_response: str        # full LLM output, for debugging


class JudgeAdapter(Protocol):
    """Common contract for every judge. ``provider`` is kept as a field so
    adding another provider later is a new adapter, not a refactor."""

    name: str          # e.g. "claude_hierarchical_v1"
    provider: str      # "anthropic"
    architecture: str  # "hierarchical" (v1); reserve "debate" for later

    def verdict(self, context: dict) -> JudgeVerdict:
        ...


# --- Context assembly --------------------------------------------------------
def _adapter_by_name(name: str):
    for adapter in MODELS:
        if adapter.name == name:
            return adapter
    return None


def assemble_judge_context(refresh: bool = True) -> dict:
    """Gather the PRIMARY model output from each family plus raw technical
    indicators and the latest headlines. Every value is already computed — the
    judge does no math on any of it.
    """
    ohlcv = load_ohlcv(refresh=refresh)
    feats = build_default_features(ohlcv, refresh=refresh, drop_na=False)
    valid = feats.dropna(subset=["adx_14"])
    as_of = valid.index[-1]
    last = valid.loc[as_of]

    context: dict = {
        "as_of_date": as_of.date().isoformat(),
        "horizon_days": JUDGE.horizon,
        "last_close": float(ohlcv.loc[as_of, "close"]),
        "indicators": {
            "adx_14": float(last["adx_14"]),
            "rsi_14": float(last.get("rsi_14", np.nan)),
            "volume": float(ohlcv.loc[as_of, "volume"]),
            "volume_sma_ratio": float(last.get("volume_sma_ratio", np.nan)),
            "close_to_sma_50": float(last.get("close_to_sma_50", np.nan)),
        },
        "primary_models": {},
    }

    # Pull each family's primary model output (already-computed numbers only).
    for target_type, model_name in PRIMARY_MODEL.items():
        adapter = _adapter_by_name(model_name)
        if adapter is None:
            continue
        try:
            out = adapter.predict(refresh=False)
            out.pop("as_of_date", None)  # already captured at top level
            context["primary_models"][target_type] = out
        except Exception as exc:  # noqa: BLE001 - a missing family shouldn't block
            context["primary_models"][target_type] = {"error": str(exc)}

    try:
        from coinpredictor.data.news import get_headlines

        context["headlines"] = get_headlines(limit=15)
    except Exception:  # noqa: BLE001 - headlines are best-effort context
        context["headlines"] = []

    return context


def _render_prompt(context: dict) -> str:
    """Build the judge prompt. All numbers are pre-computed; the model is asked
    to reason qualitatively and pick an action, NOT to calculate anything."""
    import json

    pm = context.get("primary_models", {})
    ind = context.get("indicators", {})
    headlines = context.get("headlines", [])
    headline_block = "\n".join(f"- {h}" for h in headlines[:15]) or "(none available)"

    return (
        "You are a disciplined BTC portfolio strategist. You are given ALREADY-"
        "COMPUTED numbers from several specialised models plus raw indicators. "
        "Do NOT perform any arithmetic or recompute anything — reason "
        "qualitatively over the provided values and decide an action.\n\n"
        f"As of {context.get('as_of_date')} (close ${context.get('last_close'):,.2f}), "
        f"horizon {context.get('horizon_days')} days.\n\n"
        f"Primary model outputs (JSON):\n{json.dumps(pm, indent=2, default=str)}\n\n"
        f"Raw technical indicators (JSON):\n{json.dumps(ind, indent=2, default=str)}\n\n"
        f"Recent headlines:\n{headline_block}\n\n"
        "Decide BUY, HOLD, or SELL and a suggested BTC weight in [0, 1] "
        "consistent with your conviction. Call the record_verdict tool."
    )


_VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record the final BTC trading verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
            "confidence": {
                "type": "number",
                "description": "Confidence in the action, 0 to 1.",
            },
            "suggested_weight": {
                "type": "number",
                "description": "Suggested BTC portfolio weight, 0 to 1.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief qualitative justification (2-4 sentences).",
            },
        },
        "required": ["action", "confidence", "suggested_weight", "reasoning"],
    },
}


@dataclass
class ClaudeHierarchicalJudge:
    """Single-call, structured-output judge over Claude (the only provider in
    scope). The Judges leaderboard compares ARCHITECTURE, not provider — a
    second entry (multi-step gather->reason->decide, or a debate judge) can be
    added later without touching anything outside this file."""

    name: str = "claude_hierarchical_v1"
    provider: str = "anthropic"
    architecture: str = "hierarchical"

    # Populated after each verdict() call so run_judge.py can log token usage
    # and estimated cost without the LLM ever doing the arithmetic.
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_cost_usd: float = 0.0

    def verdict(self, context: dict) -> JudgeVerdict:
        if not JUDGE.enabled:
            raise RuntimeError(
                "ClaudeHierarchicalJudge.verdict called while "
                "COINPREDICTOR_JUDGE_ENABLED is OFF. This is a paid path and "
                "must never run unless its dedicated flag is on."
            )
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "COINPREDICTOR_JUDGE_ENABLED is ON but ANTHROPIC_API_KEY is "
                "missing. Set the key in .env or turn the flag off — refusing "
                "to continue (never silently no-ops)."
            )

        import anthropic  # lazy import — only on the paid path

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=JUDGE.model,
            max_tokens=JUDGE.max_tokens,
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[{"role": "user", "content": _render_prompt(context)}],
        )

        data = None
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                data = block.input
                break
        if data is None:
            raise RuntimeError("Judge returned no structured verdict (no tool_use block).")

        self.last_input_tokens = int(getattr(msg.usage, "input_tokens", 0))
        self.last_output_tokens = int(getattr(msg.usage, "output_tokens", 0))
        self.last_cost_usd = estimate_cost(self.last_input_tokens, self.last_output_tokens)

        return JudgeVerdict(
            action=str(data["action"]).upper(),
            confidence=float(min(max(data["confidence"], 0.0), 1.0)),
            suggested_weight=float(min(max(data["suggested_weight"], 0.0), 1.0)),
            reasoning=str(data.get("reasoning", "")).strip(),
            raw_response=str(data),
        )


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Local cost estimate in USD (Python does the arithmetic, never the LLM)."""
    return float(
        input_tokens / 1_000_000 * JUDGE.input_cost_per_mtok
        + output_tokens / 1_000_000 * JUDGE.output_cost_per_mtok
    )


# Registered judges. Only Claude is in scope; add another architecture (or
# provider) here later without touching the rest of the system.
JUDGES: list[JudgeAdapter] = [
    ClaudeHierarchicalJudge(),
]
