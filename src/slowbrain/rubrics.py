"""Rubric scoring and decisions."""

from __future__ import annotations

from .data_quality import has_error
from .models import DecisionAction, FeatureVector, RubricVersion, TradeDecision

BASE_RUBRIC = RubricVersion(
    version="rubric_v1_seed",
    weights={
        "sentiment": 0.25,
        "catalyst": 0.20,
        "trend": 0.20,
        "momentum": 0.15,
        "mean_reversion": 0.10,
        "volume": 0.10,
    },
    buy_threshold=0.58,
    sell_threshold=-0.35,
    max_position_pct=0.05,
    notes=("Seed rubric; can be replaced only by guarded profit proof.",),
)


def score_feature(feature: FeatureVector, rubric: RubricVersion) -> float:
    components = {
        "sentiment": _sentiment_component(feature),
        "catalyst": _clamp01(feature.catalyst_strength),
        "trend": _trend_component(feature),
        "momentum": _clamp(feature.momentum_20d_pct / 20.0, -1.0, 1.0),
        "mean_reversion": 1.0 if feature.mean_reversion_z_20d <= -1.0 else 0.0,
        "volume": 1.0 if feature.volume_confirmed else 0.0,
    }
    return round(sum(components[name] * rubric.weights.get(name, 0.0) for name in components), 4)


# Width of the near-miss band below buy_threshold that yields WATCHLIST instead of HOLD.
WATCHLIST_BAND = 0.10


def decide_feature(feature: FeatureVector, rubric: RubricVersion) -> TradeDecision:
    score = score_feature(feature, rubric)
    action: DecisionAction
    reason: str
    clean_gates = (
        feature.quality_status != "fail"
        and feature.risk_status not in {"rejected", "fail"}
        and not has_error(feature.data_quality_issues)
    )
    if score >= rubric.buy_threshold and clean_gates:
        action = "BUY"
        reason = "score_above_buy_threshold_with_clean_gates"
    elif score <= rubric.sell_threshold or feature.sentiment == "negative":
        action = "SELL"
        reason = "negative_or_below_sell_threshold"
    elif score >= rubric.buy_threshold and not clean_gates:
        action = "AVOID"
        reason = "buy_score_blocked_by_failed_quality_or_risk_gates"
    elif score >= rubric.buy_threshold - WATCHLIST_BAND and clean_gates:
        action = "WATCHLIST"
        reason = "near_buy_threshold_pending_confirmation"
    else:
        action = "HOLD"
        reason = "insufficient_edge_for_buy_or_sell"
    max_notional = 1000.0 * rubric.max_position_pct if action == "BUY" else 0.0
    return TradeDecision(
        ticker=feature.ticker,
        action=action,
        score=score,
        rubric_version=rubric.version,
        reason=reason,
        max_notional_gbp=round(max_notional, 2),
    )


def _sentiment_component(feature: FeatureVector) -> float:
    if feature.sentiment == "positive":
        return _clamp01(feature.sentiment_confidence)
    if feature.sentiment == "negative":
        return -_clamp01(feature.sentiment_confidence)
    return 0.0


def _trend_component(feature: FeatureVector) -> float:
    if feature.trend == "uptrend":
        return 1.0
    if feature.trend == "downtrend":
        return -0.7
    return 0.0


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
