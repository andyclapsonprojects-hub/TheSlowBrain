"""Named hypothesis predicates."""

from __future__ import annotations

from collections.abc import Callable

from .models import FeatureVector, HypothesisSpec

Predicate = Callable[[FeatureVector], bool]


DEFAULT_HYPOTHESES: tuple[HypothesisSpec, ...] = (
    HypothesisSpec("H1_POSITIVE_SENTIMENT_10D", "Positive sentiment should outperform over 10 days.", "positive", 10),
    HypothesisSpec(
        "H2_POSITIVE_TREND_10D",
        "Positive sentiment plus uptrend should outperform over 10 days.",
        "positive",
        10,
    ),
    HypothesisSpec("H3_MEAN_REVERSION_VOLUME_10D", "Oversold plus volume confirmation should rebound.", "positive", 10),
    HypothesisSpec(
        "H4_NEGATIVE_AVOID_10D",
        "Negative sentiment should underperform and support avoidance.",
        "negative",
        10,
    ),
)


def hypothesis_predicates() -> dict[str, Predicate]:
    return {
        "H1_POSITIVE_SENTIMENT_10D": lambda feature: feature.sentiment == "positive",
        "H2_POSITIVE_TREND_10D": lambda feature: feature.sentiment == "positive" and feature.trend == "uptrend",
        "H3_MEAN_REVERSION_VOLUME_10D": lambda feature: (
            feature.mean_reversion_z_20d <= -1.0 and feature.volume_confirmed
        ),
        "H4_NEGATIVE_AVOID_10D": lambda feature: feature.sentiment == "negative",
    }


def matching_hypotheses(feature: FeatureVector) -> tuple[str, ...]:
    matches = [hypothesis_id for hypothesis_id, predicate in hypothesis_predicates().items() if predicate(feature)]
    return tuple(matches)
