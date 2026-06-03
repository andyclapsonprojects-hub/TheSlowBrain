"""Learned gating model types: the logistic gate, its dataset rows, and the report shape.

Training and evaluation live in :mod:`slowbrain.gating_training` (which imports from here, never the
reverse) to keep each module small and acyclic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import exp
from typing import Literal, cast

from .data_quality import has_error
from .eval_council import HumanLabel
from .models import DecisionAction, FeatureVector, RubricVersion
from .rubrics import decide_feature, score_feature

GATING_LABELS: tuple[HumanLabel, ...] = ("BUY", "SELL", "HOLD", "AVOID", "WATCHLIST")
TargetLabelMode = Literal["absolute_return", "cross_sectional_rank"]
FEATURE_NAMES = (
    "rubric_score",
    "signed_sentiment_confidence",
    "catalyst_strength",
    "momentum_20d_norm",
    "mean_reversion_norm",
    "rsi_14_norm",
    "macd_bullish",
    "macd_bearish",
    "atr_pct_14_norm",
    "momentum_63d_norm",
    "volume_ratio_20d_norm",
    "bb_percent_b",
    "bb_bandwidth",
    "macd_hist",
    "ema_trend",
    "candle_signal",
    "sma_distance",
    "rsi_14_cross_sectional_z",
    "momentum_63d_cross_sectional_z",
    "volume_ratio_cross_sectional_z",
    "value_score",
    "fundamental_quality_score",
    "size_score",
    "liquidity_score",
    "trend_up",
    "trend_down",
    "volume_confirmed",
    "quality_pass",
    "risk_pass",
    "horizon_norm",
    "cost_bps_norm",
    "penny_price",
)

BUY_RETURN_THRESHOLD = 2.0
WATCHLIST_RETURN_THRESHOLD = 0.5
AVOID_RETURN_THRESHOLD = -0.5
SELL_RETURN_THRESHOLD = -2.0


@dataclass(frozen=True)
class GatingDatasetRow:
    idea_id: str
    ticker: str
    signal_date: str
    horizon_days: int
    features: tuple[float, ...]
    target_label: HumanLabel
    baseline_label: HumanLabel
    forward_return_pct: float


@dataclass(frozen=True)
class LogisticGate:
    labels: tuple[HumanLabel, ...]
    feature_names: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]

    def probabilities(self, row: GatingDatasetRow) -> dict[HumanLabel, float]:
        logits = [_dot(weights, (1.0, *row.features)) for weights in self.weights]
        return _softmax(self.labels, logits)

    def predict_label(self, row: GatingDatasetRow) -> HumanLabel:
        probabilities = self.probabilities(row)
        return max(self.labels, key=lambda label: probabilities[label])


@dataclass(frozen=True)
class GatingModelReport:
    status: str
    selected_source: str
    fallback_active: bool
    fallback_reason: str
    sample_count: int
    train_count: int
    validation_count: int
    confirmation_count: int
    trained_row_count: int
    target_label_counts: Mapping[str, int]
    baseline_confirmation_accuracy: float
    gate_confirmation_accuracy: float
    baseline_brier: float
    gate_brier: float
    baseline_ece: float
    gate_ece: float
    anchor_count: int
    baseline_anchor_kappa: float | None
    gate_anchor_kappa: float | None
    drift_guard_passed: bool
    training_loss: float
    labels: tuple[HumanLabel, ...]
    feature_names: tuple[str, ...]
    gate_weights: tuple[tuple[float, ...], ...] = ()
    target_label_mode: TargetLabelMode = "absolute_return"


def build_gating_dataset(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    target_label_mode: TargetLabelMode = "absolute_return",
) -> tuple[GatingDatasetRow, ...]:
    return tuple(
        _row(feature, rubric, target_label_mode=target_label_mode)
        for feature in clean_ordered_features(features)
    )


def clean_ordered_features(features: Sequence[FeatureVector]) -> tuple[FeatureVector, ...]:
    return tuple(
        sorted(
            (feature for feature in features if not has_error(feature.data_quality_issues)),
            key=lambda feature: (feature.signal_date, feature.horizon_days, feature.idea_id),
        )
    )


def report_to_dict(report: GatingModelReport) -> dict[str, object]:
    return asdict(report)


def target_label_for_return(net_return_pct: float) -> HumanLabel:
    if net_return_pct >= BUY_RETURN_THRESHOLD:
        return "BUY"
    if net_return_pct >= WATCHLIST_RETURN_THRESHOLD:
        return "WATCHLIST"
    if net_return_pct <= SELL_RETURN_THRESHOLD:
        return "SELL"
    if net_return_pct <= AVOID_RETURN_THRESHOLD:
        return "AVOID"
    return "HOLD"


def target_label_for_feature(feature: FeatureVector, *, mode: TargetLabelMode = "absolute_return") -> HumanLabel:
    if mode == "cross_sectional_rank" and feature.rank_label is not None:
        return _human_label(feature.rank_label)
    return target_label_for_return(feature.net_return_pct)


def _row(feature: FeatureVector, rubric: RubricVersion, *, target_label_mode: TargetLabelMode) -> GatingDatasetRow:
    return GatingDatasetRow(
        idea_id=feature.idea_id,
        ticker=feature.ticker,
        signal_date=feature.signal_date,
        horizon_days=feature.horizon_days,
        features=feature_values(feature, rubric),
        target_label=target_label_for_feature(feature, mode=target_label_mode),
        baseline_label=cast(HumanLabel, decide_feature(feature, rubric).action),
        forward_return_pct=feature.net_return_pct,
    )


def feature_values(feature: FeatureVector, rubric: RubricVersion) -> tuple[float, ...]:
    sentiment = 0.0
    if feature.sentiment == "positive":
        sentiment = _clamp(feature.sentiment_confidence, 0.0, 1.0)
    elif feature.sentiment == "negative":
        sentiment = -_clamp(feature.sentiment_confidence, 0.0, 1.0)
    return (
        _clamp(score_feature(feature, rubric), -1.0, 1.0),
        sentiment,
        _clamp(feature.catalyst_strength, 0.0, 1.0),
        _clamp(feature.momentum_20d_pct / 20.0, -1.0, 1.0),
        _clamp(-feature.mean_reversion_z_20d / 3.0, -1.0, 1.0),
        _clamp(feature.rsi_14 / 100.0, 0.0, 1.0),
        1.0 if feature.macd_signal == "bullish" else 0.0,
        1.0 if feature.macd_signal == "bearish" else 0.0,
        _clamp(feature.atr_pct_14 / 10.0, 0.0, 1.0),
        _clamp(feature.momentum_63d_pct / 50.0, -1.0, 1.0),
        _clamp(feature.volume_ratio_20d / 3.0, 0.0, 1.0),
        _clamp(feature.bb_percent_b, 0.0, 1.0),
        _clamp(feature.bb_bandwidth * 3.0, 0.0, 1.0),
        _clamp(feature.macd_hist_pct / 2.0, -1.0, 1.0),
        _clamp(feature.ema_trend_pct / 5.0, -1.0, 1.0),
        _clamp(feature.candle_signal, -1.0, 1.0),
        _clamp(feature.sma_distance_pct / 10.0, -1.0, 1.0),
        _clamp(_zscore(feature, "rsi_14") / 3.0, -1.0, 1.0),
        _clamp(_zscore(feature, "momentum_63d_pct") / 3.0, -1.0, 1.0),
        _clamp(_zscore(feature, "volume_ratio_20d") / 3.0, -1.0, 1.0),
        _clamp(feature.value_score, -1.0, 1.0),
        _clamp(feature.fundamental_quality_score, -1.0, 1.0),
        _clamp(feature.size_score, -1.0, 1.0),
        _clamp(feature.liquidity_score, -1.0, 1.0),
        1.0 if feature.trend == "uptrend" else 0.0,
        1.0 if feature.trend == "downtrend" else 0.0,
        1.0 if feature.volume_confirmed else 0.0,
        1.0 if feature.quality_status == "pass" else 0.0,
        1.0 if feature.risk_status == "pass" else 0.0,
        _clamp(feature.horizon_days / 20.0, 0.0, 1.0),
        _clamp(feature.cost_bps / 100.0, 0.0, 1.0),
        1.0 if feature.entry_price is not None and feature.entry_price < 1.0 else 0.0,
    )


def _softmax(labels: tuple[HumanLabel, ...], logits: Sequence[float]) -> dict[HumanLabel, float]:
    peak = max(logits) if logits else 0.0
    exps = [exp(_clamp(value - peak, -50.0, 50.0)) for value in logits]
    total = sum(exps) or 1.0
    return {label: exps[index] / total for index, label in enumerate(labels)}


def _dot(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values, strict=True))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _zscore(feature: FeatureVector, name: str) -> float:
    return feature.cross_sectional_zscores.get(name, 0.0)


def _human_label(label: DecisionAction) -> HumanLabel:
    return cast(HumanLabel, label)
