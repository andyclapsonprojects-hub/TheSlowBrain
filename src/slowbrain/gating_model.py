"""PR9 learned gating model with hard fallback to the rubric baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import exp
from typing import cast

from .backtest import purged_embargoed_split
from .data_quality import has_error
from .eval_council import HumanExample, HumanLabel, calibrate_against_humans
from .microgix import Value, zero_grad
from .models import FeatureVector, RubricVersion
from .rubrics import decide_feature, score_feature

GATING_LABELS: tuple[HumanLabel, ...] = ("BUY", "SELL", "HOLD", "AVOID", "WATCHLIST")
FEATURE_NAMES = (
    "rubric_score",
    "signed_sentiment_confidence",
    "catalyst_strength",
    "momentum_20d_norm",
    "mean_reversion_norm",
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


def evaluate_gating_model(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    human_examples: tuple[HumanExample, ...] = (),
    anchor_features: Sequence[FeatureVector] = (),
    epochs: int = 4,
    learning_rate: float = 0.05,
    l2_penalty: float = 0.001,
    training_row_cap: int = 2_000,
) -> GatingModelReport:
    """Train/evaluate a tiny logistic gate and keep baseline fallback unless it earns promotion."""
    rows = build_gating_dataset(features, rubric)
    if len(rows) < 6:
        return _empty_report(len(rows), "insufficient_gating_rows")

    train_features, validation_features, confirmation_features = purged_embargoed_split(
        _clean_ordered_features(features),
        train_fraction=0.70,
        validation_fraction=0.15,
        embargo_count=min(10, max(0, len(rows) // 20)),
    )
    train_rows = build_gating_dataset(train_features, rubric)
    validation_rows = build_gating_dataset(validation_features, rubric)
    confirmation_rows = build_gating_dataset(confirmation_features, rubric)
    fit_rows = _bounded_rows(train_rows, training_row_cap)
    gate, training_loss = train_logistic_gate(
        fit_rows,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
    )
    baseline_metrics = _metrics(confirmation_rows, _baseline_probabilities)
    gate_metrics = _metrics(confirmation_rows, gate.probabilities)
    baseline_anchor_labels = _baseline_anchor_labels(anchor_features, rubric)
    gate_anchor_labels = _gate_anchor_labels(anchor_features, rubric, gate)
    baseline_anchor = calibrate_against_humans(human_examples, baseline_anchor_labels, min_kappa=0.0)
    gate_anchor = calibrate_against_humans(human_examples, gate_anchor_labels, min_kappa=0.0)
    baseline_kappa = baseline_anchor.kappa
    gate_kappa = gate_anchor.kappa
    drift_guard_passed = (
        baseline_kappa is not None
        and gate_kappa is not None
        and gate_kappa >= baseline_kappa
        and len(human_examples) > 0
    )
    performance_blockers = _fallback_reasons(
        baseline_metrics=baseline_metrics,
        gate_metrics=gate_metrics,
        drift_guard_passed=drift_guard_passed,
        confirmation_count=len(confirmation_rows),
    )
    fallback_reasons = (*performance_blockers, "pr9_shadow_only_hard_fallback")
    return GatingModelReport(
        status="fallback" if performance_blockers else "shadow_candidate",
        selected_source="baseline_fallback",
        fallback_active=True,
        fallback_reason=";".join(fallback_reasons),
        sample_count=len(rows),
        train_count=len(train_rows),
        validation_count=len(validation_rows),
        confirmation_count=len(confirmation_rows),
        trained_row_count=len(fit_rows),
        target_label_counts=dict(Counter(row.target_label for row in rows)),
        baseline_confirmation_accuracy=baseline_metrics["accuracy"],
        gate_confirmation_accuracy=gate_metrics["accuracy"],
        baseline_brier=baseline_metrics["brier"],
        gate_brier=gate_metrics["brier"],
        baseline_ece=baseline_metrics["ece"],
        gate_ece=gate_metrics["ece"],
        anchor_count=len(human_examples),
        baseline_anchor_kappa=baseline_kappa,
        gate_anchor_kappa=gate_kappa,
        drift_guard_passed=drift_guard_passed,
        training_loss=round(training_loss, 6),
        labels=GATING_LABELS,
        feature_names=FEATURE_NAMES,
        gate_weights=gate.weights,
    )


def build_gating_dataset(features: Sequence[FeatureVector], rubric: RubricVersion) -> tuple[GatingDatasetRow, ...]:
    return tuple(_row(feature, rubric) for feature in _clean_ordered_features(features))


def train_logistic_gate(
    rows: Sequence[GatingDatasetRow],
    *,
    epochs: int = 4,
    learning_rate: float = 0.05,
    l2_penalty: float = 0.001,
) -> tuple[LogisticGate, float]:
    feature_count = len(FEATURE_NAMES) + 1
    params = tuple(
        Value(_initial_weight(label_index, param_index))
        for label_index in range(len(GATING_LABELS))
        for param_index in range(feature_count)
    )
    if not rows:
        return _gate_from_params(params), 0.0
    last_loss = 0.0
    for _epoch in range(max(0, epochs)):
        for row in rows:
            zero_grad(params)
            loss = _row_loss(params, row, l2_penalty=l2_penalty)
            last_loss = loss.data
            loss.backward()
            for param in params:
                param.data -= learning_rate * param.grad
    return _gate_from_params(params), last_loss


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


def _clean_ordered_features(features: Sequence[FeatureVector]) -> tuple[FeatureVector, ...]:
    return tuple(
        sorted(
            (feature for feature in features if not has_error(feature.data_quality_issues)),
            key=lambda feature: (feature.signal_date, feature.horizon_days, feature.idea_id),
        )
    )


def _row(feature: FeatureVector, rubric: RubricVersion) -> GatingDatasetRow:
    return GatingDatasetRow(
        idea_id=feature.idea_id,
        ticker=feature.ticker,
        signal_date=feature.signal_date,
        horizon_days=feature.horizon_days,
        features=feature_values(feature, rubric),
        target_label=target_label_for_return(feature.net_return_pct),
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
        1.0 if feature.trend == "uptrend" else 0.0,
        1.0 if feature.trend == "downtrend" else 0.0,
        1.0 if feature.volume_confirmed else 0.0,
        1.0 if feature.quality_status == "pass" else 0.0,
        1.0 if feature.risk_status == "pass" else 0.0,
        _clamp(feature.horizon_days / 20.0, 0.0, 1.0),
        _clamp(feature.cost_bps / 100.0, 0.0, 1.0),
        1.0 if feature.entry_price is not None and feature.entry_price < 1.0 else 0.0,
    )


def _row_loss(params: tuple[Value, ...], row: GatingDatasetRow, *, l2_penalty: float) -> Value:
    values = (1.0, *row.features)
    loss = Value(0.0)
    for label_index, label in enumerate(GATING_LABELS):
        offset = label_index * len(values)
        logit = Value(0.0)
        for index, feature_value in enumerate(values):
            logit = logit + params[offset + index] * feature_value
        target = 1.0 if row.target_label == label else 0.0
        loss = loss + (logit.sigmoid() - target) ** 2.0
    penalty = Value(0.0)
    for param in params:
        penalty = penalty + param * param
    return (loss / float(len(GATING_LABELS))) + penalty * l2_penalty


def _gate_from_params(params: tuple[Value, ...]) -> LogisticGate:
    width = len(FEATURE_NAMES) + 1
    weights = tuple(
        tuple(params[label_index * width + offset].data for offset in range(width))
        for label_index in range(len(GATING_LABELS))
    )
    return LogisticGate(GATING_LABELS, FEATURE_NAMES, weights)


def _initial_weight(label_index: int, param_index: int) -> float:
    return ((label_index + 1) * (param_index + 3) % 11 - 5) / 500.0


def _bounded_rows(rows: Sequence[GatingDatasetRow], cap: int) -> tuple[GatingDatasetRow, ...]:
    if cap <= 0 or len(rows) <= cap:
        return tuple(rows)
    step = len(rows) / cap
    return tuple(rows[min(int(index * step), len(rows) - 1)] for index in range(cap))


def _metrics(
    rows: Sequence[GatingDatasetRow],
    probabilities: Callable[[GatingDatasetRow], dict[HumanLabel, float]],
) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "brier": 1.0, "ece": 1.0}
    correct = 0
    brier_total = 0.0
    confidence_gaps: list[float] = []
    for row in rows:
        probs = probabilities(row)
        predicted = max(GATING_LABELS, key=lambda label: probs[label])
        confidence = probs[predicted]
        accurate = 1.0 if predicted == row.target_label else 0.0
        correct += int(accurate)
        confidence_gaps.append(abs(confidence - accurate))
        brier_total += sum((probs[label] - (1.0 if label == row.target_label else 0.0)) ** 2 for label in GATING_LABELS)
    return {
        "accuracy": round(correct / len(rows), 4),
        "brier": round(brier_total / len(rows), 4),
        "ece": round(sum(confidence_gaps) / len(confidence_gaps), 4),
    }


def _baseline_probabilities(row: GatingDatasetRow) -> dict[HumanLabel, float]:
    return {label: 1.0 if label == row.baseline_label else 0.0 for label in GATING_LABELS}


def _baseline_anchor_labels(
    anchor_features: Sequence[FeatureVector],
    rubric: RubricVersion,
) -> dict[str, HumanLabel]:
    return {feature.idea_id: cast(HumanLabel, decide_feature(feature, rubric).action) for feature in anchor_features}


def _gate_anchor_labels(
    anchor_features: Sequence[FeatureVector],
    rubric: RubricVersion,
    gate: LogisticGate,
) -> dict[str, HumanLabel]:
    rows = build_gating_dataset(anchor_features, rubric)
    return {row.idea_id: gate.predict_label(row) for row in rows}


def _fallback_reasons(
    *,
    baseline_metrics: Mapping[str, float],
    gate_metrics: Mapping[str, float],
    drift_guard_passed: bool,
    confirmation_count: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if confirmation_count < 30:
        reasons.append("insufficient_confirmation_rows")
    if gate_metrics["accuracy"] <= baseline_metrics["accuracy"]:
        reasons.append("gate_confirmation_accuracy_not_above_baseline")
    if gate_metrics["brier"] > baseline_metrics["brier"]:
        reasons.append("gate_brier_worse_than_baseline")
    if not drift_guard_passed:
        reasons.append("anchor_drift_guard_failed")
    return tuple(reasons)


def _empty_report(sample_count: int, reason: str) -> GatingModelReport:
    return GatingModelReport(
        status="not_available",
        selected_source="baseline_fallback",
        fallback_active=True,
        fallback_reason=reason,
        sample_count=sample_count,
        train_count=0,
        validation_count=0,
        confirmation_count=0,
        trained_row_count=0,
        target_label_counts={},
        baseline_confirmation_accuracy=0.0,
        gate_confirmation_accuracy=0.0,
        baseline_brier=1.0,
        gate_brier=1.0,
        baseline_ece=1.0,
        gate_ece=1.0,
        anchor_count=0,
        baseline_anchor_kappa=None,
        gate_anchor_kappa=None,
        drift_guard_passed=False,
        training_loss=0.0,
        labels=GATING_LABELS,
        feature_names=FEATURE_NAMES,
        gate_weights=(),
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
