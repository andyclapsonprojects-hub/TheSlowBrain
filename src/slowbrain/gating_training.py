"""Training and evaluation for the learned gating model.

Upgrades over the first version, all pure-stdlib and deterministic:
- **warm-start continual learning** (start from the persisted gate, don't relearn from scratch);
- **softmax cross-entropy** loss (matches the softmax used at prediction time);
- **momentum** optimizer;
- **inverse-frequency class weighting** (so rare BUY/SELL are learned, not swamped by HOLD);
- **early stopping** on the validation split.

The profit-graded, earned promotion gate is unaffected: this module only makes the gate a better
*learner*; whether it deserves to trade is still decided elsewhere by out-of-sample profit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from math import log
from typing import cast

from .backtest import purged_embargoed_split
from .eval_council import HumanExample, HumanLabel, calibrate_against_humans
from .gating_model import (
    FEATURE_NAMES,
    GATING_LABELS,
    GatingDatasetRow,
    GatingModelReport,
    LogisticGate,
    TargetLabelMode,
    build_gating_dataset,
    clean_ordered_features,
)
from .microgix import Value, zero_grad
from .models import FeatureVector, RubricVersion
from .promotion import PromotionStage
from .rubrics import decide_feature

GateWeights = tuple[tuple[float, ...], ...]


def evaluate_gating_model(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    human_examples: tuple[HumanExample, ...] = (),
    anchor_features: Sequence[FeatureVector] = (),
    epochs: int = 30,
    learning_rate: float = 0.05,
    l2_penalty: float = 0.001,
    training_row_cap: int = 2_000,
    active_stage: PromotionStage = "shadow",
    warm_start_gate: LogisticGate | None = None,
    target_label_mode: TargetLabelMode = "absolute_return",
) -> GatingModelReport:
    """Train/evaluate the gate and keep baseline fallback unless it earns promotion.

    ``active_stage`` is the promotion stage governing *this* run's decisions (from persisted state):
    ``shadow`` keeps the hard baseline fallback, otherwise the report records the learned gate as the
    selected source. ``warm_start_gate`` (the persisted gate) seeds training so learning compounds.
    """
    rows = build_gating_dataset(features, rubric, target_label_mode=target_label_mode)
    if len(rows) < 6:
        return _empty_report(len(rows), "insufficient_gating_rows", target_label_mode=target_label_mode)

    train_features, validation_features, confirmation_features = purged_embargoed_split(
        clean_ordered_features(features),
        train_fraction=0.70,
        validation_fraction=0.15,
        embargo_count=min(10, max(0, len(rows) // 20)),
    )
    train_rows = build_gating_dataset(train_features, rubric, target_label_mode=target_label_mode)
    validation_rows = build_gating_dataset(validation_features, rubric, target_label_mode=target_label_mode)
    confirmation_rows = build_gating_dataset(confirmation_features, rubric, target_label_mode=target_label_mode)
    fit_rows = _bounded_rows(train_rows, training_row_cap)
    gate, training_loss = train_gate(
        fit_rows,
        validation_rows=validation_rows,
        init_weights=warm_start_gate.weights if warm_start_gate is not None else None,
        max_epochs=epochs,
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
    if active_stage != "shadow":
        status = "active"
        selected_source = "learned_gate"
        fallback_active = False
        fallback_reason = f"active_stage_{active_stage}"
    else:
        status = "fallback" if performance_blockers else "shadow_candidate"
        selected_source = "baseline_fallback"
        fallback_active = True
        fallback_reason = ";".join((*performance_blockers, "shadow_mode_hard_fallback"))
    return GatingModelReport(
        status=status,
        selected_source=selected_source,
        fallback_active=fallback_active,
        fallback_reason=fallback_reason,
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
        target_label_mode=target_label_mode,
    )


def train_gate(
    rows: Sequence[GatingDatasetRow],
    *,
    validation_rows: Sequence[GatingDatasetRow] = (),
    init_weights: GateWeights | None = None,
    max_epochs: int = 30,
    learning_rate: float = 0.05,
    momentum: float = 0.9,
    l2_penalty: float = 0.001,
    patience: int = 3,
) -> tuple[LogisticGate, float]:
    """Momentum SGD on a softmax-cross-entropy objective with class weighting and early stopping.

    ``init_weights`` warm-starts from a previously-trained gate (continual learning); ``None`` cold-
    starts from the deterministic seed. Returns the best-validation gate when validation rows exist.
    """
    width = len(FEATURE_NAMES) + 1
    params = _init_params(init_weights, width)
    if not rows:
        return LogisticGate(GATING_LABELS, FEATURE_NAMES, _weights_snapshot(params, width)), 0.0
    class_weights = _class_weights(rows)
    velocities = [0.0] * len(params)
    best_weights = _weights_snapshot(params, width)
    best_val = _validation_loss(best_weights, validation_rows, class_weights) if validation_rows else None
    no_improve = 0
    last_loss = 0.0
    for _epoch in range(max(0, max_epochs)):
        for row in rows:
            zero_grad(params)
            loss = _row_loss(
                params, row, l2_penalty=l2_penalty, class_weight=class_weights.get(row.target_label, 1.0)
            )
            last_loss = loss.data
            loss.backward()
            for index, param in enumerate(params):
                velocities[index] = momentum * velocities[index] - learning_rate * param.grad
                param.data += velocities[index]
        if validation_rows:
            current = _weights_snapshot(params, width)
            validation = _validation_loss(current, validation_rows, class_weights)
            if best_val is None or validation < best_val - 1e-9:
                best_val, best_weights, no_improve = validation, current, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    final_weights = best_weights if validation_rows else _weights_snapshot(params, width)
    return LogisticGate(GATING_LABELS, FEATURE_NAMES, final_weights), last_loss


def _row_loss(params: tuple[Value, ...], row: GatingDatasetRow, *, l2_penalty: float, class_weight: float) -> Value:
    values = (1.0, *row.features)
    width = len(values)
    logits: list[Value] = []
    for label_index in range(len(GATING_LABELS)):
        offset = label_index * width
        logit = Value(0.0)
        for index, feature_value in enumerate(values):
            logit = logit + params[offset + index] * feature_value
        logits.append(logit)
    peak = max(logit.data for logit in logits)
    exps = [(logit - peak).exp() for logit in logits]
    total = exps[0]
    for value in exps[1:]:
        total = total + value
    true_index = GATING_LABELS.index(row.target_label)
    prob_true = exps[true_index] / total
    cross_entropy = -prob_true.log() * class_weight
    penalty = Value(0.0)
    for param in params:
        penalty = penalty + param * param
    return cross_entropy + penalty * l2_penalty


def _init_params(init_weights: GateWeights | None, width: int) -> tuple[Value, ...]:
    if init_weights is not None and _weights_shape_ok(init_weights, width):
        return tuple(
            Value(init_weights[label_index][param_index])
            for label_index in range(len(GATING_LABELS))
            for param_index in range(width)
        )
    return tuple(
        Value(_initial_weight(label_index, param_index))
        for label_index in range(len(GATING_LABELS))
        for param_index in range(width)
    )


def _weights_shape_ok(weights: GateWeights, width: int) -> bool:
    return len(weights) == len(GATING_LABELS) and all(len(row) == width for row in weights)


def _weights_snapshot(params: tuple[Value, ...], width: int) -> GateWeights:
    return tuple(
        tuple(params[label_index * width + offset].data for offset in range(width))
        for label_index in range(len(GATING_LABELS))
    )


def _initial_weight(label_index: int, param_index: int) -> float:
    return ((label_index + 1) * (param_index + 3) % 11 - 5) / 500.0


def _class_weights(rows: Sequence[GatingDatasetRow]) -> dict[HumanLabel, float]:
    counts = Counter(row.target_label for row in rows)
    total = len(rows)
    label_count = len(counts) or 1
    return {label: total / (label_count * count) for label, count in counts.items()}


def _validation_loss(
    weights: GateWeights,
    rows: Sequence[GatingDatasetRow],
    class_weights: Mapping[HumanLabel, float],
) -> float:
    if not rows:
        return 0.0
    gate = LogisticGate(GATING_LABELS, FEATURE_NAMES, weights)
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        probabilities = gate.probabilities(row)
        weight = class_weights.get(row.target_label, 1.0)
        total += weight * -log(max(probabilities[row.target_label], 1e-12))
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


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


def _empty_report(
    sample_count: int,
    reason: str,
    *,
    target_label_mode: TargetLabelMode = "absolute_return",
) -> GatingModelReport:
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
        target_label_mode=target_label_mode,
    )
