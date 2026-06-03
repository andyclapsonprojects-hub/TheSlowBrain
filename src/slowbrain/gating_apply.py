"""Apply a promoted learned gate to rubric decisions, and the promotion preconditions.

``confirm_only`` may only **downgrade** a rubric BUY (veto) — it can never create or upgrade a BUY.
``co_decide`` additionally permits a bounded WATCHLIST->BUY upgrade. ``shadow`` is a no-op. These
helpers live apart from the training core in :mod:`slowbrain.gating_model` to keep each module small.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast

from .eval_council import HumanLabel
from .gating_model import GATING_LABELS, GatingModelReport, LogisticGate, build_gating_dataset
from .models import DecisionAction, FeatureVector, RubricVersion, TradeDecision
from .promotion import PromotionStage
from .rubrics import decide_feature

# When the gate is asked to confirm a rubric BUY, these are the labels it must agree are at least
# watch-worthy for the BUY to stand. Anything else (SELL/HOLD/AVOID) vetoes the BUY down to HOLD.
GATE_BUY_CONFIRMING_LABELS: frozenset[HumanLabel] = frozenset({"BUY", "WATCHLIST"})

DecisionPair = tuple[FeatureVector, TradeDecision]


@dataclass(frozen=True)
class GateInfluence:
    idea_id: str
    ticker: str
    from_action: str
    to_action: str
    gate_label: str
    kind: str  # "veto" or "upgrade"


def gate_from_state(
    weights: Sequence[Sequence[float]],
    labels: Sequence[str],
    feature_names: Sequence[str],
) -> LogisticGate | None:
    """Rebuild a usable gate from persisted weights, or None if the persisted gate is empty."""
    if not weights or not labels or not feature_names:
        return None
    typed_labels = tuple(cast(HumanLabel, label) for label in labels)
    return LogisticGate(typed_labels, tuple(feature_names), tuple(tuple(row) for row in weights))


def gate_label_for_feature(feature: FeatureVector, rubric: RubricVersion, gate: LogisticGate) -> HumanLabel | None:
    rows = build_gating_dataset((feature,), rubric)
    if not rows:
        return None
    return gate.predict_label(rows[0])


def apply_gate_to_decisions(
    pairs: Sequence[DecisionPair],
    gate: LogisticGate | None,
    rubric: RubricVersion,
    *,
    stage: PromotionStage,
) -> tuple[tuple[DecisionPair, ...], tuple[GateInfluence, ...]]:
    """Let a promoted gate adjust rubric decisions.

    ``confirm_only``/``co_decide`` keep the rubric in charge (the gate can veto a BUY, or upgrade a
    WATCHLIST at co_decide). ``gate_primary`` flips it: the NN's label becomes the decision and the
    rubric is only a guardrail that vetoes an NN BUY it actively calls SELL/AVOID.
    """
    if stage == "shadow" or gate is None:
        return tuple(pairs), ()
    adjusted: list[DecisionPair] = []
    influences: list[GateInfluence] = []
    for feature, decision in pairs:
        gate_label = gate_label_for_feature(feature, rubric, gate)
        if gate_label is None:
            adjusted.append((feature, decision))
            continue
        if stage == "gate_primary":
            new_decision, kind = _primary_decision(decision, gate_label, rubric)
        else:
            new_decision, kind = _confirm_decision(decision, gate_label, rubric, stage=stage)
        if kind is not None:
            influences.append(
                GateInfluence(feature.idea_id, feature.ticker, decision.action, new_decision.action, gate_label, kind)
            )
        adjusted.append((feature, new_decision))
    return tuple(adjusted), tuple(influences)


def _confirm_decision(
    decision: TradeDecision,
    gate_label: HumanLabel,
    rubric: RubricVersion,
    *,
    stage: PromotionStage,
) -> tuple[TradeDecision, str | None]:
    """confirm_only/co_decide: the gate may only tighten (veto a BUY) or, at co_decide, upgrade WATCHLIST."""
    if decision.action == "BUY" and gate_label not in GATE_BUY_CONFIRMING_LABELS:
        return replace(
            decision, action="HOLD", max_notional_gbp=0.0, reason=f"gate_veto_predicted_{gate_label.lower()}"
        ), "veto"
    if stage == "co_decide" and decision.action == "WATCHLIST" and gate_label == "BUY":
        return replace(
            decision,
            action="BUY",
            max_notional_gbp=round(1000.0 * rubric.max_position_pct, 2),
            reason="gate_upgrade_watchlist_to_buy",
        ), "upgrade"
    return decision, None


def _primary_decision(
    decision: TradeDecision,
    gate_label: HumanLabel,
    rubric: RubricVersion,
) -> tuple[TradeDecision, str | None]:
    """gate_primary: the NN label becomes the decision; the rubric only vetoes an NN BUY it rejects."""
    if gate_label == "BUY":
        if decision.action in {"SELL", "AVOID"}:
            return replace(
                decision, action="HOLD", max_notional_gbp=0.0, reason="gate_primary_guardrail_veto"
            ), "guardrail_veto"
        if decision.action != "BUY":
            return replace(
                decision,
                action="BUY",
                max_notional_gbp=round(1000.0 * rubric.max_position_pct, 2),
                reason="gate_primary_buy",
            ), "primary"
        return decision, None
    if decision.action != gate_label:
        return replace(
            decision,
            action=cast(DecisionAction, gate_label),
            max_notional_gbp=0.0,
            reason=f"gate_primary_{gate_label.lower()}",
        ), "primary"
    return decision, None


def build_gate_primary_pairs(
    features: Sequence[FeatureVector],
    gate: LogisticGate,
    rubric: RubricVersion,
    *,
    limit: int = 10,
) -> tuple[tuple[DecisionPair, ...], tuple[GateInfluence, ...]]:
    """NN-led decisions: relabel each candidate with the NN's call (rubric-guardrailed) and rank by the
    NN's BUY probability, so the network can surface its own BUYs rather than only the rubric's shortlist.
    """
    scored: list[tuple[FeatureVector, TradeDecision, float]] = []
    influences: list[GateInfluence] = []
    for feature in features:
        base = decide_feature(feature, rubric)
        rows = build_gating_dataset((feature,), rubric)
        if not rows:
            scored.append((feature, base, 0.0))
            continue
        probabilities = gate.probabilities(rows[0])
        nn_label = max(GATING_LABELS, key=lambda label: probabilities[label])
        new_decision, kind = _primary_decision(base, nn_label, rubric)
        if kind is not None:
            influences.append(
                GateInfluence(feature.idea_id, feature.ticker, base.action, new_decision.action, nn_label, kind)
            )
        scored.append((feature, new_decision, probabilities["BUY"]))
    ranked = sorted(scored, key=lambda item: (item[1].action != "BUY", -item[2], item[1].ticker))
    pairs = tuple((feature, decision) for feature, decision, _ in ranked[:limit])
    return pairs, tuple(influences)


def gate_decider(
    gate: LogisticGate | None,
    rubric: RubricVersion,
    *,
    stage: PromotionStage,
) -> Callable[[FeatureVector], TradeDecision]:
    """A decision function applying the gate to the rubric decision — used for economic grading."""

    def decide(feature: FeatureVector) -> TradeDecision:
        base = decide_feature(feature, rubric)
        adjusted, _ = apply_gate_to_decisions(((feature, base),), gate, rubric, stage=stage)
        return adjusted[0][1]

    return decide


def gate_secondary_guards_passed(report: GatingModelReport) -> bool:
    """Secondary (non-economic) promotion preconditions: calibration, anchor drift, sample size.

    Mirrors the thresholds in :func:`slowbrain.gating_model._fallback_reasons` so a gate is never
    promoted while it is worse calibrated than the baseline, diverges from the held-out human anchor,
    or lacks confirmation rows.
    """
    return (
        report.confirmation_count >= 30
        and report.gate_confirmation_accuracy > report.baseline_confirmation_accuracy
        and report.gate_brier <= report.baseline_brier
        and report.drift_guard_passed
    )
