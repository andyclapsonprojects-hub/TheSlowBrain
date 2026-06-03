"""Earned, reversible promotion state machine for the learned gating network.

The gate climbs a ladder (``shadow`` -> ``confirm_only`` -> ``co_decide``) only by clearing BOTH a
PRIMARY economic gate (its decisions beat the rubric on after-cost, out-of-sample, overfitting-guarded
return) AND SECONDARY safety guards (calibration / anchor drift), for ``required_streak`` consecutive
runs. Any failure demotes one stage and resets the streak; an operator override forces a stage. The
machine is a pure function of the previous state and this run's pass/fail flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

PromotionStage = Literal["shadow", "confirm_only", "co_decide", "gate_primary"]

# The ladder tops out at gate_primary: the NN proposes decisions and the rubric becomes a guardrail.
# Each rung is earned the same way — by beating the rubric on out-of-sample profit for K runs.
LADDER: tuple[PromotionStage, ...] = ("shadow", "confirm_only", "co_decide", "gate_primary")
DEFAULT_REQUIRED_STREAK = 10
_HISTORY_CAP = 20


@dataclass(frozen=True)
class PromotionEvent:
    stage: PromotionStage
    status: str
    reason: str
    qualifying_streak: int


@dataclass(frozen=True)
class PromotionState:
    stage: PromotionStage = "shadow"
    qualifying_streak: int = 0
    required_streak: int = DEFAULT_REQUIRED_STREAK
    last_status: str = "initial"
    last_reason: str = "no_evaluation_yet"
    history: tuple[PromotionEvent, ...] = field(default_factory=tuple)

    @property
    def is_promoted(self) -> bool:
        return self.stage != "shadow"


def evaluate_promotion(
    state: PromotionState,
    *,
    economic_pass: bool,
    secondary_pass: bool,
    override: PromotionStage | None = None,
) -> PromotionState:
    """Advance, hold, demote, or override the gate's promotion stage for the next run."""
    if override is not None:
        return _transition(
            state,
            stage=override,
            qualifying_streak=0,
            status="override",
            reason=f"operator_override_to_{override}",
        )

    if economic_pass and secondary_pass:
        streak = state.qualifying_streak + 1
        if streak >= state.required_streak and _can_promote(state.stage):
            promoted = next_stage(state.stage)
            return _transition(
                state,
                stage=promoted,
                qualifying_streak=0,
                status="promoted",
                reason=f"promoted_to_{promoted}_after_{streak}_qualifying_runs",
            )
        return _transition(
            state,
            stage=state.stage,
            qualifying_streak=streak,
            status="qualifying",
            reason=f"qualifying_streak_{streak}_of_{state.required_streak}",
        )

    reason = _failure_reason(economic_pass=economic_pass, secondary_pass=secondary_pass)
    if state.is_promoted:
        demoted = _prev_stage(state.stage)
        return _transition(
            state,
            stage=demoted,
            qualifying_streak=0,
            status="demoted",
            reason=f"demoted_to_{demoted}_because_{reason}",
        )
    return _transition(
        state,
        stage="shadow",
        qualifying_streak=0,
        status="not_qualifying",
        reason=reason,
    )


def _transition(
    state: PromotionState,
    *,
    stage: PromotionStage,
    qualifying_streak: int,
    status: str,
    reason: str,
) -> PromotionState:
    event = PromotionEvent(stage=stage, status=status, reason=reason, qualifying_streak=qualifying_streak)
    history = (*state.history, event)[-_HISTORY_CAP:]
    return replace(
        state,
        stage=stage,
        qualifying_streak=qualifying_streak,
        last_status=status,
        last_reason=reason,
        history=history,
    )


def next_stage(stage: PromotionStage) -> PromotionStage:
    """The stage one rung above ``stage`` (clamped at the top of the ladder)."""
    return LADDER[min(LADDER.index(stage) + 1, len(LADDER) - 1)]


def _prev_stage(stage: PromotionStage) -> PromotionStage:
    return LADDER[max(LADDER.index(stage) - 1, 0)]


def _can_promote(stage: PromotionStage) -> bool:
    return stage != LADDER[-1]


def _failure_reason(*, economic_pass: bool, secondary_pass: bool) -> str:
    if not economic_pass and not secondary_pass:
        return "economic_and_secondary_failed"
    if not economic_pass:
        return "economic_gate_failed"
    return "secondary_guard_failed"
