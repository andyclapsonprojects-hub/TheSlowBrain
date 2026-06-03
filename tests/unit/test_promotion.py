"""Slice 2: earned, reversible promotion state machine + persistence."""

from __future__ import annotations

from pathlib import Path

from slowbrain.learning_state import load_promotion_state, persist_promotion_state
from slowbrain.promotion import PromotionState, evaluate_promotion


def _qualify(state: PromotionState) -> PromotionState:
    return evaluate_promotion(state, economic_pass=True, secondary_pass=True)


def test_default_state_is_shadow() -> None:
    state = PromotionState(required_streak=3)
    assert state.stage == "shadow"
    assert state.qualifying_streak == 0
    assert state.is_promoted is False


def test_qualifying_increments_streak_but_holds_until_threshold() -> None:
    state = PromotionState(required_streak=3)
    state = _qualify(state)
    assert state.stage == "shadow" and state.qualifying_streak == 1
    state = _qualify(state)
    assert state.stage == "shadow" and state.qualifying_streak == 2


def test_promotes_one_stage_after_required_streak() -> None:
    state = PromotionState(required_streak=3)
    for _ in range(3):
        state = _qualify(state)
    assert state.stage == "confirm_only"
    assert state.qualifying_streak == 0  # streak resets; the next stage must be re-earned
    assert state.last_status == "promoted"


def test_promotes_one_stage_at_a_time_up_to_the_top() -> None:
    state = PromotionState(required_streak=2)
    for _ in range(2):
        state = _qualify(state)
    assert state.stage == "confirm_only"
    for _ in range(2):
        state = _qualify(state)
    assert state.stage == "co_decide"
    for _ in range(2):
        state = _qualify(state)
    assert state.stage == "gate_primary"
    # gate_primary is the ceiling — further qualifying never escalates beyond it.
    for _ in range(5):
        state = _qualify(state)
    assert state.stage == "gate_primary"


def test_economic_failure_demotes_and_resets() -> None:
    state = PromotionState(required_streak=2)
    state = _qualify(state)
    state = _qualify(state)
    assert state.stage == "confirm_only"
    state = evaluate_promotion(state, economic_pass=False, secondary_pass=True)
    assert state.stage == "shadow"
    assert state.qualifying_streak == 0
    assert "economic_gate_failed" in state.last_reason


def test_secondary_failure_demotes_when_promoted() -> None:
    state = PromotionState(required_streak=1)
    state = _qualify(state)  # -> confirm_only
    assert state.stage == "confirm_only"
    state = evaluate_promotion(state, economic_pass=True, secondary_pass=False)
    assert state.stage == "shadow"
    assert "secondary_guard_failed" in state.last_reason


def test_failure_while_shadow_stays_shadow() -> None:
    state = PromotionState(required_streak=2)
    state = evaluate_promotion(state, economic_pass=False, secondary_pass=False)
    assert state.stage == "shadow"
    assert state.qualifying_streak == 0


def test_override_forces_stage_and_resets_streak() -> None:
    state = PromotionState(required_streak=1)
    state = _qualify(state)  # -> confirm_only
    forced = evaluate_promotion(state, economic_pass=True, secondary_pass=True, override="shadow")
    assert forced.stage == "shadow"
    assert forced.qualifying_streak == 0
    assert "operator_override_to_shadow" in forced.last_reason


def test_history_is_recorded_and_bounded() -> None:
    state = PromotionState(required_streak=1)
    for _ in range(30):
        state = _qualify(state)
    assert len(state.history) <= 20
    assert state.history[-1].stage == "gate_primary"


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "gating_promotion.json"
    state = PromotionState(required_streak=2)
    state = _qualify(state)
    state = _qualify(state)  # -> confirm_only
    persist_promotion_state(path, state, run_id="run-1")

    loaded = load_promotion_state(path, required_streak=2)
    assert loaded.stage == "confirm_only"
    assert loaded.qualifying_streak == state.qualifying_streak
    assert loaded.required_streak == 2
    assert loaded.last_status == state.last_status
    assert tuple(event.reason for event in loaded.history) == tuple(event.reason for event in state.history)


def test_missing_state_file_returns_default(tmp_path: Path) -> None:
    loaded = load_promotion_state(tmp_path / "absent.json", required_streak=7)
    assert loaded.stage == "shadow"
    assert loaded.required_streak == 7
