"""Slice 3: a promoted gate may only tighten decisions (veto BUYs); never invent them at confirm_only."""

from __future__ import annotations

from slowbrain.gating_apply import (
    apply_gate_to_decisions,
    build_gate_primary_pairs,
    gate_decider,
    gate_from_state,
)
from slowbrain.gating_model import FEATURE_NAMES, GATING_LABELS, LogisticGate
from slowbrain.models import FeatureVector, TradeDecision
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def _force_gate(label: str) -> LogisticGate:
    width = len(FEATURE_NAMES) + 1
    rows = tuple((100.0 if name == label else 0.0, *([0.0] * (width - 1))) for name in GATING_LABELS)
    return LogisticGate(GATING_LABELS, FEATURE_NAMES, rows)


def _buy_feature() -> FeatureVector:
    return FeatureVector(
        idea_id="buy-1",
        ticker="BUYCO",
        signal_date="2026-01-01",
        sentiment="positive",
        sentiment_confidence=1.0,
        catalyst_strength=1.0,
        trend="uptrend",
        momentum_20d_pct=20.0,
        mean_reversion_z_20d=-1.5,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=5.0,
        cost_bps=30.0,
        source="unit",
        entry_price=100.0,
    )


def _watchlist_feature() -> FeatureVector:
    feature = FeatureVector(
        idea_id="watch-1",
        ticker="WCH",
        signal_date="2026-01-02",
        sentiment="positive",
        sentiment_confidence=1.0,
        catalyst_strength=0.0,
        trend="uptrend",
        momentum_20d_pct=0.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=1.0,
        cost_bps=30.0,
        source="unit",
        entry_price=100.0,
    )
    assert decide_feature(feature, BASE_RUBRIC).action == "WATCHLIST"
    return feature


def _pairs(feature: FeatureVector) -> tuple[tuple[FeatureVector, TradeDecision], ...]:
    return ((feature, decide_feature(feature, BASE_RUBRIC)),)


def test_shadow_stage_is_identity() -> None:
    pairs = _pairs(_buy_feature())
    out, influences = apply_gate_to_decisions(pairs, _force_gate("SELL"), BASE_RUBRIC, stage="shadow")
    assert out == pairs
    assert influences == ()


def test_confirm_only_vetoes_a_disagreed_buy() -> None:
    out, influences = apply_gate_to_decisions(
        _pairs(_buy_feature()), _force_gate("SELL"), BASE_RUBRIC, stage="confirm_only"
    )
    assert out[0][1].action == "HOLD"
    assert out[0][1].max_notional_gbp == 0.0
    assert len(influences) == 1 and influences[0].kind == "veto" and influences[0].gate_label == "SELL"


def test_confirm_only_keeps_a_confirmed_buy() -> None:
    for confirming in ("BUY", "WATCHLIST"):
        out, influences = apply_gate_to_decisions(
            _pairs(_buy_feature()), _force_gate(confirming), BASE_RUBRIC, stage="confirm_only"
        )
        assert out[0][1].action == "BUY"
        assert influences == ()


def test_confirm_only_never_upgrades_a_watchlist() -> None:
    out, influences = apply_gate_to_decisions(
        _pairs(_watchlist_feature()), _force_gate("BUY"), BASE_RUBRIC, stage="confirm_only"
    )
    assert out[0][1].action == "WATCHLIST"
    assert influences == ()


def test_co_decide_can_upgrade_a_watchlist_when_gate_is_confident() -> None:
    out, influences = apply_gate_to_decisions(
        _pairs(_watchlist_feature()), _force_gate("BUY"), BASE_RUBRIC, stage="co_decide"
    )
    assert out[0][1].action == "BUY"
    assert out[0][1].max_notional_gbp > 0.0
    assert len(influences) == 1 and influences[0].kind == "upgrade"


def test_empty_gate_state_is_none_and_applies_as_identity() -> None:
    assert gate_from_state((), (), ()) is None
    pairs = _pairs(_buy_feature())
    out, influences = apply_gate_to_decisions(pairs, None, BASE_RUBRIC, stage="confirm_only")
    assert out == pairs and influences == ()


def test_gate_decider_vetoes_for_economic_grading() -> None:
    decide = gate_decider(_force_gate("HOLD"), BASE_RUBRIC, stage="confirm_only")
    assert decide(_buy_feature()).action == "HOLD"
    keep = gate_decider(_force_gate("BUY"), BASE_RUBRIC, stage="confirm_only")
    assert keep(_buy_feature()).action == "BUY"


def _hold_feature() -> FeatureVector:
    feature = FeatureVector(
        idea_id="hold-1",
        ticker="HLD",
        signal_date="2026-01-03",
        sentiment="positive",
        sentiment_confidence=1.0,
        catalyst_strength=0.0,
        trend="sideways",
        momentum_20d_pct=0.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=False,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=0.0,
        cost_bps=30.0,
        source="unit",
        entry_price=100.0,
    )
    assert decide_feature(feature, BASE_RUBRIC).action == "HOLD"
    return feature


def _sell_feature() -> FeatureVector:
    feature = FeatureVector(
        idea_id="sell-1",
        ticker="SLL",
        signal_date="2026-01-04",
        sentiment="negative",
        sentiment_confidence=0.9,
        catalyst_strength=0.2,
        trend="downtrend",
        momentum_20d_pct=-10.0,
        mean_reversion_z_20d=1.0,
        volume_confirmed=False,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=-3.0,
        cost_bps=30.0,
        source="unit",
        entry_price=100.0,
    )
    assert decide_feature(feature, BASE_RUBRIC).action == "SELL"
    return feature


def test_gate_primary_makes_the_nn_label_the_decision() -> None:
    # NN is primary: it can downgrade a rubric BUY to HOLD...
    out, influences = apply_gate_to_decisions(
        _pairs(_buy_feature()), _force_gate("HOLD"), BASE_RUBRIC, stage="gate_primary"
    )
    assert out[0][1].action == "HOLD"
    assert influences[0].kind == "primary"


def test_gate_primary_can_create_a_buy_the_rubric_did_not() -> None:
    out, influences = apply_gate_to_decisions(
        _pairs(_hold_feature()), _force_gate("BUY"), BASE_RUBRIC, stage="gate_primary"
    )
    assert out[0][1].action == "BUY"
    assert out[0][1].max_notional_gbp > 0.0
    assert influences[0].kind == "primary"


def test_gate_primary_rubric_guardrail_vetoes_an_nn_buy_it_rejects() -> None:
    out, influences = apply_gate_to_decisions(
        _pairs(_sell_feature()), _force_gate("BUY"), BASE_RUBRIC, stage="gate_primary"
    )
    assert out[0][1].action == "HOLD"
    assert influences[0].kind == "guardrail_veto"


def test_build_gate_primary_pairs_surfaces_and_ranks_nn_buys() -> None:
    features = (_hold_feature(), _sell_feature())  # rubric would BUY neither
    pairs, influences = build_gate_primary_pairs(features, _force_gate("BUY"), BASE_RUBRIC, limit=10)
    actions = [decision.action for _, decision in pairs]
    # The NN surfaces a BUY from a feature the rubric did not buy; the guardrail blocks the SELL one.
    assert "BUY" in actions
    assert actions[0] == "BUY"  # BUYs rank first
    assert any(influence.kind == "primary" for influence in influences)
    assert any(influence.kind == "guardrail_veto" for influence in influences)
