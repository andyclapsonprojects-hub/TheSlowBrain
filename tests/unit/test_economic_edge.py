"""Slice 1: the gate is graded on profit, via the same backtest machinery as the rubric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from slowbrain.backtest import economic_edge, evaluate_rubric
from slowbrain.models import FeatureVector, TradeDecision
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def _buy_feature(index: int, *, net: float) -> FeatureVector:
    """A feature the seed rubric scores as a clean BUY; ``net`` is its forward outcome."""
    return FeatureVector(
        idea_id=f"idea-{index:04d}",
        ticker=f"TIC{index:04d}",
        signal_date=f"2026-01-{index % 28 + 1:02d}",
        sentiment="positive",
        sentiment_confidence=1.0,
        catalyst_strength=1.0,
        trend="uptrend",
        momentum_20d_pct=20.0,
        mean_reversion_z_20d=-1.5,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=30.0,
        source="unit_fixture",
        horizon_days=10,
        entry_price=100.0,
    )


def _features() -> tuple[FeatureVector, ...]:
    # Alternating winners (+) and losers (-) so every holdout slice contains both.
    return tuple(_buy_feature(i, net=6.0 if i % 2 == 0 else -6.0) for i in range(40))


def _veto(predicate: Callable[[FeatureVector], bool]) -> Callable[[FeatureVector], TradeDecision]:
    def decide(feature: FeatureVector) -> TradeDecision:
        decision = decide_feature(feature, BASE_RUBRIC)
        if decision.action == "BUY" and predicate(feature):
            return replace(decision, action="HOLD", max_notional_gbp=0.0)
        return decision
    return decide


def test_features_are_clean_buys() -> None:
    assert all(decide_feature(feature, BASE_RUBRIC).action == "BUY" for feature in _features())


def test_decide_none_preserves_rubric_behaviour() -> None:
    features = _features()
    explicit = evaluate_rubric(features, BASE_RUBRIC, decide=lambda f: decide_feature(f, BASE_RUBRIC))
    default = evaluate_rubric(features, BASE_RUBRIC)
    assert explicit.confirmation_total_net_profit_pct == default.confirmation_total_net_profit_pct
    assert explicit.total_net_profit_pct == default.total_net_profit_pct


def test_identity_gate_does_not_beat_the_rubric() -> None:
    edge = economic_edge(_features(), BASE_RUBRIC, gate_decide=lambda f: decide_feature(f, BASE_RUBRIC))
    assert edge.confirmation_return_delta_pct == 0.0
    assert edge.gate_confirmation_trades == edge.rubric_confirmation_trades
    assert edge.gate_beats_rubric is False


def test_vetoing_losers_raises_confirmation_return_and_cuts_trades() -> None:
    edge = economic_edge(
        _features(),
        BASE_RUBRIC,
        gate_decide=_veto(lambda feature: feature.net_return_pct < 0.0),
    )
    assert edge.confirmation_return_delta_pct >= 0.0
    assert edge.gate_confirmation_trades < edge.rubric_confirmation_trades
    assert edge.gate_confirmation_return_pct >= edge.rubric_confirmation_return_pct


def test_vetoing_winners_cannot_beat_the_rubric() -> None:
    edge = economic_edge(
        _features(),
        BASE_RUBRIC,
        gate_decide=_veto(lambda feature: feature.net_return_pct > 0.0),
    )
    assert edge.confirmation_return_delta_pct <= 0.0
    assert edge.gate_beats_rubric is False
