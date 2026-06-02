from __future__ import annotations

from slowbrain.models import FeatureVector, RubricCandidate, RubricVersion
from slowbrain.optimizer import select_rubric
from slowbrain.rubrics import BASE_RUBRIC


def test_optimizer_rejects_validation_winner_that_fails_confirmation() -> None:
    active = RubricVersion(
        version="active",
        weights=BASE_RUBRIC.weights,
        buy_threshold=0.58,
        sell_threshold=-0.35,
        max_position_pct=0.05,
    )
    candidate = RubricVersion(
        version="candidate_lower_threshold",
        weights=BASE_RUBRIC.weights,
        buy_threshold=0.40,
        sell_threshold=-0.35,
        max_position_pct=0.05,
    )
    train = [feature(f"train_{index}", signal_date=f"2026-01-{index + 1:02d}", net=1.0) for index in range(10)]
    validation = [
        feature(
            f"validation_{index}",
            signal_date=f"2026-02-{index + 1:02d}",
            confidence=0.5,
            catalyst=0.3,
            momentum=3.0,
            volume=False,
            net=4.0,
        )
        for index in range(3)
    ]
    confirmation = [
        feature(
            f"confirmation_{index}",
            signal_date=f"2026-03-{index + 1:02d}",
            confidence=0.5,
            catalyst=0.3,
            momentum=3.0,
            volume=False,
            net=-3.0,
        )
        for index in range(5)
    ]

    decision = select_rubric(
        active=active,
        candidates=(RubricCandidate("candidate", "active", candidate, "fixture"),),
        features=tuple(train + validation + confirmation),
        min_profit_improvement_pct=0.1,
    )

    assert decision.action == "reject"
    assert "confirmation_profit_improvement_below_guard" in decision.gaps


def feature(
    idea_id: str,
    *,
    signal_date: str,
    confidence: float = 0.8,
    catalyst: float = 0.7,
    momentum: float = 8.0,
    volume: bool = True,
    net: float = 1.0,
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker="AAPL",
        signal_date=signal_date,
        sentiment="positive",
        sentiment_confidence=confidence,
        catalyst_strength=catalyst,
        trend="uptrend",
        momentum_20d_pct=momentum,
        mean_reversion_z_20d=0.0,
        volume_confirmed=volume,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=45.0,
        source="fixture",
    )
