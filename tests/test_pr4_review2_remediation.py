from __future__ import annotations

import json
from pathlib import Path

from slowbrain.accounting import calculate_mark_to_market_from_positions
from slowbrain.backtest import evaluate_rubric
from slowbrain.costs import estimate_trade_cost, liquidity_profile
from slowbrain.data_quality import DataQualityIssue
from slowbrain.decision_capture import write_decision_capture
from slowbrain.eval_council import JUDGE_DIMENSIONS, review_promotion_quality
from slowbrain.models import FeatureVector, RubricCandidate, RubricVersion
from slowbrain.optimizer import select_rubric
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def test_backtest_rejects_zero_variance_confirmation_returns() -> None:
    result = evaluate_rubric(tuple(feature(f"constant_{index}", net=1.0) for index in range(100)), BASE_RUBRIC)

    assert "degenerate_zero_variance_confirmation_returns" in result.guard_failures
    assert result.p_value == 1.0
    assert not result.survived_guards


def test_error_severity_data_quality_excludes_feature_from_buy_and_backtest() -> None:
    bad = feature(
        "bad",
        issues=(DataQualityIssue("payload", "malformed_json", "error", "bad fixture"),),
    )

    decision = decide_feature(bad, BASE_RUBRIC)
    result = evaluate_rubric((bad, feature("good")), BASE_RUBRIC, min_test_trades=0)

    # A high-scoring feature blocked by an error-severity gate is an active negative
    # screen (AVOID), not a passive HOLD; the safety guarantee is that it is never BUY
    # and is excluded from the backtest.
    assert decision.action == "AVOID"
    assert result.excluded_error_feature_count == 1
    assert "data_quality_errors_excluded" in result.guard_failures


def test_per_name_liquidity_changes_costs() -> None:
    liquid = feature("liquid", ticker="AAPL")
    thin = feature("thin", ticker="ZZZZ")

    assert liquidity_profile(liquid).avg_daily_volume_gbp != liquidity_profile(thin).avg_daily_volume_gbp
    assert estimate_trade_cost(liquid).total_cost_bps != estimate_trade_cost(thin).total_cost_bps


def test_mark_to_market_uses_open_position_prices(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    path.write_text(
        "ticker,status,entry_price,quantity,current_price\nAAA,open,10,2,12\nBBB,closed,10,1,9\n",
        encoding="utf-8",
    )

    summary = calculate_mark_to_market_from_positions(path)

    assert summary.unrealized_profit_gbp == 4.0
    assert summary.open_market_value_gbp == 24.0
    assert summary.quality == "marked_to_market_from_position_prices"


def test_decision_capture_writes_labelable_jsonl(tmp_path: Path) -> None:
    current_feature = feature("capture")
    decision = decide_feature(current_feature, BASE_RUBRIC)

    path = write_decision_capture(tmp_path / "decisions.jsonl", ((current_feature, decision),), run_id="fixture")
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["schema"] == "theslowbrain.golden_decision.v1"
    assert record["human_label"] is None
    assert record["feature"]["idea_id"] == "capture"
    assert record["decision"]["rubric_version"] == BASE_RUBRIC.version
    assert record["outcome"]["realized_net_return_pct"] == current_feature.net_return_pct


def test_tier1_council_has_review2_dimensions_and_can_fail_overfit_candidate() -> None:
    review = review_promotion_quality(
        decision_id="fixture",
        candidate_metrics={
            "candidate_reason": "fixture",
            "p_value": 0.50,
            "max_drawdown_pct": 5.0,
            "alpha_vs_benchmark_pct": -1.0,
            "deflated_sharpe": -0.5,
            "probability_backtest_overfit": 0.90,
            "capacity_ok": True,
            "excluded_error_feature_count": 0,
        },
    )

    assert {"overfitting_robustness", "economic_rationale"}.issubset(set(JUDGE_DIMENSIONS))
    assert review.aggregate_outcome == "fail"
    assert "overfitting_robustness" in {vote.dimension for vote in review.votes if vote.outcome == "fail"}


def test_optimizer_requires_council_quality_and_strict_significance() -> None:
    active = RubricVersion(
        version="active",
        weights=BASE_RUBRIC.weights,
        buy_threshold=0.80,
        sell_threshold=-0.35,
        max_position_pct=0.05,
    )
    candidate = RubricVersion(
        version="candidate",
        weights=BASE_RUBRIC.weights,
        buy_threshold=0.40,
        sell_threshold=-0.35,
        max_position_pct=0.05,
    )
    decision = select_rubric(
        active=active,
        candidates=(RubricCandidate("candidate", "active", candidate, "fixture reason"),),
        features=tuple(feature(f"constant_{index}", net=1.0) for index in range(100)),
        min_profit_improvement_pct=0.0,
    )

    assert decision.action != "adopt"
    assert "multiple_testing_adjusted_significance_failed" in decision.gaps
    assert "council_quality_gate_failed" in decision.gaps


def feature(
    idea_id: str,
    *,
    ticker: str = "AAPL",
    net: float = 1.0,
    issues: tuple[DataQualityIssue, ...] = (),
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker=ticker,
        signal_date="2026-01-01",
        sentiment="positive",
        sentiment_confidence=0.9,
        catalyst_strength=0.9,
        trend="uptrend",
        momentum_20d_pct=8.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=10.0,
        source="fixture",
        data_quality_issues=issues,
    )
