from __future__ import annotations

from pathlib import Path

from slowbrain.backtest import evaluate_rubric
from slowbrain.data_quality import DataQualityIssue
from slowbrain.eval_council import CalibrationReport, HumanExample
from slowbrain.gating_model import (
    build_gating_dataset,
    evaluate_gating_model,
    target_label_for_return,
)
from slowbrain.microgix import Value
from slowbrain.models import FeatureVector, PortfolioState, PromotionDecision
from slowbrain.reporting import build_eric_brief, write_first_report
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def test_microgix_backpropagates_scalar_expression() -> None:
    x = Value(2.0)
    y = x * x + 3.0 * x + 1.0

    y.backward()

    assert y.data == 11.0
    assert x.grad == 7.0


def test_gating_targets_discretize_forward_returns_to_five_labels() -> None:
    assert target_label_for_return(3.0) == "BUY"
    assert target_label_for_return(0.7) == "WATCHLIST"
    assert target_label_for_return(0.0) == "HOLD"
    assert target_label_for_return(-0.7) == "AVOID"
    assert target_label_for_return(-3.0) == "SELL"


def test_gating_dataset_excludes_error_rows() -> None:
    clean = _feature("clean", net=1.0)
    bad = _feature(
        "bad",
        net=5.0,
        issues=(DataQualityIssue("outcome", "implausible_outcome_for_penny_stock", "error", "bad"),),
    )

    rows = build_gating_dataset((bad, clean), BASE_RUBRIC)

    assert [row.idea_id for row in rows] == ["clean"]
    assert rows[0].target_label == "WATCHLIST"


def test_gating_model_falls_back_when_confirmation_or_anchor_guard_fails() -> None:
    features = tuple(_feature(f"row-{index}", net=(-1.0 if index % 3 == 0 else 0.2)) for index in range(80))
    anchor = _feature("anchor-1", sentiment="negative", confidence=0.9, trend="downtrend", net=-3.0)
    human = (HumanExample("anchor-1", "AAPL", "2026-01-01", "AVOID", "negative screen"),)

    report = evaluate_gating_model(features, BASE_RUBRIC, human_examples=human, anchor_features=(anchor,), epochs=0)

    assert report.fallback_active
    assert report.selected_source == "baseline_fallback"
    assert "pr9_shadow_only_hard_fallback" in report.fallback_reason
    assert report.anchor_count == 1
    assert report.baseline_anchor_kappa == 1.0
    assert report.drift_guard_passed is False


def test_report_contains_pr9_gating_model_evidence(tmp_path: Path) -> None:
    feature = _feature("idea-1", net=1.0)
    decision = decide_feature(feature, BASE_RUBRIC)
    promotion = PromotionDecision(
        action="reject",
        selected_version=BASE_RUBRIC.version,
        active_version=BASE_RUBRIC.version,
        reason="fixture",
        current_result=evaluate_rubric((feature,) * 6, BASE_RUBRIC, min_test_trades=0),
    )
    gating_report = evaluate_gating_model(tuple(_feature(f"row-{index}", net=1.0) for index in range(40)), BASE_RUBRIC)
    portfolio = PortfolioState(holdings=("AAPL",), profit_since_first_trade_pct=1.0)

    payload = write_first_report(
        output_json=tmp_path / "report.json",
        output_md=tmp_path / "report.md",
        promotion=promotion,
        decisions=(decision,),
        portfolio=portfolio,
        brief=build_eric_brief((decision,), portfolio),
        human_calibration=CalibrationReport("failed", True, 30, 0.0, ("low",)),
        gating_model=gating_report,
    )

    gating = payload["gating_model"]
    assert isinstance(gating, dict)
    assert gating["selected_source"] == "baseline_fallback"
    assert gating["fallback_active"] is True
    assert "fallback_reason" in gating


def _feature(
    idea_id: str,
    *,
    sentiment: str = "positive",
    confidence: float = 0.9,
    trend: str = "uptrend",
    net: float = 1.0,
    issues: tuple[DataQualityIssue, ...] = (),
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker="AAPL",
        signal_date="2026-01-01",
        sentiment=sentiment,
        sentiment_confidence=confidence,
        catalyst_strength=0.9,
        trend=trend,
        momentum_20d_pct=8.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=10.0,
        source="fixture",
        data_quality_issues=tuple(issues),
        horizon_days=10,
        entry_price=100.0,
    )
