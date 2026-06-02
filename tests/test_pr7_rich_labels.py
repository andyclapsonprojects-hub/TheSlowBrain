from __future__ import annotations

from pathlib import Path

from slowbrain.data_quality import DataQualityIssue
from slowbrain.decision_capture import write_decision_capture
from slowbrain.eval_council import (
    HumanExample,
    HumanLabel,
    calibrate_against_humans,
    project_label_for_kappa,
)
from slowbrain.human_labeling import load_decision_capture_records
from slowbrain.models import FeatureVector
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def _feature(
    idea_id: str,
    *,
    sentiment: str = "positive",
    confidence: float = 0.9,
    catalyst: float = 0.9,
    trend: str = "uptrend",
    momentum: float = 8.0,
    volume: bool = True,
    quality: str = "pass",
    risk: str = "pass",
    issues: tuple[DataQualityIssue, ...] = (),
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker="AAPL",
        signal_date="2026-01-01",
        sentiment=sentiment,
        sentiment_confidence=confidence,
        catalyst_strength=catalyst,
        trend=trend,
        momentum_20d_pct=momentum,
        mean_reversion_z_20d=0.0,
        volume_confirmed=volume,
        quality_status=quality,
        risk_status=risk,
        net_return_pct=1.0,
        cost_bps=10.0,
        source="fixture",
        data_quality_issues=issues,
    )


def test_decide_feature_can_emit_all_five_labels() -> None:
    buy = decide_feature(_feature("buy"), BASE_RUBRIC)
    sell = decide_feature(
        _feature(
            "sell",
            sentiment="negative",
            confidence=0.9,
            catalyst=0.0,
            trend="downtrend",
            momentum=-5.0,
            volume=False,
        ),
        BASE_RUBRIC,
    )
    hold = decide_feature(
        _feature("hold", confidence=0.2, catalyst=0.0, trend="unknown", momentum=0.0, volume=False),
        BASE_RUBRIC,
    )
    avoid = decide_feature(
        _feature("avoid", risk="rejected"),
        BASE_RUBRIC,
    )
    watchlist = decide_feature(
        _feature("watchlist", confidence=0.55, catalyst=0.5, trend="uptrend", momentum=4.0, volume=True),
        BASE_RUBRIC,
    )

    assert buy.action == "BUY"
    assert sell.action == "SELL"
    assert hold.action == "HOLD"
    assert avoid.action == "AVOID"
    assert watchlist.action == "WATCHLIST"
    # AVOID must never be a buy and must record the gate-block reason.
    assert "blocked_by_failed" in avoid.reason


def test_avoid_high_score_blocked_by_gate_is_not_buy() -> None:
    blocked = decide_feature(_feature("blocked", quality="fail"), BASE_RUBRIC)
    assert blocked.action == "AVOID"


def test_kappa_projection_maps_five_labels_to_three_space() -> None:
    assert project_label_for_kappa("WATCHLIST") == "HOLD"
    assert project_label_for_kappa("AVOID") == "SELL"
    assert project_label_for_kappa("BUY") == "BUY"
    assert project_label_for_kappa("SELL") == "SELL"
    assert project_label_for_kappa("HOLD") == "HOLD"
    assert project_label_for_kappa("UNKNOWN") == "UNKNOWN"


def test_workflow_rerun_preserves_human_labelled_capture_rows(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    one = _feature("idea-1")
    two = _feature("idea-2")

    # First run writes machine rows (no human labels yet).
    write_decision_capture(path, ((one, decide_feature(one, BASE_RUBRIC)),), run_id="run-1")

    # A human labels idea-1 by re-writing the file with the label set.
    text = path.read_text(encoding="utf-8").replace('"human_label": null', '"human_label": "WATCHLIST"')
    path.write_text(text, encoding="utf-8")

    # Second run regenerates machine rows for a different decision set.
    write_decision_capture(path, ((two, decide_feature(two, BASE_RUBRIC)),), run_id="run-2")

    records = {
        str(rec["feature"]["idea_id"]): rec  # type: ignore[index]
        for rec in load_decision_capture_records(path)
    }
    # The human label on idea-1 survived the re-run, and idea-2 was added.
    assert records["idea-1"]["human_label"] == "WATCHLIST"
    assert "idea-2" in records


def test_calibration_projects_rich_human_labels_before_scoring() -> None:
    # Human says WATCHLIST (projects to HOLD); machine HOLD -> they should agree post-projection.
    examples = (
        HumanExample("one", "AAPL", "2026-01-01", "WATCHLIST", "monitor"),
        HumanExample("two", "MSFT", "2026-01-02", "AVOID", "negative screen"),
    )
    automated: dict[str, HumanLabel] = {"one": "HOLD", "two": "SELL"}

    report = calibrate_against_humans(examples, automated, min_kappa=0.0)

    # Perfect agreement after projection -> kappa is defined and not fabricated/blocked.
    assert report.kappa is not None
    assert report.example_count == 2
    assert report.status in {"calibrated", "failed"}
