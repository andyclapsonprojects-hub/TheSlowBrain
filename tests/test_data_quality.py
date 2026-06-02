from __future__ import annotations

from slowbrain.data_quality import check_outcome_plausibility
from slowbrain.features import _row_to_feature


def test_feature_loader_records_malformed_json_and_invalid_numbers() -> None:
    feature = _row_to_feature(
        (
            "idea-1",
            "msft",
            "2026-01-01",
            "positive",
            "not-a-number",
            None,
            "pass",
            "pass",
            "{bad json",
            "1.2",
            None,
            "fixture",
        )
    )

    codes = {issue.code for issue in feature.data_quality_issues}

    assert "malformed_json" in codes
    assert "invalid_float" in codes
    assert "missing_float" in codes
    assert feature.sentiment_confidence == 0.0


def test_check_outcome_plausibility_flags_penny_stock_artefacts() -> None:
    # Real artefact from the legacy cache: OTLK 0.233 close with +212% 10d return.
    otlk = check_outcome_plausibility(close_price=0.233, forward_return_pct=212.85)
    healthy = check_outcome_plausibility(close_price=68.03, forward_return_pct=10.85)
    big_move_normal_price = check_outcome_plausibility(close_price=25.0, forward_return_pct=80.0)
    missing = check_outcome_plausibility(close_price=None, forward_return_pct=212.85)

    assert otlk is not None
    assert otlk.severity == "error"
    assert otlk.code == "implausible_outcome_for_penny_stock"
    # A normal-priced stock with a big move is NOT auto-flagged by this rule.
    assert healthy is None
    assert big_move_normal_price is None
    assert missing is None
