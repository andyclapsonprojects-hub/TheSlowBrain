from __future__ import annotations

import json
from pathlib import Path

from slowbrain.decision_capture import append_decision_outcome_stream
from slowbrain.gating_model import evaluate_gating_model
from slowbrain.learning_state import (
    append_track_record,
    load_active_rubric,
    load_outcome_stream_features,
    merge_feature_evidence,
    persist_active_rubric,
    persist_gating_gate,
)
from slowbrain.models import FeatureVector
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def test_active_rubric_state_round_trips_and_falls_back_on_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "state" / "active_rubric.json"
    persist_active_rubric(path, BASE_RUBRIC, run_id="run-1", promotion_action="reject", reason="fixture")

    loaded = load_active_rubric(path, default=BASE_RUBRIC)

    assert loaded == BASE_RUBRIC
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "theslowbrain.active_rubric_state.v1"

    path.write_text("{bad json", encoding="utf-8")
    assert load_active_rubric(path, default=BASE_RUBRIC) == BASE_RUBRIC

    path.write_text("[]", encoding="utf-8")
    assert load_active_rubric(path, default=BASE_RUBRIC) == BASE_RUBRIC


def test_outcome_stream_loader_excludes_anchor_and_malformed_rows(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl"
    anchor = _feature("anchor-1")
    usable = _feature("usable-1")
    append_decision_outcome_stream(stream, ((anchor, decide_feature(anchor, BASE_RUBRIC)),), run_id="run-1")
    stream.write_text(stream.read_text(encoding="utf-8") + "\n{bad json\n{}\n", encoding="utf-8")
    append_decision_outcome_stream(stream, ((usable, decide_feature(usable, BASE_RUBRIC)),), run_id="run-2")
    lines = stream.read_text(encoding="utf-8").splitlines()
    enriched = json.loads(lines[-1])
    enriched["feature"]["data_quality_issues"] = (
        {"field": "one", "code": "info_code", "severity": "info", "message": "info"},
        {"field": "two", "code": "warning_code", "severity": "warning", "message": "warning"},
        {"field": "three", "code": "error_code", "severity": "error", "message": "error"},
        {"field": "ignored", "code": "ignored_code", "severity": "unknown", "message": "ignored"},
    )
    lines[-1] = json.dumps(enriched)
    stream.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = load_outcome_stream_features(stream, exclude_idea_ids=("anchor-1",))

    assert result.status == "loaded"
    assert result.usable_count == 1
    assert result.excluded_anchor_count == 1
    assert result.malformed_count == 2
    assert result.features[0].idea_id == "usable-1"
    assert {issue.severity for issue in result.features[0].data_quality_issues} == {"info", "warning", "error"}


def test_gate_state_and_track_record_persist_shadow_only_evidence(tmp_path: Path) -> None:
    gate_path = tmp_path / "state" / "gating_gate.json"
    track_path = tmp_path / "reports" / "track-record" / "daily-history.jsonl"
    report = evaluate_gating_model(tuple(_feature(f"row-{index}", net=1.0) for index in range(40)), BASE_RUBRIC)

    persist_gating_gate(gate_path, report, run_id="run-1")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    assert gate["fallback_active"] is True
    assert gate["selected_source"] == "baseline_fallback"
    assert gate["broker_live_execution_allowed"] is False
    assert gate["gate_weights"]

    outcome_stream = load_outcome_stream_features(tmp_path / "missing.jsonl", exclude_idea_ids=())
    payload = {
        "promotion_decision": {"action": "reject", "active_version": "a", "selected_version": "a"},
        "trade_decisions": [{"action": "BUY"}, {"action": "SELL"}, {"action": "HOLD"}],
        "gating_model": {"status": "fallback", "selected_source": "baseline_fallback", "fallback_active": True},
        "safety": {"broker_live_execution_allowed": False},
    }
    append_track_record(
        track_path,
        run_id="run-1",
        report_payload=payload,
        outcome_stream=outcome_stream,
        active_rubric_state_path=tmp_path / "state" / "active_rubric.json",
        gating_gate_state_path=gate_path,
    )
    append_track_record(
        track_path,
        run_id="run-2",
        report_payload=payload,
        outcome_stream=outcome_stream,
        active_rubric_state_path=tmp_path / "state" / "active_rubric.json",
        gating_gate_state_path=gate_path,
    )

    rows = [json.loads(line) for line in track_path.read_text(encoding="utf-8").splitlines()]
    assert [row["run_id"] for row in rows] == ["run-1", "run-2"]
    assert all(row["broker_live_execution_allowed"] is False for row in rows)
    assert rows[0]["buy_count"] == 1
    assert rows[0]["sell_count"] == 1


def test_merge_feature_evidence_uses_append_only_rows_for_existing_keys() -> None:
    old = _feature("idea-1", net=0.1)
    refreshed = _feature("idea-1", net=2.0)

    merged = merge_feature_evidence((old,), (refreshed,))

    assert len(merged) == 1
    assert merged[0].net_return_pct == 2.0


def _feature(idea_id: str, *, net: float = 1.0) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker="AAPL",
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
        horizon_days=10,
        outcome_future_date="2026-01-11",
        entry_price=100.0,
    )
