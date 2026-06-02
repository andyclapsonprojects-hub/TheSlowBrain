from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from slowbrain.backtest import evaluate_rubric
from slowbrain.decision_capture import append_decision_outcome_stream, write_decision_capture
from slowbrain.eval_council import CalibrationReport
from slowbrain.features import (
    load_features_for_idea_ids_from_legacy_sqlite,
    load_training_features_from_legacy_sqlite,
)
from slowbrain.human_anchor import HELD_OUT_ANCHOR_TRAINING_ROLE, ingest_enriched_human_anchor
from slowbrain.models import FeatureVector, PortfolioState, PromotionDecision
from slowbrain.reporting import build_eric_brief, write_first_report
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def test_ingest_rich_anchor_preserves_five_labels_and_marks_held_out(tmp_path: Path) -> None:
    source = tmp_path / "enriched.json"
    output = tmp_path / "anchor.json"
    capture = tmp_path / "capture.jsonl"
    rows = [
        _anchor_json_row("anchor-1", "OTLK", "WATCHLIST"),
        _anchor_json_row("anchor-2", "AAPL", "AVOID"),
        _anchor_json_row("anchor-3", "MSFT", "HOLD"),
        _anchor_json_row("anchor-4", "NVDA", "BUY"),
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    feature = _feature("anchor-1")
    write_decision_capture(capture, ((feature, decide_feature(feature, BASE_RUBRIC)),))

    result = ingest_enriched_human_anchor(source_path=source, output_path=output, capture_path=capture)
    ingested = json.loads(output.read_text(encoding="utf-8"))

    assert result.anchor_count == 4
    assert result.matched_capture_count == 1
    assert result.missing_capture_example_ids == ("anchor-2", "anchor-3", "anchor-4")
    assert {row["human_label"] for row in ingested} == {"WATCHLIST", "AVOID", "HOLD", "BUY"}
    assert all(row["label_source"] == "human_verified" for row in ingested)
    assert all(row["held_out_anchor"] is True for row in ingested)
    assert all(row["training_role"] == HELD_OUT_ANCHOR_TRAINING_ROLE for row in ingested)


def test_full_universe_loader_supports_horizons_and_excludes_anchor(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "pipeline_runs.sqlite"
    _write_sqlite(sqlite_path)

    training = load_training_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=(1, 5, 10, 20),
        exclude_idea_ids=("anchor-1",),
    )
    anchor = load_features_for_idea_ids_from_legacy_sqlite(sqlite_path, idea_ids=("anchor-1",), horizon_days=10)

    assert {feature.horizon_days for feature in training} == {1, 5, 10, 20}
    assert all(feature.idea_id != "anchor-1" for feature in training)
    assert anchor[0].idea_id == "anchor-1"
    assert anchor[0].horizon_days == 10


def test_outcome_quarantine_excludes_phantom_alpha_from_backtest(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "pipeline_runs.sqlite"
    _write_sqlite(sqlite_path)

    features = load_training_features_from_legacy_sqlite(sqlite_path, horizon_days=(10,))
    bad = next(feature for feature in features if feature.idea_id == "bad-outcome")
    result = evaluate_rubric(features, BASE_RUBRIC, min_test_trades=0)

    assert any(issue.code == "implausible_outcome_for_penny_stock" for issue in bad.data_quality_issues)
    assert result.excluded_error_feature_count == 1
    assert result.sample_count == len(features) - 1


def test_append_only_outcome_stream_grows_while_latest_capture_stays_reviewable(tmp_path: Path) -> None:
    capture = tmp_path / "latest.jsonl"
    stream = tmp_path / "stream.jsonl"
    feature_one = _feature("idea-1")
    feature_two = _feature("idea-2")
    write_decision_capture(capture, ((feature_one, decide_feature(feature_one, BASE_RUBRIC)),), run_id="run-1")
    labelled = capture.read_text(encoding="utf-8").replace('"human_label": null', '"human_label": "WATCHLIST"')
    capture.write_text(labelled, encoding="utf-8")

    append_decision_outcome_stream(stream, ((feature_one, decide_feature(feature_one, BASE_RUBRIC)),), run_id="run-1")
    append_decision_outcome_stream(stream, ((feature_two, decide_feature(feature_two, BASE_RUBRIC)),), run_id="run-2")
    write_decision_capture(capture, ((feature_two, decide_feature(feature_two, BASE_RUBRIC)),), run_id="run-2")

    assert len(stream.read_text(encoding="utf-8").splitlines()) == 2
    latest = capture.read_text(encoding="utf-8")
    assert '"idea_id": "idea-1"' in latest
    assert '"human_label": "WATCHLIST"' in latest
    assert '"idea_id": "idea-2"' in latest


def test_report_surfaces_low_n_non_binding_calibration(tmp_path: Path) -> None:
    feature = _feature("idea-1")
    decision = decide_feature(feature, BASE_RUBRIC)
    promotion_result = evaluate_rubric((feature,), BASE_RUBRIC, min_test_trades=0)
    promotion = PromotionDecision(
        action="reject",
        selected_version=BASE_RUBRIC.version,
        active_version=BASE_RUBRIC.version,
        reason="fixture",
        current_result=promotion_result,
    )

    payload = write_first_report(
        output_json=tmp_path / "report.json",
        output_md=tmp_path / "report.md",
        promotion=promotion,
        decisions=(decision,),
        portfolio=_portfolio(),
        brief=build_eric_brief((decision,), _portfolio()),
        human_calibration=CalibrationReport("failed", True, 30, 0.42, ("low agreement",)),
    )

    calibration = payload["human_calibration"]
    assert isinstance(calibration, dict)
    assert calibration["example_count"] == 30
    assert calibration["adoption_binding"] is False
    assert calibration["confidence_note"] == "n=30, low-confidence; not adoption-binding"


def _anchor_json_row(example_id: str, ticker: str, label: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "ticker": ticker,
        "signal_date": "2026-01-01",
        "human_label": label,
        "human_rationale": "fixture rationale",
    }


def _feature(idea_id: str) -> FeatureVector:
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
        net_return_pct=1.0,
        cost_bps=10.0,
        source="fixture",
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(holdings=("AAPL",), profit_since_first_trade_pct=1.0)


def _write_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE step2_research_ideas (
                idea_id TEXT PRIMARY KEY,
                generated_at TEXT NOT NULL,
                case_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_date TEXT,
                source_title TEXT,
                evidence TEXT,
                sentiment TEXT,
                sentiment_confidence REAL,
                catalyst_strength REAL,
                recommendation TEXT,
                quality_status TEXT,
                risk_status TEXT,
                order_created INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                signal_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                eval_stage TEXT NOT NULL
            );
            CREATE TABLE step2_forward_returns (
                idea_id TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                future_date TEXT NOT NULL,
                gross_return_pct REAL NOT NULL,
                net_return_pct REAL NOT NULL,
                cost_bps REAL NOT NULL,
                PRIMARY KEY (idea_id, horizon_days)
            );
            """
        )
        _insert_idea(conn, "anchor-1", "AAPL", "2026-01-01", 100.0)
        _insert_idea(conn, "good-1", "MSFT", "2026-01-02", 100.0)
        _insert_idea(conn, "bad-outcome", "OTLK", "2026-01-03", 0.25)
        for idea_id in ("anchor-1", "good-1"):
            for horizon in (1, 5, 10, 20):
                conn.execute(
                    "INSERT INTO step2_forward_returns VALUES (?, ?, ?, ?, ?, 10)",
                    (idea_id, horizon, "2026-02-01", 1.0, 1.0),
                )
        conn.execute(
            "INSERT INTO step2_forward_returns VALUES ('bad-outcome', 10, '2026-02-01', 90.0, 90.0, 10)"
        )
        conn.commit()
    finally:
        conn.close()


def _insert_idea(conn: sqlite3.Connection, idea_id: str, ticker: str, signal_date: str, entry_price: float) -> None:
    conn.execute(
        """
        INSERT INTO step2_research_ideas VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            idea_id,
            f"{signal_date}T00:00:00Z",
            f"case-{idea_id}",
            f"candidate-{idea_id}",
            ticker,
            signal_date,
            "Fixture",
            "Evidence",
            "positive",
            0.9,
            0.9,
            "buy",
            "pass",
            "pass",
            1,
            entry_price,
            '{"trend":"uptrend","momentum_20d_pct":8,"mean_reversion_z_20d":0,'
            '"volume_signal":"high_volume_confirmation"}',
            "{}",
            "fixture",
        ),
    )
