from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from slowbrain.enrichment import PointInTimeEnrichment, join_point_in_time_enrichment, load_pit_enrichment_records
from slowbrain.features import attach_cross_sectional_context, load_training_features_from_legacy_sqlite
from slowbrain.gating_model import FEATURE_NAMES, build_gating_dataset, target_label_for_feature
from slowbrain.models import DecisionAction, FeatureVector
from slowbrain.rubrics import BASE_RUBRIC


def test_universe_hygiene_errors_exclude_bad_rows_from_gate(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "pipeline_runs.sqlite"
    _write_sqlite(sqlite_path)

    features = load_training_features_from_legacy_sqlite(sqlite_path, horizon_days=(10,))
    rows = build_gating_dataset(features, BASE_RUBRIC)
    bad_codes = {
        issue.code
        for feature in features
        if feature.idea_id == "penny"
        for issue in feature.data_quality_issues
    }
    missing_date_codes = {
        issue.code
        for feature in features
        if feature.idea_id == "missing-date"
        for issue in feature.data_quality_issues
    }

    assert "entry_price_below_tradeable_floor" in bad_codes
    assert "implausible_outcome_for_penny_stock" in bad_codes
    assert "missing_original_signal_date" in missing_date_codes
    assert [row.idea_id for row in rows] == ["clean-a", "clean-b"]


def test_unused_signal_json_technicals_are_parsed_and_used_in_gate_features(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "pipeline_runs.sqlite"
    _write_sqlite(sqlite_path)

    feature = next(item for item in _loaded_features(sqlite_path) if item.idea_id == "clean-a")
    row = build_gating_dataset((feature,), BASE_RUBRIC)[0]

    assert feature.rsi_14 == 61.0
    assert feature.macd_signal == "bullish"
    assert feature.atr_pct_14 == 2.4
    assert feature.momentum_63d_pct == 11.0
    assert feature.volume_ratio_20d == 1.8
    assert "rsi_14_norm" in FEATURE_NAMES
    assert len(row.features) == len(FEATURE_NAMES)


def test_cross_sectional_rank_labels_are_scoped_to_date_and_horizon() -> None:
    same_date = tuple(
        _feature(f"same-{index}", signal_date="2026-01-01", net=float(index)) for index in range(10)
    )
    other_date = (_feature("other-date", signal_date="2026-01-02", net=100.0),)

    ranked = attach_cross_sectional_context((*same_date, *other_date))
    labels = {feature.idea_id: feature.rank_label for feature in ranked}
    zscores = {feature.idea_id: feature.cross_sectional_zscores["momentum_63d_pct"] for feature in ranked}

    assert labels["same-0"] == "SELL"
    assert labels["same-9"] == "BUY"
    assert labels["other-date"] == "HOLD"
    assert zscores["same-0"] < zscores["same-9"]
    assert zscores["other-date"] == 0.0


def test_rank_label_mode_is_explicit_and_default_preserves_absolute_return() -> None:
    feature = _feature("ranked", net=5.0, rank_label="SELL")

    assert target_label_for_feature(feature) == "BUY"
    assert target_label_for_feature(feature, mode="cross_sectional_rank") == "SELL"


def test_point_in_time_enrichment_uses_latest_available_record_without_future_leakage() -> None:
    feature = _feature("enrich", signal_date="2026-03-15", net=1.0)
    records = (
        PointInTimeEnrichment("AAPL", "2026-03-10", "fixture_past", "positive", 0.8, 0.7, 0.2, 0.3, 0.4, 0.5),
        PointInTimeEnrichment("AAPL", "2026-03-20", "fixture_future", "negative", 1.0, 1.0, -1.0, -1.0, -1.0, -1.0),
    )

    enriched = join_point_in_time_enrichment((feature,), records)[0]

    assert enriched.pit_enrichment_source == "fixture_past"
    assert enriched.pit_enrichment_available_date == "2026-03-10"
    assert enriched.sentiment == "positive"
    assert enriched.catalyst_strength == 0.7
    assert enriched.value_score == 0.2


def test_pit_enrichment_loader_accepts_jsonl_and_ignores_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "pit.jsonl"
    path.write_text(
        "\n".join(
            (
                "",
                "{not-json",
                json.dumps(["not", "a", "mapping"]),
                json.dumps({"ticker": "", "available_date": "2026-01-01"}),
                json.dumps(
                    {
                        "ticker": "msft",
                        "available_date": "2026-01-05",
                        "source": "json_fixture",
                        "sentiment": "excited",
                        "sentiment_confidence": "2.5",
                        "catalyst_strength": "-2.5",
                        "quality_score": "0.4",
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    rows = load_pit_enrichment_records(path)

    assert load_pit_enrichment_records(tmp_path / "missing.jsonl") == ()
    assert len(rows) == 1
    assert rows[0].ticker == "MSFT"
    assert rows[0].sentiment == ""
    assert rows[0].sentiment_confidence == 1.0
    assert rows[0].catalyst_strength == -1.0
    assert rows[0].fundamental_quality_score == 0.4


def test_pit_enrichment_loader_accepts_csv_exports(tmp_path: Path) -> None:
    path = tmp_path / "pit.csv"
    path.write_text(
        "ticker,as_of_date,source,sentiment,value_score,size_score,liquidity_score\n"
        "AAPL,2026-02-01,csv_fixture,neutral,0.2,0.3,0.4\n"
        "BAD,,csv_fixture,positive,0.9,0.9,0.9\n",
        encoding="utf-8",
    )

    rows = load_pit_enrichment_records(path)

    assert len(rows) == 1
    assert rows[0].ticker == "AAPL"
    assert rows[0].available_date == "2026-02-01"
    assert rows[0].sentiment == "neutral"
    assert rows[0].value_score == 0.2


def test_pit_enrichment_preserves_feature_when_no_safe_match_exists() -> None:
    feature = _feature("no-match", signal_date="2026-03-15", net=1.0)
    bad_date_feature = _feature("bad-date", signal_date="not-a-date", net=1.0)
    future = PointInTimeEnrichment("AAPL", "2026-03-20", "future", "negative", 1.0, 1.0)

    no_match, invalid_date = join_point_in_time_enrichment((feature, bad_date_feature), (future,))

    assert no_match.pit_enrichment_source == ""
    assert no_match.sentiment == feature.sentiment
    assert invalid_date.pit_enrichment_source == ""


def _feature(
    idea_id: str,
    *,
    signal_date: str = "2026-01-01",
    net: float,
    rank_label: DecisionAction | None = None,
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker="AAPL",
        signal_date=signal_date,
        sentiment="neutral",
        sentiment_confidence=0.0,
        catalyst_strength=0.0,
        trend="uptrend",
        momentum_20d_pct=net,
        mean_reversion_z_20d=0.0,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=10.0,
        source="fixture",
        horizon_days=10,
        entry_price=100.0,
        momentum_63d_pct=net,
        rank_label=rank_label,
    )


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
        _insert_idea(conn, "clean-a", "AAPL", "2026-01-01", 100.0, net=2.0)
        _insert_idea(conn, "clean-b", "MSFT", "2026-01-01", 100.0, net=-1.0)
        _insert_idea(conn, "penny", "OTLK", "2026-01-01", 0.25, net=90.0)
        _insert_idea(conn, "missing-date", "AAPL", "2026-01-01", 100.0, net=1.0)
        conn.execute("UPDATE step2_research_ideas SET signal_date = NULL WHERE idea_id = 'missing-date'")
        conn.commit()
    finally:
        conn.close()


def _insert_idea(
    conn: sqlite3.Connection,
    idea_id: str,
    ticker: str,
    signal_date: str,
    price: float,
    *,
    net: float,
) -> None:
    signal = {
        "trend": "uptrend",
        "momentum_20d_pct": 8,
        "mean_reversion_z_20d": 0,
        "volume_signal": "high_volume_confirmation",
        "rsi_14": 61,
        "macd_signal": "bullish",
        "atr_pct_14": 2.4,
        "momentum_63d_pct": 11,
        "volume_ratio_20d": 1.8,
    }
    conn.execute(
        "INSERT INTO step2_research_ideas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
            price,
            json.dumps(signal),
            "{}",
            "fixture",
        ),
    )
    conn.execute(
        "INSERT INTO step2_forward_returns VALUES (?, 10, ?, ?, ?, 10)",
        (idea_id, "2026-02-01", net, net),
    )


def _loaded_features(sqlite_path: Path) -> list[FeatureVector]:
    return load_training_features_from_legacy_sqlite(sqlite_path, horizon_days=(10,))
