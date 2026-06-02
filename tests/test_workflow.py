from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from slowbrain.data_import import build_import_manifest
from slowbrain.features import load_features_from_legacy_sqlite
from slowbrain.market_data import BenchmarkReturn, LiquiditySnapshot, UniverseMembership
from slowbrain.models import FeatureVector
from slowbrain.workflow import FIRST_REPORT_JSON, run_first_cycle


def test_first_workflow_generates_native_report_without_old_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLOWBRAIN_MARKET_DATA_ENABLED", "false")
    legacy = tmp_path / "legacy"
    project = tmp_path / "project"
    (legacy / "data").mkdir(parents=True)
    (legacy / "reports").mkdir()
    (legacy / "paper_trading").mkdir()
    (legacy / "data" / "raw.json").write_text('{"source": "fixture"}', encoding="utf-8")
    (legacy / "reports" / "old.json").write_text('{"old": true}', encoding="utf-8")
    _write_sqlite(legacy / "paper_trading" / "pipeline_runs.sqlite")
    (legacy / "paper_trading" / "pead_positions.csv").write_text(
        "position_id,strategy_id,ticker,entry_date,entry_price,quantity,target_exit_date,status,exit_date,exit_price,gross_return_pct,net_return_pct,cost_bps,pnl_currency,updated_at\n",
        encoding="utf-8",
    )
    (legacy / "paper_trading" / "live_fills.csv").write_text(
        "execution_id,order_id,ticker,side,order_type,filled_quantity,average_fill_price,limit_price,currency,submitted_at,filled_at,status,source,net_value,fx_rate,taxes,raw_order_status,error\n",
        encoding="utf-8",
    )
    build_import_manifest(legacy_root=legacy, project_root=project, copy_files=True)

    payload = run_first_cycle(project, feature_limit=None)

    report_path = project / FIRST_REPORT_JSON
    assert report_path.exists()
    assert payload["safety"] == {"old_reports_imported": False, "broker_live_execution_allowed": False}
    assert not (project / "data" / "imports" / "n8n_original_stock_trader_2026-05-31" / "reports").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "theslowbrain.first_report.v1"
    assert report["eric_brief"]["lines"][0] == "Eric - TheSlowBrain"
    assert report["import_record_count"] > 0
    assert all(intent["status"] == "blocked" for intent in report["blocked_order_intents"])
    assert (project / "state" / "active_rubric.json").exists()
    assert (project / "state" / "gating_gate.json").exists()
    track_record = project / "reports" / "track-record" / "daily-history.jsonl"
    assert track_record.exists()
    track_rows = [json.loads(line) for line in track_record.read_text(encoding="utf-8").splitlines()]
    assert len(track_rows) == 1
    assert track_rows[0]["broker_live_execution_allowed"] is False
    assert report["active_rubric_state_path"].endswith("state\\active_rubric.json") or report[
        "active_rubric_state_path"
    ].endswith("state/active_rubric.json")
    assert report["gating_model"]["fallback_active"] is True


def test_workflow_passes_configured_market_data_provider_into_backtest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLOWBRAIN_MARKET_DATA_ENABLED", "false")
    legacy = tmp_path / "legacy"
    project = tmp_path / "project"
    (legacy / "data").mkdir(parents=True)
    (legacy / "paper_trading").mkdir()
    (legacy / "data" / "raw.json").write_text('{"source": "fixture"}', encoding="utf-8")
    _write_sqlite(legacy / "paper_trading" / "pipeline_runs.sqlite")
    (legacy / "paper_trading" / "pead_positions.csv").write_text(
        "position_id,strategy_id,ticker,entry_date,entry_price,quantity,target_exit_date,status,exit_date,exit_price,gross_return_pct,net_return_pct,cost_bps,pnl_currency,updated_at\n",
        encoding="utf-8",
    )
    (legacy / "paper_trading" / "live_fills.csv").write_text(
        "execution_id,order_id,ticker,side,order_type,filled_quantity,average_fill_price,limit_price,currency,submitted_at,filled_at,status,source,net_value,fx_rate,taxes,raw_order_status,error\n",
        encoding="utf-8",
    )
    build_import_manifest(legacy_root=legacy, project_root=project, copy_files=True)
    provider = FixtureMarketDataProvider()
    monkeypatch.setattr("slowbrain.workflow.build_market_data_provider", lambda config, *, project_root: provider)

    payload = run_first_cycle(project, feature_limit=None)

    backtest = payload["portfolio_backtest"]
    assert isinstance(backtest, dict)
    assert backtest["benchmark_quality"] == "provider_real_benchmark"
    assert backtest["liquidity_quality"] == "provider_real_per_name_liquidity"
    assert provider.benchmark_calls > 0
    assert provider.liquidity_calls > 0


def test_bounded_feature_loading_uses_latest_rows(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "pipeline_runs.sqlite"
    _write_sqlite(sqlite_path)

    features = load_features_from_legacy_sqlite(sqlite_path, limit=3)

    assert [feature.idea_id for feature in features] == ["idea_9", "idea_10", "idea_11"]
    assert [feature.signal_date for feature in features] == ["2026-01-10", "2026-01-11", "2026-01-12"]


class FixtureMarketDataProvider:
    def __init__(self) -> None:
        self.benchmark_calls = 0
        self.liquidity_calls = 0

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn:
        self.benchmark_calls += 1
        return BenchmarkReturn("SPY", feature.signal_date, 0.2, "workflow_fixture")

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot:
        self.liquidity_calls += 1
        return LiquiditySnapshot(feature.ticker, 100_000_000.0, 1.0, 3.0, "workflow_fixture")

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        return UniverseMembership(feature.ticker, feature.signal_date, True, False, "workflow_fixture")


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
        for index in range(12):
            idea_id = f"idea_{index}"
            conn.execute(
                """
                INSERT INTO step2_research_ideas VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    idea_id,
                    "2026-01-01T00:00:00Z",
                    f"case_{index}",
                    f"candidate_{index}",
                    "AAPL" if index % 2 == 0 else "MSFT",
                    f"2026-01-{index + 1:02d}",
                    "Fixture",
                    "Evidence",
                    "positive",
                    0.8,
                    0.8,
                    "buy",
                    "pass",
                    "pass",
                    1,
                    100.0,
        (
            '{"trend": "uptrend", "momentum_20d_pct": 8, '
            '"mean_reversion_z_20d": 0, "volume_signal": "high_volume_confirmation"}'
        ),
                    "{}",
                    "fixture",
                ),
            )
            conn.execute(
                "INSERT INTO step2_forward_returns VALUES (?, 10, ?, ?, ?, 45)",
                (idea_id, "2026-02-01", 1.0, 1.0),
            )
        conn.commit()
    finally:
        conn.close()
