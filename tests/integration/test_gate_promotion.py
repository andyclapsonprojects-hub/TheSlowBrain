"""Slice 5 (integration): a promoted gate actually changes decisions, and never unblocks live trading.

The earned, reversible state-machine transitions are unit-tested in ``tests/unit/test_promotion.py``;
here we drive the full ``run_first_cycle`` to prove (a) the default run is shadow and unchanged, and
(b) when the gate is active it genuinely vetoes rubric BUYs while the broker stays blocked.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from slowbrain.data_import import build_import_manifest
from slowbrain.gating_model import FEATURE_NAMES, GATING_LABELS
from slowbrain.workflow import FIRST_REPORT_JSON, run_first_cycle


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value

_SIGNAL_JSON = (
    '{"trend": "uptrend", "momentum_20d_pct": 12, '
    '"mean_reversion_z_20d": -1.2, "volume_signal": "high_volume_confirmation"}'
)


def _write_buy_sqlite(path: Path, *, rows: int = 60) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE step2_research_ideas (
                idea_id TEXT PRIMARY KEY, generated_at TEXT NOT NULL, case_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL, ticker TEXT NOT NULL, signal_date TEXT, source_title TEXT,
                evidence TEXT, sentiment TEXT, sentiment_confidence REAL, catalyst_strength REAL,
                recommendation TEXT, quality_status TEXT, risk_status TEXT, order_created INTEGER NOT NULL,
                entry_price REAL NOT NULL, signal_json TEXT NOT NULL, payload_json TEXT NOT NULL,
                eval_stage TEXT NOT NULL
            );
            CREATE TABLE step2_forward_returns (
                idea_id TEXT NOT NULL, horizon_days INTEGER NOT NULL, future_date TEXT NOT NULL,
                gross_return_pct REAL NOT NULL, net_return_pct REAL NOT NULL, cost_bps REAL NOT NULL,
                PRIMARY KEY (idea_id, horizon_days)
            );
            """
        )
        for index in range(rows):
            idea_id = f"idea_{index:04d}"
            net = 4.0 if index % 2 == 0 else -4.0
            conn.execute(
                "INSERT INTO step2_research_ideas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    idea_id, "2026-01-01T00:00:00Z", f"case_{index}", f"cand_{index}",
                    "AAPL" if index % 2 == 0 else "MSFT", f"2026-{index:04d}", "Synthetic", "Evidence",
                    "positive", 1.0, 1.0, "buy", "pass", "pass", 1, 100.0, _SIGNAL_JSON, "{}", "fixture",
                ),
            )
            conn.execute(
                "INSERT INTO step2_forward_returns VALUES (?, 10, ?, ?, ?, 30)",
                (idea_id, "2026-03-01", net, net),
            )
        conn.commit()
    finally:
        conn.close()


def _build_project(tmp_path: Path) -> Path:
    legacy = tmp_path / "legacy"
    project = tmp_path / "project"
    (legacy / "data").mkdir(parents=True)
    (legacy / "paper_trading").mkdir(parents=True)
    (legacy / "data" / "raw.json").write_text('{"source": "fixture"}', encoding="utf-8")
    (legacy / "paper_trading" / "pead_positions.csv").write_text("status\n", encoding="utf-8")
    (legacy / "paper_trading" / "live_fills.csv").write_text("status\n", encoding="utf-8")
    _write_buy_sqlite(legacy / "paper_trading" / "pipeline_runs.sqlite")
    build_import_manifest(legacy_root=legacy, project_root=project, copy_files=True)
    return project


def _seed_force_gate(project: Path, label: str) -> None:
    """Persist a gate that always predicts ``label``."""
    width = len(FEATURE_NAMES) + 1
    weights = [[100.0 if name == label else 0.0, *([0.0] * (width - 1))] for name in GATING_LABELS]
    state = project / "state" / "gating_gate.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps({"gate_weights": weights, "labels": list(GATING_LABELS), "feature_names": list(FEATURE_NAMES)}),
        encoding="utf-8",
    )


def _seed_force_sell_gate(project: Path) -> None:
    """A gate that always predicts SELL, so it vetoes every rubric BUY."""
    _seed_force_gate(project, "SELL")


def _report(project: Path) -> dict[str, object]:
    return _dict(json.loads((project / FIRST_REPORT_JSON).read_text(encoding="utf-8")))


def test_default_run_is_shadow_and_paper_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOWBRAIN_MARKET_DATA_ENABLED", "false")
    monkeypatch.delenv("SLOWBRAIN_GATING_STAGE_OVERRIDE", raising=False)
    project = _build_project(tmp_path)

    run_first_cycle(project, feature_limit=None)
    report = _report(project)

    promotion = _dict(report["gating_promotion"])
    assert promotion["active_stage_applied"] == "shadow"
    assert promotion["decisions_changed"] == 0
    assert _dict(report["gating_model"])["selected_source"] == "baseline_fallback"
    assert _dict(report["gating_model"])["fallback_active"] is True
    assert _dict(report["safety"])["broker_live_execution_allowed"] is False
    assert (project / "state" / "gating_promotion.json").exists()


def test_active_gate_vetoes_buys_and_stays_paper_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOWBRAIN_MARKET_DATA_ENABLED", "false")
    monkeypatch.setenv("SLOWBRAIN_GATING_STAGE_OVERRIDE", "confirm_only")
    project = _build_project(tmp_path)
    _seed_force_sell_gate(project)

    run_first_cycle(project, feature_limit=None)
    report = _report(project)

    promotion = _dict(report["gating_promotion"])
    assert promotion["active_stage_applied"] == "confirm_only"
    # Every BUY the rubric proposed is vetoed by the SELL-forcing gate.
    changed = promotion["decisions_changed"]
    assert isinstance(changed, int) and changed > 0
    assert all(_dict(influence)["kind"] == "veto" for influence in _list(promotion["influences"]))
    assert all(_dict(decision)["action"] != "BUY" for decision in _list(report["trade_decisions"]))
    assert _dict(report["gating_model"])["selected_source"] == "learned_gate"
    assert _dict(report["gating_model"])["fallback_active"] is False
    # Promotion never touches live execution.
    assert _dict(report["safety"])["broker_live_execution_allowed"] is False
    assert promotion["broker_live_execution_allowed"] is False


def test_gate_primary_makes_the_network_the_decider_and_stays_paper_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLOWBRAIN_MARKET_DATA_ENABLED", "false")
    monkeypatch.setenv("SLOWBRAIN_GATING_STAGE_OVERRIDE", "gate_primary")
    project = _build_project(tmp_path)
    _seed_force_gate(project, "HOLD")  # NN overrules the rubric's BUYs down to HOLD

    run_first_cycle(project, feature_limit=None)
    report = _report(project)

    promotion = _dict(report["gating_promotion"])
    assert promotion["active_stage_applied"] == "gate_primary"
    changed = promotion["decisions_changed"]
    assert isinstance(changed, int) and changed > 0
    influences = _list(promotion["influences"])
    assert all(_dict(influence)["kind"] in {"primary", "guardrail_veto"} for influence in influences)
    # The NN is the decision source now; no rubric BUY survives its HOLD call.
    assert all(_dict(decision)["action"] != "BUY" for decision in _list(report["trade_decisions"]))
    assert _dict(report["gating_model"])["selected_source"] == "learned_gate"
    assert _dict(report["safety"])["broker_live_execution_allowed"] is False
    assert promotion["broker_live_execution_allowed"] is False


def test_kill_switch_override_forces_shadow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOWBRAIN_MARKET_DATA_ENABLED", "false")
    monkeypatch.setenv("SLOWBRAIN_GATING_STAGE_OVERRIDE", "shadow")
    project = _build_project(tmp_path)
    _seed_force_sell_gate(project)

    run_first_cycle(project, feature_limit=None)
    report = _report(project)

    promotion = _dict(report["gating_promotion"])
    assert promotion["active_stage_applied"] == "shadow"
    assert promotion["decisions_changed"] == 0
    assert _dict(report["gating_model"])["fallback_active"] is True
