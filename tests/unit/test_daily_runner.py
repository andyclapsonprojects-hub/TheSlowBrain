from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType


def test_daily_runner_writes_success_record_and_latest_summary(tmp_path: Path) -> None:
    module = _daily_module()
    ticks = _ticks()

    exit_code = module.main(
        ["--project-root", str(tmp_path), "--feature-limit", "7"],
        cycle_runner=_fake_cycle,
        clock=lambda: next(ticks),
    )

    latest = json.loads((tmp_path / "reports" / "daily-runs" / "latest-daily-run.json").read_text(encoding="utf-8"))
    records = list((tmp_path / "reports" / "daily-runs").glob("daily-*.json"))

    assert exit_code == 0
    assert len(records) == 1
    assert latest["status"] == "success"
    assert latest["feature_limit"] == 7
    assert latest["broker_live_execution_allowed"] is False
    assert latest["active_rubric_state_path"] == "state/active_rubric.json"
    assert "Broker live execution allowed: false" in (
        tmp_path / "reports" / "daily-runs" / "latest-daily-run.md"
    ).read_text(encoding="utf-8")


def test_daily_runner_logs_failure_and_returns_nonzero(tmp_path: Path) -> None:
    module = _daily_module()

    def broken(_project_root: Path, _feature_limit: int | None) -> dict[str, object]:
        raise RuntimeError("fixture boom")

    exit_code = module.main(["--project-root", str(tmp_path)], cycle_runner=broken, clock=lambda: next(_ticks()))

    latest = json.loads((tmp_path / "reports" / "daily-runs" / "latest-daily-run.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert latest["status"] == "failed"
    assert latest["reason"] == "RuntimeError"
    assert latest["broker_live_execution_allowed"] is False


def test_daily_scheduler_script_uses_safe_shadow_command() -> None:
    script = Path("scripts/install_daily_task.ps1").read_text(encoding="utf-8")

    assert "Register-ScheduledTask" in script
    assert "run_daily_slowbrain.py" in script
    assert "uv" in script
    assert "-WorkingDirectory $ProjectRoot" in script
    assert "TRADING_LIVE_ENABLED" not in script
    assert "broker live execution remains blocked" in script.lower()


def _fake_cycle(project_root: Path, feature_limit: int | None) -> dict[str, object]:
    assert project_root.exists()
    assert feature_limit == 7
    return {
        "promotion_decision": {"action": "reject"},
        "active_rubric_state_path": "state/active_rubric.json",
        "gating_gate_state_path": "state/gating_gate.json",
        "track_record_path": "reports/track-record/daily-history.jsonl",
        "decision_outcome_stream_path": "reports/decision-capture/append-only-outcome-stream.jsonl",
        "eric_brief": {"lines": ("Eric - TheSlowBrain", "Stocks to buy are: none")},
        "safety": {"broker_live_execution_allowed": False},
    }


def _daily_module() -> ModuleType:
    script_path = Path("scripts/run_daily_slowbrain.py").resolve()
    spec = importlib.util.spec_from_file_location("run_daily_slowbrain", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("daily runner spec could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ticks() -> Iterator[datetime]:
    base = datetime(2026, 6, 2, 7, 30, tzinfo=UTC)
    return (base + timedelta(seconds=index) for index in range(4))
