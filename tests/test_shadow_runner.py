from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from slowbrain.shadow_runner import run_shadow_cycles


def test_shadow_cycles_write_per_run_records_and_summary(tmp_path: Path) -> None:
    base_time = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
    ticks = (base_time + timedelta(seconds=index) for index in range(10))

    summary = run_shadow_cycles(
        tmp_path,
        count=2,
        feature_limit=7,
        cycle_runner=_fake_cycle,
        clock=lambda: next(ticks),
    )

    output_dir = tmp_path / "reports" / "shadow-runs"
    records = sorted(path for path in output_dir.glob("shadow-*.json"))
    saved_summary = json.loads((output_dir / "latest-shadow-summary.json").read_text(encoding="utf-8"))

    assert len(records) == 2
    assert summary["run_count"] == 2
    assert summary["successful_runs"] == 2
    assert summary["broker_live_execution_allowed"] is False
    assert saved_summary["latest_eric_brief_lines"][0] == "Eric - TheSlowBrain"
    assert "Broker live execution allowed: false" in (output_dir / "latest-shadow-summary.md").read_text(
        encoding="utf-8"
    )


def test_shadow_cycles_reject_invalid_count(tmp_path: Path) -> None:
    try:
        run_shadow_cycles(tmp_path, count=0, cycle_runner=_fake_cycle)
    except ValueError as exc:
        assert "count" in str(exc)
    else:
        raise AssertionError("expected invalid count to raise")


def test_shadow_cycles_record_failure_before_reraising(tmp_path: Path) -> None:
    def broken_cycle(_project_root: Path, _feature_limit: int | None) -> dict[str, object]:
        raise RuntimeError("fixture failure")

    try:
        run_shadow_cycles(tmp_path, count=1, cycle_runner=broken_cycle)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected broken cycle to raise")

    records = list((tmp_path / "reports" / "shadow-runs").glob("shadow-*.json"))
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["reason"] == "RuntimeError"
    assert record["broker_live_execution_allowed"] is False


def test_shadow_cycles_reject_excessive_count(tmp_path: Path) -> None:
    try:
        run_shadow_cycles(tmp_path, count=26, cycle_runner=_fake_cycle)
    except ValueError as exc:
        assert "25" in str(exc)
    else:
        raise AssertionError("expected excessive count to raise")


def _fake_cycle(project_root: Path, feature_limit: int | None) -> dict[str, object]:
    assert project_root.exists()
    assert feature_limit == 7
    return {
        "promotion_decision": {"action": "reject"},
        "eric_brief": {"lines": ("Eric - TheSlowBrain", "Stocks to buy are: none")},
        "safety": {"old_reports_imported": False, "broker_live_execution_allowed": False},
    }
