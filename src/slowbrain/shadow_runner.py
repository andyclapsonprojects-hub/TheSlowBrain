"""Repeated paper/shadow Slow Brain cycle runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .workflow import run_first_cycle

CycleRunner = Callable[[Path], dict[str, object]]
Clock = Callable[[], datetime]


def run_shadow_cycles(
    project_root: Path,
    *,
    count: int = 3,
    feature_limit: int | None = 5000,
    cycle_runner: Callable[[Path, int | None], dict[str, object]] | None = None,
    clock: Clock | None = None,
) -> dict[str, object]:
    """Run repeated paper/shadow cycles and write durable local evidence."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > 25:
        raise ValueError("count must be 25 or fewer")

    now = clock or _utc_now
    runner = cycle_runner or _run_cycle
    output_dir = project_root / "reports" / "shadow-runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for index in range(1, count + 1):
        started = now()
        started_at = started.isoformat()
        run_id = _run_id(index, started)
        try:
            payload = runner(project_root, feature_limit)
            record = _success_record(index, run_id, started_at, now().isoformat(), payload)
        except Exception as exc:
            record = _failure_record(index, run_id, started_at, now().isoformat(), exc)
            _write_json(output_dir / f"{_record_id(record)}.json", record)
            raise
        _write_json(output_dir / f"{_record_id(record)}.json", record)
        records.append(record)

    summary = {
        "schema": "theslowbrain.shadow_summary.v1",
        "created_at": now().isoformat(),
        "run_count": len(records),
        "successful_runs": sum(1 for record in records if record["status"] == "success"),
        "failed_runs": sum(1 for record in records if record["status"] == "failed"),
        "broker_live_execution_allowed": any(
            bool(record.get("broker_live_execution_allowed")) for record in records
        ),
        "latest_eric_brief_lines": records[-1].get("eric_brief_lines") if records else (),
        "records": records,
    }
    _write_json(output_dir / "latest-shadow-summary.json", summary)
    (output_dir / "latest-shadow-summary.md").write_text(_render_summary(summary), encoding="utf-8")
    return summary


def _run_cycle(project_root: Path, feature_limit: int | None) -> dict[str, object]:
    return run_first_cycle(project_root, feature_limit=feature_limit)


def _success_record(
    index: int,
    run_id: str,
    started_at: str,
    completed_at: str,
    payload: dict[str, object],
) -> dict[str, object]:
    safety = payload.get("safety")
    brief = payload.get("eric_brief")
    lines = brief.get("lines") if isinstance(brief, dict) else ()
    return {
        "schema": "theslowbrain.shadow_run.v1",
        "run_id": run_id,
        "run_index": index,
        "status": "success",
        "started_at": started_at,
        "completed_at": completed_at,
        "broker_live_execution_allowed": _broker_live_allowed(safety),
        "old_reports_imported": _old_reports_imported(safety),
        "promotion_action": _promotion_action(payload),
        "eric_brief_lines": tuple(str(line) for line in lines) if isinstance(lines, (list, tuple)) else (),
    }


def _failure_record(
    index: int,
    run_id: str,
    started_at: str,
    completed_at: str,
    exc: Exception,
) -> dict[str, object]:
    return {
        "schema": "theslowbrain.shadow_run.v1",
        "run_id": run_id,
        "run_index": index,
        "status": "failed",
        "started_at": started_at,
        "completed_at": completed_at,
        "reason": type(exc).__name__,
        "broker_live_execution_allowed": False,
    }


def _broker_live_allowed(safety: object) -> bool:
    return bool(safety.get("broker_live_execution_allowed")) if isinstance(safety, dict) else True


def _old_reports_imported(safety: object) -> bool:
    return bool(safety.get("old_reports_imported")) if isinstance(safety, dict) else True


def _promotion_action(payload: dict[str, object]) -> str:
    promotion = payload.get("promotion_decision")
    action = promotion.get("action") if isinstance(promotion, dict) else None
    return str(action) if action else "unknown"


def _record_id(record: dict[str, object]) -> str:
    return str(record["run_id"])


def _run_id(index: int, value: datetime) -> str:
    return f"shadow-{index:03d}-{_file_timestamp(value)}"


def _file_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _render_summary(summary: dict[str, object]) -> str:
    lines = summary.get("latest_eric_brief_lines")
    brief = "\n".join(str(line) for line in lines) if isinstance(lines, (list, tuple)) else "No Eric brief."
    return (
        "# TheSlowBrain Shadow Run Summary\n\n"
        f"Run count: {summary['run_count']}\n"
        f"Successful runs: {summary['successful_runs']}\n"
        f"Failed runs: {summary['failed_runs']}\n"
        f"Broker live execution allowed: {str(summary['broker_live_execution_allowed']).lower()}\n\n"
        f"{brief}\n"
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)
