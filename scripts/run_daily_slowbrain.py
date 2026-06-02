"""Run one safe daily Slow Brain shadow cycle and write operational logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.workflow import FIRST_REPORT_JSON, run_first_cycle

DAILY_RUNS_DIR = Path("reports/daily-runs")
LATEST_DAILY_JSON = DAILY_RUNS_DIR / "latest-daily-run.json"
LATEST_DAILY_MD = DAILY_RUNS_DIR / "latest-daily-run.md"

DailyCycleRunner = Callable[[Path, int | None], dict[str, object]]
Clock = Callable[[], datetime]


def main(
    argv: Sequence[str] | None = None,
    *,
    cycle_runner: DailyCycleRunner | None = None,
    clock: Clock | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--feature-limit", type=int, default=5000)
    parser.add_argument("--full-universe", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    now = clock or _utc_now
    runner = cycle_runner or _run_cycle
    feature_limit = None if args.full_universe else args.feature_limit
    started_at = now()
    run_id = f"daily-{started_at.astimezone(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir = args.project_root / DAILY_RUNS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = runner(args.project_root, feature_limit)
        record = _success_record(
            run_id=run_id,
            started_at=started_at,
            completed_at=now(),
            feature_limit=feature_limit,
            payload=payload,
            project_root=args.project_root,
        )
        exit_code = 0
    except Exception as exc:
        record = _failure_record(
            run_id=run_id,
            started_at=started_at,
            completed_at=now(),
            feature_limit=feature_limit,
            exc=exc,
        )
        exit_code = 1

    _write_json(output_dir / f"{run_id}.json", record, atomic=False)
    _write_json(args.project_root / LATEST_DAILY_JSON, record, atomic=True)
    _write_text(args.project_root / LATEST_DAILY_MD, _render_markdown(record), atomic=True)
    _print_summary(record)
    return exit_code


def _success_record(
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    feature_limit: int | None,
    payload: dict[str, object],
    project_root: Path,
) -> dict[str, object]:
    safety = _mapping(payload.get("safety"))
    promotion = _mapping(payload.get("promotion_decision"))
    brief = _mapping(payload.get("eric_brief"))
    lines = brief.get("lines")
    return {
        "schema": "theslowbrain.daily_run.v1",
        "run_id": run_id,
        "status": "success",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "feature_limit": feature_limit,
        "broker_live_execution_allowed": bool(safety.get("broker_live_execution_allowed")),
        "promotion_action": str(promotion.get("action") or "unknown"),
        "report_path": str(project_root / FIRST_REPORT_JSON),
        "active_rubric_state_path": payload.get("active_rubric_state_path"),
        "gating_gate_state_path": payload.get("gating_gate_state_path"),
        "track_record_path": payload.get("track_record_path"),
        "decision_outcome_stream_path": payload.get("decision_outcome_stream_path"),
        "eric_brief_lines": tuple(str(line) for line in lines) if isinstance(lines, (list, tuple)) else (),
    }


def _run_cycle(project_root: Path, feature_limit: int | None) -> dict[str, object]:
    return run_first_cycle(project_root, feature_limit=feature_limit)


def _failure_record(
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    feature_limit: int | None,
    exc: Exception,
) -> dict[str, object]:
    return {
        "schema": "theslowbrain.daily_run.v1",
        "run_id": run_id,
        "status": "failed",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "feature_limit": feature_limit,
        "broker_live_execution_allowed": False,
        "reason": type(exc).__name__,
        "message": str(exc),
    }


def _write_json(path: Path, payload: dict[str, object], *, atomic: bool) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    _write_text(path, f"{text}\n", atomic=atomic)


def _write_text(path: Path, text: str, *, atomic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _render_markdown(record: dict[str, object]) -> str:
    lines = record.get("eric_brief_lines")
    brief = "\n".join(str(line) for line in lines) if isinstance(lines, (list, tuple)) else ""
    return (
        "# TheSlowBrain Daily Run\n\n"
        f"Run ID: {record['run_id']}\n"
        f"Status: {record['status']}\n"
        f"Broker live execution allowed: {str(record['broker_live_execution_allowed']).lower()}\n"
        f"Promotion action: {record.get('promotion_action', 'unknown')}\n\n"
        f"{brief}\n"
    )


def _print_summary(record: dict[str, object]) -> None:
    print("TheSlowBrain daily shadow run complete.")
    print(f"Run ID: {record['run_id']}")
    print(f"Status: {record['status']}")
    print(f"Broker live execution allowed: {str(record['broker_live_execution_allowed']).lower()}")
    if record["status"] == "failed":
        print(f"Reason: {record.get('reason')}: {record.get('message')}", file=sys.stderr)


def _mapping(value: object) -> dict[str, object]:
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


def _utc_now() -> datetime:
    return datetime.now(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
