"""Labelable decision capture for human calibration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .models import FeatureVector, TradeDecision

DECISION_CAPTURE_JSONL = Path("reports/decision-capture/latest-decision-log.jsonl")
DECISION_OUTCOME_STREAM_JSONL = Path("reports/decision-capture/append-only-outcome-stream.jsonl")


def write_decision_capture(
    output_path: Path,
    pairs: Sequence[tuple[FeatureVector, TradeDecision]],
    *,
    run_id: str | None = None,
) -> Path:
    """Write a labelable JSONL decision set, preserving any existing human labels.

    A workflow re-run regenerates the machine rows, but must never destroy a row a
    human has already labelled. Existing non-null ``human_label``/``human_rationale``
    values are carried onto the matching new row (by ``feature.idea_id``); existing
    human-labelled rows whose decision is no longer captured are retained as well.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_labels = _existing_human_labels(output_path)
    captured_at = datetime.now(UTC).isoformat()
    active_run_id = run_id or f"capture-{captured_at}"

    new_records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for feature, decision in pairs:
        record = _record(feature, decision, captured_at, active_run_id)
        if feature.idea_id in existing_labels:
            label, rationale, _record_existing = existing_labels[feature.idea_id]
            record["human_label"] = label
            record["human_rationale"] = rationale
        new_records.append(record)
        seen_ids.add(feature.idea_id)

    preserved = [
        record for idea_id, (_label, _rationale, record) in existing_labels.items() if idea_id not in seen_ids
    ]
    lines = [json.dumps(record, sort_keys=True) for record in [*new_records, *preserved]]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def append_decision_outcome_stream(
    output_path: Path,
    pairs: Sequence[tuple[FeatureVector, TradeDecision]],
    *,
    run_id: str | None = None,
) -> Path:
    """Append decision/outcome rows without rewriting previous run evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(UTC).isoformat()
    active_run_id = run_id or f"outcome-{captured_at}"
    lines = [
        json.dumps(_record(feature, decision, captured_at, active_run_id), sort_keys=True)
        for feature, decision in pairs
    ]
    if not lines:
        return output_path
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_path


def _existing_human_labels(path: Path) -> dict[str, tuple[object, object, dict[str, object]]]:
    """Map idea_id -> (human_label, human_rationale, full_record) for labelled rows."""
    from .human_labeling import load_decision_capture_records

    labelled: dict[str, tuple[object, object, dict[str, object]]] = {}
    for record in load_decision_capture_records(path):
        if record.get("human_label") in (None, ""):
            continue
        feature = record.get("feature")
        idea_id = str(feature.get("idea_id")) if isinstance(feature, dict) else ""
        if idea_id:
            labelled[idea_id] = (record.get("human_label"), record.get("human_rationale"), record)
    return labelled


def _record(
    feature: FeatureVector,
    decision: TradeDecision,
    captured_at: str,
    run_id: str,
) -> dict[str, object]:
    return {
        "schema": "theslowbrain.golden_decision.v1",
        "run_id": run_id,
        "captured_at": captured_at,
        "feature": asdict(feature),
        "decision": asdict(decision),
        "outcome": {
            "horizon_days": feature.horizon_days,
            "future_date": feature.outcome_future_date,
            "realized_net_return_pct": feature.net_return_pct,
            "source": feature.source,
        },
        "human_label": None,
        "human_rationale": None,
    }
