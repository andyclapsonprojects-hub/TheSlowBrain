"""Held-out human anchor ingestion for PR8 calibration evidence."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .eval_council import VALID_HUMAN_LABELS
from .human_labeling import HUMAN_LABELING_DIR, load_decision_capture_records, merge_completed_human_labels

HUMAN_ANCHOR_JSON = HUMAN_LABELING_DIR / "verified_enriched_stock_labels.json"
DEFAULT_ENRICHED_HUMAN_LABELS_PATH = Path("C:/Users/AndyC/Downloads/enriched_stock_human_labels_full.json")
HELD_OUT_ANCHOR_TRAINING_ROLE = "held_out_anchor_never_train"


@dataclass(frozen=True)
class HumanAnchorIngestionResult:
    source_path: Path
    output_path: Path
    anchor_count: int
    label_counts: Mapping[str, int]
    matched_capture_count: int
    missing_capture_example_ids: tuple[str, ...]


def ingest_enriched_human_anchor(
    *,
    source_path: Path = DEFAULT_ENRICHED_HUMAN_LABELS_PATH,
    output_path: Path,
    capture_path: Path | None = None,
) -> HumanAnchorIngestionResult:
    """Normalize Andy's rich 5-label file into the project-held-out anchor."""
    rows = normalize_enriched_human_anchor_rows(load_human_anchor_rows(source_path), source_path=source_path)
    # Reuse the supported merge validator so label acceptance stays one-path.
    merge_completed_human_labels(capture_records=(), label_rows=rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    capture_ids = _capture_idea_ids(capture_path) if capture_path is not None else set()
    anchor_ids = {str(row["example_id"]) for row in rows}
    missing = tuple(sorted(anchor_ids - capture_ids)) if capture_path is not None else ()
    return HumanAnchorIngestionResult(
        source_path=source_path,
        output_path=output_path,
        anchor_count=len(rows),
        label_counts=dict(Counter(str(row["human_label"]) for row in rows)),
        matched_capture_count=len(anchor_ids & capture_ids) if capture_path is not None else 0,
        missing_capture_example_ids=missing,
    )


def load_human_anchor_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("human anchor file must be a JSON list")
    return tuple({str(key): value for key, value in row.items()} for row in payload if isinstance(row, dict))


def normalize_enriched_human_anchor_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_path: Path,
) -> tuple[dict[str, object], ...]:
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        copied = dict(row)
        example_id = str(copied.get("example_id") or "").strip()
        if not example_id:
            raise ValueError("human anchor row is missing example_id")
        if example_id in seen:
            raise ValueError(f"duplicate human anchor example_id: {example_id}")
        seen.add(example_id)
        label = str(copied.get("human_label") or "").strip().upper()
        if label not in VALID_HUMAN_LABELS - {"UNKNOWN"}:
            raise ValueError(f"invalid held-out human anchor label for {example_id}: {label}")
        copied["example_id"] = example_id
        copied["human_label"] = label
        copied["human_rationale"] = str(copied.get("human_rationale") or copied.get("rationale") or "").strip()
        copied["label_source"] = "human_verified"
        copied["held_out_anchor"] = True
        copied["training_role"] = HELD_OUT_ANCHOR_TRAINING_ROLE
        copied["rows_are_human_labels"] = True
        copied["anchor_source_path"] = str(source_path)
        normalized.append(copied)
    return tuple(normalized)


def load_human_anchor_example_ids(path: Path) -> frozenset[str]:
    return frozenset(str(row.get("example_id") or "") for row in load_human_anchor_rows(path) if row.get("example_id"))


def _capture_idea_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    ids: set[str] = set()
    for record in load_decision_capture_records(path):
        feature = record.get("feature")
        if isinstance(feature, dict):
            idea_id = str(feature.get("idea_id") or "")
            if idea_id:
                ids.add(idea_id)
    return ids
