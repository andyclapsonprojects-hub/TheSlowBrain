from __future__ import annotations

import json
from pathlib import Path

import pytest

from slowbrain.human_anchor import (
    HELD_OUT_ANCHOR_TRAINING_ROLE,
    HumanAnchorIngestionResult,
    ingest_enriched_human_anchor,
    load_human_anchor_example_ids,
    load_human_anchor_rows,
    normalize_enriched_human_anchor_rows,
)


def test_human_anchor_load_rejects_non_list_payloads(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_human_anchor_rows(missing) == ()

    path = tmp_path / "anchor.json"
    path.write_text('{"not": "a-list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        load_human_anchor_rows(path)


def test_human_anchor_normalization_rejects_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing example_id"):
        normalize_enriched_human_anchor_rows(({"human_label": "BUY"},), source_path=tmp_path / "source.json")

    with pytest.raises(ValueError, match="duplicate"):
        normalize_enriched_human_anchor_rows(
            ({"example_id": "one", "human_label": "BUY"}, {"example_id": "one", "human_label": "HOLD"}),
            source_path=tmp_path / "source.json",
        )

    with pytest.raises(ValueError, match="invalid held-out"):
        normalize_enriched_human_anchor_rows(
            ({"example_id": "one", "human_label": "UNKNOWN"},),
            source_path=tmp_path / "source.json",
        )


def test_ingest_enriched_human_anchor_reports_capture_mapping(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "out" / "anchor.json"
    capture = tmp_path / "capture.jsonl"
    source.write_text(
        json.dumps(
            [
                {"example_id": "matched", "human_label": "buy", "rationale": "strong"},
                {"example_id": "missing", "human_label": "hold", "human_rationale": "wait"},
            ]
        ),
        encoding="utf-8",
    )
    capture.write_text('{"feature": {"idea_id": "matched"}}\n', encoding="utf-8")

    result = ingest_enriched_human_anchor(source_path=source, output_path=output, capture_path=capture)

    assert isinstance(result, HumanAnchorIngestionResult)
    assert result.anchor_count == 2
    assert result.matched_capture_count == 1
    assert result.missing_capture_example_ids == ("missing",)
    assert load_human_anchor_example_ids(output) == frozenset({"matched", "missing"})
    rows = load_human_anchor_rows(output)
    assert rows[0]["training_role"] == HELD_OUT_ANCHOR_TRAINING_ROLE
    assert rows[0]["held_out_anchor"] is True
