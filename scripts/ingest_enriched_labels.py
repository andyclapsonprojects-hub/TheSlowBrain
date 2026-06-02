"""Ingest Andy's rich 5-label file as a held-out PR8 calibration anchor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.decision_capture import DECISION_CAPTURE_JSONL
from slowbrain.human_anchor import DEFAULT_ENRICHED_HUMAN_LABELS_PATH, HUMAN_ANCHOR_JSON, ingest_enriched_human_anchor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=DEFAULT_ENRICHED_HUMAN_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=HUMAN_ANCHOR_JSON)
    parser.add_argument("--capture", type=Path, default=DECISION_CAPTURE_JSONL)
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else args.project_root / args.output
    capture_path = args.capture if args.capture.is_absolute() else args.project_root / args.capture
    result = ingest_enriched_human_anchor(
        source_path=args.source,
        output_path=output_path,
        capture_path=capture_path,
    )
    print("Held-out human anchor ingested")
    print(f"- labels: {result.anchor_count}")
    print(f"- label counts: {dict(result.label_counts)}")
    print(f"- capture matches: {result.matched_capture_count}")
    if result.missing_capture_example_ids:
        print(f"- missing capture ids: {', '.join(result.missing_capture_example_ids)}")
    else:
        print("- missing capture ids: none")
    print(f"- output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
