"""Merge completed human labelling CSV rows back into decision capture JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.decision_capture import DECISION_CAPTURE_JSONL
from slowbrain.human_labeling import (
    HUMAN_LABELING_CSV,
    load_decision_capture_records,
    load_label_csv,
    merge_completed_human_labels,
    write_decision_capture_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture", type=Path, default=DECISION_CAPTURE_JSONL)
    parser.add_argument("--labels", type=Path, default=HUMAN_LABELING_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/decision-capture/latest-decision-log.with-human-labels.jsonl"),
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite the capture file after validating labels.")
    args = parser.parse_args()

    capture_path = args.capture if args.capture.is_absolute() else args.project_root / args.capture
    labels_path = args.labels if args.labels.is_absolute() else args.project_root / args.labels
    if args.in_place:
        output_path = capture_path
    elif args.output.is_absolute():
        output_path = args.output
    else:
        output_path = args.project_root / args.output

    records = load_decision_capture_records(capture_path)
    merged = merge_completed_human_labels(capture_records=records, label_rows=load_label_csv(labels_path))
    labelled_count = sum(1 for record in merged if record.get("human_label"))
    write_decision_capture_records(output_path, merged)

    print("Human labels merged")
    print(f"- labelled records: {labelled_count}")
    print(f"- output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
