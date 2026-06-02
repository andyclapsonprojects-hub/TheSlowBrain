"""Run TheSlowBrain's first native research cycle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.workflow import FIRST_REPORT_JSON, run_first_cycle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--feature-limit", type=int, default=5000)
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Load all eligible 10-day rows instead of the bounded latest-row slice.",
    )
    args = parser.parse_args()

    payload = run_first_cycle(args.project_root, feature_limit=None if args.full_universe else args.feature_limit)
    brief = payload.get("eric_brief")
    print("TheSlowBrain first cycle complete.")
    print(f"Report: {args.project_root / FIRST_REPORT_JSON}")
    if isinstance(brief, dict):
        lines = brief.get("lines")
        if isinstance(lines, (list, tuple)):
            print("\n".join(str(line) for line in lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
