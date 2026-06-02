"""Run repeated TheSlowBrain paper/shadow cycles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.shadow_runner import run_shadow_cycles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--feature-limit", type=int, default=5000)
    args = parser.parse_args()

    summary = run_shadow_cycles(args.project_root, count=args.count, feature_limit=args.feature_limit)
    print("TheSlowBrain shadow cycles complete.")
    print(f"Run count: {summary['run_count']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    print(f"Broker live execution allowed: {str(summary['broker_live_execution_allowed']).lower()}")
    lines = summary.get("latest_eric_brief_lines")
    if isinstance(lines, (list, tuple)):
        print("\n".join(str(line) for line in lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
