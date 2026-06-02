"""Run one full TheSlowBrain research cycle on a small synthetic dataset.

This is the zero-setup entrypoint for new readers: it needs no API keys and no
external data. It builds a tiny synthetic "legacy" dataset in a scratch
directory, imports it with a provenance manifest, runs the offline workflow, and
prints the resulting brief plus where the report was written.

    uv run python scripts/run_demo_cycle.py
    uv run python scripts/run_demo_cycle.py --output-dir ./demo-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.data_import import build_import_manifest
from slowbrain.workflow import FIRST_REPORT_JSON, run_first_cycle

_PEAD_HEADER = (
    "position_id,strategy_id,ticker,entry_date,entry_price,quantity,target_exit_date,"
    "status,exit_date,exit_price,gross_return_pct,net_return_pct,cost_bps,pnl_currency,updated_at\n"
)
_FILLS_HEADER = (
    "execution_id,order_id,ticker,side,order_type,filled_quantity,average_fill_price,limit_price,"
    "currency,submitted_at,filled_at,status,source,net_value,fx_rate,taxes,raw_order_status,error\n"
)
_SIGNAL_JSON = (
    '{"trend": "uptrend", "momentum_20d_pct": 8, '
    '"mean_reversion_z_20d": 0, "volume_signal": "high_volume_confirmation"}'
)


def build_synthetic_legacy(legacy_root: Path) -> None:
    """Create a minimal synthetic legacy dataset the workflow can import."""
    (legacy_root / "data").mkdir(parents=True, exist_ok=True)
    paper = legacy_root / "paper_trading"
    paper.mkdir(parents=True, exist_ok=True)
    (legacy_root / "data" / "raw.json").write_text('{"source": "synthetic-demo"}', encoding="utf-8")
    (paper / "pead_positions.csv").write_text(_PEAD_HEADER, encoding="utf-8")
    (paper / "live_fills.csv").write_text(_FILLS_HEADER, encoding="utf-8")
    _write_synthetic_sqlite(paper / "pipeline_runs.sqlite")


def _write_synthetic_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE step2_research_ideas (
                idea_id TEXT PRIMARY KEY,
                generated_at TEXT NOT NULL,
                case_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_date TEXT,
                source_title TEXT,
                evidence TEXT,
                sentiment TEXT,
                sentiment_confidence REAL,
                catalyst_strength REAL,
                recommendation TEXT,
                quality_status TEXT,
                risk_status TEXT,
                order_created INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                signal_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                eval_stage TEXT NOT NULL
            );
            CREATE TABLE step2_forward_returns (
                idea_id TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                future_date TEXT NOT NULL,
                gross_return_pct REAL NOT NULL,
                net_return_pct REAL NOT NULL,
                cost_bps REAL NOT NULL,
                PRIMARY KEY (idea_id, horizon_days)
            );
            """
        )
        for index in range(12):
            idea_id = f"idea_{index}"
            conn.execute(
                "INSERT INTO step2_research_ideas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    idea_id,
                    "2026-01-01T00:00:00Z",
                    f"case_{index}",
                    f"candidate_{index}",
                    "AAPL" if index % 2 == 0 else "MSFT",
                    f"2026-01-{index + 1:02d}",
                    "Synthetic",
                    "Evidence",
                    "positive",
                    0.8,
                    0.8,
                    "buy",
                    "pass",
                    "pass",
                    1,
                    100.0,
                    _SIGNAL_JSON,
                    "{}",
                    "synthetic",
                ),
            )
            conn.execute(
                "INSERT INTO step2_forward_returns VALUES (?, 10, ?, ?, ?, 45)",
                (idea_id, "2026-02-01", 1.0, 1.0),
            )
        conn.commit()
    finally:
        conn.close()


def run_demo(output_dir: Path | None) -> int:
    scratch = output_dir if output_dir is not None else Path(tempfile.mkdtemp(prefix="slowbrain-demo-"))
    legacy_root = scratch / "legacy"
    project_root = scratch / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    # Force an offline, deterministic run regardless of any ambient configuration.
    os.environ["SLOWBRAIN_MARKET_DATA_ENABLED"] = "false"

    build_synthetic_legacy(legacy_root)
    build_import_manifest(legacy_root=legacy_root, project_root=project_root, copy_files=True)

    payload = run_first_cycle(project_root, feature_limit=None)

    print("TheSlowBrain demo cycle complete (synthetic data, offline).")
    print(f"Scratch directory: {scratch}")
    print(f"Report: {project_root / FIRST_REPORT_JSON}")
    print("-" * 48)
    brief = payload.get("eric_brief")
    if isinstance(brief, dict):
        lines = brief.get("lines")
        if isinstance(lines, (list, tuple)):
            print("\n".join(str(line) for line in lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the scratch dataset/report (default: a temporary directory).",
    )
    args = parser.parse_args()
    return run_demo(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
