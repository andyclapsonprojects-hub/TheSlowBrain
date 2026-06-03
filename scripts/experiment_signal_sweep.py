"""Run an honest PR12 signal sweep without live trading side effects."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.backtest import economic_edge
from slowbrain.config import load_config
from slowbrain.data_import import MANIFEST_NAME, default_import_root, load_manifest
from slowbrain.data_quality import has_error
from slowbrain.enrichment import join_point_in_time_enrichment, load_pit_enrichment_records
from slowbrain.eval_council import load_human_examples
from slowbrain.features import (
    SUPPORTED_FORWARD_HORIZONS,
    attach_cross_sectional_context,
    load_training_features_from_legacy_sqlite,
)
from slowbrain.gating_apply import gate_decider, gate_from_state
from slowbrain.gating_model import TargetLabelMode
from slowbrain.gating_training import evaluate_gating_model
from slowbrain.human_anchor import HUMAN_ANCHOR_JSON
from slowbrain.learning_state import ACTIVE_RUBRIC_STATE_JSON, load_active_rubric
from slowbrain.models import FeatureVector, RubricVersion
from slowbrain.rubrics import BASE_RUBRIC

DEFAULT_OUTPUT = Path("reports/experiments/pr12-signal-sweep.json")
SequenceFeature = Sequence[FeatureVector]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    payload = build_sweep_payload(
        project_root,
        feature_limit=args.feature_limit,
        label_modes=_label_modes(args.label_mode),
    )
    output = project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote PR12 signal sweep: {output}")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


def build_sweep_payload(
    project_root: Path,
    *,
    feature_limit: int | None,
    label_modes: tuple[TargetLabelMode, ...],
) -> dict[str, object]:
    config = load_config(project_root)
    import_root = default_import_root(project_root)
    load_manifest(import_root / MANIFEST_NAME)
    sqlite_path = import_root / "paper_trading" / "pipeline_runs.sqlite"
    anchor_ids = tuple(example.example_id for example in load_human_examples(project_root / HUMAN_ANCHOR_JSON))
    features = load_training_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=SUPPORTED_FORWARD_HORIZONS,
        limit=feature_limit,
        exclude_idea_ids=anchor_ids,
    )
    if config.pit_enrichment_enabled:
        if config.pit_enrichment_path is None or not config.pit_enrichment_path.exists():
            raise FileNotFoundError(f"PIT enrichment path does not exist: {config.pit_enrichment_path}")
        records = load_pit_enrichment_records(config.pit_enrichment_path)
        features = attach_cross_sectional_context(join_point_in_time_enrichment(features, records))
    rubric = load_active_rubric(project_root / ACTIVE_RUBRIC_STATE_JSON, default=BASE_RUBRIC)
    runs = {mode: _run_mode(features, rubric, mode) for mode in label_modes}
    return {
        "schema": "theslowbrain.pr12_signal_sweep.v1",
        "summary": _summary(features, runs),
        "feature_limit": feature_limit,
        "label_modes": list(label_modes),
        "runs": runs,
        "safety": {"broker_live_execution_allowed": False, "orders_submitted": False},
    }


def _run_mode(features: SequenceFeature, rubric: RubricVersion, mode: TargetLabelMode) -> dict[str, object]:
    report = evaluate_gating_model(features, rubric, target_label_mode=mode)
    gate = gate_from_state(report.gate_weights, report.labels, report.feature_names)
    edge = (
        economic_edge(features, rubric, gate_decide=gate_decider(gate, rubric, stage="gate_primary"))
        if gate is not None
        else None
    )
    return {
        "gating_model": asdict(report),
        "economic_edge": asdict(edge) if edge is not None else None,
    }


def _summary(features: SequenceFeature, runs: dict[TargetLabelMode, dict[str, object]]) -> dict[str, object]:
    issue_counts = Counter(issue.code for feature in features for issue in feature.data_quality_issues)
    horizon_counts = Counter(feature.horizon_days for feature in features)
    return {
        "loaded_feature_count": len(features),
        "clean_feature_count": sum(1 for feature in features if not has_error(feature.data_quality_issues)),
        "horizon_counts": dict(sorted(horizon_counts.items())),
        "data_quality_issue_counts": dict(issue_counts.most_common(20)),
        "technical_coverage": {
            "rsi_14": sum(1 for feature in features if feature.rsi_14),
            "macd_signal": sum(1 for feature in features if feature.macd_signal != "unknown"),
            "atr_pct_14": sum(1 for feature in features if feature.atr_pct_14),
            "momentum_63d_pct": sum(1 for feature in features if feature.momentum_63d_pct),
            "volume_ratio_20d": sum(1 for feature in features if feature.volume_ratio_20d),
        },
        "gate_beats_by_mode": {
            mode: _edge_flag(run.get("economic_edge")) for mode, run in runs.items()
        },
    }


def _edge_flag(edge: object) -> bool:
    data = cast("dict[str, object] | None", edge)
    return bool(data and data.get("gate_beats_rubric") is True)


def _label_modes(value: str) -> tuple[TargetLabelMode, ...]:
    if value == "both":
        return ("absolute_return", "cross_sectional_rank")
    return (cast(TargetLabelMode, value),)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="TheSlowBrain project root.")
    parser.add_argument("--feature-limit", type=int, default=5000, help="Bounded latest-row sample; 0 means no rows.")
    parser.add_argument(
        "--label-mode",
        choices=("both", "absolute_return", "cross_sectional_rank"),
        default="both",
        help="Target-label mode to train/evaluate.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path under project root.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
