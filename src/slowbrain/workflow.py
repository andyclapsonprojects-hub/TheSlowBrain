"""End-to-end first TheSlowBrain cycle."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

from .backtest import economic_edge, evaluate_rubric
from .config import load_config
from .data_import import MANIFEST_NAME, default_import_root, load_manifest
from .decision_capture import (
    DECISION_CAPTURE_JSONL,
    DECISION_OUTCOME_STREAM_JSONL,
    append_decision_outcome_stream,
    write_decision_capture,
)
from .eval_council import HumanLabel, OpenAIJudgeClient, calibrate_against_humans, load_human_examples
from .features import (
    SUPPORTED_FORWARD_HORIZONS,
    load_features_for_idea_ids_from_legacy_sqlite,
    load_features_from_legacy_sqlite,
    load_training_features_from_legacy_sqlite,
)
from .gating_apply import (
    apply_gate_to_decisions,
    build_gate_primary_pairs,
    gate_decider,
    gate_from_state,
    gate_secondary_guards_passed,
)
from .gating_training import evaluate_gating_model
from .grader_council import propose_rubric_candidates
from .human_anchor import HUMAN_ANCHOR_JSON
from .learning_state import (
    ACTIVE_RUBRIC_STATE_JSON,
    GATING_GATE_STATE_JSON,
    GATING_PROMOTION_STATE_JSON,
    TRACK_RECORD_JSONL,
    append_track_record,
    load_active_rubric,
    load_gating_gate,
    load_outcome_stream_features,
    load_promotion_state,
    merge_feature_evidence,
    persist_active_rubric,
    persist_gating_gate,
    persist_promotion_state,
    workflow_run_id,
)
from .market_data import warm_market_data_provider
from .market_data_vendors import build_market_data_provider
from .optimizer import select_rubric
from .promotion import evaluate_promotion, next_stage
from .reporting import build_eric_brief, write_first_report
from .rubrics import BASE_RUBRIC, decide_feature
from .trading_flow import build_blocked_order_intents, build_ranked_trade_decision_pairs, load_portfolio_state

FIRST_REPORT_JSON = Path("reports/first-slowbrain-report.json")
FIRST_REPORT_MD = Path("reports/first-slowbrain-report.md")


def run_first_cycle(project_root: Path, *, feature_limit: int | None = 5000) -> dict[str, object]:
    run_id = workflow_run_id()
    config = load_config(project_root)
    import_root = default_import_root(project_root)
    manifest = load_manifest(import_root / MANIFEST_NAME)
    sqlite_path = import_root / "paper_trading" / "pipeline_runs.sqlite"
    anchor_path = project_root / HUMAN_ANCHOR_JSON
    active_rubric_state_target = project_root / ACTIVE_RUBRIC_STATE_JSON
    gating_gate_state_target = project_root / GATING_GATE_STATE_JSON
    track_record_target = project_root / TRACK_RECORD_JSONL
    active_rubric = load_active_rubric(active_rubric_state_target, default=BASE_RUBRIC)
    anchor_examples = load_human_examples(anchor_path)
    anchor_ids = tuple(example.example_id for example in anchor_examples)
    features = load_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=10,
        limit=feature_limit,
        exclude_idea_ids=anchor_ids,
    )
    gating_features = load_training_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=SUPPORTED_FORWARD_HORIZONS,
        limit=feature_limit,
        exclude_idea_ids=anchor_ids,
    )
    market_data_provider = build_market_data_provider(config, project_root=project_root)
    warm_market_data_provider(market_data_provider, features)
    candidates = propose_rubric_candidates(active_rubric, features)
    openai_judge = (
        OpenAIJudgeClient(api_key=config.openai_api_key, model=config.openai_model)
        if config.openai_api_key
        else None
    )
    promotion = select_rubric(
        active=active_rubric,
        candidates=candidates,
        features=features,
        openai_judge=openai_judge,
        council_cache_dir=project_root / "data" / "eval_council_cache",
        market_data_provider=market_data_provider,
    )
    selected_rubric = next(
        (candidate.rubric for candidate in candidates if candidate.rubric.version == promotion.selected_version),
        active_rubric,
    )
    active_rubric_state_path = persist_active_rubric(
        active_rubric_state_target,
        selected_rubric,
        run_id=run_id,
        promotion_action=promotion.action,
        reason=promotion.reason,
    )
    # Apply YESTERDAY's persisted, already-earned gate to TODAY's decisions (no look-ahead). The gate
    # is re-trained and re-graded below to update its promotion stage for the next run.
    promotion_state_target = project_root / GATING_PROMOTION_STATE_JSON
    persisted_gate = load_gating_gate(gating_gate_state_target)
    promotion_state = load_promotion_state(promotion_state_target, required_streak=config.gating_required_streak)
    active_stage = config.gating_stage_override or promotion_state.stage
    if active_stage == "gate_primary" and persisted_gate is not None:
        # NN is primary: it proposes (and ranks) the decisions; the rubric is only a guardrail.
        decision_pairs, gate_influences = build_gate_primary_pairs(
            features[-200:], persisted_gate, selected_rubric, limit=10
        )
    else:
        decision_pairs = build_ranked_trade_decision_pairs(features[-200:], selected_rubric, limit=10)
        decision_pairs, gate_influences = apply_gate_to_decisions(
            decision_pairs, persisted_gate, selected_rubric, stage=active_stage
        )
    label_capture_pairs = build_ranked_trade_decision_pairs(features[-1000:], selected_rubric, limit=200)
    decisions = tuple(decision for _, decision in decision_pairs)
    decision_capture_path = write_decision_capture(
        project_root / DECISION_CAPTURE_JSONL,
        label_capture_pairs,
        run_id=run_id,
    )
    decision_outcome_stream_path = append_decision_outcome_stream(
        project_root / DECISION_OUTCOME_STREAM_JSONL,
        label_capture_pairs,
        run_id=run_id,
    )
    outcome_stream = load_outcome_stream_features(decision_outcome_stream_path, exclude_idea_ids=anchor_ids)
    gating_evidence = merge_feature_evidence(gating_features, outcome_stream.features)
    anchor_features = load_features_for_idea_ids_from_legacy_sqlite(
        sqlite_path,
        idea_ids=anchor_ids,
        horizon_days=10,
    )
    automated_anchor_labels = {
        feature.idea_id: cast(HumanLabel, decide_feature(feature, selected_rubric).action)
        for feature in anchor_features
    }
    human_calibration = calibrate_against_humans(anchor_examples, automated_anchor_labels)
    gating_model = evaluate_gating_model(
        gating_evidence,
        selected_rubric,
        human_examples=anchor_examples,
        anchor_features=anchor_features,
        active_stage=active_stage,
        warm_start_gate=persisted_gate,
    )
    gating_gate_state_path = persist_gating_gate(gating_gate_state_target, gating_model, run_id=run_id)
    # Re-grade the freshly-trained gate on PROFIT (primary) + calibration/drift (secondary), then run
    # the earned, reversible promotion state machine and persist the stage for the next run.
    fresh_gate = gate_from_state(gating_model.gate_weights, gating_model.labels, gating_model.feature_names)
    # Grade the gate at the semantics of the stage it is trying to earn (or hold, at the top), so each
    # rung — including gate_primary — is justified by beating the rubric on profit at the level it operates.
    candidate_stage = next_stage(promotion_state.stage)
    economic = (
        economic_edge(
            gating_evidence,
            selected_rubric,
            gate_decide=gate_decider(fresh_gate, selected_rubric, stage=candidate_stage),
            market_data_provider=market_data_provider,
        )
        if fresh_gate is not None
        else None
    )
    economic_pass = economic.gate_beats_rubric if economic is not None else False
    secondary_pass = gate_secondary_guards_passed(gating_model)
    next_promotion_state = evaluate_promotion(
        promotion_state,
        economic_pass=economic_pass,
        secondary_pass=secondary_pass,
        override=config.gating_stage_override,
    )
    gating_promotion_state_path = persist_promotion_state(promotion_state_target, next_promotion_state, run_id=run_id)
    gating_promotion: dict[str, object] = {
        "active_stage_applied": active_stage,
        "next_stage": next_promotion_state.stage,
        "qualifying_streak": next_promotion_state.qualifying_streak,
        "required_streak": next_promotion_state.required_streak,
        "last_status": next_promotion_state.last_status,
        "last_reason": next_promotion_state.last_reason,
        "economic_pass": economic_pass,
        "secondary_pass": secondary_pass,
        "decisions_changed": len(gate_influences),
        "influences": [asdict(influence) for influence in gate_influences],
        "economic_edge": asdict(economic) if economic is not None else None,
        "state_path": str(gating_promotion_state_path),
        "broker_live_execution_allowed": False,
    }
    portfolio_backtest = evaluate_rubric(features, selected_rubric, market_data_provider=market_data_provider)
    portfolio = load_portfolio_state(import_root)
    intents = build_blocked_order_intents(decisions, portfolio)
    brief = build_eric_brief(decisions, portfolio)
    payload = write_first_report(
        output_json=project_root / FIRST_REPORT_JSON,
        output_md=project_root / FIRST_REPORT_MD,
        promotion=promotion,
        decisions=decisions,
        blocked_order_intents=intents,
        import_record_count=len(manifest.records),
        portfolio=portfolio,
        portfolio_backtest=portfolio_backtest,
        decision_capture_path=decision_capture_path,
        decision_outcome_stream_path=decision_outcome_stream_path,
        active_rubric_state_path=active_rubric_state_path,
        gating_gate_state_path=gating_gate_state_path,
        track_record_path=track_record_target,
        outcome_stream_training=outcome_stream.as_dict(),
        human_calibration=human_calibration,
        gating_model=gating_model,
        gating_promotion=gating_promotion,
        brief=brief,
    )
    append_track_record(
        track_record_target,
        run_id=run_id,
        report_payload=payload,
        outcome_stream=outcome_stream,
        active_rubric_state_path=active_rubric_state_path,
        gating_gate_state_path=gating_gate_state_path,
    )
    return payload
