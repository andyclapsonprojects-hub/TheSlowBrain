"""Durable shadow-learning state for Slow Brain runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .data_quality import DataQualityIssue, Severity
from .gating_apply import gate_from_state
from .gating_model import GatingModelReport, LogisticGate
from .models import FeatureVector, RubricVersion
from .numeric import float_or_default, optional_float
from .promotion import LADDER, PromotionEvent, PromotionStage, PromotionState

ACTIVE_RUBRIC_STATE_JSON = Path("state/active_rubric.json")
GATING_GATE_STATE_JSON = Path("state/gating_gate.json")
GATING_PROMOTION_STATE_JSON = Path("state/gating_promotion.json")
TRACK_RECORD_JSONL = Path("reports/track-record/daily-history.jsonl")


type OutcomeStreamStatus = Literal["not_found", "loaded"]


class OutcomeStreamLoadResult:
    def __init__(
        self,
        *,
        status: OutcomeStreamStatus,
        features: tuple[FeatureVector, ...],
        usable_count: int,
        excluded_anchor_count: int,
        malformed_count: int,
    ) -> None:
        self.status = status
        self.features = features
        self.usable_count = usable_count
        self.excluded_anchor_count = excluded_anchor_count
        self.malformed_count = malformed_count

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "usable_count": self.usable_count,
            "excluded_anchor_count": self.excluded_anchor_count,
            "malformed_count": self.malformed_count,
        }


def workflow_run_id(clock: datetime | None = None) -> str:
    value = clock or datetime.now(UTC)
    return f"slowbrain-{value.astimezone(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"


def load_active_rubric(path: Path, *, default: RubricVersion) -> RubricVersion:
    if not path.exists():
        return default
    try:
        payload = _load_json_object(path)
        rubric_payload = _mapping(payload.get("rubric"))
        weights = _weights(rubric_payload.get("weights"))
        return RubricVersion(
            version=_text(rubric_payload.get("version")) or default.version,
            weights=weights or dict(default.weights),
            buy_threshold=float_or_default(rubric_payload.get("buy_threshold"), default=default.buy_threshold),
            sell_threshold=float_or_default(rubric_payload.get("sell_threshold"), default=default.sell_threshold),
            max_position_pct=float_or_default(
                rubric_payload.get("max_position_pct"),
                default=default.max_position_pct,
            ),
            notes=tuple(str(item) for item in _sequence(rubric_payload.get("notes"))),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default


def persist_active_rubric(
    path: Path,
    rubric: RubricVersion,
    *,
    run_id: str,
    promotion_action: str,
    reason: str,
) -> Path:
    payload = {
        "schema": "theslowbrain.active_rubric_state.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "promotion_action": promotion_action,
        "reason": reason,
        "rubric": asdict(rubric),
    }
    return _atomic_write_json(path, payload)


def persist_gating_gate(path: Path, report: GatingModelReport, *, run_id: str) -> Path:
    payload = {
        "schema": "theslowbrain.gating_gate_state.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "status": report.status,
        "selected_source": report.selected_source,
        "fallback_active": report.fallback_active,
        "fallback_reason": report.fallback_reason,
        "sample_count": report.sample_count,
        "confirmation_count": report.confirmation_count,
        "labels": report.labels,
        "feature_names": report.feature_names,
        "gate_weights": report.gate_weights,
        "drift_guard_passed": report.drift_guard_passed,
        "broker_live_execution_allowed": False,
    }
    return _atomic_write_json(path, payload)


def load_gating_gate(path: Path) -> LogisticGate | None:
    """Rebuild the previously-trained gate from persisted state, for applying to today's decisions."""
    if not path.exists():
        return None
    try:
        payload = _load_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    weights = payload.get("gate_weights")
    labels = payload.get("labels")
    feature_names = payload.get("feature_names")
    if not isinstance(weights, list) or not isinstance(labels, list) or not isinstance(feature_names, list):
        return None
    typed_weights = [[float(value) for value in row] for row in weights if isinstance(row, (list, tuple))]
    return gate_from_state(typed_weights, [str(item) for item in labels], [str(item) for item in feature_names])


def load_promotion_state(path: Path, *, required_streak: int) -> PromotionState:
    if not path.exists():
        return PromotionState(required_streak=max(1, required_streak))
    try:
        payload = _load_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return PromotionState(required_streak=max(1, required_streak))
    return PromotionState(
        stage=_stage(payload.get("stage")),
        qualifying_streak=max(0, _int(payload.get("qualifying_streak"), default=0)),
        required_streak=max(1, _int(payload.get("required_streak"), default=required_streak)),
        last_status=_text(payload.get("last_status")) or "loaded",
        last_reason=_text(payload.get("last_reason")) or "loaded_from_state",
        history=_promotion_history(payload.get("history")),
    )


def persist_promotion_state(path: Path, state: PromotionState, *, run_id: str) -> Path:
    payload = {
        "schema": "theslowbrain.gating_promotion_state.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "stage": state.stage,
        "qualifying_streak": state.qualifying_streak,
        "required_streak": state.required_streak,
        "last_status": state.last_status,
        "last_reason": state.last_reason,
        "broker_live_execution_allowed": False,
        "history": [asdict(event) for event in state.history],
    }
    return _atomic_write_json(path, payload)


def _stage(value: object) -> PromotionStage:
    text = _text(value)
    for stage in LADDER:
        if text == stage:
            return stage
    return "shadow"


def _promotion_history(value: object) -> tuple[PromotionEvent, ...]:
    events: list[PromotionEvent] = []
    for item in _sequence(value):
        raw = _mapping(item)
        events.append(
            PromotionEvent(
                stage=_stage(raw.get("stage")),
                status=_text(raw.get("status")),
                reason=_text(raw.get("reason")),
                qualifying_streak=max(0, _int(raw.get("qualifying_streak"), default=0)),
            )
        )
    return tuple(events)


def load_outcome_stream_features(path: Path, *, exclude_idea_ids: Sequence[str]) -> OutcomeStreamLoadResult:
    excluded = {str(idea_id) for idea_id in exclude_idea_ids}
    if not path.exists():
        return OutcomeStreamLoadResult(
            status="not_found",
            features=(),
            usable_count=0,
            excluded_anchor_count=0,
            malformed_count=0,
        )
    features: list[FeatureVector] = []
    excluded_anchor_count = 0
    malformed_count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            feature = _feature_from_stream_record(_mapping(payload))
        except (json.JSONDecodeError, TypeError, ValueError):
            malformed_count += 1
            continue
        if feature.idea_id in excluded:
            excluded_anchor_count += 1
            continue
        features.append(feature)
    ordered = tuple(sorted(features, key=lambda feature: (feature.signal_date, feature.horizon_days, feature.idea_id)))
    return OutcomeStreamLoadResult(
        status="loaded",
        features=ordered,
        usable_count=len(ordered),
        excluded_anchor_count=excluded_anchor_count,
        malformed_count=malformed_count,
    )


def append_track_record(
    path: Path,
    *,
    run_id: str,
    report_payload: Mapping[str, object],
    outcome_stream: OutcomeStreamLoadResult,
    active_rubric_state_path: Path,
    gating_gate_state_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _track_record_row(
        run_id=run_id,
        report_payload=report_payload,
        outcome_stream=outcome_stream,
        active_rubric_state_path=active_rubric_state_path,
        gating_gate_state_path=gating_gate_state_path,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def merge_feature_evidence(
    primary: Sequence[FeatureVector],
    append_only: Sequence[FeatureVector],
) -> tuple[FeatureVector, ...]:
    by_key: dict[tuple[str, int, str], FeatureVector] = {}
    for feature in (*primary, *append_only):
        by_key[(feature.idea_id, feature.horizon_days, feature.signal_date)] = feature
    return tuple(
        sorted(
            by_key.values(),
            key=lambda feature: (feature.signal_date, feature.horizon_days, feature.idea_id),
        )
    )


def _feature_from_stream_record(record: Mapping[str, object]) -> FeatureVector:
    feature = _mapping(record.get("feature"))
    outcome = _mapping(record.get("outcome"))
    idea_id = _required_text(feature.get("idea_id"), "feature.idea_id")
    return FeatureVector(
        idea_id=idea_id,
        ticker=_required_text(feature.get("ticker"), "feature.ticker").upper(),
        signal_date=_required_text(feature.get("signal_date"), "feature.signal_date"),
        sentiment=_text(feature.get("sentiment")) or "neutral",
        sentiment_confidence=float_or_default(feature.get("sentiment_confidence")),
        catalyst_strength=float_or_default(feature.get("catalyst_strength")),
        trend=_text(feature.get("trend")) or "sideways",
        momentum_20d_pct=float_or_default(feature.get("momentum_20d_pct")),
        mean_reversion_z_20d=float_or_default(feature.get("mean_reversion_z_20d")),
        volume_confirmed=bool(feature.get("volume_confirmed")),
        quality_status=_text(feature.get("quality_status")) or "unknown",
        risk_status=_text(feature.get("risk_status")) or "unknown",
        net_return_pct=float_or_default(
            outcome.get("realized_net_return_pct"),
            default=float_or_default(feature.get("net_return_pct")),
        ),
        cost_bps=float_or_default(feature.get("cost_bps")),
        source=_text(outcome.get("source")) or _text(feature.get("source")) or "append_only_outcome_stream",
        data_quality_issues=_issues(feature.get("data_quality_issues")),
        horizon_days=_int(outcome.get("horizon_days"), default=_int(feature.get("horizon_days"), default=10)),
        outcome_future_date=_text(outcome.get("future_date")) or _text(feature.get("outcome_future_date")),
        entry_price=optional_float(feature.get("entry_price")),
    )


def _track_record_row(
    *,
    run_id: str,
    report_payload: Mapping[str, object],
    outcome_stream: OutcomeStreamLoadResult,
    active_rubric_state_path: Path,
    gating_gate_state_path: Path,
) -> dict[str, object]:
    promotion = _mapping(report_payload.get("promotion_decision"))
    safety = _mapping(report_payload.get("safety"))
    gating = _mapping(report_payload.get("gating_model"))
    gating_promotion = _mapping(report_payload.get("gating_promotion"))
    decisions = _sequence(report_payload.get("trade_decisions"))
    return {
        "schema": "theslowbrain.daily_track_record.v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "promotion_action": _text(promotion.get("action")) or "unknown",
        "active_rubric_version": _text(promotion.get("active_version")),
        "selected_rubric_version": _text(promotion.get("selected_version")),
        "trade_decision_count": len(decisions),
        "buy_count": _decision_count(decisions, "BUY"),
        "sell_count": _decision_count(decisions, "SELL"),
        "outcome_stream": outcome_stream.as_dict(),
        "gating_status": _text(gating.get("status")),
        "gating_selected_source": _text(gating.get("selected_source")),
        "gating_fallback_active": bool(gating.get("fallback_active")),
        "promotion_active_stage": _text(gating_promotion.get("active_stage_applied")),
        "promotion_next_stage": _text(gating_promotion.get("next_stage")),
        "promotion_qualifying_streak": _int(gating_promotion.get("qualifying_streak"), default=0),
        "gate_decisions_changed": _int(gating_promotion.get("decisions_changed"), default=0),
        "gate_economic_pass": bool(gating_promotion.get("economic_pass")),
        "broker_live_execution_allowed": bool(safety.get("broker_live_execution_allowed")),
        "active_rubric_state_path": str(active_rubric_state_path),
        "gating_gate_state_path": str(gating_gate_state_path),
    }


def _decision_count(decisions: Sequence[object], action: str) -> int:
    return sum(1 for item in decisions if isinstance(item, Mapping) and item.get("action") == action)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)
    return path


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("state payload must be an object")
    return {str(key): item for key, item in value.items()}


def _weights(value: object) -> dict[str, float]:
    mapping = _mapping(value)
    return {str(key): float_or_default(item) for key, item in mapping.items()}


def _issues(value: object) -> tuple[DataQualityIssue, ...]:
    issues: list[DataQualityIssue] = []
    for item in _sequence(value):
        raw = _mapping(item)
        severity = _severity(raw.get("severity"))
        if severity is None:
            continue
        issues.append(
            DataQualityIssue(
                field=_text(raw.get("field")),
                code=_text(raw.get("code")),
                severity=severity,
                message=_text(raw.get("message")),
            )
        )
    return tuple(issues)


def _severity(value: object) -> Severity | None:
    text = _text(value)
    if text == "info":
        return "info"
    if text == "warning":
        return "warning"
    if text == "error":
        return "error"
    return None


def _required_text(value: object, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: object) -> str:
    return str(value or "").strip()


def _int(value: object, *, default: int) -> int:
    parsed = optional_float(value, allow_bool=True)
    return int(parsed) if parsed is not None else default


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()
