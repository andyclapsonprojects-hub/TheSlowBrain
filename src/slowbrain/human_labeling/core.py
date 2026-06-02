"""Build, write, and merge human-labeling packs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..eval_council import VALID_HUMAN_LABELS
from ..market_data import PriceHistoryProvider
from ..technical_context import build_technical_context
from .models import (
    CSV_COLUMNS,
    HUMAN_LABELING_CSV,
    HUMAN_LABELING_HTML,
    HUMAN_LABELING_JSON,
    HumanLabelingCase,
    HumanLabelingOutputs,
    HumanLabelingPack,
)
from .render import _html
from .utils import _blank_none, _existing_label, _float, _mapping, _optional_float, _resolve, _text


def build_human_labeling_pack(
    *,
    capture_path: Path,
    records: Sequence[Mapping[str, object]],
    price_provider: PriceHistoryProvider | None,
    limit: int | None = None,
    generated_at: datetime | None = None,
) -> HumanLabelingPack:
    """Build the review pack Andy should actually use for golden labels."""
    all_cases = tuple(_case_from_record(index + 1, record, price_provider) for index, record in enumerate(records))
    cases = tuple(sorted(all_cases, key=_review_priority)[:limit]) if limit is not None else all_cases
    created = generated_at or datetime.now(UTC)
    return HumanLabelingPack(
        schema="theslowbrain.human_labeling_pack.v1",
        generated_at=created.isoformat(),
        source_capture_path=str(capture_path),
        case_count=len(cases),
        mode="assisted_review_context_not_human_labels",
        rows_are_human_labels=False,
        label_values_allowed=tuple(sorted(VALID_HUMAN_LABELS)),
        notes=(
            "Use the market-context columns and HTML cards to decide BUY, SELL, HOLD, or UNKNOWN.",
            "The pack includes decision-time OHLCV context; 10-day outcome is labelled as calibration context.",
            "Fill human_label and human_rationale only when Andy has made the judgement.",
        ),
        cases=cases,
    )


def write_human_labeling_pack(
    pack: HumanLabelingPack,
    *,
    project_root: Path,
    output_json: Path = HUMAN_LABELING_JSON,
    output_csv: Path = HUMAN_LABELING_CSV,
    output_html: Path = HUMAN_LABELING_HTML,
) -> HumanLabelingOutputs:
    """Write JSON, CSV, and bright HTML versions of the same review pack."""
    json_path = _resolve(project_root, output_json)
    csv_path = _resolve(project_root, output_csv)
    html_path = _resolve(project_root, output_html)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(pack), indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(_csv_row(case) for case in pack.cases)
    html_path.write_text(_html(pack), encoding="utf-8")
    return HumanLabelingOutputs(json_path=json_path, csv_path=csv_path, html_path=html_path)


def merge_completed_human_labels(
    *,
    capture_records: Sequence[Mapping[str, object]],
    label_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Merge completed CSV labels back into decision-capture JSONL records."""
    labels = _valid_label_rows(label_rows)
    merged: list[dict[str, object]] = []
    for index, record in enumerate(capture_records, start=1):
        copied = dict(record)
        label = labels.get(_example_id(record, index))
        if label is not None:
            copied["human_label"] = label[0]
            copied["human_rationale"] = label[1]
        merged.append(copied)
    return tuple(merged)


def load_label_csv(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple({str(key): value for key, value in row.items()} for row in csv.DictReader(handle))


def write_decision_capture_records(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _case_from_record(
    line_number: int,
    record: Mapping[str, object],
    price_provider: PriceHistoryProvider | None,
) -> HumanLabelingCase:
    feature = _mapping(record.get("feature"))
    decision = _mapping(record.get("decision"))
    outcome = _mapping(record.get("outcome"))
    ticker = _text(feature.get("ticker")).upper()
    signal_date = _text(feature.get("signal_date"))
    prices = price_provider.daily_prices(ticker) if price_provider is not None and ticker else ()
    context = build_technical_context(symbol=ticker, signal_date=signal_date, prices=prices)
    return HumanLabelingCase(
        example_id=_example_id(record, line_number),
        ticker=ticker,
        signal_date=signal_date,
        line_number=line_number,
        slowbrain_action=_text(decision.get("action")),
        slowbrain_score=_float(decision.get("score")),
        slowbrain_reason=_text(decision.get("reason")),
        sentiment=_text(feature.get("sentiment")),
        sentiment_confidence=_float(feature.get("sentiment_confidence")),
        catalyst_strength=_float(feature.get("catalyst_strength")),
        feature_momentum_20d_pct=_float(feature.get("momentum_20d_pct")),
        feature_mean_reversion_z_20d=_float(feature.get("mean_reversion_z_20d")),
        outcome_10d_net_return_pct=_optional_float(outcome.get("realized_net_return_pct")),
        source=_text(feature.get("source")),
        technical_context=context,
        human_label=_existing_label(record.get("human_label")),
        human_rationale=_text(record.get("human_rationale")),
    )


def _review_priority(case: HumanLabelingCase) -> tuple[int, float, str]:
    status_rank = {"available": 0, "partial": 1, "unavailable": 2}.get(case.technical_context.status, 3)
    outcome = abs(case.outcome_10d_net_return_pct or 0.0)
    return (status_rank, -outcome, case.ticker)


def _example_id(record: Mapping[str, object], line_number: int) -> str:
    feature = _mapping(record.get("feature"))
    idea_id = _text(feature.get("idea_id"))
    if idea_id:
        return idea_id
    run_id = _text(record.get("run_id"))
    return f"{run_id or 'capture'}:{line_number}"


def _csv_row(case: HumanLabelingCase) -> dict[str, object]:
    context = case.technical_context
    return {
        "example_id": case.example_id,
        "ticker": case.ticker,
        "signal_date": case.signal_date,
        "slowbrain_action": case.slowbrain_action,
        "slowbrain_score": case.slowbrain_score,
        "slowbrain_reason": case.slowbrain_reason,
        "sentiment": case.sentiment,
        "sentiment_confidence": case.sentiment_confidence,
        "catalyst_strength": case.catalyst_strength,
        "feature_momentum_20d_pct": case.feature_momentum_20d_pct,
        "feature_mean_reversion_z_20d": case.feature_mean_reversion_z_20d,
        "outcome_10d_net_return_pct": _blank_none(case.outcome_10d_net_return_pct),
        "market_context_status": context.status,
        "market_context_reason": context.reason,
        "price_asof_date": _blank_none(context.price_asof_date),
        "price_source": context.price_source,
        "open": _blank_none(context.open),
        "high": _blank_none(context.high),
        "low": _blank_none(context.low),
        "close": _blank_none(context.close),
        "adjusted_close": _blank_none(context.adjusted_close),
        "volume": _blank_none(context.volume),
        "previous_close": _blank_none(context.previous_close),
        "day_change_pct": _blank_none(context.day_change_pct),
        "gap_pct": _blank_none(context.gap_pct),
        "intraday_return_pct": _blank_none(context.intraday_return_pct),
        "candle_range_pct": _blank_none(context.candle_range_pct),
        "candle_body_pct": _blank_none(context.candle_body_pct),
        "close_location_pct": _blank_none(context.close_location_pct),
        "volume_ratio_20d": _blank_none(context.volume_ratio_20d),
        "volume_signal": context.volume_signal,
        "sma_5": _blank_none(context.sma_5),
        "sma_20": _blank_none(context.sma_20),
        "technical_trend": context.trend,
        "technical_momentum_5d_pct": _blank_none(context.momentum_5d_pct),
        "technical_momentum_20d_pct": _blank_none(context.momentum_20d_pct),
        "distance_from_20d_high_pct": _blank_none(context.distance_from_20d_high_pct),
        "distance_from_20d_low_pct": _blank_none(context.distance_from_20d_low_pct),
        "candlestick_patterns": ", ".join(context.pattern_names),
        "candlestick_summary": context.pattern_summary,
        "human_label": case.human_label,
        "human_rationale": case.human_rationale,
    }


def _valid_label_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    for row in rows:
        example_id = _text(row.get("example_id"))
        label = _text(row.get("human_label")).upper()
        rationale = _text(row.get("human_rationale"))
        if not example_id or not label:
            continue
        if label not in VALID_HUMAN_LABELS:
            raise ValueError(f"invalid human_label for {example_id}: {label}")
        labels[example_id] = (label, rationale)
    return labels


