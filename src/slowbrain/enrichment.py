"""Point-in-time enrichment records for fundamentals, sentiment, and catalyst features."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from .models import FeatureVector
from .numeric import optional_float


@dataclass(frozen=True)
class PointInTimeEnrichment:
    ticker: str
    available_date: str
    source: str
    sentiment: str = ""
    sentiment_confidence: float | None = None
    catalyst_strength: float | None = None
    value_score: float | None = None
    fundamental_quality_score: float | None = None
    size_score: float | None = None
    liquidity_score: float | None = None


def load_pit_enrichment_records(path: Path) -> tuple[PointInTimeEnrichment, ...]:
    """Load local PIT enrichment exports from JSONL or CSV."""
    if not path.exists():
        return ()
    if path.suffix.lower() == ".csv":
        csv_rows: list[PointInTimeEnrichment] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = _record_from_mapping(row)
                if record is not None:
                    csv_rows.append(record)
        return tuple(csv_rows)
    json_rows: list[PointInTimeEnrichment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            record = _record_from_mapping(value)
            if record is not None:
                json_rows.append(record)
    return tuple(json_rows)


def join_point_in_time_enrichment(
    features: Sequence[FeatureVector],
    records: Sequence[PointInTimeEnrichment],
) -> tuple[FeatureVector, ...]:
    """Apply the latest enrichment record whose available date is not after the feature date."""
    by_ticker: dict[str, list[PointInTimeEnrichment]] = defaultdict(list)
    for record in records:
        by_ticker[record.ticker].append(record)
    for ticker_records in by_ticker.values():
        ticker_records.sort(key=lambda record: record.available_date)
    return tuple(_enrich_feature(feature, by_ticker.get(feature.ticker, ())) for feature in features)


def _record_from_mapping(raw: Mapping[str, object]) -> PointInTimeEnrichment | None:
    ticker = str(raw.get("ticker") or "").strip().upper()
    available_date = str(raw.get("available_date") or raw.get("as_of_date") or "").strip()
    source = str(raw.get("source") or "pit_enrichment_export").strip()
    if not ticker or not _valid_date(available_date):
        return None
    return PointInTimeEnrichment(
        ticker=ticker,
        available_date=available_date,
        source=source,
        sentiment=_sentiment(raw.get("sentiment")),
        sentiment_confidence=_score(raw.get("sentiment_confidence")),
        catalyst_strength=_score(raw.get("catalyst_strength")),
        value_score=_score(raw.get("value_score")),
        fundamental_quality_score=_score(raw.get("fundamental_quality_score") or raw.get("quality_score")),
        size_score=_score(raw.get("size_score")),
        liquidity_score=_score(raw.get("liquidity_score")),
    )


def _enrich_feature(feature: FeatureVector, records: Iterable[PointInTimeEnrichment]) -> FeatureVector:
    signal_date = _date(feature.signal_date)
    if signal_date is None:
        return feature
    match: PointInTimeEnrichment | None = None
    for record in records:
        available_date = _date(record.available_date)
        if available_date is not None and available_date <= signal_date:
            match = record
    if match is None:
        return feature
    return replace(
        feature,
        sentiment=match.sentiment or feature.sentiment,
        sentiment_confidence=_coalesce(match.sentiment_confidence, feature.sentiment_confidence),
        catalyst_strength=_coalesce(match.catalyst_strength, feature.catalyst_strength),
        value_score=_coalesce(match.value_score, feature.value_score),
        fundamental_quality_score=_coalesce(match.fundamental_quality_score, feature.fundamental_quality_score),
        size_score=_coalesce(match.size_score, feature.size_score),
        liquidity_score=_coalesce(match.liquidity_score, feature.liquidity_score),
        pit_enrichment_source=match.source,
        pit_enrichment_available_date=match.available_date,
    )


def _score(value: object) -> float | None:
    parsed = optional_float(value, allow_bool=False)
    if parsed is None:
        return None
    return max(-1.0, min(1.0, parsed))


def _coalesce(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _sentiment(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"positive", "negative", "neutral"} else ""


def _valid_date(value: str) -> bool:
    return _date(value) is not None


def _date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
