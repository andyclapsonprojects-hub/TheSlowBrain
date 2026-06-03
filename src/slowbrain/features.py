"""Feature loading and normalization."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from math import tanh
from pathlib import Path
from typing import Any

from .data_quality import (
    DataQualityIssue,
    check_outcome_plausibility,
    check_tradeable_universe,
    parse_float,
    parse_json_object,
)
from .models import DecisionAction, FeatureVector
from .numeric import optional_float

SUPPORTED_FORWARD_HORIZONS = (1, 5, 10, 20)
# Scales an absolute earnings-surprise percentage into a [0,1) materiality via tanh: a ~25% surprise
# maps to ~0.76, larger surprises saturate toward 1.0 so outliers (e.g. -265%) do not dominate.
EARNINGS_SURPRISE_SCALE = 25.0
CROSS_SECTIONAL_NUMERIC_FIELDS = (
    "sentiment_confidence",
    "catalyst_strength",
    "momentum_20d_pct",
    "mean_reversion_z_20d",
    "rsi_14",
    "atr_pct_14",
    "momentum_63d_pct",
    "volume_ratio_20d",
    "value_score",
    "fundamental_quality_score",
    "size_score",
    "liquidity_score",
)


def load_features_from_legacy_sqlite(
    sqlite_path: Path,
    *,
    horizon_days: int = 10,
    limit: int | None = None,
    exclude_idea_ids: Sequence[str] = (),
) -> list[FeatureVector]:
    """Load backtestable features from the imported legacy SQLite ledger."""
    return load_training_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=(horizon_days,),
        limit=limit,
        exclude_idea_ids=exclude_idea_ids,
    )


def load_training_features_from_legacy_sqlite(
    sqlite_path: Path,
    *,
    horizon_days: Sequence[int] = SUPPORTED_FORWARD_HORIZONS,
    limit: int | None = None,
    exclude_idea_ids: Sequence[str] = (),
) -> list[FeatureVector]:
    """Load the leakage-controlled training universe for one or more forward horizons."""
    return _load_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=horizon_days,
        limit=limit,
        exclude_idea_ids=exclude_idea_ids,
        include_idea_ids=None,
    )


def load_features_for_idea_ids_from_legacy_sqlite(
    sqlite_path: Path,
    *,
    idea_ids: Sequence[str],
    horizon_days: int = 10,
) -> list[FeatureVector]:
    """Load held-out anchor features for scoring only, not training."""
    return _load_features_from_legacy_sqlite(
        sqlite_path,
        horizon_days=(horizon_days,),
        limit=None,
        exclude_idea_ids=(),
        include_idea_ids=idea_ids,
    )


def _load_features_from_legacy_sqlite(
    sqlite_path: Path,
    *,
    horizon_days: Sequence[int],
    limit: int | None,
    exclude_idea_ids: Sequence[str],
    include_idea_ids: Sequence[str] | None,
) -> list[FeatureVector]:
    horizons = tuple(dict.fromkeys(int(horizon) for horizon in horizon_days))
    if not horizons or (limit is not None and limit <= 0):
        return []
    where_clauses = [
        f"r.horizon_days IN ({_placeholders(len(horizons))})",
        "i.ticker IS NOT NULL",
        "r.net_return_pct IS NOT NULL",
    ]
    params: list[object] = list(horizons)
    excluded = tuple(str(item) for item in exclude_idea_ids if str(item))
    if excluded:
        where_clauses.append(f"i.idea_id NOT IN ({_placeholders(len(excluded))})")
        params.extend(excluded)
    included = tuple(str(item) for item in include_idea_ids or () if str(item))
    if include_idea_ids is not None:
        if not included:
            return []
        where_clauses.append(f"i.idea_id IN ({_placeholders(len(included))})")
        params.extend(included)
    base_query = """
        SELECT
            i.idea_id,
            i.ticker,
            COALESCE(i.signal_date, substr(i.generated_at, 1, 10)) AS signal_date,
            i.sentiment,
            i.sentiment_confidence,
            i.catalyst_strength,
            i.quality_status,
            i.risk_status,
            i.signal_json,
            r.net_return_pct,
            r.cost_bps,
            i.eval_stage,
            i.entry_price,
            r.horizon_days,
            r.future_date,
            i.signal_date AS raw_signal_date
        FROM step2_research_ideas i
        JOIN step2_forward_returns r ON r.idea_id = i.idea_id
        WHERE
    """
    base_query = f"{base_query} {' AND '.join(where_clauses)}"
    if limit is not None:
        params.append(limit)
        query = f"""
            SELECT *
            FROM ({base_query} ORDER BY signal_date DESC, horizon_days DESC, i.idea_id DESC LIMIT ?)
            ORDER BY signal_date, horizon_days, idea_id
        """
    else:
        query = f"{base_query} ORDER BY signal_date, horizon_days, i.idea_id"
    conn = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(query, tuple(params)).fetchall()
    finally:
        conn.close()
    return attach_cross_sectional_context(_row_to_feature(row) for row in rows)


def attach_cross_sectional_context(features: Iterable[FeatureVector]) -> list[FeatureVector]:
    """Attach leakage-safe within-date/horizon z-scores and rank labels."""
    grouped: dict[tuple[str, int], list[FeatureVector]] = defaultdict(list)
    for feature in features:
        grouped[(feature.signal_date, feature.horizon_days)].append(feature)
    enriched: list[FeatureVector] = []
    for group in grouped.values():
        zscores = _group_zscores(group)
        labels = _rank_labels(group)
        enriched.extend(
            replace(
                feature,
                cross_sectional_zscores=zscores.get(feature.idea_id, {}),
                rank_label=labels.get(feature.idea_id),
            )
            for feature in group
        )
    return sorted(enriched, key=lambda feature: (feature.signal_date, feature.horizon_days, feature.idea_id))


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _optional_float(value: object) -> float | None:
    return optional_float(value, allow_bool=True)


def _row_to_feature(row: tuple[Any, ...]) -> FeatureVector:
    issues: list[DataQualityIssue] = []
    signal = parse_json_object(row[8], field="signal_json", issues=issues)
    net_return_pct = parse_float(row[9], field="net_return_pct", issues=issues, required=True)
    entry_price = _optional_float(row[12]) if len(row) > 12 else None
    horizon_days = int(row[13]) if len(row) > 13 else 10
    outcome_issue = check_outcome_plausibility(
        close_price=entry_price,
        forward_return_pct=net_return_pct,
        field=f"outcome_{horizon_days}d_net_return_pct",
    )
    if outcome_issue is not None:
        issues.append(outcome_issue)
    ticker = str(row[1]).upper()
    signal_date = str(row[2] or "")
    quality_status = str(row[6] or "unknown").lower()
    risk_status = str(row[7] or "unknown").lower()
    issues.extend(
        check_tradeable_universe(
            ticker=ticker,
            signal_date=signal_date,
            raw_signal_date=str(row[15] or "") if len(row) > 15 else signal_date,
            entry_price=entry_price,
            quality_status=quality_status,
            risk_status=risk_status,
        )
    )
    sentiment = str(row[3] or "unknown").lower()
    sentiment_confidence = parse_float(row[4], field="sentiment_confidence", issues=issues)
    catalyst_strength = parse_float(row[5], field="catalyst_strength", issues=issues)
    earnings = _earnings_signal(signal)
    if earnings is not None:
        sentiment, sentiment_confidence, catalyst_strength = earnings
    return FeatureVector(
        idea_id=str(row[0]),
        ticker=ticker,
        signal_date=signal_date,
        sentiment=sentiment,
        sentiment_confidence=sentiment_confidence,
        catalyst_strength=catalyst_strength,
        quality_status=quality_status,
        risk_status=risk_status,
        trend=str(signal.get("trend") or "unknown").lower(),
        momentum_20d_pct=parse_float(
            signal.get("momentum_20d_pct"),
            field="signal_json.momentum_20d_pct",
            issues=issues,
        ),
        mean_reversion_z_20d=parse_float(
            signal.get("mean_reversion_z_20d"),
            field="signal_json.mean_reversion_z_20d",
            issues=issues,
        ),
        volume_confirmed=str(signal.get("volume_signal") or "").lower()
        in {"high_volume_confirmation", "above_average_breakout_volume"},
        net_return_pct=net_return_pct,
        cost_bps=parse_float(row[10], field="cost_bps", issues=issues),
        source=str(row[11] or "unknown"),
        data_quality_issues=tuple(issues),
        horizon_days=horizon_days,
        outcome_future_date=str(row[14] or "") if len(row) > 14 else "",
        entry_price=entry_price,
        rsi_14=_signal_float(signal, "rsi_14", issues),
        macd_signal=str(signal.get("macd_signal") or "unknown").lower(),
        atr_pct_14=_signal_float(signal, "atr_pct_14", issues),
        momentum_63d_pct=_signal_float(signal, "momentum_63d_pct", issues),
        volume_ratio_20d=_signal_float(signal, "volume_ratio_20d", issues),
    )


def _signal_float(signal: dict[str, Any], key: str, issues: list[DataQualityIssue]) -> float:
    if key not in signal or signal.get(key) in (None, ""):
        return 0.0
    return parse_float(signal.get(key), field=f"signal_json.{key}", issues=issues)


def _earnings_signal(signal: dict[str, Any]) -> tuple[str, float, float] | None:
    """Derive (sentiment, sentiment_confidence, catalyst_strength) from a genuine earnings surprise.

    The rubric treats catalyst as bullish-only, so the surprise *magnitude* becomes ``catalyst_strength``
    (how material the event is) and its *direction* becomes ``sentiment`` (beat -> positive, miss ->
    negative, inline -> neutral). Returns ``None`` when the row carries no earnings surprise. This is a
    point-in-time signal: the source rows are dated on/after the earnings ``report_date``.
    """
    surprise_pct = optional_float(signal.get("surprise_pct"), allow_bool=False)
    if surprise_pct is None:
        return None
    strength = round(tanh(abs(surprise_pct) / EARNINGS_SURPRISE_SCALE), 6)
    surprise_class = str(signal.get("surprise_class") or "").lower()
    if surprise_class == "beat" or (surprise_class not in {"miss", "inline"} and surprise_pct > 0.0):
        return "positive", strength, strength
    if surprise_class == "miss" or (surprise_class not in {"beat", "inline"} and surprise_pct < 0.0):
        return "negative", strength, strength
    return "neutral", 0.0, strength


def _group_zscores(group: Sequence[FeatureVector]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {feature.idea_id: {} for feature in group}
    for field_name in CROSS_SECTIONAL_NUMERIC_FIELDS:
        values = [_feature_number(feature, field_name) for feature in group]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        stdev = variance**0.5
        for feature, value in zip(group, values, strict=True):
            result[feature.idea_id][field_name] = 0.0 if stdev <= 1e-12 else round((value - mean) / stdev, 6)
    return result


def _feature_number(feature: FeatureVector, field_name: str) -> float:
    value = getattr(feature, field_name)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _rank_labels(group: Sequence[FeatureVector]) -> dict[str, DecisionAction]:
    ranked = sorted(group, key=lambda feature: (feature.net_return_pct, feature.idea_id))
    if len(ranked) == 1:
        return {ranked[0].idea_id: "HOLD"}
    labels: dict[str, DecisionAction] = {}
    denominator = max(len(ranked) - 1, 1)
    for index, feature in enumerate(ranked):
        percentile = index / denominator
        labels[feature.idea_id] = _label_for_percentile(percentile)
    return labels


def _label_for_percentile(percentile: float) -> DecisionAction:
    if percentile < 0.20:
        return "SELL"
    if percentile < 0.40:
        return "AVOID"
    if percentile < 0.60:
        return "HOLD"
    if percentile < 0.80:
        return "WATCHLIST"
    return "BUY"
