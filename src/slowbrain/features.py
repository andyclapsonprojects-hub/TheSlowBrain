"""Feature loading and normalization."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .data_quality import DataQualityIssue, check_outcome_plausibility, parse_float, parse_json_object
from .models import FeatureVector
from .numeric import optional_float

SUPPORTED_FORWARD_HORIZONS = (1, 5, 10, 20)


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
            r.future_date
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
    return [_row_to_feature(row) for row in rows]


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
    return FeatureVector(
        idea_id=str(row[0]),
        ticker=str(row[1]).upper(),
        signal_date=str(row[2]),
        sentiment=str(row[3] or "unknown").lower(),
        sentiment_confidence=parse_float(row[4], field="sentiment_confidence", issues=issues),
        catalyst_strength=parse_float(row[5], field="catalyst_strength", issues=issues),
        quality_status=str(row[6] or "unknown").lower(),
        risk_status=str(row[7] or "unknown").lower(),
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
    )
