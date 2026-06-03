"""Data-quality records and strict boundary coercion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from .numeric import optional_float

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class DataQualityIssue:
    field: str
    code: str
    severity: Severity
    message: str


def parse_json_object(raw: object, *, field: str, issues: list[DataQualityIssue]) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        issues.append(DataQualityIssue(field, "missing_json", "warning", f"{field} is missing."))
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(DataQualityIssue(field, "malformed_json", "error", f"{field} is not valid JSON: {exc.msg}."))
        return {}
    if not isinstance(value, dict):
        issues.append(DataQualityIssue(field, "json_not_object", "error", f"{field} is not a JSON object."))
        return {}
    return value


def parse_float(
    value: object,
    *,
    field: str,
    issues: list[DataQualityIssue],
    default: float = 0.0,
    required: bool = False,
) -> float:
    parsed = optional_float(value, allow_bool=True)
    if parsed is not None:
        return parsed
    if isinstance(value, str) and value.strip():
        issues.append(DataQualityIssue(field, "invalid_float", "error", f"{field} is not numeric."))
        return default
    severity: Severity = "error" if required else "warning"
    issues.append(DataQualityIssue(field, "missing_float", severity, f"{field} is missing."))
    return default


def has_error(issues: tuple[DataQualityIssue, ...]) -> bool:
    return any(issue.severity == "error" for issue in issues)


# Defaults for flagging phantom-alpha outcome artefacts: a very cheap stock posting an
# enormous forward return is almost always a split/adjustment or bad-print artefact in the
# legacy price cache (e.g. OTLK close 0.233 with +212% 10d, PSNL +69%), not real edge.
PENNY_PRICE_THRESHOLD = 1.0
IMPLAUSIBLE_RETURN_PCT = 50.0
MIN_TRADEABLE_ENTRY_PRICE = 5.0
MAX_PLAUSIBLE_ENTRY_PRICE = 100_000.0
BLOCKED_RISK_STATUSES = frozenset({"rejected", "fail"})


def check_outcome_plausibility(
    *,
    close_price: float | None,
    forward_return_pct: float | None,
    field: str = "outcome_10d_net_return_pct",
    penny_price_threshold: float = PENNY_PRICE_THRESHOLD,
    implausible_return_pct: float = IMPLAUSIBLE_RETURN_PCT,
) -> DataQualityIssue | None:
    """Flag a sub-threshold-price name with an implausibly large forward return.

    Returns an ``error``-severity issue when the price is below ``penny_price_threshold``
    and the absolute forward return exceeds ``implausible_return_pct``; otherwise ``None``.
    Pure and side-effect free so callers decide whether to exclude the row.
    """
    if close_price is None or forward_return_pct is None:
        return None
    if close_price < penny_price_threshold and abs(forward_return_pct) > implausible_return_pct:
        return DataQualityIssue(
            field,
            "implausible_outcome_for_penny_stock",
            "error",
            (
                f"{field} {forward_return_pct:.2f}% on a sub-{penny_price_threshold:g} price "
                f"({close_price:g}) is implausible; likely a split/adjustment artefact."
            ),
        )
    return None


def check_tradeable_universe(
    *,
    ticker: str,
    signal_date: str,
    raw_signal_date: str | None = None,
    entry_price: float | None,
    quality_status: str,
    risk_status: str,
) -> tuple[DataQualityIssue, ...]:
    """Return error issues for rows that should not drive tradeable-universe evidence."""
    issues: list[DataQualityIssue] = []
    normalized_ticker = ticker.strip().upper()
    normalized_quality = quality_status.strip().lower()
    normalized_risk = risk_status.strip().lower()
    if not normalized_ticker or normalized_ticker == "ACME":
        issues.append(DataQualityIssue("ticker", "placeholder_or_missing_ticker", "error", "Ticker is not tradeable."))
    if raw_signal_date is not None and not raw_signal_date.strip():
        issues.append(
            DataQualityIssue(
                "signal_date",
                "missing_original_signal_date",
                "error",
                "Original signal date is missing.",
            )
        )
    if not _valid_iso_date(signal_date):
        issues.append(
            DataQualityIssue("signal_date", "invalid_signal_date", "error", "Signal date is missing or invalid.")
        )
    if entry_price is None:
        issues.append(DataQualityIssue("entry_price", "missing_entry_price", "error", "Entry price is missing."))
    elif entry_price < MIN_TRADEABLE_ENTRY_PRICE:
        issues.append(
            DataQualityIssue(
                "entry_price",
                "entry_price_below_tradeable_floor",
                "error",
                f"Entry price {entry_price:g} is below the ${MIN_TRADEABLE_ENTRY_PRICE:g} tradeable floor.",
            )
        )
    elif entry_price > MAX_PLAUSIBLE_ENTRY_PRICE:
        issues.append(
            DataQualityIssue(
                "entry_price",
                "entry_price_implausibly_high",
                "error",
                f"Entry price {entry_price:g} is above the ${MAX_PLAUSIBLE_ENTRY_PRICE:g} sanity ceiling.",
            )
        )
    if normalized_quality != "pass":
        issues.append(
            DataQualityIssue(
                "quality_status",
                "quality_status_not_pass",
                "error",
                f"Quality status {normalized_quality or 'missing'} is not pass.",
            )
        )
    if normalized_risk in BLOCKED_RISK_STATUSES:
        issues.append(
            DataQualityIssue(
                "risk_status",
                "risk_status_blocked",
                "error",
                f"Risk status {normalized_risk} blocks tradeable-universe evidence.",
            )
        )
    return tuple(issues)


def _valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True
