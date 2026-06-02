"""Shared helpers for approval-gated live execution."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from .numeric import optional_float
from .trading212 import Trading212Response

BLOCKING_LEDGER_STATUSES = {"intended", "sent", "accepted", "reconciled"}
VALID_DECISION_ACTIONS = {"BUY", "SELL", "HOLD", "AVOID", "WATCHLIST"}


def has_duplicate_ledger_entry(ledger_path: Path, ready_orders: Sequence[Mapping[str, object]]) -> bool:
    if not ledger_path.exists():
        return False
    ready_keys = {(text(order.get("preview_id")), text(order.get("intent_id"))) for order in ready_orders}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = mapping(json.loads(line))
        except json.JSONDecodeError:
            continue
        key = (text(row.get("preview_id")), text(row.get("intent_id")))
        if key in ready_keys and text(row.get("status")) in BLOCKING_LEDGER_STATUSES:
            return True
    return False


def resolve_instrument(
    ticker: str,
    instruments: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object] | None, str]:
    normalized = ticker.upper()
    matches = [
        instrument
        for instrument in instruments
        if _instrument_matches(normalized, instrument)
        and (text(instrument.get("type")) or "").upper() in {"STOCK", "ETF"}
    ]
    if not matches:
        return None, "instrument_not_found"
    exact = [instrument for instrument in matches if (text(instrument.get("ticker")) or "").upper() == normalized]
    preferred = exact or [
        instrument for instrument in matches if (text(instrument.get("ticker")) or "").upper().endswith("_US_EQ")
    ]
    candidates = preferred or matches
    if len(candidates) > 1:
        return None, "ambiguous_instrument_match"
    return candidates[0], "ok"


def position_map(positions: Sequence[object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw_position in positions:
        position = mapping(raw_position)
        instrument = mapping(position.get("instrument"))
        ticker = text(instrument.get("ticker")) or text(position.get("ticker"))
        if ticker:
            result[ticker.upper()] = position
    return result


def active_order_tickers(active_orders: Sequence[object]) -> set[str]:
    tickers: set[str] = set()
    for raw_order in active_orders:
        order = mapping(raw_order)
        ticker = text(order.get("ticker"))
        instrument = mapping(order.get("instrument"))
        ticker = ticker or text(instrument.get("ticker"))
        if ticker:
            tickers.add(ticker.upper())
    return tickers


def approval_token(preview: Mapping[str, object]) -> str:
    orders = [
        {
            "intent_id": order.get("intent_id"),
            "order_payload": order.get("order_payload"),
            "side": order.get("side"),
        }
        for order in (mapping(item) for item in sequence(preview.get("orders")))
        if order.get("status") == "ready"
    ]
    raw = json.dumps(
        {
            "preview_id": preview.get("preview_id"),
            "created_at": preview.get("created_at"),
            "orders": orders,
        },
        sort_keys=True,
    )
    return sha256(raw.encode("utf-8")).hexdigest()[:20]


def preview_expired(preview: Mapping[str, object], now: datetime) -> bool:
    expires_at = parse_dt(text(preview.get("expires_at")))
    return expires_at is None or now > expires_at


def append_ledger(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = path.exists() and path.stat().st_size > 0 and not path.read_text(encoding="utf-8").endswith("\n")
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def ledger_row(
    order: Mapping[str, object],
    *,
    status: str,
    broker_response: Trading212Response | None,
    now: datetime,
) -> dict[str, object]:
    return {
        "schema": "theslowbrain.live_execution_ledger.v1",
        "recorded_at": now.isoformat(),
        "preview_id": text(order.get("preview_id")),
        "intent_id": order.get("intent_id"),
        "broker_ticker": order.get("broker_ticker"),
        "side": order.get("side"),
        "quantity": order.get("quantity"),
        "status": status,
        "broker_status_code": broker_response.status_code if broker_response is not None else None,
        "broker_order_id": broker_order_id(broker_response.payload) if broker_response is not None else None,
    }


def response_status(response: Trading212Response) -> str:
    if 200 <= response.status_code < 300:
        return "accepted"
    if response.status_code in {401, 403, 408, 429} or response.status_code >= 500:
        return "rejected"
    return "failed"


def broker_order_id(payload: object) -> object:
    mapped = mapping(payload)
    return mapped.get("id") or mapped.get("orderId")


def safe_decision(decision: Mapping[str, object]) -> dict[str, object]:
    action = text(decision.get("action"))
    return {
        "ticker": text(decision.get("ticker")),
        "action": action,
        "score": optional_float(decision.get("score")),
        "rubric_version": text(decision.get("rubric_version")),
        "reason": text(decision.get("reason")),
    }


def order_value(order: Mapping[str, object]) -> float:
    value = optional_float(order.get("estimated_notional_gbp"))
    return value if value is not None else 0.0


def cash_available(payload: Mapping[str, object]) -> float | None:
    cash = mapping(payload.get("cash"))
    value = optional_float(cash.get("availableToTrade"))
    if value is not None:
        return value
    return optional_float(payload.get("freeFunds"))


def intent_id(preview_id: str, broker_ticker: str, side: str, quantity: float) -> str:
    raw = f"{preview_id}:{broker_ticker}:{side}:{quantity:.6f}"
    return sha256(raw.encode("utf-8")).hexdigest()[:24]


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC)


def utc(value: datetime | None) -> datetime:
    return (value or datetime.now(UTC)).astimezone(UTC)


def sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def mapping(value: object) -> dict[str, object]:
    return {str(key): item for key, item in value.items()} if isinstance(value, Mapping) else {}


def text(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def required_text(value: object, field: str) -> str:
    rendered = text(value)
    if rendered is None:
        raise ValueError(f"Missing required field: {field}")
    return rendered


def _instrument_matches(ticker: str, instrument: Mapping[str, object]) -> bool:
    broker_ticker = (text(instrument.get("ticker")) or "").upper()
    short_name = (text(instrument.get("shortName")) or "").upper()
    return broker_ticker == ticker or short_name == ticker or broker_ticker.startswith(f"{ticker}_")
