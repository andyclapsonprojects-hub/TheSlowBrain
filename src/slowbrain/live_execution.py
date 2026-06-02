"""Approval-gated Trading 212 live execution readiness."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from .config import AppConfig
from .live_execution_support import (
    active_order_tickers,
    append_ledger,
    approval_token,
    broker_order_id,
    cash_available,
    has_duplicate_ledger_entry,
    intent_id,
    ledger_row,
    mapping,
    order_value,
    position_map,
    preview_expired,
    required_text,
    resolve_instrument,
    response_status,
    safe_decision,
    sequence,
    text,
    utc,
)
from .numeric import float_or_default, optional_float
from .trading212 import Trading212Gateway, credentials_available, response_summary

LIVE_EXECUTION_DIR = Path("reports/live-execution")
BROKER_HEALTH_JSON = LIVE_EXECUTION_DIR / "broker-health.json"
LATEST_PREVIEW_JSON = LIVE_EXECUTION_DIR / "latest-preview.json"
LATEST_SUBMISSION_JSON = LIVE_EXECUTION_DIR / "latest-submission.json"
EXECUTION_LEDGER_JSONL = LIVE_EXECUTION_DIR / "execution-ledger.jsonl"

BUY_NOTIONAL_GBP = 10.0
PREVIEW_TTL_MINUTES = 30
READY_STATUSES = {"ready"}

type PriceLookup = Callable[[str, str], "PriceSnapshot | None"]


@dataclass(frozen=True)
class PriceSnapshot:
    price_gbp: float
    source: str
    as_of: str


def build_broker_health_report(
    *,
    config: AppConfig,
    client: Trading212Gateway | None,
    now: datetime | None = None,
) -> dict[str, object]:
    recorded_at = utc(now)
    if not credentials_available(config) or client is None:
        return {
            "schema": "theslowbrain.trading212_broker_health.v1",
            "recorded_at": recorded_at.isoformat(),
            "status": "blocked",
            "reason": "missing_trading212_credentials",
            "environment": config.trading212_env,
            "orders_submitted": False,
            "broker_live_execution_allowed": False,
        }
    account = client.account_summary()
    positions = client.positions()
    orders = client.active_orders()
    ok = account.status_code == 200 and positions.status_code == 200 and orders.status_code == 200
    account_payload = mapping(account.payload)
    return {
        "schema": "theslowbrain.trading212_broker_health.v1",
        "recorded_at": recorded_at.isoformat(),
        "status": "ok" if ok else "failed",
        "environment": config.trading212_env,
        "orders_submitted": False,
        "broker_live_execution_allowed": False,
        "account_currency": text(account_payload.get("currency")),
        "cash_available_to_trade": cash_available(account_payload),
        "positions_count": len(sequence(positions.payload)),
        "active_orders_count": len(sequence(orders.payload)),
        "responses": {
            "account_summary": response_summary(account),
            "positions": response_summary(positions),
            "active_orders": response_summary(orders),
        },
    }


def build_execution_preview(
    *,
    report_payload: Mapping[str, object],
    config: AppConfig,
    instruments: Sequence[Mapping[str, object]],
    positions: Sequence[Mapping[str, object]],
    active_orders: Sequence[Mapping[str, object]] = (),
    price_lookup: PriceLookup | None = None,
    now: datetime | None = None,
    buy_notional_gbp: float = BUY_NOTIONAL_GBP,
) -> dict[str, object]:
    created_at = utc(now)
    preview_id = f"live-preview-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    decisions = sequence(report_payload.get("trade_decisions"))
    current_positions = position_map(positions)
    current_active_order_tickers = active_order_tickers(active_orders)
    orders = [
        _decision_preview(
            preview_id=preview_id,
            decision=mapping(decision),
            instruments=instruments,
            position_map=current_positions,
            active_order_tickers=current_active_order_tickers,
            price_lookup=price_lookup,
            buy_notional_gbp=buy_notional_gbp,
            usd_gbp_rate=config.market_data_usd_gbp_rate,
        )
        for decision in decisions
    ]
    actionable = [order for order in orders if order["status"] in READY_STATUSES]
    preview_without_token = {
        "schema": "theslowbrain.live_execution_preview.v1",
        "preview_id": preview_id,
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(minutes=PREVIEW_TTL_MINUTES)).isoformat(),
        "status": "ready" if actionable else "blocked",
        "environment": config.trading212_env,
        "execute_command": "uv run python scripts/submit_live_orders.py --execute --approval-token <token>",
        "broker_live_execution_allowed": False,
        "target_buy_notional_gbp": buy_notional_gbp,
        "ready_order_count": len(actionable),
        "blocked_order_count": sum(1 for order in orders if order["status"] == "blocked"),
        "orders": orders,
    }
    token = approval_token(preview_without_token) if actionable else None
    return {**preview_without_token, "approval_token": token}


def submit_execution_preview(
    *,
    preview: Mapping[str, object],
    config: AppConfig,
    client: Trading212Gateway | None,
    ledger_path: Path,
    execute: bool,
    approval_token: str | None,
    now: datetime | None = None,
) -> dict[str, object]:
    submitted_at = utc(now)
    ready_orders = [
        mapping(order)
        for order in sequence(preview.get("orders"))
        if text(mapping(order).get("status")) == "ready"
    ]
    base_record: dict[str, object] = {
        "schema": "theslowbrain.live_execution_submission.v1",
        "submitted_at": submitted_at.isoformat(),
        "preview_id": text(preview.get("preview_id")),
        "environment": config.trading212_env,
        "execute_requested": execute,
        "broker_live_execution_allowed": bool(config.trading_live_enabled and execute),
        "orders_attempted": 0,
        "orders_submitted": False,
        "results": [],
    }
    gate_reason = _submit_gate_reason(
        preview=preview,
        config=config,
        client=client,
        execute=execute,
        approval_token=approval_token,
        ready_orders=ready_orders,
        ledger_path=ledger_path,
        now=submitted_at,
    )
    if gate_reason is not None:
        return {**base_record, "status": "blocked", "reason": gate_reason}
    if client is None:
        return {**base_record, "status": "blocked", "reason": "missing_trading212_client"}

    preflight_positions = client.positions()
    active_orders = client.active_orders()
    current_positions = position_map(sequence(preflight_positions.payload))
    current_active_order_tickers = active_order_tickers(sequence(active_orders.payload))
    stale_reason = _fresh_broker_state_reason(ready_orders, current_positions, current_active_order_tickers)
    if stale_reason is not None:
        return {**base_record, "status": "blocked", "reason": stale_reason}

    results: list[dict[str, object]] = []
    for order in ready_orders:
        order_intent_id = required_text(order.get("intent_id"), "intent_id")
        append_ledger(ledger_path, ledger_row(order, status="intended", broker_response=None, now=submitted_at))
        payload = mapping(order.get("order_payload"))
        response = client.place_market_order(
            ticker=required_text(payload.get("ticker"), "order_payload.ticker"),
            quantity=float_or_default(payload.get("quantity")),
        )
        status = response_status(response)
        append_ledger(ledger_path, ledger_row(order, status=status, broker_response=response, now=submitted_at))
        results.append(
            {
                "intent_id": order_intent_id,
                "status": status,
                "broker_status_code": response.status_code,
                "broker_order_id": broker_order_id(response.payload),
                "ticker": payload.get("ticker"),
                "quantity": payload.get("quantity"),
            }
        )

    post_positions = client.positions()
    post_orders = client.active_orders()
    post_history = client.history_orders(limit=50)
    return {
        **base_record,
        "status": "submitted",
        "orders_attempted": len(ready_orders),
        "orders_submitted": bool(results),
        "results": results,
        "reconciliation": {
            "positions_status_code": post_positions.status_code,
            "active_orders_status_code": post_orders.status_code,
            "history_orders_status_code": post_history.status_code,
            "positions_count": len(sequence(post_positions.payload)),
            "active_orders_count": len(sequence(post_orders.payload)),
            "history_orders_count": len(sequence(post_history.payload)),
        },
    }


def write_json(path: Path, payload: Mapping[str, object], *, atomic: bool = True) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        path.write_text(f"{text}\n", encoding="utf-8")
        return path
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(f"{text}\n", encoding="utf-8")
    temp_path.replace(path)
    return path


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _decision_preview(
    *,
    preview_id: str,
    decision: Mapping[str, object],
    instruments: Sequence[Mapping[str, object]],
    position_map: Mapping[str, Mapping[str, object]],
    active_order_tickers: set[str],
    price_lookup: PriceLookup | None,
    buy_notional_gbp: float,
    usd_gbp_rate: float | None,
) -> dict[str, object]:
    action = text(decision.get("action"))
    ticker = required_text(decision.get("ticker"), "decision.ticker").upper()
    if action not in {"BUY", "SELL"}:
        return _blocked_order(preview_id, decision, ticker, action, "decision_not_buy_or_sell")
    instrument, reason = resolve_instrument(ticker, instruments)
    if instrument is None:
        return _blocked_order(preview_id, decision, ticker, action, reason)
    broker_ticker = required_text(instrument.get("ticker"), "instrument.ticker")
    if broker_ticker in active_order_tickers:
        return _blocked_order(preview_id, decision, ticker, action, "active_order_already_exists_for_ticker")
    if action == "SELL":
        return _sell_preview(preview_id, decision, ticker, instrument, position_map)
    return _buy_preview(
        preview_id,
        decision,
        ticker,
        instrument,
        position_map,
        price_lookup,
        buy_notional_gbp,
        usd_gbp_rate,
    )


def _buy_preview(
    preview_id: str,
    decision: Mapping[str, object],
    ticker: str,
    instrument: Mapping[str, object],
    position_map: Mapping[str, Mapping[str, object]],
    price_lookup: PriceLookup | None,
    buy_notional_gbp: float,
    usd_gbp_rate: float | None,
) -> dict[str, object]:
    broker_ticker = required_text(instrument.get("ticker"), "instrument.ticker")
    if broker_ticker in position_map:
        return _blocked_order(preview_id, decision, ticker, "BUY", "ticker_already_held")
    if price_lookup is None:
        return _blocked_order(preview_id, decision, ticker, "BUY", "missing_price_lookup_for_buy_quantity")
    price = price_lookup(ticker, text(instrument.get("currencyCode")) or "GBP")
    if price is None or price.price_gbp <= 0:
        return _blocked_order(preview_id, decision, ticker, "BUY", "missing_fresh_price_for_buy_quantity")
    quantity = round(buy_notional_gbp / price.price_gbp, 6)
    if quantity <= 0:
        return _blocked_order(preview_id, decision, ticker, "BUY", "calculated_buy_quantity_not_positive")
    return _ready_order(
        preview_id=preview_id,
        decision=decision,
        ticker=ticker,
        broker_ticker=broker_ticker,
        side="BUY",
        quantity=quantity,
        target_notional_gbp=buy_notional_gbp,
        estimated_notional_gbp=round(quantity * price.price_gbp, 4),
        reason="gbp10_buy_market_order_ready",
        price_source=price.source,
        price_as_of=price.as_of,
        usd_gbp_rate=usd_gbp_rate,
    )


def _sell_preview(
    preview_id: str,
    decision: Mapping[str, object],
    ticker: str,
    instrument: Mapping[str, object],
    position_map: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    broker_ticker = required_text(instrument.get("ticker"), "instrument.ticker")
    position = position_map.get(broker_ticker)
    if position is None:
        return _blocked_order(preview_id, decision, ticker, "SELL", "sell_decision_not_currently_held")
    available = float_or_default(position.get("quantityAvailableForTrading"))
    if available <= 0:
        return _blocked_order(preview_id, decision, ticker, "SELL", "no_quantity_available_for_trading")
    current_price = optional_float(position.get("currentPrice"))
    estimated = round(abs(available) * current_price, 4) if current_price is not None else None
    return _ready_order(
        preview_id=preview_id,
        decision=decision,
        ticker=ticker,
        broker_ticker=broker_ticker,
        side="SELL",
        quantity=round(-abs(available), 6),
        target_notional_gbp=None,
        estimated_notional_gbp=estimated,
        reason="held_position_sell_market_order_ready",
        price_source="trading212_position_current_price",
        price_as_of=text(position.get("createdAt")),
        usd_gbp_rate=None,
    )


def _ready_order(
    *,
    preview_id: str,
    decision: Mapping[str, object],
    ticker: str,
    broker_ticker: str,
    side: Literal["BUY", "SELL"],
    quantity: float,
    target_notional_gbp: float | None,
    estimated_notional_gbp: float | None,
    reason: str,
    price_source: str | None,
    price_as_of: str | None,
    usd_gbp_rate: float | None,
) -> dict[str, object]:
    order_intent_id = intent_id(preview_id, broker_ticker, side, quantity)
    return {
        "preview_id": preview_id,
        "intent_id": order_intent_id,
        "source_decision": safe_decision(decision),
        "ticker": ticker,
        "broker_ticker": broker_ticker,
        "side": side,
        "quantity": quantity,
        "target_notional_gbp": target_notional_gbp,
        "estimated_notional_gbp": estimated_notional_gbp,
        "status": "ready",
        "reason": reason,
        "price_source": price_source,
        "price_as_of": price_as_of,
        "usd_gbp_rate": usd_gbp_rate,
        "order_type": "market",
        "order_payload": {"ticker": broker_ticker, "quantity": quantity},
    }


def _blocked_order(
    preview_id: str,
    decision: Mapping[str, object],
    ticker: str,
    action: str | None,
    reason: str,
) -> dict[str, object]:
    return {
        "preview_id": preview_id,
        "intent_id": intent_id(preview_id, ticker, action or "NONE", 0.0),
        "source_decision": safe_decision(decision),
        "ticker": ticker,
        "broker_ticker": None,
        "side": action,
        "quantity": 0.0,
        "target_notional_gbp": None,
        "estimated_notional_gbp": None,
        "status": "blocked",
        "reason": reason,
        "order_type": "market",
        "order_payload": None,
    }


def _submit_gate_reason(
    *,
    preview: Mapping[str, object],
    config: AppConfig,
    client: Trading212Gateway | None,
    execute: bool,
    approval_token: str | None,
    ready_orders: Sequence[Mapping[str, object]],
    ledger_path: Path,
    now: datetime,
) -> str | None:
    if not ready_orders:
        return "preview_has_no_ready_orders"
    if not execute:
        return "execute_flag_not_set"
    if not config.trading_live_enabled:
        return "trading_live_enabled_false"
    if client is None or not credentials_available(config):
        return "missing_trading212_credentials"
    if config.trading_require_manual_approval and approval_token != text(preview.get("approval_token")):
        return "approval_token_mismatch"
    if preview_expired(preview, now):
        return "preview_expired"
    if config.trading_max_daily_orders is not None and len(ready_orders) > config.trading_max_daily_orders:
        return "max_daily_orders_exceeded"
    max_order_value = config.trading_max_order_value
    if max_order_value is not None and any(order_value(order) > max_order_value for order in ready_orders):
        return "max_order_value_exceeded"
    if has_duplicate_ledger_entry(ledger_path, ready_orders):
        return "duplicate_execution_intent_blocked"
    return None


def _fresh_broker_state_reason(
    ready_orders: Sequence[Mapping[str, object]],
    position_map: Mapping[str, Mapping[str, object]],
    active_order_tickers: set[str],
) -> str | None:
    for order in ready_orders:
        broker_ticker = required_text(order.get("broker_ticker"), "broker_ticker")
        if broker_ticker in active_order_tickers:
            return "active_order_detected_before_submit"
        if text(order.get("side")) == "SELL":
            position = position_map.get(broker_ticker)
            available = float_or_default(position.get("quantityAvailableForTrading")) if position else 0.0
            if available < abs(float_or_default(order.get("quantity"))):
                return "sell_quantity_no_longer_available"
    return None

