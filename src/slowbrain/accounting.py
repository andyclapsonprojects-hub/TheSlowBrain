"""FIFO portfolio accounting from broker audit fills."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .numeric import float_or_default


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: str
    quantity: float
    value_gbp: float
    fees_gbp: float


@dataclass(frozen=True)
class AccountingSummary:
    realized_profit_gbp: float
    matched_cost_gbp: float
    realized_profit_pct: float | None
    open_cost_gbp: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MarkToMarketSummary:
    unrealized_profit_gbp: float | None
    open_market_value_gbp: float | None
    quality: str
    warnings: tuple[str, ...]


@dataclass
class _Lot:
    quantity: float
    cost_gbp: float


def load_fills(path: Path) -> tuple[Fill, ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(_row_to_fill(row) for row in rows if _is_filled(row))


def calculate_fifo_pnl(fills: tuple[Fill, ...]) -> AccountingSummary:
    """Calculate realized P&L using FIFO lot matching."""
    lots: dict[str, list[_Lot]] = {}
    warnings: list[str] = []
    realized_profit = 0.0
    matched_cost = 0.0
    for fill in fills:
        if fill.side == "buy":
            lots.setdefault(fill.ticker, []).append(_Lot(fill.quantity, fill.value_gbp + fill.fees_gbp))
            continue
        if fill.side != "sell":
            warnings.append(f"unknown_side:{fill.ticker}:{fill.side}")
            continue
        remaining = fill.quantity
        proceeds_per_unit = fill.value_gbp / fill.quantity if fill.quantity else 0.0
        while remaining > 0 and lots.get(fill.ticker):
            lot = lots[fill.ticker][0]
            matched_quantity = min(remaining, lot.quantity)
            cost = lot.cost_gbp * (matched_quantity / lot.quantity)
            proceeds = proceeds_per_unit * matched_quantity
            realized_profit += proceeds - cost - (fill.fees_gbp * (matched_quantity / fill.quantity))
            matched_cost += cost
            lot.quantity -= matched_quantity
            lot.cost_gbp -= cost
            remaining -= matched_quantity
            if lot.quantity <= 1e-9:
                lots[fill.ticker].pop(0)
        if remaining > 1e-9:
            warnings.append(f"unmatched_sell:{fill.ticker}:{remaining:.6f}")
    open_cost = sum(lot.cost_gbp for ticker_lots in lots.values() for lot in ticker_lots)
    realized_pct = (realized_profit / matched_cost) * 100.0 if matched_cost > 0 else None
    return AccountingSummary(
        realized_profit_gbp=round(realized_profit, 4),
        matched_cost_gbp=round(matched_cost, 4),
        realized_profit_pct=round(realized_pct, 4) if realized_pct is not None else None,
        open_cost_gbp=round(open_cost, 4),
        warnings=tuple(warnings),
    )


def calculate_fifo_pnl_from_csv(path: Path) -> AccountingSummary:
    return calculate_fifo_pnl(load_fills(path))


def calculate_mark_to_market_from_positions(path: Path) -> MarkToMarketSummary:
    if not path.exists():
        return MarkToMarketSummary(None, None, "not_available", ("positions_file_missing",))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unrealized = 0.0
    market_value = 0.0
    priced_count = 0
    warnings: list[str] = []
    for row in rows:
        status = str(row.get("status") or row.get("position_status") or "").lower()
        if "open" not in status:
            continue
        ticker = str(row.get("ticker") or "").upper()
        entry_price = _float(row.get("entry_price"))
        quantity = abs(_float(row.get("quantity")))
        current_price = _current_price(row)
        if entry_price <= 0 or quantity <= 0:
            warnings.append(f"invalid_open_position:{ticker or 'UNKNOWN'}")
            continue
        if current_price is None:
            warnings.append(f"missing_market_price:{ticker or 'UNKNOWN'}")
            continue
        priced_count += 1
        market_value += current_price * quantity
        unrealized += (current_price - entry_price) * quantity
    if priced_count == 0:
        return MarkToMarketSummary(None, None, "missing_market_prices", tuple(warnings))
    return MarkToMarketSummary(
        round(unrealized, 4),
        round(market_value, 4),
        "marked_to_market_from_position_prices",
        tuple(warnings),
    )


def load_open_position_market_values(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[str, float] = {}
    for row in rows:
        status = str(row.get("status") or row.get("position_status") or "").lower()
        if "open" not in status:
            continue
        ticker = str(row.get("ticker") or "").upper()
        quantity = abs(_float(row.get("quantity")))
        price = _current_price(row) or _float(row.get("entry_price"))
        if ticker and quantity > 0.0 and price > 0.0:
            values[ticker] = round(values.get(ticker, 0.0) + (quantity * price), 4)
    return values


def _row_to_fill(row: dict[str, str]) -> Fill:
    quantity = abs(_float(row.get("filled_quantity")))
    side = str(row.get("side") or "").lower()
    currency = str(row.get("currency") or "GBP").upper()
    fx_rate = 1.0 if currency == "GBP" else (_float(row.get("fx_rate"), default=1.0) or 1.0)
    value_gbp = abs(_float(row.get("net_value"))) * fx_rate
    fees_gbp = _taxes_gbp(row.get("taxes"), fx_rate=fx_rate)
    return Fill(
        ticker=str(row.get("ticker") or "").upper(),
        side=side,
        quantity=quantity,
        value_gbp=value_gbp,
        fees_gbp=fees_gbp,
    )


def _is_filled(row: dict[str, str]) -> bool:
    status = str(row.get("status") or "").upper()
    raw_status = str(row.get("raw_order_status") or "").upper()
    return status == "FILLED" or raw_status == "FILLED"


def _taxes_gbp(raw: str | None, *, fx_rate: float) -> float:
    if not raw:
        return 0.0
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(value, list):
        return 0.0
    total = 0.0
    for item in value:
        if isinstance(item, dict):
            total += abs(_float(item.get("amount") or item.get("quantity"))) * fx_rate
    return total


def _current_price(row: dict[str, str]) -> float | None:
    for field in ("current_price", "last_price", "market_price", "mark_price"):
        value = _float(row.get(field), default=-1.0)
        if value > 0:
            return value
    return None


def _float(value: object, *, default: float = 0.0) -> float:
    return float_or_default(value, default=default, allow_bool=True)
