"""Vendor payload parsers and tolerant JSON coercion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from ..market_data import DailyPrice
from ..market_data_cache import JsonValue
from .types import FinnhubQuote, _is_json_value


def _parse_alpha_daily_payload(symbol: str, payload: JsonValue | None) -> tuple[DailyPrice, ...]:
    data = _mapping(payload)
    if data is None or _alpha_payload_is_miss(data):
        return ()
    series = _mapping(data.get("Time Series (Daily)"))
    if series is None:
        return ()
    prices = [
        price
        for raw_date, raw_fields in series.items()
        if isinstance(raw_date, str)
        for price in [_alpha_daily_price(symbol, raw_date, raw_fields)]
        if price is not None
    ]
    return tuple(sorted(prices, key=lambda price: price.date))


def _alpha_daily_price(symbol: str, date: str, raw_fields: object) -> DailyPrice | None:
    fields = _mapping(raw_fields)
    if fields is None:
        return None
    values = (
        _float_field(fields, "1. open"),
        _float_field(fields, "2. high"),
        _float_field(fields, "3. low"),
        _float_field(fields, "4. close"),
        _float_field(fields, "5. adjusted close"),
        _float_field(fields, "6. volume"),
    )
    if any(value is None for value in values):
        return None
    open_, high, low, close, adjusted_close, volume = cast("tuple[float, float, float, float, float, float]", values)
    return DailyPrice(symbol, date, open_, high, low, close, adjusted_close, volume, "alpha_vantage:daily_adjusted")


def _parse_finnhub_candle_payload(symbol: str, payload: JsonValue | None) -> tuple[DailyPrice, ...]:
    data = _mapping(payload)
    if data is None or _finnhub_payload_is_miss(data):
        return ()
    closes = _float_sequence(data.get("c"))
    highs = _float_sequence(data.get("h"))
    lows = _float_sequence(data.get("l"))
    opens = _float_sequence(data.get("o"))
    timestamps = _int_sequence(data.get("t"))
    volumes = _float_sequence(data.get("v"))
    if any(value is None for value in (closes, highs, lows, opens, timestamps, volumes)):
        return ()
    rows = zip(
        cast("list[int]", timestamps),
        cast("list[float]", opens),
        cast("list[float]", highs),
        cast("list[float]", lows),
        cast("list[float]", closes),
        cast("list[float]", volumes),
        strict=False,
    )
    prices = [
        DailyPrice(
            symbol,
            datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat(),
            open_,
            high,
            low,
            close,
            close,
            volume,
            "finnhub:stock_candle",
        )
        for timestamp, open_, high, low, close, volume in rows
        if close > 0.0 and volume >= 0.0
    ]
    return tuple(sorted(prices, key=lambda price: price.date))


def _parse_finnhub_quote_payload(symbol: str, payload: JsonValue | None) -> FinnhubQuote | None:
    data = _mapping(payload)
    if data is None or _finnhub_quote_payload_is_miss(data):
        return None
    values = (
        _float_field(data, "c"),
        _float_field(data, "d"),
        _float_field(data, "dp"),
        _float_field(data, "h"),
        _float_field(data, "l"),
        _float_field(data, "o"),
        _float_field(data, "pc"),
        _int_field(data, "t"),
    )
    if any(value is None for value in values):
        return None
    current, change, change_pct, high, low, open_, previous_close, timestamp = cast(
        "tuple[float, float, float, float, float, float, float, int]",
        values,
    )
    return FinnhubQuote(
        symbol,
        current,
        change,
        change_pct,
        high,
        low,
        open_,
        previous_close,
        timestamp,
        "finnhub:quote",
    )


def _parse_yahoo_chart_payload(symbol: str, payload: JsonValue | None) -> tuple[DailyPrice, ...]:
    data = _mapping(payload)
    if data is None or _yahoo_payload_is_miss(data):
        return ()
    chart = _mapping(data.get("chart"))
    if chart is None or chart.get("error") is not None:
        return ()
    result = _first_mapping(chart.get("result"))
    if result is None:
        return ()
    timestamps = _int_sequence(result.get("timestamp"))
    indicators = _mapping(result.get("indicators"))
    quote_fields = _first_mapping(indicators.get("quote") if indicators is not None else None)
    if timestamps is None or quote_fields is None:
        return ()
    opens = _optional_float_sequence(quote_fields.get("open"))
    highs = _optional_float_sequence(quote_fields.get("high"))
    lows = _optional_float_sequence(quote_fields.get("low"))
    closes = _optional_float_sequence(quote_fields.get("close"))
    volumes = _optional_float_sequence(quote_fields.get("volume"))
    adjclose_fields = _first_mapping(indicators.get("adjclose") if indicators is not None else None)
    adjusted = _optional_float_sequence(adjclose_fields.get("adjclose")) if adjclose_fields is not None else closes
    if any(value is None for value in (opens, highs, lows, closes, volumes, adjusted)):
        return ()
    prices: list[DailyPrice] = []
    for index, timestamp in enumerate(timestamps):
        price = _yahoo_daily_price(
            symbol,
            timestamp,
            cast("list[float | None]", opens),
            cast("list[float | None]", highs),
            cast("list[float | None]", lows),
            cast("list[float | None]", closes),
            cast("list[float | None]", adjusted),
            cast("list[float | None]", volumes),
            index,
        )
        if price is not None:
            prices.append(price)
    return tuple(sorted(prices, key=lambda price: price.date))


def _yahoo_daily_price(
    symbol: str,
    timestamp: int,
    opens: list[float | None],
    highs: list[float | None],
    lows: list[float | None],
    closes: list[float | None],
    adjusted: list[float | None],
    volumes: list[float | None],
    index: int,
) -> DailyPrice | None:
    values = (
        _list_value(opens, index),
        _list_value(highs, index),
        _list_value(lows, index),
        _list_value(closes, index),
        _list_value(adjusted, index),
        _list_value(volumes, index),
    )
    if any(value is None for value in values):
        return None
    open_, high, low, close, adjusted_close, volume = cast("tuple[float, float, float, float, float, float]", values)
    if close <= 0.0 or volume < 0.0:
        return None
    return DailyPrice(
        symbol,
        datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat(),
        open_,
        high,
        low,
        close,
        adjusted_close,
        volume,
        "yahoo:chart",
    )


def _alpha_payload_is_miss(payload: object) -> bool:
    data = _mapping(payload)
    if data is None:
        return True
    return any(key in data for key in ("Note", "Information", "Error Message"))


def _finnhub_payload_is_miss(payload: object) -> bool:
    if isinstance(payload, str):
        return _access_denied(payload)
    data = _mapping(payload)
    if data is None:
        return True
    status = data.get("s")
    if status == "no_data" or status is None:
        return True
    if isinstance(status, str) and status.lower() != "ok":
        return True
    return any(_access_denied(str(value)) for value in data.values() if isinstance(value, str))


def _finnhub_quote_payload_is_miss(payload: object) -> bool:
    if isinstance(payload, str):
        return _access_denied(payload)
    data = _mapping(payload)
    if data is None:
        return True
    return any(_access_denied(str(value)) for value in data.values() if isinstance(value, str))


def _yahoo_payload_is_miss(payload: object) -> bool:
    if isinstance(payload, str):
        return _access_denied(payload) or "too many" in payload.lower()
    data = _mapping(payload)
    if data is None:
        return True
    chart = _mapping(data.get("chart"))
    return chart is not None and chart.get("error") is not None


def _access_denied(text: str) -> bool:
    lowered = text.lower()
    return "access" in lowered and ("denied" in lowered or "resource" in lowered)


def _mapping(value: object) -> Mapping[str, JsonValue] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items() if _is_json_value(item)}


def _first_mapping(value: object) -> Mapping[str, JsonValue] | None:
    if not isinstance(value, list) or not value:
        return None
    return _mapping(value[0])


def _float_field(mapping: Mapping[str, object], key: str) -> float | None:
    return _float_value(mapping.get(key))


def _int_field(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_sequence(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or item is None:
            return None
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            return None
    return parsed


def _optional_float_sequence(value: object) -> list[float | None] | None:
    if not isinstance(value, list):
        return None
    parsed: list[float | None] = []
    for item in value:
        if item is None:
            parsed.append(None)
            continue
        if isinstance(item, bool):
            return None
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            return None
    return parsed


def _int_sequence(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool) or item is None:
            return None
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            return None
    return parsed


def _list_value(values: list[float | None], index: int) -> float | None:
    return values[index] if index < len(values) else None

