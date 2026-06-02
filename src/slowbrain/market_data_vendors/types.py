"""Shared types and read-only HTTP transport for market-data vendors."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast
from urllib.request import HTTPCookieProcessor, Request, build_opener

from ..market_data import MarketDataProvider, PriceHistoryProvider
from ..market_data_cache import JsonValue

type MarketDataTransport = Callable[[str, float], JsonValue]

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
YAHOO_BASE_URL = "https://query1.finance.yahoo.com"
YAHOO_CRUMB_URL = f"{YAHOO_BASE_URL}/v1/test/getcrumb"
DEFAULT_HISTORY_START = datetime(2000, 1, 1, tzinfo=UTC)
DEFAULT_SPREAD_PROXY_BPS = 5.0


class VendorProvider(PriceHistoryProvider, MarketDataProvider, Protocol):
    """Concrete vendor adapters expose both evidence snapshots and daily prices."""


@dataclass(frozen=True)
class FinnhubQuote:
    symbol: str
    current_price: float
    change: float
    change_pct: float
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: int
    source: str


@dataclass(frozen=True)
class FxRate:
    rate: float
    source: str


class StdlibHttpTransport:
    """Small JSON/text GET transport with cookie support for read-only market-data calls."""

    def __init__(self) -> None:
        self._opener = build_opener(HTTPCookieProcessor())

    def __call__(self, url: str, timeout_seconds: float) -> JsonValue:
        request = Request(url, headers={"User-Agent": "TheSlowBrain/0.1 market-data research"})
        with self._opener.open(request, timeout=timeout_seconds) as response:
            body = cast("bytes", response.read())
        text = body.decode("utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        return parsed if _is_json_value(parsed) else None


type FxRateSource = None | float | Callable[[str], FxRate | None]


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
