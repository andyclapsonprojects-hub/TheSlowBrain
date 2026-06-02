"""Read-only adapter for the original n8n trader daily OHLCV cache."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .market_data import DailyPrice, PriceHistoryProvider
from .numeric import optional_float


class LegacyN8nPriceCacheProvider(PriceHistoryProvider):
    """Use the original project cache as labelled decision-review evidence."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    @classmethod
    def from_legacy_project_root(cls, legacy_project_root: Path) -> LegacyN8nPriceCacheProvider:
        return cls(legacy_project_root / "reports" / "price-cache" / "daily")

    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        path = self.cache_dir / f"{_safe_symbol(symbol)}.json"
        if not path.exists():
            return ()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return ()
        root = _mapping(payload)
        raw_prices = _mapping(root.get("prices"))
        if not raw_prices:
            return ()
        prices = [
            price
            for date, row in raw_prices.items()
            if isinstance(date, str)
            for price in [_daily_price(symbol.upper(), date, row)]
            if price is not None
        ]
        return tuple(sorted(prices, key=lambda price: price.date))


def _daily_price(symbol: str, date: str, value: object) -> DailyPrice | None:
    row = _mapping(value)
    open_ = _float(row.get("1. open"))
    high = _float(row.get("2. high"))
    low = _float(row.get("3. low"))
    close = _float(row.get("4. close"))
    volume = _float(row.get("5. volume"))
    if any(item is None for item in (open_, high, low, close, volume)):
        return None
    return DailyPrice(
        symbol=symbol,
        date=date,
        open=open_ or 0.0,
        high=high or 0.0,
        low=low or 0.0,
        close=close or 0.0,
        adjusted_close=close or 0.0,
        volume=volume or 0.0,
        source="legacy_n8n_price_cache",
    )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _float(value: object) -> float | None:
    return optional_float(value)


def _safe_symbol(symbol: str) -> str:
    cleaned = "".join(character for character in symbol.upper() if character.isalnum() or character in {".", "-"})
    return cleaned or "UNKNOWN"
