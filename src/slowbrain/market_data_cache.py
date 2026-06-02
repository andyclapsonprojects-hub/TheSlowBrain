"""Disk cache for read-only market-data vendor responses."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonFetcher = Callable[[], JsonValue | None]


@dataclass(frozen=True)
class MarketDataCache:
    root: Path
    vendor: str
    max_age_days: int | None = None

    def get_or_fetch(self, symbol: str, fetch: JsonFetcher) -> JsonValue | None:
        """Return cached payload for a symbol, fetching and persisting on a miss."""
        path = self.path_for_symbol(symbol)
        cached = self._read_payload(path)
        if cached is not None:
            return cached

        payload = fetch()
        if payload is None:
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        envelope: dict[str, JsonValue] = {
            "schema": "theslowbrain.market_data_cache.v1",
            "vendor": self.vendor,
            "symbol": symbol.upper(),
            "fetched_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def path_for_symbol(self, symbol: str) -> Path:
        return self.root / self.vendor / f"{_safe_symbol(symbol)}.json"

    def _read_payload(self, path: Path) -> JsonValue | None:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        if raw.get("schema") != "theslowbrain.market_data_cache.v1":
            return None
        if raw.get("vendor") != self.vendor:
            return None
        if not self._fresh_enough(raw.get("fetched_at")):
            return None
        payload = raw.get("payload")
        return payload if _is_json_value(payload) else None

    def _fresh_enough(self, fetched_at: object) -> bool:
        if self.max_age_days is None:
            return True
        if not isinstance(fetched_at, str):
            return False
        try:
            parsed = datetime.fromisoformat(fetched_at)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return datetime.now(UTC) - parsed <= timedelta(days=self.max_age_days)


def _safe_symbol(symbol: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in symbol.upper())
    return cleaned or "UNKNOWN"


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
