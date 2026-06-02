from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from slowbrain.market_data import DailyPrice
from slowbrain.market_data_cache import MarketDataCache, _is_json_value
from slowbrain.market_data_vendors.parsers import (
    _alpha_payload_is_miss,
    _finnhub_payload_is_miss,
    _finnhub_quote_payload_is_miss,
    _float_sequence,
    _int_sequence,
    _optional_float_sequence,
    _parse_alpha_daily_payload,
    _parse_finnhub_candle_payload,
    _parse_finnhub_quote_payload,
    _parse_yahoo_chart_payload,
    _yahoo_payload_is_miss,
)
from slowbrain.market_data_vendors.pricing import _benchmark_return_from_prices, _liquidity_from_prices
from slowbrain.market_data_vendors.types import StdlibHttpTransport
from slowbrain.market_data_vendors.types import _is_json_value as _vendor_is_json_value


def test_market_data_cache_rejects_stale_wrong_or_invalid_envelopes(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path, "vendor", max_age_days=1)
    path = cache.path_for_symbol("AAPL")
    path.parent.mkdir(parents=True)

    stale = _envelope("vendor", datetime.now(UTC) - timedelta(days=3), {"old": True})
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert cache.get_or_fetch("AAPL", lambda: {"fresh": True}) == {"fresh": True}

    path.write_text(json.dumps(_envelope("other", datetime.now(UTC), {"old": True})), encoding="utf-8")
    assert cache.get_or_fetch("AAPL", lambda: {"vendor": True}) == {"vendor": True}

    path.write_text(json.dumps({"schema": "wrong", "payload": {"old": True}}), encoding="utf-8")
    assert cache.get_or_fetch("AAPL", lambda: None) is None

    path.write_text(json.dumps(_envelope("vendor", "not-a-date", {"old": True})), encoding="utf-8")
    assert cache.get_or_fetch("AAPL", lambda: {"date": True}) == {"date": True}


def test_json_value_helpers_reject_non_json_objects() -> None:
    assert _is_json_value({"ok": [1, None, "x"]})
    assert not _is_json_value({1: "bad-key"})
    assert not _is_json_value(object())
    assert not _vendor_is_json_value({"bad": object()})


def test_vendor_payload_parsers_cover_defensive_miss_paths() -> None:
    assert _parse_alpha_daily_payload("AAPL", {"Note": "limited"}) == ()
    assert _parse_alpha_daily_payload("AAPL", {"Time Series (Daily)": {"2026-01-01": {"1. open": "bad"}}}) == ()
    assert _alpha_payload_is_miss(None)

    assert _parse_finnhub_candle_payload("AAPL", {"s": "no_data"}) == ()
    assert _parse_finnhub_candle_payload("AAPL", {"s": "ok", "c": [1.0], "h": [1.0]}) == ()
    assert _parse_finnhub_quote_payload("AAPL", {"c": True}) is None
    assert _finnhub_payload_is_miss("Access denied")
    assert _finnhub_payload_is_miss({"s": "error"})
    assert _finnhub_quote_payload_is_miss({"message": "resource access denied"})

    assert _parse_yahoo_chart_payload("AAPL", "Too Many Requests") == ()
    assert _parse_yahoo_chart_payload("AAPL", {"chart": {"error": {"code": "bad"}}}) == ()
    assert _parse_yahoo_chart_payload("AAPL", {"chart": {"result": []}}) == ()
    assert _yahoo_payload_is_miss("access denied")

    assert _float_sequence([1, "bad"]) is None
    assert _float_sequence([True]) is None
    assert _optional_float_sequence([1.0, None, "bad"]) is None
    assert _optional_float_sequence([False]) is None
    assert _int_sequence([1, "bad"]) is None
    assert _int_sequence([None]) is None


def test_vendor_pricing_handles_missing_context_and_callable_fx() -> None:
    prices = (
        DailyPrice("AAPL", "2026-01-01", 1, 1, 1, 1, 0, 100, "fixture"),
        DailyPrice("AAPL", "2026-01-02", 1, 1, 1, 2, 2, 200, "fixture"),
    )
    assert _benchmark_return_from_prices("AAPL", "2026-01-02", prices, source="fixture") is None
    assert _liquidity_from_prices("AAPL", "2026-01-02", prices, quote_to_gbp_rate=None, source="fixture") is None
    assert _liquidity_from_prices("AAPL", "2026-01-01", prices, quote_to_gbp_rate=0.8, source="fixture") is None

    clean_prices = (
        DailyPrice("AAPL", "2026-01-01", 1, 1, 1, 1, 1, 100, "fixture"),
        DailyPrice("AAPL", "2026-01-02", 1, 1, 1, 2, 2, 200, "fixture"),
    )
    benchmark = _benchmark_return_from_prices("AAPL", "2026-01-02", clean_prices, source="fixture")
    liquidity = _liquidity_from_prices(
        "AAPL",
        "2026-01-02",
        clean_prices,
        quote_to_gbp_rate=lambda _date: None,
        source="fixture",
    )

    assert benchmark is not None
    assert benchmark.return_pct == 100.0
    assert liquidity is None


def test_stdlib_transport_returns_json_or_text_payloads() -> None:
    transport = StdlibHttpTransport()
    cast(Any, transport)._opener = _FakeOpener(b'{"ok": true}')
    assert transport("https://example.test/json", 1.0) == {"ok": True}

    cast(Any, transport)._opener = _FakeOpener(b"plain text")
    assert transport("https://example.test/text", 1.0) == "plain text"


def _envelope(vendor: str, fetched_at: datetime | str, payload: object) -> dict[str, object]:
    return {
        "schema": "theslowbrain.market_data_cache.v1",
        "vendor": vendor,
        "fetched_at": fetched_at if isinstance(fetched_at, str) else fetched_at.isoformat(),
        "payload": payload,
    }


class _FakeOpener:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def open(self, _request: object, *, timeout: float) -> _FakeResponse:
        assert timeout == 1.0
        return _FakeResponse(self.body)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body
