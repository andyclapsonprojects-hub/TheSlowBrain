from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from urllib.error import HTTPError, URLError

import pytest

import slowbrain.trading212 as trading212
from slowbrain.config import AppConfig
from slowbrain.trading212 import (
    DEMO_BASE_URL,
    LIVE_BASE_URL,
    Trading212Client,
    Trading212Response,
    build_trading212_client,
    response_summary,
)


def test_trading212_client_shapes_live_market_order_and_basic_auth() -> None:
    calls: list[tuple[str, str, Mapping[str, str], bytes | None, float]] = []

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> Trading212Response:
        calls.append((method, url, headers, body, timeout_seconds))
        return Trading212Response(method, "/equity/orders/market", 200, {"id": "broker-1"}, {})

    client = Trading212Client(
        api_key="api-key",
        api_secret="api-secret",
        environment="live",
        transport=transport,
        timeout_seconds=3.5,
    )

    response = client.place_market_order(ticker="AVGO_US_EQ", quantity=0.125)

    method, url, headers, body, timeout_seconds = calls[0]
    expected_auth = base64.b64encode(b"api-key:api-secret").decode("ascii")
    assert response.status_code == 200
    assert method == "POST"
    assert url == f"{LIVE_BASE_URL}/equity/orders/market"
    assert headers["Authorization"] == f"Basic {expected_auth}"
    assert json.loads((body or b"{}").decode("utf-8")) == {"ticker": "AVGO_US_EQ", "quantity": 0.125}
    assert timeout_seconds == 3.5


def test_response_summary_never_includes_credentials_or_payload() -> None:
    response = Trading212Response(
        "GET",
        "/equity/account/summary",
        200,
        {"secret": "payload"},
        {"Authorization": "Basic secret", "X-RateLimit-Remaining": "9"},
    )

    summary = response_summary(response)

    assert summary == {
        "method": "GET",
        "path": "/equity/account/summary",
        "status_code": 200,
        "rate_limit": {"X-RateLimit-Remaining": "9"},
    }


def test_trading212_client_endpoint_helpers_and_config_builder() -> None:
    calls: list[str] = []

    def transport(
        method: str,
        url: str,
        _headers: Mapping[str, str],
        _body: bytes | None,
        _timeout_seconds: float,
    ) -> Trading212Response:
        calls.append(f"{method} {url}")
        return Trading212Response(method, url, 200, {"ok": True}, {})

    client = build_trading212_client(_config(api=True), transport=transport)

    assert client.base_url == DEMO_BASE_URL
    assert client.account_summary().status_code == 200
    assert client.instruments().status_code == 200
    assert client.positions(ticker="AVGO_US_EQ").status_code == 200
    assert client.active_orders().status_code == 200
    assert client.history_orders(limit=7).status_code == 200
    assert calls == [
        f"GET {DEMO_BASE_URL}/equity/account/summary",
        f"GET {DEMO_BASE_URL}/equity/metadata/instruments",
        f"GET {DEMO_BASE_URL}/equity/positions?ticker=AVGO_US_EQ",
        f"GET {DEMO_BASE_URL}/equity/orders",
        f"GET {DEMO_BASE_URL}/equity/history/orders?limit=7",
    ]


def test_trading212_client_rejects_missing_credentials() -> None:
    with pytest.raises(ValueError, match="API key and secret"):
        Trading212Client(api_key="", api_secret="", environment="demo")
    with pytest.raises(ValueError, match="not configured"):
        build_trading212_client(_config(api=False))


def test_stdlib_transport_parses_success_http_error_and_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status: ClassVar[int] = 200
        headers: ClassVar[dict[str, str]] = {"X-Test": "ok"}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    monkeypatch.setattr(trading212, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    success = trading212._stdlib_transport("GET", f"{LIVE_BASE_URL}/equity/orders", {}, None, 1.0)

    headers = Message()
    headers["X-RateLimit-Remaining"] = "0"

    def http_error(*_args: object, **_kwargs: object) -> object:
        raise HTTPError(f"{LIVE_BASE_URL}/equity/orders", 429, "rate limited", headers, BytesIO(b'{"error":"rate"}'))

    monkeypatch.setattr(trading212, "urlopen", http_error)
    rejected = trading212._stdlib_transport("GET", f"{LIVE_BASE_URL}/equity/orders", {}, None, 1.0)

    def network_error(*_args: object, **_kwargs: object) -> object:
        raise URLError("offline")

    monkeypatch.setattr(trading212, "urlopen", network_error)
    network = trading212._stdlib_transport("GET", f"{LIVE_BASE_URL}/equity/orders", {}, None, 1.0)

    assert success.payload == {"ok": True}
    assert success.path == "/equity/orders"
    assert rejected.status_code == 429
    assert rejected.payload == {"error": "rate"}
    assert network.status_code == 0
    assert network.payload == {"error": "trading212_network_error"}


def test_payload_and_path_helpers_cover_non_json_and_invalid_shapes() -> None:
    assert trading212._parse_payload(b"") is None
    assert trading212._parse_payload(b"not-json") == "not-json"
    assert trading212._parse_payload(b'{"ok": true}') == {"ok": True}
    assert trading212._parse_payload(b"\xff") == "�"
    assert trading212._path_from_url("https://example.test/custom") == "https://example.test/custom"
    assert trading212._is_json_value({1: "bad"}) is False


def _config(*, api: bool) -> AppConfig:
    return AppConfig(
        legacy_stock_project_root=Path("legacy"),
        alpha_vantage_api_key=None,
        finnhub_api_key=None,
        openai_api_key=None,
        openai_model=None,
        trading212_api_key="key" if api else None,
        trading212_api_secret="secret" if api else None,
        trading212_env="demo",
        trading_live_enabled=False,
        trading_require_manual_approval=True,
        trading_max_daily_orders=None,
        trading_max_order_value=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        telegram_message_thread_id=None,
    )
