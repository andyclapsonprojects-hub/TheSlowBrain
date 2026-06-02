"""Trading 212 Public API boundary.

This module deliberately contains only transport and endpoint mechanics. Order
approval, duplicate prevention, and reconciliation policy live in
``slowbrain.live_execution``.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AppConfig

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

DEMO_BASE_URL = "https://demo.trading212.com/api/v0"
LIVE_BASE_URL = "https://live.trading212.com/api/v0"


@dataclass(frozen=True)
class Trading212Response:
    method: str
    path: str
    status_code: int
    payload: JsonValue
    headers: Mapping[str, str]


Trading212Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], Trading212Response]


class Trading212Client:
    """Small fakeable client for Trading 212's REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        environment: str,
        transport: Trading212Transport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("Trading 212 API key and secret are required")
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = _base_url(environment)
        self._transport = transport or _stdlib_transport
        self._timeout_seconds = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    def account_summary(self) -> Trading212Response:
        return self.get("/equity/account/summary")

    def instruments(self) -> Trading212Response:
        return self.get("/equity/metadata/instruments")

    def positions(self, *, ticker: str | None = None) -> Trading212Response:
        query = f"?{urlencode({'ticker': ticker})}" if ticker else ""
        return self.get(f"/equity/positions{query}")

    def active_orders(self) -> Trading212Response:
        return self.get("/equity/orders")

    def history_orders(self, *, limit: int = 50) -> Trading212Response:
        return self.get(f"/equity/history/orders?{urlencode({'limit': limit})}")

    def place_market_order(self, *, ticker: str, quantity: float) -> Trading212Response:
        return self.post("/equity/orders/market", {"ticker": ticker, "quantity": quantity})

    def get(self, path: str) -> Trading212Response:
        return self._request("GET", path, None)

    def post(self, path: str, payload: Mapping[str, object]) -> Trading212Response:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: Mapping[str, object] | None) -> Trading212Response:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Authorization": _basic_auth_header(self._api_key, self._api_secret),
            "Content-Type": "application/json",
            "User-Agent": "TheSlowBrain/0.1 broker-readiness",
        }
        return self._transport(method, self._url(path), headers, body, self._timeout_seconds)

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        normalized = path if path.startswith("/") else f"/{path}"
        return f"{self._base_url}{normalized}"


class Trading212Gateway(Protocol):
    def account_summary(self) -> Trading212Response: ...
    def instruments(self) -> Trading212Response: ...
    def positions(self, *, ticker: str | None = None) -> Trading212Response: ...
    def active_orders(self) -> Trading212Response: ...
    def history_orders(self, *, limit: int = 50) -> Trading212Response: ...
    def place_market_order(self, *, ticker: str, quantity: float) -> Trading212Response: ...


def credentials_available(config: AppConfig) -> bool:
    return bool(config.trading212_api_key and config.trading212_api_secret)


def build_trading212_client(
    config: AppConfig,
    *,
    transport: Trading212Transport | None = None,
    timeout_seconds: float = 10.0,
) -> Trading212Client:
    if not config.trading212_api_key or not config.trading212_api_secret:
        raise ValueError("Trading 212 credentials are not configured")
    return Trading212Client(
        api_key=config.trading212_api_key,
        api_secret=config.trading212_api_secret,
        environment=config.trading212_env,
        transport=transport,
        timeout_seconds=timeout_seconds,
    )


def response_summary(response: Trading212Response) -> dict[str, object]:
    """Return non-secret response metadata suitable for reports."""
    rate_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower()
        in {
            "x-ratelimit-limit",
            "x-ratelimit-period",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "x-ratelimit-used",
        }
    }
    return {
        "method": response.method,
        "path": response.path,
        "status_code": response.status_code,
        "rate_limit": rate_headers,
    }


def _base_url(environment: str) -> str:
    normalized = environment.strip().lower()
    return LIVE_BASE_URL if normalized in {"live", "real", "real_money"} else DEMO_BASE_URL


def _basic_auth_header(api_key: str, api_secret: str) -> str:
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode("ascii")
    return f"Basic {token}"


def _stdlib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: float,
) -> Trading212Response:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
            status_code = int(response.status)
            response_headers = dict(response.headers.items())
    except HTTPError as exc:
        raw_body = exc.read()
        status_code = int(exc.code)
        response_headers = dict(exc.headers.items()) if exc.headers is not None else {}
    except (TimeoutError, URLError, OSError):
        return Trading212Response(method, _path_from_url(url), 0, {"error": "trading212_network_error"}, {})
    return Trading212Response(method, _path_from_url(url), status_code, _parse_payload(raw_body), response_headers)


def _parse_payload(raw_body: bytes) -> JsonValue:
    if not raw_body:
        return None
    text = raw_body.decode("utf-8", errors="replace")
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        return text
    return value if _is_json_value(value) else None


def _path_from_url(url: str) -> str:
    for base_url in (DEMO_BASE_URL, LIVE_BASE_URL):
        if url.startswith(base_url):
            return url[len(base_url) :]
    return url


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
