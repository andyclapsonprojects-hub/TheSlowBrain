from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

from slowbrain.config import AppConfig
from slowbrain.telegram import extract_eric_message, send_telegram_message


def test_extract_eric_message_from_report() -> None:
    message = extract_eric_message({"eric_brief": {"lines": ["Eric - TheSlowBrain", "Stocks to buy are: none"]}})

    assert message == "Eric - TheSlowBrain\nStocks to buy are: none"


def test_extract_eric_message_rejects_bad_report_shape() -> None:
    for report in ({}, {"eric_brief": {"lines": "Eric - TheSlowBrain"}}):
        try:
            extract_eric_message(report)
        except ValueError:
            pass
        else:
            raise AssertionError("expected bad report shape to raise")


def test_telegram_preview_does_not_call_transport() -> None:
    calls: list[tuple[str, bytes, float]] = []

    def fake_transport(url: str, body: bytes, timeout: float) -> bytes:
        calls.append((url, body, timeout))
        return b"{}"

    result = send_telegram_message(
        _config(token="fixture-token", chat_id="123"),
        "Eric - TheSlowBrain",
        send=False,
        transport=fake_transport,
    )

    assert result.status == "preview_only"
    assert calls == []


def test_telegram_blocks_missing_credentials() -> None:
    result = send_telegram_message(_config(token=None, chat_id="123"), "Eric - TheSlowBrain", send=True)

    assert result.status == "blocked"
    assert result.reason == "missing_telegram_bot_token"


def test_telegram_blocks_missing_chat_and_invalid_text() -> None:
    missing_chat = send_telegram_message(_config(token="fixture-token", chat_id=None), "Eric", send=True)
    empty = send_telegram_message(_config(token="fixture-token", chat_id="123"), "", send=True)
    too_long = send_telegram_message(_config(token="fixture-token", chat_id="123"), "x" * 4097, send=True)

    assert missing_chat.reason == "missing_telegram_chat_id"
    assert empty.reason == "telegram_text_empty"
    assert too_long.reason == "telegram_text_too_long"


def test_telegram_sends_json_payload_without_token_in_result() -> None:
    calls: list[tuple[str, bytes, float]] = []

    def fake_transport(url: str, body: bytes, timeout: float) -> bytes:
        calls.append((url, body, timeout))
        return b'{"ok": true, "result": {"message_id": 42}}'

    result = send_telegram_message(
        _config(token="fixture-token", chat_id="123"),
        "Eric - TheSlowBrain",
        send=True,
        transport=fake_transport,
    )

    assert result.status == "sent"
    assert result.message_id == 42
    assert "fixture-token" not in repr(result)
    payload = json.loads(calls[0][1].decode("utf-8"))
    assert payload == {
        "chat_id": "123",
        "text": "Eric - TheSlowBrain",
        "disable_web_page_preview": True,
    }
    assert "fixture-token" not in calls[0][1].decode("utf-8")


def test_telegram_sends_to_forum_topic_when_thread_id_configured() -> None:
    calls: list[tuple[str, bytes, float]] = []

    def fake_transport(url: str, body: bytes, timeout: float) -> bytes:
        calls.append((url, body, timeout))
        return b'{"ok": true, "result": {"message_id": 43}}'

    result = send_telegram_message(
        _config(token="fixture-token", chat_id="-1003869293930", message_thread_id="2937"),
        "Eric - TheSlowBrain",
        send=True,
        transport=fake_transport,
    )

    assert result.status == "sent"
    payload = json.loads(calls[0][1].decode("utf-8"))
    assert payload == {
        "chat_id": "-1003869293930",
        "text": "Eric - TheSlowBrain",
        "disable_web_page_preview": True,
        "message_thread_id": 2937,
    }


def test_telegram_blocks_invalid_forum_topic_before_transport() -> None:
    calls: list[tuple[str, bytes, float]] = []

    def fake_transport(url: str, body: bytes, timeout: float) -> bytes:
        calls.append((url, body, timeout))
        return b'{"ok": true}'

    result = send_telegram_message(
        _config(token="fixture-token", chat_id="-1003869293930", message_thread_id="not-a-number"),
        "Eric - TheSlowBrain",
        send=True,
        transport=fake_transport,
    )

    assert result.status == "blocked"
    assert result.reason == "invalid_telegram_message_thread_id"
    assert calls == []


def test_telegram_reports_safe_failure_statuses() -> None:
    config = _config(token="fixture-token", chat_id="123")

    def http_error(_url: str, _body: bytes, _timeout: float) -> bytes:
        raise HTTPError("redacted-url", 401, "bad token", Message(), None)

    def network_error(_url: str, _body: bytes, _timeout: float) -> bytes:
        raise URLError("offline")

    assert send_telegram_message(config, "Eric", send=True, transport=http_error).status == "rejected"
    assert send_telegram_message(config, "Eric", send=True, transport=network_error).status == "failed"
    assert (
        send_telegram_message(config, "Eric", send=True, transport=lambda _u, _b, _t: b"not-json").reason
        == "telegram_invalid_json_response"
    )
    assert (
        send_telegram_message(config, "Eric", send=True, transport=lambda _u, _b, _t: b"[]").reason
        == "telegram_non_object_response"
    )
    assert (
        send_telegram_message(config, "Eric", send=True, transport=lambda _u, _b, _t: b'{"ok": false}').reason
        == "telegram_rejected"
    )


def _config(*, token: str | None, chat_id: str | None, message_thread_id: str | None = None) -> AppConfig:
    return AppConfig(
        legacy_stock_project_root=Path("legacy"),
        alpha_vantage_api_key=None,
        finnhub_api_key=None,
        openai_api_key=None,
        openai_model=None,
        trading212_api_key=None,
        trading212_api_secret=None,
        trading212_env="demo",
        trading_live_enabled=False,
        trading_require_manual_approval=True,
        trading_max_daily_orders=None,
        trading_max_order_value=None,
        telegram_bot_token=token,
        telegram_chat_id=chat_id,
        telegram_message_thread_id=message_thread_id,
    )
