"""Environment-gated Telegram delivery for Eric briefs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import AppConfig

TelegramStatus = Literal["preview_only", "blocked", "sent", "rejected", "failed"]
TelegramTransport = Callable[[str, bytes, float], bytes]

MAX_TELEGRAM_TEXT_CHARS = 4096


@dataclass(frozen=True)
class TelegramDeliveryResult:
    status: TelegramStatus
    reason: str
    message_id: int | None = None


def extract_eric_message(report: Mapping[str, object]) -> str:
    """Extract the concise Eric brief text from a workflow report."""
    brief = report.get("eric_brief")
    if not isinstance(brief, Mapping):
        raise ValueError("report does not contain eric_brief")
    lines = brief.get("lines")
    if not isinstance(lines, Sequence) or isinstance(lines, str):
        raise ValueError("eric_brief.lines must be a sequence")
    return "\n".join(str(line) for line in lines)


def send_telegram_message(
    config: AppConfig,
    text: str,
    *,
    send: bool = False,
    transport: TelegramTransport | None = None,
    timeout_seconds: float = 10.0,
) -> TelegramDeliveryResult:
    """Send text to Telegram only when explicitly enabled and configured."""
    validation_error = _validate_text(text)
    if validation_error is not None:
        return TelegramDeliveryResult(status="blocked", reason=validation_error)
    if not send:
        return TelegramDeliveryResult(status="preview_only", reason="dry_run_no_external_message_sent")
    if not config.telegram_bot_token:
        return TelegramDeliveryResult(status="blocked", reason="missing_telegram_bot_token")
    if not config.telegram_chat_id:
        return TelegramDeliveryResult(status="blocked", reason="missing_telegram_chat_id")
    message_thread_id = _parse_message_thread_id(config.telegram_message_thread_id)
    if message_thread_id == "invalid":
        return TelegramDeliveryResult(status="blocked", reason="invalid_telegram_message_thread_id")

    payload = {
        "chat_id": config.telegram_chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if isinstance(message_thread_id, int):
        payload["message_thread_id"] = message_thread_id
    body = json.dumps(payload).encode("utf-8")
    post_json = transport or _post_json
    try:
        response = post_json(_telegram_url(config.telegram_bot_token), body, timeout_seconds)
    except HTTPError as exc:
        return TelegramDeliveryResult(status="rejected", reason=f"telegram_http_{exc.code}")
    except (TimeoutError, URLError):
        return TelegramDeliveryResult(status="failed", reason="telegram_network_error")

    return _parse_telegram_response(response)


def _validate_text(text: str) -> str | None:
    if not text.strip():
        return "telegram_text_empty"
    if len(text) > MAX_TELEGRAM_TEXT_CHARS:
        return "telegram_text_too_long"
    return None


def _parse_message_thread_id(value: str | None) -> int | Literal["invalid"] | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return "invalid"
    return parsed if parsed > 0 else "invalid"


def _telegram_url(token: str) -> str:
    return f"https://api.telegram.org/bot{token}/sendMessage"


def _post_json(url: str, body: bytes, timeout_seconds: float) -> bytes:
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return cast("bytes", response.read())


def _parse_telegram_response(response: bytes) -> TelegramDeliveryResult:
    try:
        payload = json.loads(response.decode("utf-8"))
    except json.JSONDecodeError:
        return TelegramDeliveryResult(status="failed", reason="telegram_invalid_json_response")
    if not isinstance(payload, dict):
        return TelegramDeliveryResult(status="failed", reason="telegram_non_object_response")
    if payload.get("ok") is not True:
        return TelegramDeliveryResult(status="rejected", reason="telegram_rejected")
    result = payload.get("result")
    message_id = result.get("message_id") if isinstance(result, dict) else None
    return TelegramDeliveryResult(
        status="sent",
        reason="telegram_ok",
        message_id=message_id if isinstance(message_id, int) else None,
    )
