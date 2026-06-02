from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import slowbrain.live_execution_support as support
from slowbrain.config import AppConfig
from slowbrain.live_execution import (
    PriceSnapshot,
    build_broker_health_report,
    build_execution_preview,
    submit_execution_preview,
)
from slowbrain.trading212 import JsonValue, Trading212Response


def test_broker_health_blocks_without_credentials() -> None:
    report = build_broker_health_report(config=_config(api=False), client=None)

    assert report["status"] == "blocked"
    assert report["reason"] == "missing_trading212_credentials"
    assert report["orders_submitted"] is False
    assert report["broker_live_execution_allowed"] is False


def test_broker_health_ok_reports_sanitized_counts() -> None:
    report = build_broker_health_report(config=_config(), client=FakeTrading212Gateway(), now=_now())

    assert report["status"] == "ok"
    assert report["orders_submitted"] is False
    assert report["cash_available_to_trade"] == 100.0
    assert report["positions_count"] == 1
    assert report["active_orders_count"] == 0


def test_preview_builds_positive_buy_and_negative_sell_orders() -> None:
    preview = _preview()
    orders = _orders(preview)
    ready = {str(order["ticker"]): order for order in orders if order["status"] == "ready"}
    blocked = {str(order["ticker"]): order for order in orders if order["status"] == "blocked"}

    assert preview["status"] == "ready"
    assert preview["ready_order_count"] == 2
    assert preview["approval_token"]
    assert ready["AVGO"]["preview_id"] == preview["preview_id"]
    assert ready["AVGO"]["broker_ticker"] == "AVGO_US_EQ"
    assert ready["AVGO"]["quantity"] == 0.5
    assert ready["AVGO"]["order_payload"] == {"ticker": "AVGO_US_EQ", "quantity": 0.5}
    assert ready["ADSK"]["quantity"] == -2.5
    assert ready["ADSK"]["order_payload"] == {"ticker": "ADSK_US_EQ", "quantity": -2.5}
    assert blocked["MSFT"]["reason"] == "sell_decision_not_currently_held"
    assert blocked["HOLDME"]["reason"] == "decision_not_buy_or_sell"


def test_preview_blocks_buy_when_price_lookup_is_missing() -> None:
    preview = build_execution_preview(
        report_payload={"trade_decisions": [{"ticker": "AVGO", "action": "BUY", "score": 0.9}]},
        config=_config(),
        instruments=_instruments(),
        positions=(),
        price_lookup=None,
        now=_now(),
    )

    order = _orders(preview)[0]
    assert preview["status"] == "blocked"
    assert order["status"] == "blocked"
    assert order["reason"] == "missing_price_lookup_for_buy_quantity"


def test_preview_blocks_active_orders_existing_holding_bad_prices_and_zero_quantity() -> None:
    active_order_preview = build_execution_preview(
        report_payload={"trade_decisions": [{"ticker": "AVGO", "action": "BUY", "score": 0.9}]},
        config=_config(),
        instruments=_instruments(),
        positions=(),
        active_orders=({"ticker": "AVGO_US_EQ"},),
        price_lookup=lambda _symbol, _currency: PriceSnapshot(20.0, "fixture", "2026-06-02"),
        now=_now(),
    )
    held_preview = build_execution_preview(
        report_payload={"trade_decisions": [{"ticker": "AVGO", "action": "BUY", "score": 0.9}]},
        config=_config(),
        instruments=_instruments(),
        positions=({"instrument": {"ticker": "AVGO_US_EQ"}, "quantityAvailableForTrading": 1.0},),
        price_lookup=lambda _symbol, _currency: PriceSnapshot(20.0, "fixture", "2026-06-02"),
        now=_now(),
    )
    stale_price_preview = build_execution_preview(
        report_payload={"trade_decisions": [{"ticker": "AVGO", "action": "BUY", "score": 0.9}]},
        config=_config(),
        instruments=_instruments(),
        positions=(),
        price_lookup=lambda _symbol, _currency: None,
        now=_now(),
    )
    zero_quantity_preview = build_execution_preview(
        report_payload={"trade_decisions": [{"ticker": "AVGO", "action": "BUY", "score": 0.9}]},
        config=_config(),
        instruments=_instruments(),
        positions=(),
        price_lookup=lambda _symbol, _currency: PriceSnapshot(20.0, "fixture", "2026-06-02"),
        buy_notional_gbp=0.0,
        now=_now(),
    )
    no_sell_quantity_preview = build_execution_preview(
        report_payload={"trade_decisions": [{"ticker": "ADSK", "action": "SELL", "score": -0.9}]},
        config=_config(),
        instruments=_instruments(),
        positions=({"instrument": {"ticker": "ADSK_US_EQ"}, "quantityAvailableForTrading": 0.0},),
        now=_now(),
    )

    assert _orders(active_order_preview)[0]["reason"] == "active_order_already_exists_for_ticker"
    assert _orders(held_preview)[0]["reason"] == "ticker_already_held"
    assert _orders(stale_price_preview)[0]["reason"] == "missing_fresh_price_for_buy_quantity"
    assert _orders(zero_quantity_preview)[0]["reason"] == "calculated_buy_quantity_not_positive"
    assert _orders(no_sell_quantity_preview)[0]["reason"] == "no_quantity_available_for_trading"


def test_submit_gates_block_before_any_market_order(tmp_path: Path) -> None:
    preview = _preview()
    client = FakeTrading212Gateway()
    ledger_path = tmp_path / "ledger.jsonl"

    no_execute = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=client,
        ledger_path=ledger_path,
        execute=False,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )
    live_disabled = submit_execution_preview(
        preview=preview,
        config=_config(live=False),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )
    bad_token = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token="wrong-token",
        now=_now(),
    )

    assert no_execute["reason"] == "execute_flag_not_set"
    assert live_disabled["reason"] == "trading_live_enabled_false"
    assert bad_token["reason"] == "approval_token_mismatch"
    assert client.market_orders == []
    assert not ledger_path.exists()


def test_submit_blocks_stale_preview_caps_missing_credentials_and_fresh_state(tmp_path: Path) -> None:
    preview = _preview()
    client = FakeTrading212Gateway()
    ledger_path = tmp_path / "ledger.jsonl"
    missing_credentials = submit_execution_preview(
        preview=preview,
        config=_config(api=False, live=True),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )
    expired = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now() + timedelta(minutes=31),
    )
    max_orders = submit_execution_preview(
        preview=preview,
        config=_config(live=True, max_daily_orders=1),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )
    max_value = submit_execution_preview(
        preview=preview,
        config=_config(live=True, max_order_value=5.0),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )
    active_order = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=FakeTrading212Gateway(active_orders_payload=[{"ticker": "AVGO_US_EQ"}]),
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )
    missing_sell = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=FakeTrading212Gateway(positions_payload=[]),
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now(),
    )

    assert missing_credentials["reason"] == "missing_trading212_credentials"
    assert expired["reason"] == "preview_expired"
    assert max_orders["reason"] == "max_daily_orders_exceeded"
    assert max_value["reason"] == "max_order_value_exceeded"
    assert active_order["reason"] == "active_order_detected_before_submit"
    assert missing_sell["reason"] == "sell_quantity_no_longer_available"
    assert client.market_orders == []
    assert not ledger_path.exists()


def test_submit_records_ledger_reconciles_and_blocks_duplicate_replay(tmp_path: Path) -> None:
    preview = _preview()
    client = FakeTrading212Gateway()
    ledger_path = tmp_path / "ledger.jsonl"

    result = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now() + timedelta(minutes=1),
    )
    replay = submit_execution_preview(
        preview=preview,
        config=_config(live=True),
        client=client,
        ledger_path=ledger_path,
        execute=True,
        approval_token=str(preview["approval_token"]),
        now=_now() + timedelta(minutes=2),
    )

    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    reconciliation = result["reconciliation"]
    assert isinstance(reconciliation, dict)
    assert result["status"] == "submitted"
    assert result["orders_submitted"] is True
    assert result["orders_attempted"] == 2
    assert reconciliation["history_orders_status_code"] == 200
    assert client.market_orders == [("AVGO_US_EQ", 0.5), ("ADSK_US_EQ", -2.5)]
    assert [row["status"] for row in ledger_rows] == ["intended", "accepted", "intended", "accepted"]
    assert {row["preview_id"] for row in ledger_rows} == {preview["preview_id"]}
    assert replay["status"] == "blocked"
    assert replay["reason"] == "duplicate_execution_intent_blocked"
    assert client.market_orders == [("AVGO_US_EQ", 0.5), ("ADSK_US_EQ", -2.5)]


def test_live_execution_support_helpers_cover_edge_cases(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ready_order = {"preview_id": "preview-1", "intent_id": "intent-1"}
    ledger_path.write_text(
        "\n".join(
            (
                "",
                "{not-json",
                json.dumps({"preview_id": "preview-1", "intent_id": "intent-1", "status": "failed"}),
            )
        ),
        encoding="utf-8",
    )

    assert support.has_duplicate_ledger_entry(tmp_path / "missing.jsonl", [ready_order]) is False
    assert support.has_duplicate_ledger_entry(ledger_path, [ready_order]) is False
    support.append_ledger(ledger_path, {"preview_id": "preview-1", "intent_id": "intent-1", "status": "accepted"})
    assert support.has_duplicate_ledger_entry(ledger_path, [ready_order]) is True
    assert support.resolve_instrument("NOPE", _instruments()) == (None, "instrument_not_found")
    assert support.resolve_instrument(
        "DUP",
        (
            {"ticker": "DUP_A_EQ", "shortName": "DUP", "type": "STOCK"},
            {"ticker": "DUP_B_EQ", "shortName": "DUP", "type": "ETF"},
        ),
    ) == (None, "ambiguous_instrument_match")
    assert support.position_map(({"ticker": "DIRECT_EQ"},))["DIRECT_EQ"]["ticker"] == "DIRECT_EQ"
    assert support.active_order_tickers(({"instrument": {"ticker": "NESTED_EQ"}},)) == {"NESTED_EQ"}
    assert support.preview_expired({"expires_at": "bad-date"}, _now()) is True
    assert support.cash_available({"freeFunds": 12.5}) == 12.5
    assert support.response_status(Trading212Response("POST", "/x", 300, None, {})) == "failed"
    assert support.broker_order_id({"orderId": "order-2"}) == "order-2"
    assert support.sequence({"not": "a-list"}) == ()
    assert support.mapping([("not", "mapping")]) == {}
    assert support.text("   ") is None
    try:
        support.required_text(None, "fixture")
    except ValueError as exc:
        assert "fixture" in str(exc)
    else:
        raise AssertionError("required_text should reject empty values")


class FakeTrading212Gateway:
    def __init__(
        self,
        *,
        positions_payload: list[JsonValue] | None = None,
        active_orders_payload: list[JsonValue] | None = None,
    ) -> None:
        self.market_orders: list[tuple[str, float]] = []
        self._positions_payload = positions_payload
        self._active_orders_payload = active_orders_payload

    def account_summary(self) -> Trading212Response:
        return Trading212Response("GET", "/equity/account/summary", 200, {"cash": {"availableToTrade": 100.0}}, {})

    def instruments(self) -> Trading212Response:
        return Trading212Response(
            "GET",
            "/equity/metadata/instruments",
            200,
            cast("JsonValue", list(_instruments())),
            {},
        )

    def positions(self, *, ticker: str | None = None) -> Trading212Response:
        del ticker
        payload = (
            self._positions_payload
            if self._positions_payload is not None
            else cast("list[JsonValue]", list(_positions()))
        )
        return Trading212Response("GET", "/equity/positions", 200, cast("JsonValue", payload), {})

    def active_orders(self) -> Trading212Response:
        payload = self._active_orders_payload if self._active_orders_payload is not None else []
        return Trading212Response("GET", "/equity/orders", 200, cast("JsonValue", payload), {})

    def history_orders(self, *, limit: int = 50) -> Trading212Response:
        del limit
        return Trading212Response("GET", "/equity/history/orders?limit=50", 200, [{"id": "broker-1"}], {})

    def place_market_order(self, *, ticker: str, quantity: float) -> Trading212Response:
        self.market_orders.append((ticker, quantity))
        return Trading212Response("POST", "/equity/orders/market", 200, {"id": f"broker-{len(self.market_orders)}"}, {})


def _preview() -> dict[str, object]:
    return build_execution_preview(
        report_payload={
            "trade_decisions": [
                {"ticker": "AVGO", "action": "BUY", "score": 0.95, "rubric_version": "fixture"},
                {"ticker": "ADSK", "action": "SELL", "score": -0.5, "rubric_version": "fixture"},
                {"ticker": "MSFT", "action": "SELL", "score": -0.5, "rubric_version": "fixture"},
                {"ticker": "HOLDME", "action": "HOLD", "score": 0.1, "rubric_version": "fixture"},
            ]
        },
        config=_config(),
        instruments=_instruments(),
        positions=_positions(),
        active_orders=(),
        price_lookup=lambda _symbol, _currency: PriceSnapshot(20.0, "fixture_price", "2026-06-02"),
        now=_now(),
    )


def _instruments() -> tuple[dict[str, object], ...]:
    return (
        {"ticker": "AVGO_US_EQ", "shortName": "AVGO", "currencyCode": "USD", "type": "STOCK"},
        {"ticker": "ADSK_US_EQ", "shortName": "ADSK", "currencyCode": "USD", "type": "STOCK"},
        {"ticker": "MSFT_US_EQ", "shortName": "MSFT", "currencyCode": "USD", "type": "STOCK"},
    )


def _positions() -> tuple[dict[str, object], ...]:
    return (
        {
            "instrument": {"ticker": "ADSK_US_EQ"},
            "quantityAvailableForTrading": 2.5,
            "currentPrice": 100.0,
            "createdAt": "2026-06-02T08:00:00Z",
        },
    )


def _config(
    *,
    api: bool = True,
    live: bool = False,
    max_daily_orders: int | None = 5,
    max_order_value: float | None = 500.0,
) -> AppConfig:
    return AppConfig(
        legacy_stock_project_root=Path("legacy"),
        alpha_vantage_api_key=None,
        finnhub_api_key=None,
        openai_api_key=None,
        openai_model=None,
        trading212_api_key="key" if api else None,
        trading212_api_secret="secret" if api else None,
        trading212_env="demo",
        trading_live_enabled=live,
        trading_require_manual_approval=True,
        trading_max_daily_orders=max_daily_orders,
        trading_max_order_value=max_order_value,
        telegram_bot_token=None,
        telegram_chat_id=None,
        telegram_message_thread_id=None,
        market_data_enabled=False,
        market_data_cache_dir=None,
        market_data_usd_gbp_rate=0.8,
    )


def _now() -> datetime:
    return datetime(2026, 6, 2, 10, 0, tzinfo=UTC)


def _orders(preview: dict[str, object]) -> list[dict[str, object]]:
    orders = preview["orders"]
    assert isinstance(orders, list)
    assert all(isinstance(order, dict) for order in orders)
    return orders
