"""Build a no-submit Trading 212 execution preview from the latest Slow Brain report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import AppConfig, load_config
from slowbrain.live_execution import (
    BUY_NOTIONAL_GBP,
    LATEST_PREVIEW_JSON,
    PriceSnapshot,
    build_execution_preview,
    load_json_object,
    write_json,
)
from slowbrain.market_data import PriceHistoryProvider
from slowbrain.market_data_vendors import build_price_history_provider
from slowbrain.trading212 import Trading212Gateway, build_trading212_client, credentials_available, response_summary
from slowbrain.workflow import FIRST_REPORT_JSON

ClientFactory = Callable[[AppConfig], Trading212Gateway]


def main(argv: Sequence[str] | None = None, *, client_factory: ClientFactory | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=FIRST_REPORT_JSON)
    parser.add_argument("--out", type=Path, default=LATEST_PREVIEW_JSON)
    parser.add_argument("--buy-notional-gbp", type=float, default=BUY_NOTIONAL_GBP)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(project_root=args.project_root)
    report_payload = load_json_object(args.project_root / args.report)
    client = _build_client(config, client_factory)
    preview = (
        _blocked_preview(config, "missing_trading212_credentials")
        if client is None
        else _build_preview(args.project_root, config, report_payload, client, args.buy_notional_gbp)
    )
    out_path = args.project_root / args.out
    write_json(out_path, preview)
    print("TheSlowBrain live execution preview complete.")
    print(f"Status: {preview.get('status')}")
    print(f"Ready orders: {preview.get('ready_order_count', 0)}")
    print(f"Blocked orders: {preview.get('blocked_order_count', 0)}")
    print(f"Orders submitted: {str(preview.get('orders_submitted', False)).lower()}")
    if preview.get("approval_token"):
        print(f"Approval token: {preview['approval_token']}")
    print(f"Preview: {out_path}")
    return 0 if preview.get("status") in {"ready", "blocked"} else 2


def _build_client(config: AppConfig, client_factory: ClientFactory | None) -> Trading212Gateway | None:
    if not credentials_available(config):
        return None
    factory = client_factory or build_trading212_client
    return factory(config)


def _build_preview(
    project_root: Path,
    config: AppConfig,
    report_payload: dict[str, object],
    client: Trading212Gateway,
    buy_notional_gbp: float,
) -> dict[str, object]:
    instruments_response = client.instruments()
    positions_response = client.positions()
    active_orders_response = client.active_orders()
    broker_reads = (instruments_response, positions_response, active_orders_response)
    if any(response.status_code != 200 for response in broker_reads):
        return {
            **_blocked_preview(config, "broker_read_failed"),
            "responses": {
                "instruments": response_summary(instruments_response),
                "positions": response_summary(positions_response),
                "active_orders": response_summary(active_orders_response),
            },
        }
    price_provider = build_price_history_provider(config, project_root=project_root)
    return build_execution_preview(
        report_payload=report_payload,
        config=config,
        instruments=_sequence(instruments_response.payload),
        positions=_sequence(positions_response.payload),
        active_orders=_sequence(active_orders_response.payload),
        price_lookup=_price_lookup(config, price_provider),
        buy_notional_gbp=buy_notional_gbp,
    )


def _blocked_preview(config: AppConfig, reason: str) -> dict[str, object]:
    return {
        "schema": "theslowbrain.live_execution_preview.v1",
        "status": "blocked",
        "reason": reason,
        "environment": config.trading212_env,
        "broker_live_execution_allowed": False,
        "orders_submitted": False,
        "ready_order_count": 0,
        "blocked_order_count": 0,
        "orders": [],
        "approval_token": None,
    }


def _price_lookup(
    config: AppConfig,
    provider: PriceHistoryProvider | None,
) -> Callable[[str, str], PriceSnapshot | None] | None:
    if provider is None:
        return None

    def lookup(symbol: str, currency: str) -> PriceSnapshot | None:
        prices = provider.daily_prices(symbol)
        if not prices:
            return None
        latest = sorted(prices, key=lambda price: price.date)[-1]
        if latest.adjusted_close <= 0.0:
            return None
        fx_rate, fx_source = _gbp_rate(config, provider, currency)
        if fx_rate is None:
            return None
        return PriceSnapshot(
            price_gbp=round(latest.adjusted_close * fx_rate, 6),
            source=f"{latest.source}:latest_price:{fx_source}",
            as_of=latest.date,
        )

    return lookup


def _gbp_rate(
    config: AppConfig,
    provider: PriceHistoryProvider,
    currency: str,
) -> tuple[float | None, str]:
    normalized = currency.upper()
    if normalized in {"GBP", "GBX"}:
        return 1.0, f"{normalized.lower()}_native"
    if normalized != "USD":
        return None, "unsupported_currency"
    if config.market_data_usd_gbp_rate is not None:
        return config.market_data_usd_gbp_rate, "configured_usd_gbp"
    prices = provider.daily_prices("GBPUSD=X")
    if not prices:
        return None, "missing_gbpusd_fx"
    latest = sorted(prices, key=lambda price: price.date)[-1]
    if latest.adjusted_close <= 0.0:
        return None, "invalid_gbpusd_fx"
    return round(1.0 / latest.adjusted_close, 8), "yahoo_gbpusd_latest"


def _sequence(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple({str(key): item for key, item in row.items()} for row in value if isinstance(row, dict))


if __name__ == "__main__":
    raise SystemExit(main())
