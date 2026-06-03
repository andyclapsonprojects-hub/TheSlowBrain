"""Centralized environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .promotion import DEFAULT_REQUIRED_STREAK, LADDER, PromotionStage


@dataclass(frozen=True)
class AppConfig:
    legacy_stock_project_root: Path
    alpha_vantage_api_key: str | None
    finnhub_api_key: str | None
    openai_api_key: str | None
    openai_model: str | None
    trading212_api_key: str | None
    trading212_api_secret: str | None
    trading212_env: str
    trading_live_enabled: bool
    trading_require_manual_approval: bool
    trading_max_daily_orders: int | None
    trading_max_order_value: float | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    telegram_message_thread_id: str | None
    market_data_enabled: bool = False
    market_data_cache_dir: Path | None = None
    market_data_usd_gbp_rate: float | None = None
    market_data_benchmark_symbol: str = "SPY"
    gating_stage_override: PromotionStage | None = None
    gating_required_streak: int = DEFAULT_REQUIRED_STREAK
    gating_label_mode: str = "absolute_return"
    pit_enrichment_enabled: bool = False
    pit_enrichment_path: Path | None = None


def load_config(project_root: Path | None = None) -> AppConfig:
    load_dotenv(project_root=project_root)
    legacy_root = os.environ.get(
        "SLOWBRAIN_LEGACY_STOCK_PROJECT",
        "C:/Users/AndyC/projects/n8n original Stock Trader",
    )
    return AppConfig(
        legacy_stock_project_root=Path(legacy_root),
        alpha_vantage_api_key=_optional_env("ALPHA_VANTAGE_API_KEY"),
        finnhub_api_key=_optional_env("FINNHUB_API_KEY"),
        openai_api_key=_optional_env("OPENAI_API_KEY"),
        openai_model=_optional_env("OPENAI_MODEL"),
        trading212_api_key=_optional_env("TRADING212_API_KEY"),
        trading212_api_secret=_optional_env("TRADING212_API_SECRET"),
        trading212_env=os.environ.get("TRADING212_ENV", "demo"),
        trading_live_enabled=_bool_env("TRADING_LIVE_ENABLED", default=False),
        trading_require_manual_approval=_bool_env("TRADING_REQUIRE_MANUAL_APPROVAL", default=True),
        trading_max_daily_orders=_int_env("TRADING_MAX_DAILY_ORDERS"),
        trading_max_order_value=_float_env("TRADING_MAX_ORDER_VALUE"),
        telegram_bot_token=_optional_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_optional_env("TELEGRAM_CHAT_ID"),
        telegram_message_thread_id=_optional_env("TELEGRAM_MESSAGE_THREAD_ID"),
        market_data_enabled=_bool_env("SLOWBRAIN_MARKET_DATA_ENABLED", default=True),
        market_data_cache_dir=_optional_path_env("SLOWBRAIN_MARKET_DATA_CACHE_DIR"),
        market_data_usd_gbp_rate=_positive_float_env("SLOWBRAIN_MARKET_DATA_USD_GBP_RATE"),
        market_data_benchmark_symbol=os.environ.get("SLOWBRAIN_MARKET_DATA_BENCHMARK_SYMBOL", "SPY").strip().upper()
        or "SPY",
        gating_stage_override=_gating_stage_override(),
        gating_required_streak=_positive_int_env("SLOWBRAIN_GATING_REQUIRED_STREAK", default=DEFAULT_REQUIRED_STREAK),
        gating_label_mode=_gating_label_mode(),
        pit_enrichment_enabled=_bool_env("SLOWBRAIN_PIT_ENRICHMENT_ENABLED", default=False),
        pit_enrichment_path=_optional_path_env("SLOWBRAIN_PIT_ENRICHMENT_PATH"),
    )


def _gating_stage_override() -> PromotionStage | None:
    value = os.environ.get("SLOWBRAIN_GATING_STAGE_OVERRIDE")
    if value is None:
        return None
    text = value.strip().lower()
    for stage in LADDER:
        if text == stage:
            return stage
    return None


def _positive_int_env(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _gating_label_mode() -> str:
    value = os.environ.get("SLOWBRAIN_GATING_LABEL_MODE", "absolute_return").strip().lower()
    if value in {"absolute_return", "cross_sectional_rank"}:
        return value
    return "absolute_return"


def load_dotenv(project_root: Path | None = None) -> None:
    env_path = (project_root or _project_root()) / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_quotes(value.strip()))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return float(value)


def _positive_float_env(name: str) -> float | None:
    value = _float_env(name)
    return value if value is not None and value > 0.0 else None


def _optional_path_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value and value.strip() else None
