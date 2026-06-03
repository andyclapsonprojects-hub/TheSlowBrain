"""Provider adapters and provider-chain builders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode

from ..config import AppConfig
from ..market_data import (
    BenchmarkReturn,
    DailyPrice,
    FallbackMarketDataProvider,
    FallbackPriceHistoryProvider,
    LiquiditySnapshot,
    MarketDataProvider,
    PriceHistoryProvider,
    UniverseMembership,
)
from ..market_data_cache import JsonValue, MarketDataCache
from ..models import FeatureVector
from .parsers import (
    _alpha_payload_is_miss,
    _finnhub_payload_is_miss,
    _parse_alpha_daily_payload,
    _parse_finnhub_candle_payload,
    _parse_finnhub_quote_payload,
    _parse_yahoo_chart_payload,
    _yahoo_payload_is_miss,
)
from .pricing import _benchmark_return_from_prices, _liquidity_from_prices, _prices_on_or_before
from .types import (
    ALPHA_VANTAGE_BASE_URL,
    DEFAULT_HISTORY_START,
    FINNHUB_BASE_URL,
    YAHOO_BASE_URL,
    FinnhubQuote,
    FxRate,
    FxRateSource,
    MarketDataTransport,
    StdlibHttpTransport,
    VendorProvider,
    _is_json_value,
)


class AlphaVantageProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        cache: MarketDataCache,
        benchmark_symbol: str,
        quote_to_gbp_rate: FxRateSource,
        transport: MarketDataTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._benchmark_symbol = benchmark_symbol
        self._quote_to_gbp_rate = quote_to_gbp_rate
        self._transport = transport or StdlibHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._series_cache: dict[str, tuple[DailyPrice, ...]] = {}

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        return _benchmark_return_from_prices(
            self._benchmark_symbol,
            feature.signal_date,
            self.daily_prices(self._benchmark_symbol),
            source="alpha_vantage:daily_adjusted",
        )

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        return _liquidity_from_prices(
            feature.ticker,
            feature.signal_date,
            self.daily_prices(feature.ticker),
            quote_to_gbp_rate=self._quote_to_gbp_rate,
            source="alpha_vantage:daily_adjusted",
        )

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        return None

    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        normalized = symbol.upper()
        if normalized in self._series_cache:
            return self._series_cache[normalized]
        if not self._api_key:
            self._series_cache[normalized] = ()
            return ()
        payload = self._cache.get_or_fetch(normalized, lambda: self._fetch_daily_payload(normalized))
        prices = _parse_alpha_daily_payload(normalized, payload)
        self._series_cache[normalized] = prices
        return prices

    def _fetch_daily_payload(self, symbol: str) -> JsonValue | None:
        params = urlencode(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self._api_key or "",
            }
        )
        payload = _safe_transport_call(self._transport, f"{ALPHA_VANTAGE_BASE_URL}?{params}", self._timeout_seconds)
        if _alpha_payload_is_miss(payload):
            return None
        return payload


class FinnhubProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        cache: MarketDataCache,
        benchmark_symbol: str,
        quote_to_gbp_rate: FxRateSource,
        transport: MarketDataTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._benchmark_symbol = benchmark_symbol
        self._quote_to_gbp_rate = quote_to_gbp_rate
        self._transport = transport or StdlibHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._series_cache: dict[str, tuple[DailyPrice, ...]] = {}

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        return _benchmark_return_from_prices(
            self._benchmark_symbol,
            feature.signal_date,
            self.daily_prices(self._benchmark_symbol),
            source="finnhub:stock_candle",
        )

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        return _liquidity_from_prices(
            feature.ticker,
            feature.signal_date,
            self.daily_prices(feature.ticker),
            quote_to_gbp_rate=self._quote_to_gbp_rate,
            source="finnhub:stock_candle",
        )

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        return None

    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        normalized = symbol.upper()
        if normalized in self._series_cache:
            return self._series_cache[normalized]
        if not self._api_key:
            self._series_cache[normalized] = ()
            return ()
        payload = self._cache.get_or_fetch(normalized, lambda: self._fetch_candle_payload(normalized))
        prices = _parse_finnhub_candle_payload(normalized, payload)
        self._series_cache[normalized] = prices
        return prices

    def quote(self, symbol: str) -> FinnhubQuote | None:
        if not self._api_key:
            return None
        params = urlencode({"symbol": symbol.upper(), "token": self._api_key})
        payload = _safe_transport_call(self._transport, f"{FINNHUB_BASE_URL}/quote?{params}", self._timeout_seconds)
        return _parse_finnhub_quote_payload(symbol.upper(), payload)

    def _fetch_candle_payload(self, symbol: str) -> JsonValue | None:
        params = urlencode(
            {
                "symbol": symbol,
                "resolution": "D",
                "from": str(int(DEFAULT_HISTORY_START.timestamp())),
                "to": str(int(datetime.now(UTC).timestamp())),
                "token": self._api_key or "",
            }
        )
        payload = _safe_transport_call(
            self._transport,
            f"{FINNHUB_BASE_URL}/stock/candle?{params}",
            self._timeout_seconds,
        )
        if _finnhub_payload_is_miss(payload):
            return None
        return payload


class YahooProvider:
    def __init__(
        self,
        *,
        cache: MarketDataCache,
        benchmark_symbol: str,
        quote_to_gbp_rate: FxRateSource,
        transport: MarketDataTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._cache = cache
        self._benchmark_symbol = benchmark_symbol
        self._quote_to_gbp_rate = quote_to_gbp_rate
        self._transport = transport or StdlibHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._series_cache: dict[str, tuple[DailyPrice, ...]] = {}

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        return _benchmark_return_from_prices(
            self._benchmark_symbol,
            feature.signal_date,
            self.daily_prices(self._benchmark_symbol),
            source="yahoo:chart",
        )

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        return _liquidity_from_prices(
            feature.ticker,
            feature.signal_date,
            self.daily_prices(feature.ticker),
            quote_to_gbp_rate=self._quote_to_gbp_rate,
            source="yahoo:chart",
        )

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        return None

    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        normalized = symbol.upper()
        if normalized in self._series_cache:
            return self._series_cache[normalized]
        payload = self._cache.get_or_fetch(normalized, lambda: self._fetch_chart_payload(normalized))
        prices = _parse_yahoo_chart_payload(normalized, payload)
        self._series_cache[normalized] = prices
        return prices

    def _fetch_chart_payload(self, symbol: str) -> JsonValue | None:
        # The public v8 chart endpoint serves OHLCV without a crumb; only a browser User-Agent
        # (set in the transport) is required. The crumb handshake is unreliable and unnecessary here.
        params = urlencode(
            {
                "period1": str(int(DEFAULT_HISTORY_START.timestamp())),
                "period2": str(int(datetime.now(UTC).timestamp())),
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
        payload = _safe_transport_call(
            self._transport,
            f"{YAHOO_BASE_URL}/v8/finance/chart/{quote(symbol)}?{params}",
            self._timeout_seconds,
        )
        if _yahoo_payload_is_miss(payload):
            return None
        return payload


class YahooUsdGbpRateProvider:
    def __init__(
        self,
        *,
        cache: MarketDataCache,
        transport: MarketDataTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._provider = YahooProvider(
            cache=cache,
            benchmark_symbol="GBPUSD=X",
            quote_to_gbp_rate=None,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )

    def __call__(self, signal_date: str) -> FxRate | None:
        prices = _prices_on_or_before(self._provider.daily_prices("GBPUSD=X"), signal_date)
        if not prices:
            return None
        gbp_usd = prices[-1].adjusted_close
        if gbp_usd <= 0.0:
            return None
        return FxRate(rate=1.0 / gbp_usd, source="yahoo_gbpusd_asof")


def build_market_data_provider(
    config: AppConfig,
    *,
    project_root: Path,
    transport: MarketDataTransport | None = None,
) -> MarketDataProvider | None:
    """Build the optional read-only fallback provider chain from central config."""
    if not config.market_data_enabled:
        return None
    providers = _build_vendor_provider_chain(config, project_root=project_root, transport=transport)
    return FallbackMarketDataProvider(tuple(providers))


def build_price_history_provider(
    config: AppConfig,
    *,
    project_root: Path,
    transport: MarketDataTransport | None = None,
) -> PriceHistoryProvider | None:
    """Build the optional read-only daily price-history provider chain."""
    if not config.market_data_enabled:
        return None
    providers = _build_vendor_provider_chain(config, project_root=project_root, transport=transport)
    return FallbackPriceHistoryProvider(tuple(providers))


def _build_vendor_provider_chain(
    config: AppConfig,
    *,
    project_root: Path,
    transport: MarketDataTransport | None,
) -> list[VendorProvider]:
    root = config.market_data_cache_dir or (project_root / "data" / "market_data_cache")
    fx_rate_source = _build_usd_gbp_rate_source(config, root, transport)
    providers: list[VendorProvider] = []
    if config.alpha_vantage_api_key:
        providers.append(
            AlphaVantageProvider(
                api_key=config.alpha_vantage_api_key,
                cache=MarketDataCache(root, "alpha_vantage"),
                benchmark_symbol=config.market_data_benchmark_symbol,
                quote_to_gbp_rate=fx_rate_source,
                transport=transport,
            )
        )
    if config.finnhub_api_key:
        providers.append(
            FinnhubProvider(
                api_key=config.finnhub_api_key,
                cache=MarketDataCache(root, "finnhub"),
                benchmark_symbol=config.market_data_benchmark_symbol,
                quote_to_gbp_rate=fx_rate_source,
                transport=transport,
            )
        )
    providers.append(
        YahooProvider(
            cache=MarketDataCache(root, "yahoo"),
            benchmark_symbol=config.market_data_benchmark_symbol,
            quote_to_gbp_rate=fx_rate_source,
            transport=transport,
        )
    )
    return providers


def _build_usd_gbp_rate_source(
    config: AppConfig,
    cache_root: Path,
    transport: MarketDataTransport | None,
) -> FxRateSource:
    if config.market_data_usd_gbp_rate is not None:
        return config.market_data_usd_gbp_rate
    return YahooUsdGbpRateProvider(
        cache=MarketDataCache(cache_root, "yahoo_fx"),
        transport=transport,
    )


def _safe_transport_call(transport: MarketDataTransport, url: str, timeout_seconds: float) -> JsonValue | None:
    try:
        payload = transport(url, timeout_seconds)
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    return payload if _is_json_value(payload) else None
