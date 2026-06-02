from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from slowbrain.backtest import evaluate_rubric
from slowbrain.config import AppConfig
from slowbrain.costs import estimate_trade_cost
from slowbrain.market_data import (
    BenchmarkReturn,
    FallbackMarketDataProvider,
    LiquiditySnapshot,
    UniverseMembership,
    unique_feature_symbol_dates,
    warm_market_data_provider,
)
from slowbrain.market_data_cache import JsonValue, MarketDataCache
from slowbrain.market_data_vendors import (
    AlphaVantageProvider,
    FinnhubProvider,
    YahooProvider,
    build_market_data_provider,
)
from slowbrain.models import FeatureVector
from slowbrain.rubrics import BASE_RUBRIC


def test_market_data_cache_fetches_symbol_once_and_reuses_disk_payload(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path, "fixture")
    calls = 0

    def fetch() -> JsonValue:
        nonlocal calls
        calls += 1
        return {"prices": [1, 2, 3]}

    assert cache.get_or_fetch("BRK.B", fetch) == {"prices": [1, 2, 3]}
    assert cache.get_or_fetch("BRK.B", fetch) == {"prices": [1, 2, 3]}

    assert calls == 1
    assert cache.path_for_symbol("BRK.B").name == "BRK_B.json"


def test_market_data_cache_refetches_corrupt_payload(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path, "fixture")
    cache.path_for_symbol("AAPL").parent.mkdir(parents=True)
    cache.path_for_symbol("AAPL").write_text("{not-json", encoding="utf-8")

    assert cache.get_or_fetch("AAPL", lambda: {"ok": True}) == {"ok": True}


def test_unique_feature_symbol_dates_bounds_vendor_fetch_scope() -> None:
    first = feature("one", ticker="aapl", signal_date="2026-01-02")
    duplicate = feature("two", ticker="AAPL", signal_date="2026-01-02")
    second = feature("three", ticker="MSFT", signal_date="2026-01-03")

    assert unique_feature_symbol_dates((first, duplicate, second)) == (
        ("AAPL", "2026-01-02"),
        ("MSFT", "2026-01-03"),
    )


def test_warm_market_data_provider_fetches_unique_feature_pairs_once() -> None:
    provider = CountingProvider()
    first = feature("one", ticker="AAPL", signal_date="2026-01-02")
    duplicate = feature("two", ticker="AAPL", signal_date="2026-01-02")
    second = feature("three", ticker="MSFT", signal_date="2026-01-03")

    warm_market_data_provider(provider, (first, duplicate, second))

    assert provider.benchmark_calls == 2
    assert provider.liquidity_calls == 2
    assert provider.universe_calls == 2


def test_alpha_vantage_daily_adjusted_fixture_supplies_benchmark_and_liquidity(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "symbol=SPY": alpha_payload("SPY"),
            "symbol=AAPL": alpha_payload("AAPL"),
        }
    )
    provider = AlphaVantageProvider(
        api_key="fixture-key",
        cache=MarketDataCache(tmp_path, "alpha"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=transport,
    )

    benchmark = provider.benchmark_return(feature("one", signal_date="2026-01-03"))
    liquidity = provider.liquidity_snapshot(feature("two", signal_date="2026-01-03"))

    assert benchmark is not None
    assert round(benchmark.return_pct, 4) == 1.0
    assert benchmark.source == "alpha_vantage:daily_adjusted"
    assert liquidity is not None
    assert liquidity.source == "alpha_vantage:daily_adjusted:fx_configured_usd_gbp:spread_proxy_5.0bps"
    assert liquidity.avg_daily_volume_gbp > 0
    assert len([url for url in transport.urls if "symbol=AAPL" in url]) == 1


def test_alpha_vantage_throttle_and_no_key_return_misses(tmp_path: Path) -> None:
    throttled = AlphaVantageProvider(
        api_key="fixture-key",
        cache=MarketDataCache(tmp_path, "alpha"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=FakeTransport({"TIME_SERIES_DAILY_ADJUSTED": {"Note": "rate limit"}}),
    )
    no_key = AlphaVantageProvider(
        api_key=None,
        cache=MarketDataCache(tmp_path, "alpha-empty"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=FakeTransport({}),
    )

    assert throttled.benchmark_return(feature("one", signal_date="2026-01-03")) is None
    assert no_key.benchmark_return(feature("one", signal_date="2026-01-03")) is None


def test_finnhub_candle_success_no_data_and_quote_fixture(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "/stock/candle": finnhub_candle_payload(),
            "/quote": {
                "c": 151.55,
                "d": 1.08,
                "dp": 0.7178,
                "h": 153.4,
                "l": 150.43,
                "o": 151.28,
                "pc": 150.47,
                "t": 1,
            },
        }
    )
    provider = FinnhubProvider(
        api_key="fixture-key",
        cache=MarketDataCache(tmp_path, "finnhub"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=transport,
    )
    missed = FinnhubProvider(
        api_key="fixture-key",
        cache=MarketDataCache(tmp_path, "finnhub-miss"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=FakeTransport({"/stock/candle": {"s": "no_data"}}),
    )

    assert provider.benchmark_return(feature("one", signal_date="2026-01-03")) is not None
    assert provider.quote("AAPL") is not None
    assert missed.benchmark_return(feature("one", signal_date="2026-01-03")) is None


def test_yahoo_crumb_handshake_and_chart_parse(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "/v1/test/getcrumb": "fixture-crumb",
            "/v8/finance/chart/SPY": yahoo_chart_payload(),
        }
    )
    provider = YahooProvider(
        cache=MarketDataCache(tmp_path, "yahoo"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=transport,
    )

    benchmark = provider.benchmark_return(feature("one", signal_date="2026-01-03"))

    assert benchmark is not None
    assert benchmark.source == "yahoo:chart"
    assert any("/v1/test/getcrumb" in url for url in transport.urls)
    assert any("crumb=fixture-crumb" in url for url in transport.urls)


def test_yahoo_malformed_or_throttled_response_misses(tmp_path: Path) -> None:
    provider = YahooProvider(
        cache=MarketDataCache(tmp_path, "yahoo"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=FakeTransport({"/v1/test/getcrumb": "Too Many Requests"}),
    )

    assert provider.benchmark_return(feature("one", signal_date="2026-01-03")) is None


def test_fallback_provider_preserves_first_available_vendor_source() -> None:
    feature_row = feature("one", signal_date="2026-01-03")
    provider = FallbackMarketDataProvider(
        (
            FixtureProvider(benchmark=None, liquidity=None),
            FixtureProvider(
                benchmark=BenchmarkReturn("SPY", "2026-01-03", 0.5, "finnhub"),
                liquidity=LiquiditySnapshot("AAPL", 1_000_000, 1.0, 4.0, "finnhub"),
            ),
            FixtureProvider(
                benchmark=BenchmarkReturn("SPY", "2026-01-03", 0.7, "yahoo"),
                liquidity=LiquiditySnapshot("AAPL", 2_000_000, 1.0, 4.0, "yahoo"),
            ),
        )
    )

    benchmark = provider.benchmark_return(feature_row)
    liquidity = provider.liquidity_snapshot(feature_row)

    assert benchmark is not None
    assert liquidity is not None
    assert benchmark.source == "finnhub"
    assert liquidity.source == "finnhub"


def test_vendor_provider_flips_backtest_benchmark_and_liquidity_quality(tmp_path: Path) -> None:
    transport = FakeTransport(
        {
            "symbol=SPY": alpha_payload("SPY"),
            "symbol=AAPL": alpha_payload("AAPL"),
        }
    )
    provider = AlphaVantageProvider(
        api_key="fixture-key",
        cache=MarketDataCache(tmp_path, "alpha"),
        benchmark_symbol="SPY",
        quote_to_gbp_rate=0.80,
        transport=transport,
    )
    features = tuple(feature(f"row-{index}", signal_date="2026-01-03") for index in range(12))

    result = evaluate_rubric(features, BASE_RUBRIC, min_test_trades=0, market_data_provider=provider)
    cost = estimate_trade_cost(features[0], liquidity_snapshot=provider.liquidity_snapshot(features[0]))

    assert result.benchmark_quality == "provider_real_benchmark"
    assert result.liquidity_quality == "provider_real_per_name_liquidity"
    proxy_alpha = evaluate_rubric(features, BASE_RUBRIC, min_test_trades=0).alpha_vs_benchmark_pct
    assert result.alpha_vs_benchmark_pct != proxy_alpha
    assert cost.spread_bps == 5.0


def test_provider_builder_can_be_explicitly_disabled_for_deterministic_tests(tmp_path: Path) -> None:
    config = app_config(alpha_key="present", finnhub_key="present", market_data_enabled=False)

    assert build_market_data_provider(config, project_root=tmp_path, transport=FakeTransport({})) is None


def test_provider_builder_is_automatic_by_default_with_yahoo_benchmark_and_fx(tmp_path: Path) -> None:
    config = app_config()
    provider = build_market_data_provider(
        config,
        project_root=tmp_path,
        transport=FakeTransport(
            {
                "/v1/test/getcrumb": "fixture-crumb",
                "/v8/finance/chart/SPY": yahoo_chart_payload(),
                "/v8/finance/chart/AAPL": yahoo_chart_payload(),
                "/v8/finance/chart/GBPUSD%3DX": yahoo_fx_chart_payload(),
            }
        ),
    )
    feature_row = feature("one", signal_date="2026-01-03")

    assert provider is not None
    assert provider.benchmark_return(feature_row) is not None
    liquidity = provider.liquidity_snapshot(feature_row)
    assert liquidity is not None
    assert "fx_yahoo_gbpusd_asof" in liquidity.source


class FakeTransport:
    def __init__(self, responses: dict[str, JsonValue]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str, timeout_seconds: float) -> JsonValue:
        self.urls.append(url)
        for needle, response in self.responses.items():
            if needle in url:
                return response
        raise AssertionError(f"unexpected market-data URL: {url}")


class FixtureProvider:
    def __init__(self, *, benchmark: BenchmarkReturn | None, liquidity: LiquiditySnapshot | None) -> None:
        self.benchmark = benchmark
        self.liquidity = liquidity

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        return self.benchmark

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        return self.liquidity

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        return None


class CountingProvider:
    def __init__(self) -> None:
        self.benchmark_calls = 0
        self.liquidity_calls = 0
        self.universe_calls = 0

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        self.benchmark_calls += 1
        return None

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        self.liquidity_calls += 1
        return None

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        self.universe_calls += 1
        return None


def app_config(
    *,
    alpha_key: str | None = None,
    finnhub_key: str | None = None,
    market_data_enabled: bool = True,
    usd_gbp_rate: float | None = None,
) -> AppConfig:
    return AppConfig(
        legacy_stock_project_root=Path("legacy"),
        alpha_vantage_api_key=alpha_key,
        finnhub_api_key=finnhub_key,
        openai_api_key=None,
        openai_model=None,
        trading212_api_key=None,
        trading212_api_secret=None,
        trading212_env="demo",
        trading_live_enabled=False,
        trading_require_manual_approval=True,
        trading_max_daily_orders=None,
        trading_max_order_value=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        telegram_message_thread_id=None,
        market_data_enabled=market_data_enabled,
        market_data_usd_gbp_rate=usd_gbp_rate,
    )


def feature(
    idea_id: str,
    *,
    ticker: str = "AAPL",
    signal_date: str = "2026-01-03",
    net: float = 1.0,
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker=ticker,
        signal_date=signal_date,
        sentiment="positive",
        sentiment_confidence=0.9,
        catalyst_strength=0.9,
        trend="uptrend",
        momentum_20d_pct=8.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=10.0,
        source="fixture",
    )


def alpha_payload(symbol: str) -> JsonValue:
    series: dict[str, JsonValue] = {
        "2026-01-03": alpha_bar(102.01, 1_020_000),
        "2026-01-02": alpha_bar(101.00, 1_010_000),
        "2026-01-01": alpha_bar(100.00, 1_000_000),
    }
    return {
        "Meta Data": {"2. Symbol": symbol},
        "Time Series (Daily)": series,
    }


def alpha_bar(close: float, volume: int) -> dict[str, JsonValue]:
    return {
        "1. open": str(close - 1.0),
        "2. high": str(close + 1.0),
        "3. low": str(close - 2.0),
        "4. close": str(close),
        "5. adjusted close": str(close),
        "6. volume": str(volume),
    }


def finnhub_candle_payload() -> JsonValue:
    return {
        "s": "ok",
        "t": [epoch("2026-01-01"), epoch("2026-01-02"), epoch("2026-01-03")],
        "o": [99.0, 100.0, 101.0],
        "h": [101.0, 102.0, 103.0],
        "l": [98.0, 99.0, 100.0],
        "c": [100.0, 101.0, 102.01],
        "v": [1_000_000, 1_010_000, 1_020_000],
    }


def yahoo_chart_payload() -> JsonValue:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [epoch("2026-01-01"), epoch("2026-01-02"), epoch("2026-01-03")],
                    "indicators": {
                        "quote": [
                            {
                                "open": [99.0, 100.0, 101.0],
                                "high": [101.0, 102.0, 103.0],
                                "low": [98.0, 99.0, 100.0],
                                "close": [100.0, 101.0, 102.01],
                                "volume": [1_000_000, 1_010_000, 1_020_000],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.0, 101.0, 102.01]}],
                    },
                }
            ],
            "error": None,
        }
    }


def yahoo_fx_chart_payload() -> JsonValue:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [epoch("2026-01-01"), epoch("2026-01-02"), epoch("2026-01-03")],
                    "indicators": {
                        "quote": [
                            {
                                "open": [1.24, 1.24, 1.25],
                                "high": [1.26, 1.26, 1.27],
                                "low": [1.23, 1.23, 1.24],
                                "close": [1.25, 1.25, 1.25],
                                "volume": [0, 0, 0],
                            }
                        ],
                        "adjclose": [{"adjclose": [1.25, 1.25, 1.25]}],
                    },
                }
            ],
            "error": None,
        }
    }


def epoch(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp())
