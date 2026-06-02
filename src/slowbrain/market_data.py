"""Market-data provider boundaries for benchmark, liquidity, and PIT universe evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .models import FeatureVector


@dataclass(frozen=True)
class BenchmarkReturn:
    ticker: str
    signal_date: str
    return_pct: float
    source: str


@dataclass(frozen=True)
class LiquiditySnapshot:
    ticker: str
    avg_daily_volume_gbp: float
    volatility_pct: float
    spread_bps: float
    source: str


@dataclass(frozen=True)
class UniverseMembership:
    ticker: str
    signal_date: str
    is_member: bool
    is_delisted: bool
    source: str


@dataclass(frozen=True)
class DailyPrice:
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float
    source: str


class PriceHistoryProvider(Protocol):
    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        """Return daily OHLCV history for a symbol."""


class MarketDataProvider(Protocol):
    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        """Return benchmark evidence aligned to the feature date."""

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        """Return per-name liquidity evidence aligned to the feature date."""

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        """Return point-in-time universe membership for the feature."""


@dataclass(frozen=True)
class StaticMarketDataProvider:
    benchmark_returns: Mapping[tuple[str, str], BenchmarkReturn]
    liquidity_snapshots: Mapping[str, LiquiditySnapshot]
    universe_memberships: Mapping[tuple[str, str], UniverseMembership]

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        return self.benchmark_returns.get((feature.ticker, feature.signal_date))

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        return self.liquidity_snapshots.get(feature.ticker)

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        return self.universe_memberships.get((feature.ticker, feature.signal_date))


@dataclass(frozen=True)
class FallbackPriceHistoryProvider:
    providers: Sequence[PriceHistoryProvider]

    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        for provider in self.providers:
            prices = provider.daily_prices(symbol)
            if prices:
                return prices
        return ()


@dataclass(frozen=True)
class FallbackMarketDataProvider:
    providers: Sequence[MarketDataProvider]

    def benchmark_return(self, feature: FeatureVector) -> BenchmarkReturn | None:
        for provider in self.providers:
            result = provider.benchmark_return(feature)
            if result is not None:
                return result
        return None

    def liquidity_snapshot(self, feature: FeatureVector) -> LiquiditySnapshot | None:
        for provider in self.providers:
            result = provider.liquidity_snapshot(feature)
            if result is not None:
                return result
        return None

    def universe_membership(self, feature: FeatureVector) -> UniverseMembership | None:
        for provider in self.providers:
            result = provider.universe_membership(feature)
            if result is not None:
                return result
        return None


def unique_feature_symbol_dates(features: Sequence[FeatureVector]) -> tuple[tuple[str, str], ...]:
    """Return deterministic symbol/date pairs present in the feature evidence."""
    return tuple(
        sorted(
            {
                (feature.ticker.upper(), feature.signal_date)
                for feature in features
                if feature.ticker.strip() and feature.signal_date.strip()
            }
        )
    )


def warm_market_data_provider(provider: MarketDataProvider | None, features: Sequence[FeatureVector]) -> None:
    """Warm bounded provider evidence for the feature set before scoring."""
    if provider is None:
        return
    representatives: dict[tuple[str, str], FeatureVector] = {}
    for feature in features:
        representatives.setdefault((feature.ticker.upper(), feature.signal_date), feature)
    for feature in representatives.values():
        provider.benchmark_return(feature)
        provider.liquidity_snapshot(feature)
        provider.universe_membership(feature)
