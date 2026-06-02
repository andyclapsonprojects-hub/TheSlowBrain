"""Derived benchmark and liquidity evidence from daily prices."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from statistics import mean, stdev

from ..market_data import BenchmarkReturn, DailyPrice, LiquiditySnapshot
from .types import DEFAULT_SPREAD_PROXY_BPS, FxRate, FxRateSource


def _benchmark_return_from_prices(
    ticker: str,
    signal_date: str,
    prices: Sequence[DailyPrice],
    *,
    source: str,
) -> BenchmarkReturn | None:
    selected = _prices_on_or_before(prices, signal_date)
    if len(selected) < 2:
        return None
    previous, current = selected[-2], selected[-1]
    if previous.adjusted_close <= 0.0:
        return None
    return BenchmarkReturn(
        ticker=ticker,
        signal_date=signal_date,
        return_pct=((current.adjusted_close / previous.adjusted_close) - 1.0) * 100.0,
        source=source,
    )


def _liquidity_from_prices(
    ticker: str,
    signal_date: str,
    prices: Sequence[DailyPrice],
    *,
    quote_to_gbp_rate: FxRateSource,
    source: str,
    window: int = 20,
) -> LiquiditySnapshot | None:
    fx_rate = _resolve_fx_rate(quote_to_gbp_rate, signal_date)
    if fx_rate is None:
        return None
    selected = _prices_on_or_before(prices, signal_date)[-window:]
    if len(selected) < 2:
        return None
    traded_values = [
        price.volume * price.adjusted_close * fx_rate.rate
        for price in selected
        if price.volume > 0.0 and price.adjusted_close > 0.0
    ]
    returns = [
        ((current.adjusted_close / previous.adjusted_close) - 1.0) * 100.0
        for previous, current in pairwise(selected)
        if previous.adjusted_close > 0.0
    ]
    if not traded_values:
        return None
    volatility = stdev(returns) if len(returns) >= 2 else 0.0
    return LiquiditySnapshot(
        ticker=ticker,
        avg_daily_volume_gbp=mean(traded_values),
        volatility_pct=volatility,
        spread_bps=DEFAULT_SPREAD_PROXY_BPS,
        source=f"{source}:fx_{fx_rate.source}:spread_proxy_{DEFAULT_SPREAD_PROXY_BPS:.1f}bps",
    )


def _resolve_fx_rate(rate_source: FxRateSource, signal_date: str) -> FxRate | None:
    if rate_source is None:
        return None
    if isinstance(rate_source, int | float):
        return FxRate(rate=float(rate_source), source="configured_usd_gbp")
    return rate_source(signal_date)


def _prices_on_or_before(prices: Sequence[DailyPrice], signal_date: str) -> tuple[DailyPrice, ...]:
    return tuple(sorted((price for price in prices if price.date <= signal_date), key=lambda price: price.date))
