"""Human-readable technical context from point-in-time OHLCV bars."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .market_data import DailyPrice

ContextStatus = Literal["available", "partial", "unavailable"]
PatternDirection = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class RecentBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float


@dataclass(frozen=True)
class CandlePattern:
    name: str
    direction: PatternDirection
    strength: int
    meaning: str
    gap_pct: float | None = None


@dataclass(frozen=True)
class TechnicalContext:
    status: ContextStatus
    reason: str
    symbol: str
    signal_date: str
    price_asof_date: str | None
    price_source: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adjusted_close: float | None
    volume: float | None
    previous_close: float | None
    day_change_pct: float | None
    gap_pct: float | None
    intraday_return_pct: float | None
    candle_range_pct: float | None
    candle_body_pct: float | None
    close_location_pct: float | None
    volume_ratio_20d: float | None
    volume_signal: str
    sma_5: float | None
    sma_20: float | None
    momentum_5d_pct: float | None
    momentum_20d_pct: float | None
    distance_from_20d_high_pct: float | None
    distance_from_20d_low_pct: float | None
    trend: str
    pattern_names: tuple[str, ...]
    pattern_summary: str
    recent_bars: tuple[RecentBar, ...]


def build_technical_context(
    *,
    symbol: str,
    signal_date: str,
    prices: Sequence[DailyPrice],
    window: int = 20,
) -> TechnicalContext:
    """Build decision-time price, volume, trend, and candle context."""
    selected = tuple(sorted((price for price in prices if price.date <= signal_date), key=lambda price: price.date))
    if len(selected) < 2:
        return _unavailable(symbol=symbol, signal_date=signal_date, reason="not_enough_price_bars")

    recent = selected[-window:]
    latest = recent[-1]
    previous = selected[-2]
    closes = [price.adjusted_close for price in recent if price.adjusted_close > 0.0]
    highs = [price.high for price in recent if price.high > 0.0]
    lows = [price.low for price in recent if price.low > 0.0]
    volumes = [price.volume for price in recent if price.volume >= 0.0]
    patterns = _patterns(previous, latest)
    volume_ratio = _volume_ratio(volumes)
    volume_signal = _volume_label(volume_ratio)
    status: ContextStatus = "available" if len(recent) >= window else "partial"

    return TechnicalContext(
        status=status,
        reason="ok" if status == "available" else f"only_{len(recent)}_bars_available",
        symbol=symbol.upper(),
        signal_date=signal_date,
        price_asof_date=latest.date,
        price_source=latest.source,
        open=_round(latest.open),
        high=_round(latest.high),
        low=_round(latest.low),
        close=_round(latest.close),
        adjusted_close=_round(latest.adjusted_close),
        volume=_round(latest.volume, 0),
        previous_close=_round(previous.adjusted_close),
        day_change_pct=_pct(latest.adjusted_close, previous.adjusted_close),
        gap_pct=_pct(latest.open, previous.close),
        intraday_return_pct=_pct(latest.close, latest.open),
        candle_range_pct=_pct(latest.high - latest.low, latest.close, ratio=False),
        candle_body_pct=_pct(abs(latest.close - latest.open), latest.open, ratio=False),
        close_location_pct=_close_location(latest),
        volume_ratio_20d=volume_ratio,
        volume_signal=volume_signal,
        sma_5=_sma(closes, 5),
        sma_20=_sma(closes, 20),
        momentum_5d_pct=_momentum(closes, 5),
        momentum_20d_pct=_momentum(closes, 20),
        distance_from_20d_high_pct=_distance_from_high(latest.adjusted_close, highs),
        distance_from_20d_low_pct=_distance_from_low(latest.adjusted_close, lows),
        trend=_trend(closes),
        pattern_names=tuple(pattern.name for pattern in patterns),
        pattern_summary=_pattern_summary(patterns, volume_signal),
        recent_bars=tuple(_recent_bar(price) for price in recent),
    )


def _unavailable(*, symbol: str, signal_date: str, reason: str) -> TechnicalContext:
    return TechnicalContext(
        status="unavailable",
        reason=reason,
        symbol=symbol.upper(),
        signal_date=signal_date,
        price_asof_date=None,
        price_source="not_available",
        open=None,
        high=None,
        low=None,
        close=None,
        adjusted_close=None,
        volume=None,
        previous_close=None,
        day_change_pct=None,
        gap_pct=None,
        intraday_return_pct=None,
        candle_range_pct=None,
        candle_body_pct=None,
        close_location_pct=None,
        volume_ratio_20d=None,
        volume_signal="unknown",
        sma_5=None,
        sma_20=None,
        momentum_5d_pct=None,
        momentum_20d_pct=None,
        distance_from_20d_high_pct=None,
        distance_from_20d_low_pct=None,
        trend="unknown",
        pattern_names=(),
        pattern_summary="Market price context unavailable.",
        recent_bars=(),
    )


def _recent_bar(price: DailyPrice) -> RecentBar:
    return RecentBar(
        date=price.date,
        open=_round(price.open),
        high=_round(price.high),
        low=_round(price.low),
        close=_round(price.close),
        adjusted_close=_round(price.adjusted_close),
        volume=_round(price.volume, 0),
    )


def _patterns(previous: DailyPrice, latest: DailyPrice) -> tuple[CandlePattern, ...]:
    patterns: list[CandlePattern] = []
    engulfing = _engulfing(previous, latest)
    if engulfing is not None:
        patterns.append(engulfing)
    if latest.high <= previous.high and latest.low >= previous.low:
        patterns.append(CandlePattern("inside_candle", "neutral", 1, "volatility compression or indecision"))
    gap = _gap(previous, latest)
    if gap is not None:
        patterns.append(gap)
    single = _single_candle_pattern(latest)
    if single is not None:
        patterns.append(single)
    return tuple(patterns)


def _engulfing(previous: DailyPrice, latest: DailyPrice) -> CandlePattern | None:
    if (
        previous.close < previous.open
        and latest.close > latest.open
        and latest.open <= previous.close
        and latest.close >= previous.open
    ):
        return CandlePattern("bullish_engulfing", "bullish", 2, "buyers overwhelmed the prior down candle")
    if (
        previous.close > previous.open
        and latest.close < latest.open
        and latest.open >= previous.close
        and latest.close <= previous.open
    ):
        return CandlePattern("bearish_engulfing", "bearish", 2, "sellers overwhelmed the prior up candle")
    return None


def _gap(previous: DailyPrice, latest: DailyPrice) -> CandlePattern | None:
    gap_pct = _pct(latest.open, previous.close)
    if gap_pct is None:
        return None
    if gap_pct >= 1.0:
        return CandlePattern("gap_up", "bullish", 1, "opened materially above prior close", gap_pct)
    if gap_pct <= -1.0:
        return CandlePattern("gap_down", "bearish", 1, "opened materially below prior close", gap_pct)
    return None


def _single_candle_pattern(bar: DailyPrice) -> CandlePattern | None:
    candle_range = max(bar.high - bar.low, 0.0001)
    body_ratio = abs(bar.close - bar.open) / candle_range
    upper_ratio = (bar.high - max(bar.open, bar.close)) / candle_range
    lower_ratio = (min(bar.open, bar.close) - bar.low) / candle_range
    direction = _direction(bar)

    if body_ratio <= 0.10:
        return CandlePattern("doji", "neutral", 1, "open and close nearly equal; indecision")
    if lower_ratio >= 0.55 and upper_ratio <= 0.20 and body_ratio <= 0.35:
        return CandlePattern("hammer", "bullish", 1, "lower-price rejection; buyers recovered the candle")
    if upper_ratio >= 0.55 and lower_ratio <= 0.20 and body_ratio <= 0.35:
        if direction == "bullish":
            return CandlePattern("inverted_hammer", "neutral", 1, "upper-price rejection or possible exhaustion")
        return CandlePattern("shooting_star", "bearish", 1, "upper-price rejection or possible exhaustion")
    return None


def _direction(bar: DailyPrice) -> PatternDirection:
    if bar.close > bar.open:
        return "bullish"
    if bar.close < bar.open:
        return "bearish"
    return "neutral"


def _pattern_summary(patterns: tuple[CandlePattern, ...], volume_signal: str) -> str:
    if not patterns:
        return f"No whitelisted candlestick pattern detected; volume is {volume_signal}."
    names = ", ".join(pattern.name for pattern in patterns)
    bullish = sum(1 for pattern in patterns if pattern.direction == "bullish")
    bearish = sum(1 for pattern in patterns if pattern.direction == "bearish")
    if bullish > bearish:
        context = "bullish context"
    elif bearish > bullish:
        context = "bearish or exhaustion warning"
    else:
        context = "mixed or neutral context"
    return f"Detected {names}; {context}; volume is {volume_signal}."


def _volume_ratio(volumes: Sequence[float]) -> float | None:
    if len(volumes) < 6:
        return None
    baseline = mean(volumes[:-1])
    if baseline <= 0.0:
        return None
    return _round(volumes[-1] / baseline, 2)


def _volume_label(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio >= 1.5:
        return "high_volume_confirmation"
    if ratio <= 0.7:
        return "low_volume_warning"
    return "normal_volume"


def _sma(values: Sequence[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return _round(mean(values[-period:]))


def _momentum(values: Sequence[float], lookback: int) -> float | None:
    if len(values) <= lookback:
        return None
    return _pct(values[-1], values[-lookback - 1])


def _trend(closes: Sequence[float]) -> str:
    if len(closes) < 5:
        return "unknown"
    sma_5 = _sma(closes, 5)
    sma_20 = _sma(closes, 20)
    latest = closes[-1]
    if sma_20 is None:
        return "uptrend" if latest > closes[0] else "downtrend" if latest < closes[0] else "sideways"
    if sma_5 is not None and latest > sma_5 > sma_20:
        return "uptrend"
    if sma_5 is not None and latest < sma_5 < sma_20:
        return "downtrend"
    return "sideways"


def _distance_from_high(close: float, highs: Sequence[float]) -> float | None:
    if close <= 0.0 or not highs:
        return None
    high = max(highs)
    return _pct(close, high)


def _distance_from_low(close: float, lows: Sequence[float]) -> float | None:
    if close <= 0.0 or not lows:
        return None
    low = min(lows)
    return _pct(close, low)


def _close_location(price: DailyPrice) -> float | None:
    candle_range = price.high - price.low
    if candle_range <= 0.0:
        return None
    return _round(((price.close - price.low) / candle_range) * 100.0, 2)


def _pct(numerator: float, denominator: float, *, ratio: bool = True) -> float | None:
    if denominator <= 0.0:
        return None
    value = ((numerator / denominator) - 1.0) * 100.0 if ratio else (numerator / denominator) * 100.0
    return _round(value, 2)


def _round(value: float, ndigits: int = 4) -> float:
    return round(value, ndigits)
