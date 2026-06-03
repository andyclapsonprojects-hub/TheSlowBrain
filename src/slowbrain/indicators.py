"""Quantitative price indicators computed from point-in-time OHLCV bars.

Pure functions over numeric series plus a latest-value snapshot. These complement the candlestick /
trend context in :mod:`slowbrain.technical_context` with the oscillator and band indicators it lacks:
EMA, RSI (Wilder), MACD, Bollinger Bands, and ATR (Wilder). No network, no third-party dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, pstdev

from .market_data import DailyPrice


@dataclass(frozen=True)
class IndicatorSnapshot:
    symbol: str
    as_of_date: str
    bars: int
    close: float
    sma_20: float | None
    ema_12: float | None
    ema_26: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    bb_lower: float | None
    bb_mid: float | None
    bb_upper: float | None
    bb_percent_b: float | None
    bb_bandwidth: float | None
    atr_14: float | None
    atr_pct_14: float | None
    volume_ratio_20: float | None


def simple_moving_average(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return mean(values[-period:])


def ema_series(values: Sequence[float], period: int) -> list[float]:
    """Exponential moving average aligned to ``values[period-1:]`` (seeded with the first SMA)."""
    if period <= 0 or len(values) < period:
        return []
    multiplier = 2.0 / (period + 1.0)
    series = [mean(values[:period])]
    for value in values[period:]:
        series.append(value * multiplier + series[-1] * (1.0 - multiplier))
    return series


def exponential_moving_average(values: Sequence[float], period: int) -> float | None:
    series = ema_series(values, period)
    return series[-1] if series else None


def relative_strength_index(closes: Sequence[float], period: int = 14) -> float | None:
    """Wilder's RSI in [0, 100]; needs ``period + 1`` closes."""
    if period <= 0 or len(closes) < period + 1:
        return None
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    for index in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[index]) / period
        avg_loss = (avg_loss * (period - 1) + losses[index]) / period
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    closes: Sequence[float],
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float] | None:
    """Return (macd_line, signal_line, histogram) latest values, or None if too few bars."""
    if fast >= slow or len(closes) < slow + signal:
        return None
    fast_series = ema_series(closes, fast)
    slow_series = ema_series(closes, slow)
    offset = slow - fast  # fast_series is longer; align both to the slow start index
    macd_line = [fast_series[index + offset] - slow_series[index] for index in range(len(slow_series))]
    signal_series = ema_series(macd_line, signal)
    if not signal_series:
        return None
    macd_latest = macd_line[-1]
    signal_latest = signal_series[-1]
    return macd_latest, signal_latest, macd_latest - signal_latest


def bollinger_bands(
    closes: Sequence[float],
    *,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float, float, float, float, float] | None:
    """Return (lower, mid, upper, percent_b, bandwidth) for the latest bar."""
    if period <= 0 or len(closes) < period:
        return None
    window = closes[-period:]
    mid = mean(window)
    deviation = pstdev(window)
    lower = mid - num_std * deviation
    upper = mid + num_std * deviation
    width = upper - lower
    percent_b = 0.5 if width == 0.0 else (closes[-1] - lower) / width
    bandwidth = 0.0 if mid == 0.0 else width / mid
    return lower, mid, upper, percent_b, bandwidth


def average_true_range(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Wilder's ATR using true range; needs ``period + 1`` aligned bars."""
    count = min(len(highs), len(lows), len(closes))
    if period <= 0 or count < period + 1:
        return None
    true_ranges = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, count)
    ]
    atr = mean(true_ranges[:period])
    for index in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[index]) / period
    return atr


def volume_ratio(volumes: Sequence[float], period: int = 20) -> float | None:
    """Latest volume relative to the average of the prior ``period`` bars."""
    if period <= 0 or len(volumes) < period + 1:
        return None
    baseline = mean(volumes[-period - 1 : -1])
    if baseline <= 0.0:
        return None
    return volumes[-1] / baseline


def compute_indicators(
    prices: Sequence[DailyPrice],
    *,
    signal_date: str | None = None,
) -> IndicatorSnapshot | None:
    """Compute the full indicator snapshot as of ``signal_date`` (point-in-time; no future bars)."""
    selected = sorted(
        (price for price in prices if signal_date is None or price.date <= signal_date),
        key=lambda price: price.date,
    )
    if not selected:
        return None
    closes = [price.adjusted_close if price.adjusted_close > 0.0 else price.close for price in selected]
    highs = [price.high for price in selected]
    lows = [price.low for price in selected]
    volumes = [price.volume for price in selected]
    latest = selected[-1]
    band = bollinger_bands(closes)
    macd_values = macd(closes)
    atr = average_true_range(highs, lows, closes)
    return IndicatorSnapshot(
        symbol=latest.symbol.upper(),
        as_of_date=latest.date,
        bars=len(selected),
        close=closes[-1],
        sma_20=simple_moving_average(closes, 20),
        ema_12=exponential_moving_average(closes, 12),
        ema_26=exponential_moving_average(closes, 26),
        rsi_14=relative_strength_index(closes, 14),
        macd=macd_values[0] if macd_values else None,
        macd_signal=macd_values[1] if macd_values else None,
        macd_hist=macd_values[2] if macd_values else None,
        bb_lower=band[0] if band else None,
        bb_mid=band[1] if band else None,
        bb_upper=band[2] if band else None,
        bb_percent_b=band[3] if band else None,
        bb_bandwidth=band[4] if band else None,
        atr_14=atr,
        atr_pct_14=(atr / closes[-1] * 100.0) if atr is not None and closes[-1] > 0.0 else None,
        volume_ratio_20=volume_ratio(volumes),
    )
