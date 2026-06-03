"""Populate a FeatureVector's technical fields from real OHLCV history, point-in-time.

For each feature, the indicators (RSI/MACD/Bollinger/ATR/EMA + candlesticks) are computed from the
ticker's daily bars dated on or before the feature's ``signal_date`` — never from future bars.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from .indicators import compute_indicators
from .market_data import DailyPrice, PriceHistoryProvider
from .models import FeatureVector
from .technical_context import build_technical_context

_BULLISH_PATTERNS = frozenset({"bullish_engulfing", "hammer", "gap_up"})
_BEARISH_PATTERNS = frozenset({"bearish_engulfing", "shooting_star", "gap_down"})
_MIN_BARS = 26


def attach_indicators_from_history(
    features: Sequence[FeatureVector],
    provider: PriceHistoryProvider,
) -> list[FeatureVector]:
    """Recompute each feature's technical fields from its ticker's real price history (PIT)."""
    by_ticker: dict[str, list[FeatureVector]] = defaultdict(list)
    for feature in features:
        by_ticker[feature.ticker.upper()].append(feature)
    enriched: list[FeatureVector] = []
    for ticker, rows in by_ticker.items():
        prices = provider.daily_prices(ticker)
        enriched.extend(_attach_one(feature, prices) for feature in rows)
    return enriched


def _attach_one(feature: FeatureVector, prices: Sequence[DailyPrice]) -> FeatureVector:
    bars = sorted((price for price in prices if price.date <= feature.signal_date), key=lambda price: price.date)
    if len(bars) < _MIN_BARS:
        return feature
    snapshot = compute_indicators(bars)
    if snapshot is None:
        return feature
    closes = [bar.adjusted_close if bar.adjusted_close > 0.0 else bar.close for bar in bars]
    context = build_technical_context(symbol=feature.ticker, signal_date=feature.signal_date, prices=bars)
    return replace(
        feature,
        rsi_14=snapshot.rsi_14 or 0.0,
        atr_pct_14=snapshot.atr_pct_14 or 0.0,
        volume_ratio_20d=snapshot.volume_ratio_20 or 0.0,
        momentum_63d_pct=_momentum(closes, 63),
        macd_signal=_macd_label(snapshot.macd, snapshot.macd_signal),
        bb_percent_b=snapshot.bb_percent_b if snapshot.bb_percent_b is not None else 0.5,
        bb_bandwidth=snapshot.bb_bandwidth or 0.0,
        macd_hist_pct=_ratio_pct(snapshot.macd_hist, snapshot.close),
        ema_trend_pct=_spread_pct(snapshot.ema_12, snapshot.ema_26),
        candle_signal=_candle_signal(context.pattern_names),
        sma_distance_pct=_spread_pct(snapshot.close, snapshot.sma_20),
        trend=context.trend if context.trend != "unknown" else feature.trend,
        volume_confirmed=context.volume_signal == "high_volume_confirmation",
    )


def _macd_label(macd: float | None, signal: float | None) -> str:
    if macd is None or signal is None:
        return "unknown"
    if macd > signal:
        return "bullish"
    if macd < signal:
        return "bearish"
    return "unknown"


def _candle_signal(pattern_names: Sequence[str]) -> float:
    score = sum(1 for name in pattern_names if name in _BULLISH_PATTERNS)
    score -= sum(1 for name in pattern_names if name in _BEARISH_PATTERNS)
    return max(-1.0, min(1.0, float(score)))


def _momentum(closes: Sequence[float], lookback: int) -> float:
    if len(closes) <= lookback or closes[-lookback - 1] <= 0.0:
        return 0.0
    return (closes[-1] / closes[-lookback - 1] - 1.0) * 100.0


def _ratio_pct(value: float | None, base: float) -> float:
    if value is None or base <= 0.0:
        return 0.0
    return value / base * 100.0


def _spread_pct(value: float | None, base: float | None) -> float:
    if value is None or base is None or base <= 0.0:
        return 0.0
    return (value / base - 1.0) * 100.0
