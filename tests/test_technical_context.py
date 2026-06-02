from __future__ import annotations

from slowbrain.market_data import DailyPrice
from slowbrain.technical_context import build_technical_context


def test_technical_context_includes_price_volume_and_candlestick_patterns() -> None:
    prices = _prices_with_bullish_engulfing()

    context = build_technical_context(symbol="ABCD", signal_date="2026-01-20", prices=prices)

    assert context.status == "available"
    assert context.price_asof_date == "2026-01-20"
    assert context.close == 106.0
    assert context.previous_close == 102.0
    assert context.volume_ratio_20d is not None and context.volume_ratio_20d > 1.5
    assert context.volume_signal == "high_volume_confirmation"
    assert "bullish_engulfing" in context.pattern_names
    assert "bullish context" in context.pattern_summary
    assert context.recent_bars[-1].volume == 2_000_000.0


def test_technical_context_is_honest_when_prices_are_missing() -> None:
    context = build_technical_context(symbol="MISS", signal_date="2026-01-20", prices=())

    assert context.status == "unavailable"
    assert context.reason == "not_enough_price_bars"
    assert context.close is None
    assert context.pattern_names == ()


def test_technical_context_handles_bearish_gap_and_low_volume() -> None:
    prices = _decreasing_prices()

    context = build_technical_context(symbol="BEAR", signal_date="2026-02-20", prices=prices)

    assert context.status == "available"
    assert context.trend == "downtrend"
    assert context.volume_signal == "low_volume_warning"
    assert "gap_down" in context.pattern_names
    assert "shooting_star" in context.pattern_names
    assert "bearish or exhaustion warning" in context.pattern_summary


def test_technical_context_detects_bearish_engulfing() -> None:
    prices = list(_prices_with_bullish_engulfing())
    prices[-2] = DailyPrice("ABCD", "2026-01-19", 100.0, 106.0, 99.0, 105.0, 105.0, 1_000_000.0, "fixture")
    prices[-1] = DailyPrice("ABCD", "2026-01-20", 106.0, 107.0, 98.0, 99.0, 99.0, 1_000_000.0, "fixture")

    context = build_technical_context(symbol="ABCD", signal_date="2026-01-20", prices=tuple(prices))

    assert "bearish_engulfing" in context.pattern_names


def test_technical_context_marks_partial_context_and_doji() -> None:
    prices = (
        DailyPrice("DOJI", "2026-03-01", 10.0, 11.0, 9.0, 10.5, 10.5, 1000.0, "fixture"),
        DailyPrice("DOJI", "2026-03-02", 10.0, 11.0, 9.0, 10.01, 10.01, 1000.0, "fixture"),
        DailyPrice("DOJI", "2026-03-03", 10.0, 11.0, 9.0, 10.0, 10.0, 1000.0, "fixture"),
        DailyPrice("DOJI", "2026-03-04", 10.0, 11.0, 9.0, 10.0, 10.0, 1000.0, "fixture"),
        DailyPrice("DOJI", "2026-03-05", 10.0, 11.0, 9.0, 10.0, 10.0, 1000.0, "fixture"),
        DailyPrice("DOJI", "2026-03-06", 10.0, 11.0, 9.0, 10.01, 10.01, 1000.0, "fixture"),
    )

    context = build_technical_context(symbol="DOJI", signal_date="2026-03-06", prices=prices)

    assert context.status == "partial"
    assert context.reason == "only_6_bars_available"
    assert context.volume_signal == "normal_volume"
    assert "doji" in context.pattern_names
    assert context.sma_20 is None


def _prices_with_bullish_engulfing() -> tuple[DailyPrice, ...]:
    prices: list[DailyPrice] = []
    for day in range(1, 21):
        close = 100.0 + day * 0.2
        prices.append(
            DailyPrice(
                symbol="ABCD",
                date=f"2026-01-{day:02d}",
                open=close - 0.1,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                adjusted_close=close,
                volume=1_000_000.0,
                source="fixture",
            )
        )
    prices[-2] = DailyPrice("ABCD", "2026-01-19", 105.0, 106.0, 101.0, 102.0, 102.0, 1_000_000.0, "fixture")
    prices[-1] = DailyPrice("ABCD", "2026-01-20", 101.5, 107.0, 101.0, 106.0, 106.0, 2_000_000.0, "fixture")
    return tuple(prices)


def _decreasing_prices() -> tuple[DailyPrice, ...]:
    prices: list[DailyPrice] = []
    for day in range(1, 21):
        close = 130.0 - day
        prices.append(
            DailyPrice(
                symbol="BEAR",
                date=f"2026-02-{day:02d}",
                open=close + 0.2,
                high=close + 0.7,
                low=close - 0.7,
                close=close,
                adjusted_close=close,
                volume=1_000_000.0,
                source="fixture",
            )
        )
    prices[-2] = DailyPrice("BEAR", "2026-02-19", 105.0, 106.0, 101.0, 104.0, 104.0, 1_000_000.0, "fixture")
    prices[-1] = DailyPrice("BEAR", "2026-02-20", 102.0, 111.0, 98.0, 99.5, 99.5, 100_000.0, "fixture")
    return tuple(prices)
