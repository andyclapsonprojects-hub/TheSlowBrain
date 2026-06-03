from __future__ import annotations

from datetime import date, timedelta

from slowbrain.indicators import (
    average_true_range,
    bollinger_bands,
    compute_indicators,
    ema_series,
    exponential_moving_average,
    macd,
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)
from slowbrain.market_data import DailyPrice


def test_simple_moving_average() -> None:
    assert simple_moving_average([1.0, 2.0, 3.0, 4.0], 2) == 3.5
    assert simple_moving_average([1.0, 2.0], 3) is None


def test_ema_series_matches_hand_computation() -> None:
    # period 2, multiplier 2/3, seeded with SMA([1,2])=1.5
    series = ema_series([1.0, 2.0, 3.0, 4.0, 5.0], 2)
    assert [round(value, 6) for value in series] == [1.5, 2.5, 3.5, 4.5]
    assert exponential_moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 2) == 4.5
    assert ema_series([1.0], 2) == []


def test_rsi_bounds_and_direction() -> None:
    rising = [float(i) for i in range(1, 17)]
    falling = [float(i) for i in range(16, 0, -1)]
    flat = [5.0] * 16
    assert relative_strength_index(rising, 14) == 100.0  # only gains
    assert relative_strength_index(falling, 14) == 0.0  # only losses
    assert relative_strength_index(flat, 14) == 50.0  # no movement
    assert relative_strength_index([1.0] * 14, 14) is None  # need period+1


def test_macd_uptrend_is_positive_and_needs_enough_bars() -> None:
    rising = [float(i) for i in range(1, 41)]
    result = macd(rising)
    assert result is not None
    macd_line, signal_line, hist = result
    assert macd_line > 0.0  # fast EMA above slow EMA in an uptrend
    assert round(hist, 9) == round(macd_line - signal_line, 9)
    assert macd([float(i) for i in range(1, 31)]) is None  # < slow + signal bars


def test_bollinger_constant_series_has_zero_bandwidth() -> None:
    lower, mid, upper, percent_b, bandwidth = bollinger_bands([5.0] * 20)  # type: ignore[misc]
    assert (lower, mid, upper) == (5.0, 5.0, 5.0)
    assert percent_b == 0.5 and bandwidth == 0.0
    assert bollinger_bands([1.0] * 19) is None


def test_bollinger_rising_series_sits_near_upper_band() -> None:
    result = bollinger_bands([float(i) for i in range(1, 21)])
    assert result is not None
    _lower, _mid, _upper, percent_b, bandwidth = result
    assert percent_b > 0.85 and bandwidth > 0.0


def test_atr_hand_computed() -> None:
    highs = [10.0, 11.0, 12.0]
    lows = [9.0, 10.0, 11.0]
    closes = [9.5, 10.5, 11.5]
    assert average_true_range(highs, lows, closes, period=2) == 1.5
    assert average_true_range(highs, lows, closes, period=3) is None  # needs period+1 bars


def test_volume_ratio() -> None:
    volumes = [100.0] * 20 + [300.0]
    assert volume_ratio(volumes, 20) == 3.0
    assert volume_ratio([100.0] * 20, 20) is None  # need period+1


def test_compute_indicators_on_real_shaped_series_and_point_in_time() -> None:
    bars = _rising_bars(40)
    snap = compute_indicators(bars)
    assert snap is not None
    assert snap.bars == 40
    assert snap.sma_20 is not None and snap.rsi_14 == 100.0
    assert snap.macd is not None and snap.macd > 0.0
    assert snap.bb_percent_b is not None and snap.bb_percent_b > 0.85
    assert snap.atr_14 is not None and snap.atr_pct_14 is not None
    assert snap.volume_ratio_20 == 1.0  # flat volume

    # point-in-time: a cutoff date must drop later bars and never look ahead
    cutoff = bars[19].date
    pit = compute_indicators(bars, signal_date=cutoff)
    assert pit is not None and pit.bars == 20 and pit.as_of_date == cutoff
    assert compute_indicators(()) is None


def _rising_bars(count: int) -> tuple[DailyPrice, ...]:
    start = date(2026, 1, 1)
    bars: list[DailyPrice] = []
    for index in range(count):
        close = float(index + 1)
        bars.append(
            DailyPrice(
                symbol="TEST",
                date=(start + timedelta(days=index)).isoformat(),
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                adjusted_close=close,
                volume=100.0,
                source="fixture",
            )
        )
    return tuple(bars)
