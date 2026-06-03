from __future__ import annotations

from datetime import date, timedelta

from slowbrain.indicator_features import attach_indicators_from_history
from slowbrain.market_data import DailyPrice
from slowbrain.models import FeatureVector


class _FakeProvider:
    def __init__(self, prices: dict[str, tuple[DailyPrice, ...]]) -> None:
        self._prices = prices

    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        return self._prices.get(symbol.upper(), ())


def test_attach_populates_technical_fields_from_real_bars() -> None:
    bars = _rising_bars("TEST", 80)
    provider = _FakeProvider({"TEST": bars})
    feature = _feature("TEST", signal_date=bars[-1].date)

    enriched = attach_indicators_from_history([feature], provider)[0]

    assert enriched.rsi_14 == 100.0  # strictly rising closes
    assert enriched.bb_percent_b > 0.7  # price riding the upper band
    assert enriched.ema_trend_pct > 0.0  # fast EMA above slow EMA in an uptrend
    assert enriched.sma_distance_pct > 0.0  # close above its SMA20
    assert enriched.momentum_63d_pct > 0.0
    assert enriched.macd_signal in {"bullish", "bearish"}
    assert enriched.trend == "uptrend"
    assert -1.0 <= enriched.candle_signal <= 1.0


def test_attach_is_point_in_time_and_skips_thin_history() -> None:
    bars = _rising_bars("TEST", 80)
    provider = _FakeProvider({"TEST": bars})

    # a signal date before any bar -> no usable history -> fields stay at their defaults
    before = attach_indicators_from_history([_feature("TEST", signal_date="2000-01-01")], provider)[0]
    assert before.rsi_14 == 0.0 and before.bb_percent_b == 0.5

    # an as-of cutoff midway only sees earlier bars (no look-ahead)
    midpoint = bars[40].date
    mid = attach_indicators_from_history([_feature("TEST", signal_date=midpoint)], provider)[0]
    assert mid.rsi_14 == 100.0  # still computed, but only from bars[:41]

    # an unknown ticker (no history) is returned unchanged
    unchanged = attach_indicators_from_history([_feature("NONE", signal_date=bars[-1].date)], provider)[0]
    assert unchanged.bb_percent_b == 0.5 and unchanged.macd_signal == "unknown"


def _rising_bars(symbol: str, count: int) -> tuple[DailyPrice, ...]:
    start = date(2024, 1, 1)
    bars: list[DailyPrice] = []
    for index in range(count):
        close = 100.0 + index
        bars.append(
            DailyPrice(
                symbol=symbol,
                date=(start + timedelta(days=index)).isoformat(),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                adjusted_close=close,
                volume=1_000_000.0,
                source="fixture",
            )
        )
    return tuple(bars)


def _feature(ticker: str, *, signal_date: str) -> FeatureVector:
    return FeatureVector(
        idea_id=f"{ticker}-{signal_date}",
        ticker=ticker,
        signal_date=signal_date,
        sentiment="neutral",
        sentiment_confidence=0.0,
        catalyst_strength=0.0,
        trend="unknown",
        momentum_20d_pct=0.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=False,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=1.0,
        cost_bps=10.0,
        source="fixture",
        horizon_days=20,
        entry_price=100.0,
    )
