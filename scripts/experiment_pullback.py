"""Across breadth: does buying a 'registered pullback' beat entering at random?

A registered pullback = an oversold dip (RSI14 below a threshold) WITHIN an uptrend (close above
the 200-day SMA and the 50-day SMA above the 200-day SMA). For each pullback day we measure the
forward return at several horizons and compare it to all days and to uptrend-only days, across ~75
liquid tickers. Pure price math on the cached Yahoo bars; read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider

_SYMBOL_TEXT = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ADBE CRM ORCL CSCO INTC AMD QCOM TXN IBM NOW INTU "
    "JPM BAC WFC GS MS C AXP BLK SCHW USB PNC UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD "
    "CVS HD LOW MCD SBUX NKE TGT COST WMT PG KO PEP PM MO XOM CVX COP SLB EOG BA CAT GE HON UNP "
    "UPS LMT RTX DE DIS CMCSA NFLX T VZ"
)
SYMBOLS = _SYMBOL_TEXT.split()
HORIZONS = (10, 20, 40, 60)
COST_BPS = 15.0


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    buckets: dict[str, dict[int, list[float]]] = {
        name: {horizon: [] for horizon in HORIZONS} for name in ("all", "uptrend", "pullback", "strong")
    }
    for symbol in SYMBOLS:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)
        if len(bars) < 260:
            continue
        closes = [bar.adjusted_close if bar.adjusted_close > 0.0 else bar.close for bar in bars]
        sma50, sma200, rsi = _sma(closes, 50), _sma(closes, 200), _rsi(closes, 14)
        for index in range(len(closes) - max(HORIZONS)):
            if sma200[index] is None or sma50[index] is None or rsi[index] is None:
                continue
            uptrend = closes[index] > sma200[index] and sma50[index] > sma200[index]
            for horizon in HORIZONS:
                forward = (closes[index + horizon] / closes[index] - 1.0) * 100.0 - COST_BPS / 100.0 * 2.0
                buckets["all"][horizon].append(forward)
                if uptrend:
                    buckets["uptrend"][horizon].append(forward)
                    if rsi[index] < 40.0:
                        buckets["pullback"][horizon].append(forward)
                    if rsi[index] < 30.0:
                        buckets["strong"][horizon].append(forward)

    print(f"\nBreadth pullback test: {len(SYMBOLS)} tickers (overlapping windows -> averages are indicative)\n")
    columns = (("horizon", 7), ("all days", 22), ("uptrend only", 22), ("pullback RSI<40", 24), ("strong RSI<30", 22))
    header = " | ".join(f"{label:>{width}}" for label, width in columns)
    print(header)
    print("-" * len(header))
    for horizon in HORIZONS:
        cells = " | ".join(_cell(buckets[name][horizon]) for name in ("all", "uptrend", "pullback", "strong"))
        print(f"{horizon:>5}d  | {cells}")
    return 0


def _cell(values: list[float]) -> str:
    if not values:
        return f"{'n=0':>22}"
    hit = 100.0 * sum(value > 0 for value in values) / len(values)
    return f"{mean(values):+.2f}% n={len(values):>6} hit={hit:.0f}%"


def _sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= period:
            total -= values[index - period]
        if index >= period - 1:
            out[index] = total / period
    return out


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain, avg_loss = sum(gains[:period]) / period, sum(losses[:period]) / period
    out[period] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
