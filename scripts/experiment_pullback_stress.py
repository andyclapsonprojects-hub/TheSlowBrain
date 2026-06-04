"""Pressure-test the pullback signal: NON-overlapping trades, path/drawdown risk, and significance.

Honest version of the pullback test. Trades are non-overlapping (one position at a time per ticker),
so observations are roughly independent. For each config we report return, hit rate, per-trade Sharpe
and t-stat, the drawdown endured during the hold (max adverse excursion), and a matched random-entry
control. Read-only, on the cached Yahoo bars. Survivorship caveat still applies (survivor large-caps).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

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
CONFIGS = ((20, 40.0), (20, 30.0), (60, 40.0), (60, 30.0))  # (hold_days, rsi_threshold)
TRIALS = len(CONFIGS)
COST_BPS = 15.0


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    series = []
    for symbol in SYMBOLS:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)
        if len(bars) < 300:
            continue
        closes = [bar.adjusted_close if bar.adjusted_close > 0.0 else bar.close for bar in bars]
        series.append((closes, _sma(closes, 50), _sma(closes, 200), _rsi(closes, 14)))

    rng = random.Random(11)
    deflated_t = (2.0 * math.log(TRIALS)) ** 0.5  # expected max |t| under the null for TRIALS configs
    print(f"\nPullback pressure-test: {len(series)} tickers, NON-overlapping trades, cost {COST_BPS:.0f}bps "
          f"round-trip. Multiple-testing threshold |t| ~ {deflated_t:.1f} (for {TRIALS} configs).")
    for hold, threshold in CONFIGS:
        pull: list[tuple[float, float]] = []
        rand: list[tuple[float, float]] = []
        for data in series:
            ticker_trades = _pullback_trades(data, threshold, hold)
            pull.extend(ticker_trades)
            rand.extend(_random_trades(data, hold, len(ticker_trades), rng))
        print(f"\n=== hold {hold}d, RSI<{threshold:.0f} ===")
        _print_stats("pullback", pull, deflated_t)
        _print_stats("random  ", rand, deflated_t)
        if pull and rand:
            excess = mean(r for r, _ in pull) - mean(r for r, _ in rand)
            print(f"  excess return (pullback - random): {excess:+.2f}%")
    return 0


def _print_stats(label: str, trades: list[tuple[float, float]], deflated_t: float) -> None:
    if not trades:
        print(f"  {label}: no trades")
        return
    returns = sorted(value for value, _ in trades)
    maes = sorted(value for _, value in trades)
    count = len(returns)
    avg = mean(returns)
    std = pstdev(returns) or 1e-9
    t_stat = avg / (std / count**0.5)
    hit = 100.0 * sum(value > 0 for value in returns) / count
    survives = "YES" if abs(t_stat) > deflated_t else "no"
    print(f"  {label}: n={count:>5} mean={avg:+.2f}% med={returns[count // 2]:+.2f}% hit={hit:.0f}% "
          f"Sharpe={avg / std:.2f} t={t_stat:.1f} (survives MT: {survives})")
    print(f"           drawdown in hold: avg MAE={mean(maes):+.1f}%  worst-5% MAE={maes[count // 20]:+.1f}%  "
          f"worst trade={returns[0]:+.1f}%")


def _pullback_trades(data: tuple, threshold: float, hold: int) -> list[tuple[float, float]]:
    closes, sma50, sma200, rsi = data
    trades: list[tuple[float, float]] = []
    index = 200
    while index < len(closes) - hold:
        if (
            sma200[index] is not None and sma50[index] is not None and rsi[index] is not None
            and closes[index] > sma200[index] and sma50[index] > sma200[index] and rsi[index] < threshold
        ):
            trades.append(_trade(closes, index, hold))
            index += hold  # non-overlapping: one position at a time
        else:
            index += 1
    return trades


def _random_trades(data: tuple, hold: int, count: int, rng: random.Random) -> list[tuple[float, float]]:
    closes = data[0]
    low, high = 200, len(closes) - hold - 1
    if high <= low or count <= 0:
        return []
    return [_trade(closes, rng.randint(low, high), hold) for _ in range(count)]


def _trade(closes: list[float], index: int, hold: int) -> tuple[float, float]:
    entry = closes[index]
    ret = (closes[index + hold] / entry - 1.0) * 100.0 - COST_BPS / 100.0 * 2.0
    mae = (min(closes[index : index + hold + 1]) / entry - 1.0) * 100.0
    return ret, mae


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
