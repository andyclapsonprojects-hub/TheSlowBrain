"""Does professional RISK MANAGEMENT make the pullback strategy survive? An ablation.

Data: real daily OHLCV from Yahoo Finance (the project's read-only YahooProvider), split/dividend
adjusted, cached on disk. Survivor large-caps only -- so dip-buying is FLATTERED here (the names that
fell and never came back are absent); read every number with that caveat.

Entry (rule): pullback in an uptrend -- close > 200d SMA, RSI(14) < 35 (oversold dip), and price ticked
up (kick off the low). Non-overlapping trades. Then we ADD, one at a time, the tools a real trader uses:
  + ATR stop      -- initial stop at entry - 2*ATR(14) (cut losers fast)
  + trailing stop -- trail at peak - 2.5*ATR (let winners run)
  + regime filter -- only enter when SPY is above its own 200d SMA (don't buy dips in a bear market)
Report per variant: win%, avg win/loss, payoff, expectancy, worst trade, compounded return, max drawdown.
Read-only; nothing is traded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

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
COST_BPS = 15.0
MAX_HOLD = 60
ATR_STOP = 2.0
ATR_TRAIL = 2.5


@dataclass
class Trade:
    ret: float
    mae: float
    date: str
    risk: float  # stop distance % (= 2*ATR/entry) -> used for risk-unit (R) sizing


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


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    atr = sum(trs[:period]) / period
    out[period] = atr
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        out[i + 1] = atr
    return out


@dataclass
class Series:
    dates: list[str]
    closes: list[float]
    rsi: list[float | None]
    sma200: list[float | None]
    atr: list[float | None]


def _series(bars: list[Any]) -> Series:
    closes = [b.adjusted_close if b.adjusted_close > 0.0 else b.close for b in bars]
    highs, lows = [b.high for b in bars], [b.low for b in bars]
    return Series([b.date for b in bars], closes, _rsi(closes), _sma(closes, 200), _atr(highs, lows, closes))


def _spy_regime(provider: Any) -> dict[str, bool]:
    bars = sorted(provider.daily_prices("SPY"), key=lambda bar: bar.date)
    closes = [b.adjusted_close if b.adjusted_close > 0.0 else b.close for b in bars]
    sma = _sma(closes, 200)
    return {b.date: (sma[i] is not None and closes[i] > (sma[i] or 0.0)) for i, b in enumerate(bars)}


def _trades(s: Series, regime: dict[str, bool], *, stop: bool, trail: bool, use_regime: bool) -> list[Trade]:
    trades: list[Trade] = []
    cursor = -1
    for i in range(200, len(s.closes) - 1):
        if i <= cursor or s.sma200[i] is None or s.atr[i] is None or s.rsi[i] is None:
            continue
        entry_ok = s.closes[i] > (s.sma200[i] or 0.0) and (s.rsi[i] or 100.0) < 35.0 and s.closes[i] > s.closes[i - 1]
        if not entry_ok or (use_regime and not regime.get(s.dates[i], False)):
            continue
        trade, exit_index = _simulate(s, i, stop=stop, trail=trail)
        trades.append(trade)
        cursor = exit_index
    return trades


def _simulate(s: Series, i: int, *, stop: bool, trail: bool) -> tuple[Trade, int]:
    entry = s.closes[i]
    hard_stop = entry - ATR_STOP * (s.atr[i] or 0.0)
    peak = entry
    last = min(i + MAX_HOLD, len(s.closes) - 1)
    for j in range(i + 1, last + 1):
        peak = max(peak, s.closes[j])
        level = hard_stop
        if trail and s.atr[j] is not None:
            level = max(level, peak - ATR_TRAIL * (s.atr[j] or 0.0))
        if (stop or trail) and s.closes[j] <= level:
            return _close(s, i, j, level), j
        if not (stop or trail) and s.rsi[j] is not None and (s.rsi[j] or 0.0) >= 70.0:
            return _close(s, i, j, s.closes[j]), j
    return _close(s, i, last, s.closes[last]), last


def _close(s: Series, i: int, j: int, exit_price: float) -> Trade:
    ret = (exit_price / s.closes[i] - 1.0) * 100.0 - COST_BPS / 100.0 * 2.0
    mae = (min(s.closes[i : j + 1]) / s.closes[i] - 1.0) * 100.0
    risk = max(ATR_STOP * (s.atr[i] or 0.0) / s.closes[i] * 100.0, 0.5)
    return Trade(ret, mae, s.dates[i], risk)


def _report(label: str, trades: list[Trade]) -> None:
    if not trades:
        print(f"  {label:32} n=0")
        return
    rets = [t.ret for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    payoff = avg_win / abs(avg_loss) if losses and avg_loss != 0.0 else float("inf")
    # equity curve sized so EACH trade risks 1% of capital (R = return / stop-distance)
    r_expectancy = mean(trade.ret / trade.risk for trade in trades)
    equity, peak, max_dd = 1.0, 1.0, 0.0
    for trade in sorted(trades, key=lambda t: t.date):
        equity *= 1.0 + 0.01 * (trade.ret / trade.risk)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
    print(f"  {label:32} n={len(trades):4} win%={100 * len(wins) / len(rets):.0f} "
          f"avgWin={avg_win:+.1f}% avgLoss={avg_loss:+.1f}% payoff={payoff:.2f} expR={r_expectancy:+.2f} "
          f"worstR={min(t.ret / t.risk for t in trades):+.1f} "
          f"equity(1%risk)={(equity - 1) * 100:+.0f}% maxDD={max_dd * 100:.0f}%")


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    regime = _spy_regime(provider)
    series = []
    bh_returns, bh_drawdowns = [], []
    for symbol in SYMBOLS:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)
        if len(bars) < 300:
            continue
        s = _series(bars)
        series.append(s)
        closes = s.closes
        bh_returns.append((closes[-1] / closes[0] - 1.0) * 100.0)
        peak, dd = closes[0], 0.0
        for price in closes:
            peak = max(peak, price)
            dd = min(dd, price / peak - 1.0)
        bh_drawdowns.append(dd * 100.0)

    print(f"\nRisk-management ablation: {len(series)} survivor tickers, real Yahoo OHLCV, full history.")
    print(f"buy-and-hold context: avg total {mean(bh_returns):+.0f}%  but avg max drawdown {mean(bh_drawdowns):.0f}% "
          f"(holding hurts in crashes -- the point of stops)\n")

    variants = (
        ("naive (no stop, RSI>70 exit)", {"stop": False, "trail": False, "use_regime": False}),
        ("+ ATR stop", {"stop": True, "trail": False, "use_regime": False}),
        ("+ trailing stop (let run)", {"stop": True, "trail": True, "use_regime": False}),
        ("+ regime filter (SPY>200d)", {"stop": True, "trail": True, "use_regime": True}),
    )
    for label, flags in variants:
        trades = [t for s in series for t in _trades(s, regime, **flags)]
        _report(label, trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
