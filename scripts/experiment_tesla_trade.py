"""Trade TSLA from 2024 onward (established, post-hypergrowth, in the S&P 500) with the full strategy.

Train the return-maximising (profit) model on Tesla's history BEFORE 2024, then SIMULATE trading only
2024-01-01 -> now -- a fairer window than the cherry-picked 2022-crash-then-recovery. Full strategy:
profit-net entry (conviction above the model's own training median) + the validated ATR stop & trailing
stop on the exit. Start with GBP 10,000, compound the equity curve, and report trades / drawdown /
biggest loss / biggest win / total %. One ticker, one window -> indicative, not robust. Real Yahoo
OHLCV (already cached), read-only -- nothing is actually traded.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_mlp_pullback as mlp
import experiment_profit_loss as pl
import experiment_pullback_managed as pm

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider

TICKER = "TSLA"
TEST_START = "2024-01-01"  # only trade the established, post-hypergrowth period
START_CASH = 10_000.0
RISK_PER_TRADE = 0.01  # risk 1% of current equity per trade (the standard pro rule), sized by the stop


def _simulate(
    test: list[mlp.Sample], snapshot: pl.Snapshot, threshold: float, series: pm.Series
) -> tuple[list[float], float, float]:
    """Full strategy + 1%-risk position sizing. Returns (per-trade %-of-account, final GBP equity, maxDD%).

    Each trade risks RISK_PER_TRADE of the account: position size = (risk budget) / (stop distance), so a
    stop-out loses ~1% of equity regardless of the stock, and a big winner pays a multiple of that risk.
    """
    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    trade_pcts: list[float] = []
    cursor = -1
    for sample in sorted(test, key=lambda s: s.entry):
        if sample.entry > cursor and pl._position(snapshot, sample.x) >= threshold:
            trade, exit_index = pm._simulate(series, sample.entry, stop=True, trail=True)
            atr = series.atr[sample.entry] or 0.0
            stop_dist = max(pm.ATR_STOP * atr / series.closes[sample.entry], 0.005)  # fraction of price
            position = min(RISK_PER_TRADE * equity / stop_dist, equity)  # 1% risk, no leverage
            pnl = position * (trade.ret / 100.0)
            trade_pcts.append(100.0 * pnl / equity)
            equity += pnl
            cursor = exit_index
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
    return trade_pcts, equity, max_dd * 100.0


def _relabel_managed(samples: list[mlp.Sample], series: pm.Series) -> None:
    """LOCK the rules: make each sample's outcome the ATR-stop exit, so the model TRAINS on the rule it TRADES."""
    for sample in samples:
        trade, exit_index = pm._simulate(series, sample.entry, stop=True, trail=True)
        sample.ret = trade.ret
        sample.exit = exit_index


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    bars = sorted(provider.daily_prices(TICKER), key=lambda bar: bar.date)
    samples = mlp._build_ticker_samples(bars)
    series = pm._series(bars)
    _relabel_managed(samples, series)  # one consistent exit for training AND trading
    train = [s for s in samples if s.date < TEST_START]
    test = [s for s in samples if s.date >= TEST_START]
    closes = [(b.adjusted_close if b.adjusted_close > 0 else b.close) for b in bars if b.date >= TEST_START]
    buy_hold = (closes[-1] / closes[0] - 1.0) * 100.0

    print(f"\nTrading {TICKER} from {TEST_START} (established / S&P 500) | trained on {len(train)} pre-2024 samples")
    print(f"window: {test[0].date} -> {test[-1].date}  ({len(test)} candidate setups)")
    print(f"benchmark -- buy & hold: {buy_hold:+.1f}%   "
          f"GBP {START_CASH:,.0f} -> GBP {START_CASH * (1 + buy_hold / 100):,.0f}\n")

    for deep in (False, True):
        snapshot = pl._train_return(train, random.Random(7), deep=deep)
        threshold = median(pl._position(snapshot, s.x) for s in train)
        trade_pcts, equity, max_dd = _simulate(test, snapshot, threshold, series)
        label = "DEEP profit-net" if deep else "LINEAR profit-net"
        if not trade_pcts:
            print(f"{label} + ATR stop + 1%-risk sizing:  no trades taken in the window\n")
            continue
        wins = [p for p in trade_pcts if p > 0]
        total = (equity / START_CASH - 1.0) * 100.0
        verdict = "PROFIT" if equity > START_CASH else "LOSS"
        print(f"{label} + ATR stop + 1%-risk sizing  ({verdict}):")
        print(f"  trades:        {len(trade_pcts)}   win rate: {100 * len(wins) / len(trade_pcts):.0f}%")
        print(f"  biggest win:   {max(trade_pcts):+.1f}% of acct     biggest loss: {min(trade_pcts):+.1f}% of acct")
        print(f"  max drawdown:  {max_dd:.0f}%")
        print(f"  TOTAL RETURN:  {total:+.1f}%   GBP {START_CASH:,.0f} -> GBP {equity:,.0f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
