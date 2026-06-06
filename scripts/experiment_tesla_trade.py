"""Trade ONE ticker (TSLA) with the profit-trained network and see if it ends in profit.

Train the return-maximising (profit) model on Tesla's early years, then SIMULATE trading the years it
has never seen: start with $10,000, enter when the model's conviction beats its own training median,
hold to the thesis exit, compound the result into an equity curve. Compare to simply buying and holding
Tesla over the same window. One ticker, one period -> indicative, not robust (regime-dependent). Real
Yahoo OHLCV (the bars already cached), read-only -- nothing is actually traded.
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

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider

TICKER = "TSLA"
START_CASH = 10_000.0


def _simulate(test: list[mlp.Sample], snapshot: pl.Snapshot, threshold: float) -> tuple[int, float, float, float]:
    equity, peak, max_dd, wins, total = 1.0, 1.0, 0.0, 0, 0
    cursor = -1
    for sample in sorted(test, key=lambda s: s.entry):
        if sample.entry > cursor and pl._position(snapshot, sample.x) >= threshold:
            equity *= 1.0 + sample.ret / 100.0
            cursor = sample.exit
            total += 1
            wins += sample.ret > 0
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
    hit = 100.0 * wins / total if total else 0.0
    return total, (equity - 1.0) * 100.0, max_dd * 100.0, hit


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    bars = sorted(provider.daily_prices(TICKER), key=lambda bar: bar.date)
    if len(bars) < 600:
        print(f"{TICKER}: only {len(bars)} bars.")
        return 1
    samples = mlp._build_ticker_samples(bars)
    dates = sorted(s.date for s in samples)
    split = dates[int(len(dates) * 0.7)]
    train = [s for s in samples if s.date < split]
    test = [s for s in samples if s.date >= split]
    closes = [(b.adjusted_close if b.adjusted_close > 0 else b.close) for b in bars if b.date >= split]
    buy_hold = (closes[-1] / closes[0] - 1.0) * 100.0

    print(f"\nTrading {TICKER} | train {len(train)} / test {len(test)} samples | "
          f"OOS {test[0].date}..{test[-1].date}")
    print(f"benchmark -- buy & hold {TICKER} over the OOS window: {buy_hold:+.1f}%  "
          f"(${START_CASH:,.0f} -> ${START_CASH * (1 + buy_hold / 100):,.0f})\n")

    for deep in (False, True):
        snapshot = pl._train_return(train, random.Random(7), deep=deep)
        threshold = median(pl._position(snapshot, s.x) for s in train)  # the model's own typical conviction
        trades, total, max_dd, hit = _simulate(test, snapshot, threshold)
        label = "DEEP profit-net " if deep else "LINEAR profit-net"
        verdict = "PROFIT" if total > 0 else "LOSS"
        print(f"  {label}: {trades:3} trades  hit={hit:.0f}%  total={total:+.1f}%  maxDD={max_dd:.0f}%  "
              f"${START_CASH:,.0f} -> ${START_CASH * (1 + total / 100):,.0f}  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
