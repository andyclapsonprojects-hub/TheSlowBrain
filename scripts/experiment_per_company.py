"""Per-company: train the model on each stock's early years, test it out-of-sample on the recent years.

Exactly the workflow Andy asked for -- pick a company (Tesla first), build the pullback features, train
the (linear) entry model on the first ~70% of its history, then test the trained model on the last ~30%
it has never seen. Repeat for a basket of liquid names. The honest tell is CONSISTENCY: does the trained
model's entries beat random entry (and buy-and-hold) on MOST companies, or just a lucky one?

Reuses the feature/sample/training machinery from experiment_mlp_pullback. Real Yahoo OHLCV, read-only.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_mlp_pullback as mlp  # reuse Sample, _build_ticker_samples, _train, _walk_trades, _random_trades

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider

# Long-standing S&P 500 blue-chips (in the index 15+ years, established the whole time -- no recent-IPO
# "story" phase). Override on the command line: `python scripts/experiment_per_company.py JNJ KO PG ...`.
DEFAULT_TICKERS = (
    "JPM", "JNJ", "PG", "KO", "XOM", "CVX", "WMT", "HD", "MCD", "PEP", "IBM", "CAT",
    "MMM", "DIS", "MRK", "PFE", "CSCO", "INTC", "AXP", "GS", "UNH", "ABT", "LOW", "COST",
)


def _avg(trades: list[mlp.Sample]) -> tuple[int, float, float]:
    if not trades:
        return 0, 0.0, 0.0
    rets = [t.ret for t in trades]
    return len(rets), mean(rets), 100.0 * sum(r > 0 for r in rets) / len(rets)


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    tickers = tuple(t.upper() for t in sys.argv[1:]) or DEFAULT_TICKERS
    rng = random.Random(11)
    print(f"\n{'ticker':6} | {'OOS test window':23} | {'trained-model entry':26} | {'random entry':16} | buy & hold")
    print("-" * 100)
    nn_beats_random = 0
    nn_beats_hold = 0
    tested = 0
    for ticker in tickers:
        bars = sorted(provider.daily_prices(ticker), key=lambda bar: bar.date)
        if len(bars) < 600:
            continue
        samples = mlp._build_ticker_samples(bars)
        dates = sorted(s.date for s in samples)
        split = dates[int(len(dates) * 0.7)]
        train = [s for s in samples if s.date < split]
        test = [s for s in samples if s.date >= split]
        if len(train) < 100 or len(test) < 30:
            continue
        snapshot = mlp._train(train, random.Random(7), deep=False)  # linear model, trained on THIS stock's past
        nn = mlp._walk_trades(test, snapshot, deep=False)
        rnd = mlp._random_trades(test, len(nn), rng)
        closes = [
            (b.adjusted_close if b.adjusted_close > 0 else b.close) for b in bars if b.date >= split
        ]
        hold = (closes[-1] / closes[0] - 1.0) * 100.0 if len(closes) >= 2 else 0.0

        n_nn, avg_nn, hit_nn = _avg(nn)
        _n_r, avg_r, hit_r = _avg(rnd)
        tested += 1
        nn_beats_random += avg_nn > avg_r
        nn_beats_hold += avg_nn > hold
        print(f"{ticker:6} | {split}..{dates[-1]:11} | n={n_nn:3} avg={avg_nn:+5.2f}%/trade hit={hit_nn:.0f}% | "
              f"avg={avg_r:+5.2f}% hit={hit_r:.0f}% | total {hold:+.0f}%")

    print("-" * 100)
    print(f"trained model's entries beat RANDOM entry on {nn_beats_random}/{tested} companies "
          f"(per-trade); beat buy-and-hold total on {nn_beats_hold}/{tested}.")
    print("Note: 'avg %/trade' is PER trade; buy-and-hold 'total' is the whole window held continuously.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
