"""Give the hidden-layer scouts the TIME dimension: lagged features so they can see the dip FORM.

The earlier MLP saw only a single-day snapshot, which can't represent a sequence ("volume was heavy
during the fall, THEN faded, THEN price kicked up"). Here we add the trajectory -- range position,
volume ratio, and RSI at lags 0/3/6 days, plus an explicit volume-fade-after-selloff ratio and a 3-day
kick. Now a scout CAN combine "RSI was deep and is turning up AND volume faded AND price is climbing out
of the low". We train the LINEAR and the DEEP (hidden-layer) model on these richer inputs and compare
NN-entry vs random-entry out-of-sample -- to see if the time dimension finally lets depth beat the line.

Reuses the network/train/eval machinery from experiment_mlp_pullback (real Yahoo OHLCV, read-only).
Proper validation/early-stopping is deliberately deferred (Andy: run it first, tune later).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_mlp_pullback as mlp

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider

SEQ_FEATURES = (
    "range_pos", "range_pos_l3", "range_pos_l6", "vol_ratio", "vol_l3", "vol_l6", "vol_fade",
    "rsi", "rsi_l3", "rsi_l6", "kick_up", "lower_tail", "mom20", "mom63", "ret3",
)


def _build_samples_seq(bars: list) -> list[mlp.Sample]:
    closes = [b.adjusted_close if b.adjusted_close > 0.0 else b.close for b in bars]
    highs, lows = [b.high for b in bars], [b.low for b in bars]
    opens, raw_close = [b.open for b in bars], [b.close for b in bars]
    volumes = [float(b.volume) for b in bars]
    rsi = mlp._rsi(closes)
    clamp = mlp._clamp

    def rp(i: int) -> float:
        lo, hi = min(closes[i - 19 : i + 1]), max(closes[i - 19 : i + 1])
        return clamp((closes[i] - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    def vr(i: int) -> float:
        return clamp(mean(volumes[i - 2 : i + 1]) / (mean(volumes[i - 19 : i + 1]) or 1.0), 0.0, 2.0) / 2.0

    def rn(i: int) -> float:
        return (rsi[i] or 50.0) / 100.0

    samples: list[mlp.Sample] = []
    for i in range(210, len(closes) - 1, 3):
        if rsi[i] is None or rsi[i - 6] is None:
            continue
        selloff_vol = mean(volumes[i - 10 : i - 2]) or 1.0
        x = [
            rp(i), rp(i - 3), rp(i - 6),
            vr(i), vr(i - 3), vr(i - 6),
            clamp(mean(volumes[i - 2 : i + 1]) / selloff_vol, 0.0, 2.0) / 2.0,  # vol_fade (low = faded)
            rn(i), rn(i - 3), rn(i - 6),
            clamp((closes[i] / min(closes[i - 5 : i + 1]) - 1.0) / 0.10, 0.0, 1.0),  # kick_up
            clamp((min(opens[i], raw_close[i]) - lows[i]) / max(highs[i] - lows[i], 1e-9), 0.0, 1.0),  # lower_tail
            clamp((closes[i] / closes[i - 20] - 1.0) / 0.20, -1.0, 1.0),  # mom20
            clamp((closes[i] / closes[i - 63] - 1.0) / 0.50, -1.0, 1.0),  # mom63
            clamp((closes[i] / closes[i - 3] - 1.0) / 0.05, -1.0, 1.0),  # ret3 (the kick)
        ]
        exit_index = mlp._simulate_exit(i, closes, volumes, rsi)
        ret = (closes[exit_index] / closes[i] - 1.0) * 100.0 - mlp.COST_BPS / 100.0 * 2.0
        mae = (min(closes[i : exit_index + 1]) / closes[i] - 1.0) * 100.0
        samples.append(mlp.Sample(bars[i].date, x, mlp.target_label_for_return(ret), ret, mae, i, exit_index))
    return samples


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    mlp.FEATURES = SEQ_FEATURES  # so mlp._train builds the network with the right number of inputs
    per_ticker: dict[str, list[mlp.Sample]] = {}
    for symbol in mlp.SYMBOLS:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)
        if len(bars) >= 400:
            per_ticker[symbol] = _build_samples_seq(bars)

    dates = sorted(s.date for samples in per_ticker.values() for s in samples)
    split = dates[int(len(dates) * 0.7)]
    train_samples = [s for samples in per_ticker.values() for s in samples if s.date < split]
    test_by_ticker = {t: [s for s in samples if s.date >= split] for t, samples in per_ticker.items()}
    print(f"\nSEQUENTIAL features ({len(SEQ_FEATURES)} inputs, lags 0/3/6) | tickers={len(per_ticker)} "
          f"train={len(train_samples)} test={sum(len(v) for v in test_by_ticker.values())} OOS from {split}\n")

    rng = random.Random(11)
    for deep in (False, True):
        snapshot = mlp._train(train_samples, random.Random(7), deep=deep)
        nn: list[mlp.Sample] = []
        rnd: list[mlp.Sample] = []
        for samples in test_by_ticker.values():
            taken = mlp._walk_trades(samples, snapshot, deep=deep)
            nn += taken
            rnd += mlp._random_trades(samples, len(taken), rng)
        print(f"=== {'DEEP (hidden layer)' if deep else 'LINEAR (no hidden layer)'} ===")
        mlp._report("NN-entry + thesis-exit", nn)
        mlp._report("random-entry + same-exit", rnd)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
