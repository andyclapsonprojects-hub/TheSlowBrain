"""Train the network ON PROFIT itself (return-maximising loss) vs the usual cross-entropy, head to head.

Cross-entropy asks "did you pick the right label?". A return-maximising loss asks "how much money did
your bet make?": the net outputs a position p in [0,1] via a sigmoid, the bet earns p * forward_return,
and the loss is -(p * (return - average_return)) -- so minimising it MAXIMISES (excess) profit, and the
gradient pushes the weights to bet more on winners and less on losers. No labels at all.

We train BOTH on the same pullback features / same stocks and compare each model's entry picks (out of
sample, non-overlapping) against a random-entry control and buy-and-hold. The question: does training
directly on profit beat training on the label proxy? Real Yahoo OHLCV, read-only, nothing traded.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_mlp_pullback as mlp

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider
from slowbrain.microgix import Value, zero_grad

ReturnSnapshot = tuple[list[float], float]  # (weights, bias) for the linear position model


def _train_return(samples: list[mlp.Sample], rng: random.Random) -> ReturnSnapshot:
    """Linear position model trained to MAXIMISE excess return: loss = -(p * (ret - baseline))."""
    n_in = len(samples[0].x)
    weights = [Value(rng.uniform(-1, 1) * 0.3) for _ in range(n_in)]
    bias = Value(0.0)
    params = [*weights, bias]
    velocity = [0.0] * len(params)
    baseline = mean(s.ret for s in samples)  # average forward return -> learn to beat it
    fit = samples if len(samples) <= mlp.TRAIN_CAP else rng.sample(samples, mlp.TRAIN_CAP)
    for _ in range(mlp.EPOCHS):
        rng.shuffle(fit)
        for sample in fit:
            zero_grad(params)
            logit = bias
            for weight, feature in zip(weights, sample.x, strict=True):
                logit = logit + weight * feature
            position = logit.sigmoid()  # bet size in [0,1]
            loss = -(position * (sample.ret - baseline))  # maximise excess profit
            loss.backward()
            for index, param in enumerate(params):
                velocity[index] = 0.9 * velocity[index] - 0.05 * (param.grad + 0.001 * param.data)
                param.data += velocity[index]
    return [w.data for w in weights], bias.data


def _position(snapshot: ReturnSnapshot, x: list[float]) -> float:
    weights, bias = snapshot
    logit = bias + sum(w * xi for w, xi in zip(weights, x, strict=True))
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))


def _walk_position(samples: list[mlp.Sample], snapshot: ReturnSnapshot, threshold: float) -> list[mlp.Sample]:
    taken: list[mlp.Sample] = []
    cursor = -1
    for sample in sorted(samples, key=lambda s: s.entry):
        if sample.entry > cursor and _position(snapshot, sample.x) >= threshold:
            taken.append(sample)
            cursor = sample.exit
    return taken


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    per_ticker: dict[str, list[mlp.Sample]] = {}
    for symbol in mlp.SYMBOLS:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)
        if len(bars) >= 400:
            per_ticker[symbol] = mlp._build_ticker_samples(bars)

    dates = sorted(s.date for samples in per_ticker.values() for s in samples)
    split = dates[int(len(dates) * 0.7)]
    train_samples = [s for samples in per_ticker.values() for s in samples if s.date < split]
    test_by_ticker = {t: [s for s in samples if s.date >= split] for t, samples in per_ticker.items()}
    print(f"\nSame features, two loss functions | tickers={len(per_ticker)} train={len(train_samples)} "
          f"test={sum(len(v) for v in test_by_ticker.values())} OOS from {split}\n")

    ce_snapshot = mlp._train(train_samples, random.Random(7), deep=False)  # cross-entropy (label proxy)
    ret_snapshot = _train_return(train_samples, random.Random(7))  # return-maximising (profit itself)

    rng = random.Random(11)
    ce: list[mlp.Sample] = []
    profit: list[mlp.Sample] = []
    rnd: list[mlp.Sample] = []
    for samples in test_by_ticker.values():
        ce_trades = mlp._walk_trades(samples, ce_snapshot, deep=False)
        ce += ce_trades
        profit += _walk_position(samples, ret_snapshot, threshold=0.5)
        rnd += mlp._random_trades(samples, len(ce_trades), rng)

    print("=== loss = cross-entropy (predict the right label) ===")
    mlp._report("CE-entry", ce)
    print("\n=== loss = MAXIMISE PROFIT (return-maximising) ===")
    mlp._report("profit-entry", profit)
    print("\n=== control ===")
    mlp._report("random-entry", rnd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
