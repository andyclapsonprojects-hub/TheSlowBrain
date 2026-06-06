"""The full 2x2: {cross-entropy loss, profit-maximising loss} x {linear, deep hidden layer}.

Cross-entropy asks "did you pick the right label?". A return-maximising loss asks "how much money did
your bet make?": the net outputs a position p in [0,1] (sigmoid), the bet earns p * forward_return, and
the loss is -(p * (return - average_return)) -- so minimising it MAXIMISES excess profit, no labels.

We train all four combinations on the same pullback features / same stocks, judge each by its top-k
highest-conviction non-overlapping trades (matched count, so the comparison is fair), and compare to a
random-entry control. Questions: does training on profit beat the label loss? And does a hidden layer
help EITHER loss? Real Yahoo OHLCV, OOS split, read-only -- nothing traded.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_mlp_pullback as mlp

from slowbrain.config import load_config
from slowbrain.market_data_vendors.providers import build_price_history_provider
from slowbrain.microgix import Value, zero_grad

Snapshot = tuple[list[list[float]], list[float], list[float], float]  # (w1, b1, w2, b2); w1/b1 empty = linear


class _ReturnNet:
    """A position model (1 output, sigmoid) trained to maximise profit; linear or with a hidden layer."""

    def __init__(self, n_in: int, n_hidden: int, rng: random.Random, *, deep: bool) -> None:
        self.deep = deep
        self.w1 = [[Value(rng.uniform(-1, 1) * 0.3) for _ in range(n_in)] for _ in range(n_hidden)] if deep else []
        self.b1 = [Value(0.0) for _ in range(n_hidden)] if deep else []
        out_in = n_hidden if deep else n_in
        self.w2 = [Value(rng.uniform(-1, 1) * 0.3) for _ in range(out_in)]
        self.b2 = Value(0.0)

    def parameters(self) -> list[Value]:
        return [p for row in self.w1 for p in row] + list(self.b1) + [*self.w2, self.b2]

    def position(self, x: list[float]) -> Value:
        hidden: list[Value | float]
        if self.deep:
            hidden = []
            for weights, bias in zip(self.w1, self.b1, strict=True):
                total: Value = bias
                for wi, xi in zip(weights, x, strict=True):
                    total = total + wi * xi
                hidden.append(total.tanh())
        else:
            hidden = list(x)
        logit: Value = self.b2
        for wi, hi in zip(self.w2, hidden, strict=True):
            logit = logit + wi * hi
        return logit.sigmoid()

    def snapshot(self) -> Snapshot:
        return ([[w.data for w in row] for row in self.w1], [b.data for b in self.b1],
                [w.data for w in self.w2], self.b2.data)


def _train_return(samples: list[mlp.Sample], rng: random.Random, *, deep: bool) -> Snapshot:
    net = _ReturnNet(len(samples[0].x), mlp.HIDDEN, rng, deep=deep)
    params = net.parameters()
    velocity = [0.0] * len(params)
    baseline = mean(s.ret for s in samples)  # learn to beat the average forward return
    fit = samples if len(samples) <= mlp.TRAIN_CAP else rng.sample(samples, mlp.TRAIN_CAP)
    for _ in range(mlp.EPOCHS):
        rng.shuffle(fit)
        for sample in fit:
            zero_grad(params)
            loss = -(net.position(sample.x) * (sample.ret - baseline))  # maximise excess profit
            loss.backward()
            for index, param in enumerate(params):
                velocity[index] = 0.9 * velocity[index] - 0.05 * (param.grad + 0.001 * param.data)
                param.data += velocity[index]
    return net.snapshot()


def _hidden_floats(w1: list[list[float]], b1: list[float], x: list[float]) -> list[float]:
    if not w1:
        return x
    return [
        math.tanh(bias + sum(wi * xi for wi, xi in zip(row, x, strict=True)))
        for row, bias in zip(w1, b1, strict=True)
    ]


def _position(snapshot: Snapshot, x: list[float]) -> float:
    w1, b1, w2, b2 = snapshot
    hidden = _hidden_floats(w1, b1, x)
    logit = b2 + sum(wi * hi for wi, hi in zip(w2, hidden, strict=True))
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))


def _ce_pbuy(snapshot: Snapshot, x: list[float]) -> float:
    """The cross-entropy model's P(BUY) -- its conviction, for ranking. Handles linear or deep."""
    w1, b1, w2, b2 = snapshot
    hidden = _hidden_floats(w1, b1, x)
    logits = [b + sum(wi * hi for wi, hi in zip(row, hidden, strict=True)) for row, b in zip(w2, b2, strict=True)]
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    return exps[mlp.GATING_LABELS.index("BUY")] / sum(exps)


def _walk_topk(samples: list[mlp.Sample], score: Callable[[mlp.Sample], float], k: int) -> list[mlp.Sample]:
    """The k highest-scoring NON-overlapping trades -- every model judged the same way at the same count."""
    if k <= 0:
        return []
    taken: list[mlp.Sample] = []
    for sample in sorted(samples, key=score, reverse=True):
        if len(taken) >= k:
            break
        if all(not (sample.entry <= t.exit and sample.exit >= t.entry) for t in taken):
            taken.append(sample)
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
    print(f"\n2x2: loss x depth | tickers={len(per_ticker)} train={len(train_samples)} "
          f"test={sum(len(v) for v in test_by_ticker.values())} OOS from {split}\n")

    ce_lin = mlp._train(train_samples, random.Random(7), deep=False)
    ce_deep = mlp._train(train_samples, random.Random(7), deep=True)
    pr_lin = _train_return(train_samples, random.Random(7), deep=False)
    pr_deep = _train_return(train_samples, random.Random(7), deep=True)

    rng = random.Random(11)
    out: dict[str, list[mlp.Sample]] = {k: [] for k in ("ce_lin", "ce_deep", "pr_lin", "pr_deep", "rnd")}
    for samples in test_by_ticker.values():
        k = len(mlp._walk_trades(samples, ce_lin, deep=False))  # matched count = CE-linear's natural BUYs
        out["ce_lin"] += _walk_topk(samples, lambda s, sn=ce_lin: _ce_pbuy(sn, s.x), k)
        out["ce_deep"] += _walk_topk(samples, lambda s, sn=ce_deep: _ce_pbuy(sn, s.x), k)
        out["pr_lin"] += _walk_topk(samples, lambda s, sn=pr_lin: _position(sn, s.x), k)
        out["pr_deep"] += _walk_topk(samples, lambda s, sn=pr_deep: _position(sn, s.x), k)
        out["rnd"] += mlp._random_trades(samples, k, rng)

    print("=== CROSS-ENTROPY loss (predict the label) ===")
    mlp._report("  linear", out["ce_lin"])
    mlp._report("  deep (hidden layer)", out["ce_deep"])
    print("\n=== PROFIT-MAXIMISING loss (train on money) ===")
    mlp._report("  linear", out["pr_lin"])
    mlp._report("  deep (hidden layer)", out["pr_deep"])
    print("\n=== control ===")
    mlp._report("  random", out["rnd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
