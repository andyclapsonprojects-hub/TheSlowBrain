"""Hidden-layer (deep) gate vs the linear gate, on Andy's pullback-reversal swing thesis.

The thesis becomes FEATURES first (so a hidden scout can combine them):
  range_pos      -- where price sits in its 20d range (0 = at the low, 1 = at the high)
  vol_ratio_3_20 -- recent 3d volume vs 20d average (LOW = volume drying up / exhaustion)
  kick_up        -- how far price has bounced off its recent 5d low (the reversal "kick")
  lower_tail     -- size of the lower wick (hammer-ish rejection of the lows)
  rsi_norm       -- RSI/100 (HIGH = overbought, the exit zone)
  mom20, mom63   -- momentum (the ride)

A hidden node can now fire on "range_pos low AND vol fading AND kick_up AND lower_tail" = Andy's bottom.

Strategy tested: the NN picks the ENTRY; a thesis RULE exits (RSI>=70, or volume fades near the high, or
a 60-day cap). Measured with NON-overlapping trades + a matched random-entry control + buy-and-hold, and
the SAME model trained linear (no hidden layer) vs deep (hidden layer) so we see if depth adds anything.
Read-only Yahoo bars, survivor large-caps (the falling-knife caveat applies). Nothing is traded.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import load_config
from slowbrain.gating_model import GATING_LABELS, target_label_for_return
from slowbrain.market_data_vendors.providers import build_price_history_provider
from slowbrain.microgix import Value, zero_grad

_SYMBOL_TEXT = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ADBE CRM ORCL CSCO INTC AMD QCOM TXN IBM NOW INTU "
    "JPM BAC WFC GS MS C AXP BLK SCHW USB PNC UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD "
    "CVS HD LOW MCD SBUX NKE TGT COST WMT PG KO PEP PM MO XOM CVX COP SLB EOG BA CAT GE HON UNP "
    "UPS LMT RTX DE DIS CMCSA NFLX T VZ"
)
SYMBOLS = _SYMBOL_TEXT.split()
FEATURES = ("range_pos", "vol_ratio_3_20", "kick_up", "lower_tail", "rsi_norm", "mom20", "mom63")
COST_BPS = 15.0
MAX_HOLD = 60
RSI_OVERBOUGHT = 70.0
HIDDEN = 12
TRAIN_CAP = 1500
EPOCHS = 35


@dataclass
class Sample:
    date: str
    x: list[float]
    label: str
    ret: float
    mae: float
    entry: int
    exit: int


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _build_ticker_samples(bars: list) -> list[Sample]:
    closes = [b.adjusted_close if b.adjusted_close > 0.0 else b.close for b in bars]
    highs, lows = [b.high for b in bars], [b.low for b in bars]
    opens, raw_close = [b.open for b in bars], [b.close for b in bars]
    volumes = [float(b.volume) for b in bars]
    rsi = _rsi(closes)
    samples: list[Sample] = []
    for i in range(200, len(closes) - 1, 3):
        if rsi[i] is None:
            continue
        hi20, lo20 = max(closes[i - 19 : i + 1]), min(closes[i - 19 : i + 1])
        vol20 = mean(volumes[i - 19 : i + 1]) or 1.0
        low5 = min(closes[i - 5 : i + 1])
        candle_range = max(highs[i] - lows[i], 1e-9)
        x = [
            _clamp((closes[i] - lo20) / (hi20 - lo20 + 1e-9), 0.0, 1.0),  # range_pos: 0=low, 1=high
            _clamp(mean(volumes[i - 2 : i + 1]) / vol20, 0.0, 2.0) / 2.0,  # vol_ratio_3_20 -> [0,1]
            _clamp((closes[i] / low5 - 1.0) / 0.10, 0.0, 1.0),  # kick_up off recent low (cap +10%)
            _clamp((min(opens[i], raw_close[i]) - lows[i]) / candle_range, 0.0, 1.0),  # lower_tail (hammer-ish)
            (rsi[i] or 50.0) / 100.0,  # rsi_norm: high=overbought
            _clamp((closes[i] / closes[i - 20] - 1.0) / 0.20, -1.0, 1.0),  # mom20
            _clamp((closes[i] / closes[i - 63] - 1.0) / 0.50, -1.0, 1.0),  # mom63
        ]
        exit_index = _simulate_exit(i, closes, volumes, rsi)
        ret = (closes[exit_index] / closes[i] - 1.0) * 100.0 - COST_BPS / 100.0 * 2.0
        mae = (min(closes[i : exit_index + 1]) / closes[i] - 1.0) * 100.0
        samples.append(Sample(bars[i].date, x, target_label_for_return(ret), ret, mae, i, exit_index))
    return samples


def _simulate_exit(entry: int, closes: list[float], volumes: list[float], rsi: list[float | None]) -> int:
    for j in range(entry + 1, min(entry + MAX_HOLD, len(closes) - 1) + 1):
        overbought = rsi[j] is not None and (rsi[j] or 0.0) >= RSI_OVERBOUGHT
        near_high = closes[j] >= max(closes[max(0, j - 19) : j + 1]) * 0.98
        vol_fade = mean(volumes[j - 2 : j + 1]) < (mean(volumes[j - 19 : j + 1]) or 1.0) * 0.8
        if overbought or (near_high and vol_fade):
            return j
    return min(entry + MAX_HOLD, len(closes) - 1)


class Net:
    def __init__(self, n_in: int, n_hidden: int, rng: random.Random, *, deep: bool) -> None:
        self.deep = deep
        self.w1 = [[Value(rng.uniform(-1, 1) * 0.5) for _ in range(n_in)] for _ in range(n_hidden)] if deep else []
        self.b1 = [Value(0.0) for _ in range(n_hidden)] if deep else []
        out_in = n_hidden if deep else n_in
        self.w2 = [[Value(rng.uniform(-1, 1) * 0.5) for _ in range(out_in)] for _ in range(len(GATING_LABELS))]
        self.b2 = [Value(0.0) for _ in range(len(GATING_LABELS))]

    def parameters(self) -> list[Value]:
        flat = [p for row in self.w1 for p in row] + list(self.b1)
        return flat + [p for row in self.w2 for p in row] + list(self.b2)

    def logits(self, x: list[float]) -> list[Value]:
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
        outputs: list[Value] = []
        for weights, bias in zip(self.w2, self.b2, strict=True):
            total = bias
            for wi, hi in zip(weights, hidden, strict=True):
                total = total + wi * hi
            outputs.append(total)
        return outputs

    def snapshot(self) -> tuple:
        return (
            [[w.data for w in row] for row in self.w1],
            [b.data for b in self.b1],
            [[w.data for w in row] for row in self.w2],
            [b.data for b in self.b2],
        )


def _ce_loss(logits: list[Value], target_index: int, class_weight: float) -> Value:
    peak = max(logit.data for logit in logits)
    exps = [(logit - peak).exp() for logit in logits]
    total = exps[0]
    for value in exps[1:]:
        total = total + value
    return (-(exps[target_index] / total).log()) * class_weight


def _train(samples: list[Sample], rng: random.Random, *, deep: bool) -> tuple:
    net = Net(len(FEATURES), HIDDEN, rng, deep=deep)
    params = net.parameters()
    velocity = [0.0] * len(params)
    counts = {label: sum(1 for s in samples if s.label == label) for label in GATING_LABELS}
    weights = {label: len(samples) / (len(GATING_LABELS) * max(count, 1)) for label, count in counts.items()}
    fit = samples if len(samples) <= TRAIN_CAP else rng.sample(samples, TRAIN_CAP)
    for _ in range(EPOCHS):
        rng.shuffle(fit)
        for sample in fit:
            zero_grad(params)
            _ce_loss(net.logits(sample.x), GATING_LABELS.index(sample.label), weights[sample.label]).backward()
            for index, param in enumerate(params):
                velocity[index] = 0.9 * velocity[index] - 0.05 * (param.grad + 0.001 * param.data)
                param.data += velocity[index]
    return net.snapshot()


def _predict_label(snapshot: tuple, x: list[float], *, deep: bool) -> str:
    w1, b1, w2, b2 = snapshot
    if deep:
        hidden = [
            math.tanh(bias + sum(wi * xi for wi, xi in zip(row, x, strict=True)))
            for row, bias in zip(w1, b1, strict=True)
        ]
    else:
        hidden = x
    logits = [bias + sum(wi * hi for wi, hi in zip(row, hidden, strict=True)) for row, bias in zip(w2, b2, strict=True)]
    return GATING_LABELS[max(range(len(GATING_LABELS)), key=lambda k: logits[k])]


def _walk_trades(samples: list[Sample], snapshot: tuple, *, deep: bool) -> list[Sample]:
    taken: list[Sample] = []
    cursor = -1
    for sample in sorted(samples, key=lambda s: s.entry):
        if sample.entry > cursor and _predict_label(snapshot, sample.x, deep=deep) == "BUY":
            taken.append(sample)
            cursor = sample.exit
    return taken


def _random_trades(samples: list[Sample], count: int, rng: random.Random) -> list[Sample]:
    taken: list[Sample] = []
    cursor = -1
    for sample in sorted(samples, key=lambda s: s.entry):
        if len(taken) >= count:
            break
        if sample.entry > cursor and rng.random() < 0.5:
            taken.append(sample)
            cursor = sample.exit
    return taken


def _report(label: str, trades: list[Sample]) -> None:
    if not trades:
        print(f"  {label:28} n=0")
        return
    rets = [t.ret for t in trades]
    maes = sorted(t.mae for t in trades)
    avg, std = mean(rets), pstdev(rets) or 1e-9
    hit = 100.0 * sum(r > 0 for r in rets) / len(rets)
    print(f"  {label:28} n={len(trades):4} avg={avg:+.2f}% hit={hit:.0f}% t={avg / (std / len(rets) ** 0.5):.1f} "
          f"avgMAE={mean(maes):+.1f}% worstMAE={maes[0]:+.1f}%")


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    per_ticker: dict[str, list[Sample]] = {}
    prices: dict[str, list[tuple[str, float]]] = {}
    for symbol in SYMBOLS:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)
        if len(bars) >= 400:
            per_ticker[symbol] = _build_ticker_samples(bars)
            prices[symbol] = [(b.date, b.adjusted_close if b.adjusted_close > 0.0 else b.close) for b in bars]

    dates = sorted(s.date for samples in per_ticker.values() for s in samples)
    split = dates[int(len(dates) * 0.7)]
    train_samples = [s for samples in per_ticker.values() for s in samples if s.date < split]
    test_by_ticker = {t: [s for s in samples if s.date >= split] for t, samples in per_ticker.items()}
    buy_hold = [
        (test[-1][1] / test[0][1] - 1.0) * 100.0
        for series in prices.values()
        if len(test := [(d, p) for d, p in series if d >= split and p > 0.0]) >= 2
    ]
    print(f"\ntickers={len(per_ticker)}  train trades={len(train_samples)}  "
          f"test trades={sum(len(v) for v in test_by_ticker.values())}  OOS from {split}")
    print(f"buy-and-hold over the OOS window: {mean(buy_hold):+.1f}% avg per ticker  "
          f"(exit rule: RSI>=70 / volume-fade-near-high / {MAX_HOLD}d cap)\n")

    rng = random.Random(11)
    for deep in (False, True):
        snapshot = _train(train_samples, random.Random(7), deep=deep)
        nn: list[Sample] = []
        rnd: list[Sample] = []
        for samples in test_by_ticker.values():
            taken = _walk_trades(samples, snapshot, deep=deep)
            nn += taken
            rnd += _random_trades(samples, len(taken), rng)
        print(f"=== {'DEEP (hidden layer)' if deep else 'LINEAR (no hidden layer)'} ===")
        _report("NN-entry + thesis-exit", nn)
        _report("random-entry + same-exit", rnd)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
