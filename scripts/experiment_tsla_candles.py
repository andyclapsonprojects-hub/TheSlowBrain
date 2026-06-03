"""Single-stock test: can the NN learn candlestick/technical buy signals on TSLA and beat holding it?

One clean split-adjusted source (Yahoo). For each day the NN sees the candlestick shape plus the full
technical context (RSI/MACD/Bollinger/EMA/ATR), labelled by forward return. Train on the early window,
test out-of-sample on the recent window. The honest benchmark is buy-and-hold TSLA. Read-only.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import load_config
from slowbrain.gating_model import build_gating_dataset
from slowbrain.gating_training import train_gate
from slowbrain.indicator_features import attach_indicators_from_history
from slowbrain.market_data_vendors.providers import build_price_history_provider
from slowbrain.models import FeatureVector

TICKER = "TSLA"
HOLD = 10
COST_BPS = 15.0


def main() -> int:
    provider = build_price_history_provider(load_config(Path.cwd()), project_root=Path.cwd())
    if provider is None:
        print("Market data disabled.")
        return 1
    bars = sorted(provider.daily_prices(TICKER), key=lambda bar: bar.date)
    if len(bars) < 400:
        print(f"{TICKER}: only {len(bars)} bars (feed offline/rate-limited?).")
        return 1
    closes = [bar.adjusted_close if bar.adjusted_close > 0.0 else bar.close for bar in bars]
    rows = [
        FeatureVector(
            idea_id=f"{TICKER}-{bars[i].date}", ticker=TICKER, signal_date=bars[i].date,
            sentiment="neutral", sentiment_confidence=0.0, catalyst_strength=0.0, trend="unknown",
            momentum_20d_pct=_pct(closes[i], closes[i - 20]), mean_reversion_z_20d=0.0,
            volume_confirmed=False, quality_status="pass", risk_status="pass",
            net_return_pct=_pct(closes[i + HOLD], closes[i]) - COST_BPS / 100.0 * 2.0,
            cost_bps=COST_BPS, source="yahoo", horizon_days=HOLD, entry_price=closes[i],
        )
        for i in range(60, len(bars) - HOLD)
    ]
    rows = attach_indicators_from_history(rows, provider)

    cut = int(len(rows) * 0.7)
    train_features, test_features = rows[:cut], rows[cut + HOLD :]
    random.seed(11)
    train_ds = build_gating_dataset(train_features, _rubric())
    test_ds = build_gating_dataset(test_features, _rubric())
    gate, _ = train_gate(random.sample(train_ds, min(3000, len(train_ds))), max_epochs=40)

    base = mean(row.forward_return_pct for row in test_ds)
    buy_hold = _pct(closes[len(bars) - 1], test_features[0].entry_price or closes[0])
    print(f"\n{TICKER}: {len(bars)} bars {bars[0].date}..{bars[-1].date} | train {len(train_features)} / "
          f"test {len(test_features)} (OOS {test_features[0].signal_date}..{test_features[-1].signal_date})")
    print(f"buy-and-hold TSLA over OOS window: {buy_hold:+.1f}%")
    print(f"avg {HOLD}d forward return of EVERY OOS day (after cost): {base:+.2f}%  <- the bar to beat per signal\n")

    nn_buys = [row for row in test_ds if gate.predict_label(row) == "BUY"]
    _report("NN BUY days (candles+technicals)", nn_buys, base)
    k = max(20, len(test_ds) // 10)
    top = sorted(test_ds, key=lambda row: gate.probabilities(row)["BUY"], reverse=True)[:k]
    _report("NN top-decile by P(BUY)", top, base)
    return 0


def _report(name: str, picks: list[object], base: float) -> None:
    if not picks:
        print(f"  {name:34} BUYs=0")
        return
    returns = [row.forward_return_pct for row in picks]  # type: ignore[attr-defined]
    hit = 100.0 * sum(value > 0 for value in returns) / len(returns)
    verdict = "beats hold" if mean(returns) > base else "WORSE than hold"
    print(f"  {name:34} n={len(picks):4} avg={mean(returns):+.2f}% hit={hit:.0f}%  ({verdict})")


def _pct(numerator: float, denominator: float) -> float:
    return (numerator / denominator - 1.0) * 100.0 if denominator > 0.0 else 0.0


def _rubric() -> object:
    from slowbrain.rubrics import BASE_RUBRIC

    return BASE_RUBRIC


if __name__ == "__main__":
    raise SystemExit(main())
