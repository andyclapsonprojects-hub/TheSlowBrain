"""Train the NN on REAL Yahoo technicals and measure whether its picks make money (after cost).

Builds (ticker, date) samples from real daily bars, computes the full indicator feature set
point-in-time, labels each with its forward return, trains the gate on an earlier window, and
measures profit on a later out-of-sample window. Read-only; nothing is traded.
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
from slowbrain.rubrics import BASE_RUBRIC

TICKERS = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ADBE CRM ORCL CSCO INTC AMD QCOM TXN IBM NOW INTU",
    "JPM BAC WFC GS MS C AXP BLK SCHW USB PNC",
    "UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD CVS",
    "HD LOW MCD SBUX NKE TGT COST WMT PG KO PEP PM MO",
    "XOM CVX COP SLB EOG", "BA CAT GE HON UNP UPS LMT RTX DE", "DIS CMCSA NFLX T VZ",
)
HOLD = 20
COST_BPS = 15.0


def main() -> int:
    root = Path.cwd()
    provider = build_price_history_provider(load_config(root), project_root=root)
    if provider is None:
        print("Market data disabled.")
        return 1
    symbols = [symbol for line in TICKERS for symbol in line.split()]
    rows = _build_rows(provider, symbols)
    if len(rows) < 500:
        print(f"Only {len(rows)} rows built (feed offline/rate-limited?). Aborting.")
        return 1
    rows = attach_indicators_from_history(rows, provider)
    rows.sort(key=lambda feature: feature.signal_date)
    cut = int(len(rows) * 0.7)
    train_features, test_features = rows[:cut], rows[cut + 50 :]

    train_ds_full = build_gating_dataset(train_features, BASE_RUBRIC)
    test_ds = build_gating_dataset(test_features, BASE_RUBRIC)
    random.seed(11)
    gate, _ = train_gate(random.sample(train_ds_full, min(3000, len(train_ds_full))), max_epochs=40)

    overall = mean(row.forward_return_pct for row in test_ds)
    print(f"\nrows={len(rows)} (train {len(train_features)} / test {len(test_ds)})  "
          f"dates {rows[0].signal_date}..{rows[-1].signal_date}  symbols={len(symbols)}")
    print(f"TEST baseline (buy everything, {HOLD}d after-cost): {overall:+.2f}%\n")

    _report("Rubric", [row for row in test_ds if row.baseline_label == "BUY"])
    _report("NN (full technicals)", [row for row in test_ds if gate.predict_label(row) == "BUY"])
    k = max(20, len(test_ds) // 10)
    ranked = sorted(test_ds, key=lambda row: gate.probabilities(row)["BUY"], reverse=True)[:k]
    returns = [row.forward_return_pct for row in ranked]
    hit = 100.0 * sum(value > 0 for value in returns) / len(returns)
    print(f"  {'NN top-decile P(BUY)':24} n={k:4} avg={mean(returns):+.2f}% hit={hit:.0f}%  (vs {overall:+.2f}% base)")
    return 0


def _build_rows(provider: object, symbols: list[str]) -> list[FeatureVector]:
    rows: list[FeatureVector] = []
    for symbol in symbols:
        bars = sorted(provider.daily_prices(symbol), key=lambda bar: bar.date)  # type: ignore[attr-defined]
        if len(bars) < 150:
            continue
        closes = [bar.adjusted_close if bar.adjusted_close > 0.0 else bar.close for bar in bars]
        for index in range(80, len(bars) - HOLD, 12):
            entry = closes[index]
            if entry <= 0.0:
                continue
            forward = (closes[index + HOLD] / entry - 1.0) * 100.0 - COST_BPS / 100.0 * 2.0
            momentum = (closes[index] / closes[index - 20] - 1.0) * 100.0 if closes[index - 20] > 0.0 else 0.0
            rows.append(
                FeatureVector(
                    idea_id=f"{symbol}-{bars[index].date}", ticker=symbol, signal_date=bars[index].date,
                    sentiment="neutral", sentiment_confidence=0.0, catalyst_strength=0.0, trend="unknown",
                    momentum_20d_pct=momentum, mean_reversion_z_20d=0.0, volume_confirmed=False,
                    quality_status="pass", risk_status="pass", net_return_pct=forward, cost_bps=COST_BPS,
                    source="yahoo_experiment", horizon_days=HOLD, entry_price=entry,
                )
            )
    return rows


def _report(name: str, picks: list[object]) -> None:
    if not picks:
        print(f"  {name:24} BUYs=0")
        return
    returns = [row.forward_return_pct for row in picks]  # type: ignore[attr-defined]
    hit = 100.0 * sum(value > 0 for value in returns) / len(returns)
    print(f"  {name:24} BUYs={len(picks):4} avg={mean(returns):+.2f}% hit={hit:.0f}%")


if __name__ == "__main__":
    raise SystemExit(main())
