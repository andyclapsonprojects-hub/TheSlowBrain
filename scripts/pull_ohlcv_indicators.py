"""Pull real daily OHLCV via the existing read-only Yahoo feed and compute the full indicator suite.

No API key and no extra dependency: reuses the project's YahooProvider (stdlib HTTP). Read-only; it
fetches prices, computes EMA/RSI/MACD/Bollinger/ATR plus the candlestick/trend context, and prints
them. Nothing is traded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import load_config
from slowbrain.indicators import IndicatorSnapshot, compute_indicators
from slowbrain.market_data_vendors.providers import build_price_history_provider
from slowbrain.technical_context import build_technical_context


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    provider = build_price_history_provider(load_config(root), project_root=root)
    if provider is None:
        print("Market data is disabled (SLOWBRAIN_MARKET_DATA_ENABLED=false).")
        return 1
    for symbol in args.symbols:
        prices = provider.daily_prices(symbol)
        if not prices:
            print(f"\n{symbol}: no price bars (provider miss, rate-limited, or offline).")
            continue
        as_of = max(price.date for price in prices)
        snapshot = compute_indicators(prices, signal_date=as_of)
        context = build_technical_context(symbol=symbol, signal_date=as_of, prices=prices)
        _print_symbol(symbol, snapshot, context.trend, context.pattern_names)
    return 0


def _print_symbol(symbol: str, snapshot: IndicatorSnapshot | None, trend: str, patterns: tuple[str, ...]) -> None:
    if snapshot is None:
        print(f"\n{symbol}: bars present but snapshot could not be computed.")
        return
    print(f"\n{symbol}  (as of {snapshot.as_of_date}, {snapshot.bars} bars)  close={snapshot.close:.2f}")
    print(f"  trend={trend}  candles={', '.join(patterns) or 'none'}")
    print(f"  SMA20={_fmt(snapshot.sma_20)}  EMA12={_fmt(snapshot.ema_12)}  EMA26={_fmt(snapshot.ema_26)}")
    print(f"  RSI14={_fmt(snapshot.rsi_14)}  MACD={_fmt(snapshot.macd)}/sig {_fmt(snapshot.macd_signal)}"
          f"/hist {_fmt(snapshot.macd_hist)}")
    print(f"  Bollinger(20,2): lower={_fmt(snapshot.bb_lower)} mid={_fmt(snapshot.bb_mid)} "
          f"upper={_fmt(snapshot.bb_upper)} %B={_fmt(snapshot.bb_percent_b)} bw={_fmt(snapshot.bb_bandwidth)}")
    print(f"  ATR14={_fmt(snapshot.atr_14)} ({_fmt(snapshot.atr_pct_14)}%)  "
          f"vol_ratio20={_fmt(snapshot.volume_ratio_20)}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="TheSlowBrain project root.")
    parser.add_argument("symbols", nargs="*", default=["AAPL", "MSFT", "NVDA"], help="Tickers to fetch.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
