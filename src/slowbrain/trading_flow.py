"""CIO trading flow and blocked broker preview."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from .accounting import (
    AccountingSummary,
    MarkToMarketSummary,
    calculate_fifo_pnl_from_csv,
    calculate_mark_to_market_from_positions,
    load_open_position_market_values,
)
from .cio import CioPolicy, blocked_order_intents_from_policy
from .models import FeatureVector, OrderIntent, PortfolioState, RubricVersion, TradeDecision
from .rubrics import decide_feature


def build_ranked_trade_decision_pairs(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    limit: int = 10,
) -> tuple[tuple[FeatureVector, TradeDecision], ...]:
    pairs = [(feature, decide_feature(feature, rubric)) for feature in features]
    ranked = sorted(pairs, key=lambda item: (item[1].action != "BUY", -item[1].score, item[1].ticker))
    return tuple(ranked[:limit])


def build_trade_decisions(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    limit: int = 10,
) -> tuple[TradeDecision, ...]:
    return tuple(decision for _, decision in build_ranked_trade_decision_pairs(features, rubric, limit=limit))


def build_blocked_order_intents(
    decisions: Sequence[TradeDecision],
    portfolio: PortfolioState | None = None,
    *,
    policy: CioPolicy | None = None,
) -> tuple[OrderIntent, ...]:
    return blocked_order_intents_from_policy(tuple(decisions), portfolio or PortfolioState(), policy=policy)


def load_portfolio_state(import_root: Path) -> PortfolioState:
    pead_positions = import_root / "paper_trading" / "pead_positions.csv"
    paper_positions = import_root / "paper_trading" / "paper_positions.csv"
    holdings = _open_holdings(pead_positions) + _open_holdings(paper_positions)
    accounting = calculate_fifo_pnl_from_csv(import_root / "paper_trading" / "live_fills.csv")
    holding_values = _combine_holding_values(
        load_open_position_market_values(pead_positions),
        load_open_position_market_values(paper_positions),
    )
    marks = _combine_marks(
        calculate_mark_to_market_from_positions(pead_positions),
        calculate_mark_to_market_from_positions(paper_positions),
    )
    profit_pct, profit_quality = _portfolio_profit_pct(accounting, marks)
    return PortfolioState(
        holdings=tuple(sorted(set(holdings))),
        holding_market_values_gbp=holding_values,
        profit_since_first_trade_pct=profit_pct,
        realized_profit_gbp=accounting.realized_profit_gbp,
        open_cost_gbp=accounting.open_cost_gbp,
        unrealized_profit_gbp=marks.unrealized_profit_gbp,
        mark_to_market_quality=marks.quality,
        profit_quality=profit_quality,
        accounting_warnings=accounting.warnings + marks.warnings,
        notes=("Imported legacy ledgers are evidence for PR1 reporting; live trading remains blocked.",),
    )


def _portfolio_profit_pct(accounting: AccountingSummary, marks: MarkToMarketSummary) -> tuple[float | None, str]:
    realized_profit = accounting.realized_profit_gbp
    matched_cost = accounting.matched_cost_gbp
    open_cost = accounting.open_cost_gbp
    if marks.unrealized_profit_gbp is not None:
        basis = matched_cost + open_cost
        if basis > 0.0:
            total_profit = realized_profit + marks.unrealized_profit_gbp
            return round((total_profit / basis) * 100.0, 4), "realized_fifo_plus_mark_to_market"
    if accounting.realized_profit_pct is not None:
        return accounting.realized_profit_pct, "ledger_realized_fifo"
    return None, "not_available"


def _combine_holding_values(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    combined = dict(first)
    for ticker, value in second.items():
        combined[ticker] = round(combined.get(ticker, 0.0) + value, 4)
    return combined


def _combine_marks(first: MarkToMarketSummary, second: MarkToMarketSummary) -> MarkToMarketSummary:
    unrealized_values = [
        value
        for value in (
            getattr(first, "unrealized_profit_gbp", None),
            getattr(second, "unrealized_profit_gbp", None),
        )
        if isinstance(value, (int, float))
    ]
    market_values = [
        value
        for value in (
            getattr(first, "open_market_value_gbp", None),
            getattr(second, "open_market_value_gbp", None),
        )
        if isinstance(value, (int, float))
    ]
    warnings = tuple(getattr(first, "warnings", ())) + tuple(getattr(second, "warnings", ()))
    if not unrealized_values:
        return MarkToMarketSummary(None, None, "missing_market_prices", warnings)
    return MarkToMarketSummary(
        round(sum(unrealized_values), 4),
        round(sum(market_values), 4),
        "marked_to_market_from_position_prices",
        warnings,
    )


def _open_holdings(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    holdings: list[str] = []
    for row in rows:
        status = str(row.get("status") or row.get("position_status") or "").lower()
        if "open" in status:
            ticker = str(row.get("ticker") or "").upper()
            if ticker:
                holdings.append(ticker)
    return holdings
