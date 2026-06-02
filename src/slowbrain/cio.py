"""Chief Investment Officer risk policy."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .models import OrderIntent, PortfolioState, TradeDecision


@dataclass(frozen=True)
class CioPolicy:
    portfolio_value_gbp: float = 10_000.0
    cash_reserve_pct: float = 0.20
    max_position_pct: float = 0.05
    max_single_ticker_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_gross_exposure_pct: float = 0.80
    stop_loss_pct: float = 8.0


def size_order_intents_from_policy(
    decisions: tuple[TradeDecision, ...],
    portfolio: PortfolioState,
    *,
    policy: CioPolicy | None = None,
    blocked: bool = True,
) -> tuple[OrderIntent, ...]:
    active_policy = policy or CioPolicy()
    available_cash = active_policy.portfolio_value_gbp * (1.0 - active_policy.cash_reserve_pct)
    max_position_notional = min(
        active_policy.portfolio_value_gbp * active_policy.max_position_pct,
        active_policy.portfolio_value_gbp * active_policy.max_single_ticker_pct,
        available_cash,
    )
    gross_cap = active_policy.portfolio_value_gbp * active_policy.max_gross_exposure_pct
    sector_cap = active_policy.portfolio_value_gbp * active_policy.max_sector_pct
    gross_used = round(sum(portfolio.holding_market_values_gbp.values()), 4)
    sector_used = _sector_exposure(portfolio)
    intents: list[OrderIntent] = []
    for decision in decisions:
        if decision.action != "BUY":
            continue
        sector = sector_for_ticker(decision.ticker)
        requested_notional = decision.max_notional_gbp or max_position_notional
        remaining_gross = max(0.0, gross_cap - gross_used)
        remaining_sector = max(0.0, sector_cap - sector_used.get(sector, 0.0))
        notional = round(min(requested_notional, max_position_notional, remaining_gross, remaining_sector), 2)
        reason = "Trading 212 live execution is blocked; CIO policy produced preview only."
        if decision.ticker in portfolio.holdings:
            reason = "Ticker already held; additional exposure blocked pending portfolio review."
            notional = 0.0
        elif notional <= 0.0:
            reason = "CIO exposure cap blocked this order preview."
        else:
            gross_used += notional
            sector_used[sector] = sector_used.get(sector, 0.0) + notional
        intents.append(
            OrderIntent(
                ticker=decision.ticker,
                action=decision.action,
                notional_gbp=notional,
                status="blocked" if blocked else "preview_only",
                reason=reason,
                idempotency_key=_idempotency_key(decision, active_policy),
            )
        )
    return tuple(intents)


def blocked_order_intents_from_policy(
    decisions: tuple[TradeDecision, ...],
    portfolio: PortfolioState,
    *,
    policy: CioPolicy | None = None,
) -> tuple[OrderIntent, ...]:
    return size_order_intents_from_policy(decisions, portfolio, policy=policy, blocked=True)


def sector_for_ticker(ticker: str) -> str:
    first = (ticker or "?")[0].upper()
    if first in "ABCDE":
        return "cyclical_growth"
    if first in "FGHIJ":
        return "financial_industrial"
    if first in "KLMNO":
        return "technology_health"
    if first in "PQRST":
        return "consumer_services"
    return "other"


def _sector_exposure(portfolio: PortfolioState) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for ticker, value in portfolio.holding_market_values_gbp.items():
        sector = sector_for_ticker(ticker)
        exposure[sector] = round(exposure.get(sector, 0.0) + float(value), 4)
    return exposure


def _idempotency_key(decision: TradeDecision, policy: CioPolicy) -> str:
    raw = f"{decision.ticker}:{decision.rubric_version}:{decision.action}:{policy.portfolio_value_gbp}"
    return sha256(raw.encode("utf-8")).hexdigest()[:24]
