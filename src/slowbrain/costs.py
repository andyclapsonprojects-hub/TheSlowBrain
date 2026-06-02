"""Trading cost and capacity estimates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

from .market_data import LiquiditySnapshot
from .models import FeatureVector


@dataclass(frozen=True)
class CostEstimate:
    explicit_cost_bps: float
    spread_bps: float
    commission_bps: float
    market_impact_bps: float
    total_cost_bps: float
    total_cost_pct: float
    participation_rate: float
    capacity_ok: bool
    warnings: tuple[str, ...] = ()


def estimate_trade_cost(
    feature: FeatureVector,
    *,
    notional_gbp: float = 1_000.0,
    avg_daily_volume_gbp: float | None = None,
    volatility_pct: float | None = None,
    spread_bps: float | None = None,
    commission_bps: float = 1.0,
    liquidity_snapshot: LiquiditySnapshot | None = None,
) -> CostEstimate:
    profile = liquidity_profile(feature)
    active_adv = _first_configured(
        avg_daily_volume_gbp,
        liquidity_snapshot.avg_daily_volume_gbp if liquidity_snapshot else None,
        profile.avg_daily_volume_gbp,
    )
    active_volatility = _first_configured(
        volatility_pct,
        liquidity_snapshot.volatility_pct if liquidity_snapshot else None,
        profile.volatility_pct,
    )
    active_spread = _first_configured(
        spread_bps,
        liquidity_snapshot.spread_bps if liquidity_snapshot else None,
        profile.spread_bps,
    )
    participation_rate = notional_gbp / active_adv if active_adv > 0 else 1.0
    market_impact_bps = active_volatility * 100.0 * sqrt(max(participation_rate, 0.0))
    explicit_cost_bps = max(feature.cost_bps, 0.0)
    total_cost_bps = explicit_cost_bps + active_spread + commission_bps + market_impact_bps
    warnings: list[str] = []
    capacity_ok = participation_rate <= 0.01
    if not capacity_ok:
        warnings.append("capacity_participation_above_one_percent_adv")
    if explicit_cost_bps == 0:
        warnings.append("missing_explicit_cost_bps")
    return CostEstimate(
        explicit_cost_bps=round(explicit_cost_bps, 4),
        spread_bps=round(active_spread, 4),
        commission_bps=round(commission_bps, 4),
        market_impact_bps=round(market_impact_bps, 4),
        total_cost_bps=round(total_cost_bps, 4),
        total_cost_pct=round(total_cost_bps / 100.0, 4),
        participation_rate=round(participation_rate, 6),
        capacity_ok=capacity_ok,
        warnings=tuple(warnings),
    )


def after_cost_return_pct(feature: FeatureVector, *, notional_gbp: float = 1_000.0) -> float:
    return feature.net_return_pct - estimate_trade_cost(feature, notional_gbp=notional_gbp).total_cost_pct


def after_cost_return_pct_with_liquidity(
    feature: FeatureVector,
    *,
    notional_gbp: float = 1_000.0,
    liquidity_snapshot: LiquiditySnapshot | None = None,
) -> float:
    return feature.net_return_pct - estimate_trade_cost(
        feature,
        notional_gbp=notional_gbp,
        liquidity_snapshot=liquidity_snapshot,
    ).total_cost_pct


@dataclass(frozen=True)
class LiquidityProfile:
    ticker: str
    avg_daily_volume_gbp: float
    volatility_pct: float
    spread_bps: float


def liquidity_profile(feature: FeatureVector) -> LiquidityProfile:
    """Return a deterministic per-name liquidity profile until vendor ADV is wired."""
    digest = int(sha256(feature.ticker.encode("utf-8")).hexdigest()[:8], 16)
    avg_daily_volume = 750_000.0 + float(digest % 75_000_000)
    volatility = 1.2 + float(digest % 500) / 100.0
    spread = 1.0 + float(digest % 25)
    if feature.volume_confirmed:
        avg_daily_volume *= 1.25
        spread = max(1.0, spread - 2.0)
    return LiquidityProfile(
        ticker=feature.ticker,
        avg_daily_volume_gbp=round(avg_daily_volume, 2),
        volatility_pct=round(volatility, 4),
        spread_bps=round(spread, 4),
    )


def _first_configured(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return value
    return 0.0
