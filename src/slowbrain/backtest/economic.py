"""Profit comparison between the rubric's decisions and a learned gate's decisions.

This is the PRIMARY promotion gate: the gate's decisions must beat the rubric's on after-cost,
out-of-sample (confirmation-holdout) return AND survive the same guards (deflated Sharpe, probability
of backtest overfitting, drawdown, capacity, significance) that :func:`evaluate_rubric` enforces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from slowbrain.market_data import MarketDataProvider
from slowbrain.models import FeatureVector, RubricVersion

from .core import DecisionFn, evaluate_rubric


@dataclass(frozen=True)
class EconomicEdge:
    rubric_confirmation_return_pct: float
    gate_confirmation_return_pct: float
    confirmation_return_delta_pct: float
    rubric_confirmation_trades: int
    gate_confirmation_trades: int
    gate_deflated_sharpe: float
    gate_deflated_sharpe_p_value: float
    gate_probability_backtest_overfit: float
    gate_survived_guards: bool
    gate_guard_failures: tuple[str, ...]
    gate_beats_rubric: bool


def economic_edge(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    gate_decide: DecisionFn,
    market_data_provider: MarketDataProvider | None = None,
    effective_trial_count: int = 2,
) -> EconomicEdge:
    """Grade a gate's decisions against the rubric's on profit, using the same backtest machinery."""
    rubric_result = evaluate_rubric(
        features,
        rubric,
        market_data_provider=market_data_provider,
        effective_trial_count=effective_trial_count,
    )
    gate_result = evaluate_rubric(
        features,
        rubric,
        market_data_provider=market_data_provider,
        effective_trial_count=effective_trial_count,
        decide=gate_decide,
    )
    delta = gate_result.confirmation_total_net_profit_pct - rubric_result.confirmation_total_net_profit_pct
    gate_beats = delta > 0.0 and gate_result.survived_guards
    return EconomicEdge(
        rubric_confirmation_return_pct=rubric_result.confirmation_total_net_profit_pct,
        gate_confirmation_return_pct=gate_result.confirmation_total_net_profit_pct,
        confirmation_return_delta_pct=round(delta, 4),
        rubric_confirmation_trades=rubric_result.confirmation_trade_count,
        gate_confirmation_trades=gate_result.confirmation_trade_count,
        gate_deflated_sharpe=gate_result.deflated_sharpe,
        gate_deflated_sharpe_p_value=gate_result.deflated_sharpe_p_value,
        gate_probability_backtest_overfit=gate_result.probability_backtest_overfit,
        gate_survived_guards=gate_result.survived_guards,
        gate_guard_failures=gate_result.guard_failures,
        gate_beats_rubric=gate_beats,
    )
