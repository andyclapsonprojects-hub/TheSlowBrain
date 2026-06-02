"""Slow Brain rubric adoption logic."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from .backtest import evaluate_rubric
from .eval_council import OpenAIJudgeClient, review_promotion_quality
from .market_data import MarketDataProvider
from .models import FeatureVector, PromotionDecision, RubricCandidate, RubricVersion


def select_rubric(
    *,
    active: RubricVersion,
    candidates: Sequence[RubricCandidate],
    features: Sequence[FeatureVector],
    min_profit_improvement_pct: float = 0.25,
    max_corrected_p_value: float = 0.05,
    min_council_quality_score: float = 0.70,
    openai_judge: OpenAIJudgeClient | None = None,
    council_cache_dir: Path | None = None,
    market_data_provider: MarketDataProvider | None = None,
) -> PromotionDecision:
    """Select the best rubric candidate or keep the active rubric."""
    current = evaluate_rubric(features, active, market_data_provider=market_data_provider)
    best_candidate: RubricCandidate | None = None
    best_result = None
    candidate_results = []
    for candidate in candidates:
        result = evaluate_rubric(
            features,
            candidate.rubric,
            effective_trial_count=len(candidates),
            market_data_provider=market_data_provider,
        )
        candidate_results.append(result)
        if (
            best_result is None
            or result.validation_total_net_profit_pct > best_result.validation_total_net_profit_pct
        ):
            best_candidate = candidate
            best_result = result
    if best_candidate is None or best_result is None:
        return PromotionDecision(
            action="reject",
            selected_version=active.version,
            active_version=active.version,
            reason="no candidate rubrics were available",
            current_result=current,
            gaps=("candidate_generation_empty",),
        )
    improvement = best_result.confirmation_total_net_profit_pct - current.confirmation_total_net_profit_pct
    corrected_p = _bonferroni(best_result.p_value, trials=len(candidate_results))
    council = review_promotion_quality(
        decision_id=f"{active.version}->{best_candidate.rubric.version}",
        candidate_metrics=_candidate_metrics(best_candidate, best_result, corrected_p),
        cache_dir=council_cache_dir,
        openai_judge=openai_judge,
    )
    if (
        best_result.survived_guards
        and improvement >= min_profit_improvement_pct
        and corrected_p <= max_corrected_p_value
        and council.aggregate_outcome == "pass"
        and council.aggregate_score >= min_council_quality_score
    ):
        return PromotionDecision(
            action="adopt",
            selected_version=best_candidate.rubric.version,
            active_version=active.version,
            reason=(
                f"{best_candidate.candidate_id} improved confirmation net profit by "
                f"{improvement:.4f} percentage points after {len(candidate_results)} trials"
            ),
            current_result=current,
            candidate_result=best_result,
            council_quality_score=council.aggregate_score,
            council_quality_status=council.aggregate_outcome,
        )
    gaps = list(best_result.guard_failures)
    if improvement < min_profit_improvement_pct:
        gaps.append("confirmation_profit_improvement_below_guard")
    if corrected_p > max_corrected_p_value:
        gaps.append("multiple_testing_adjusted_significance_failed")
    if council.aggregate_outcome != "pass" or council.aggregate_score < min_council_quality_score:
        gaps.append("council_quality_gate_failed")
    action: Literal["reject", "try_variation"] = "try_variation" if improvement > 0 else "reject"
    return PromotionDecision(
        action=action,
        selected_version=active.version,
        active_version=active.version,
        reason=f"{best_candidate.candidate_id} did not clear the adoption guard",
        current_result=current,
        candidate_result=best_result,
        gaps=tuple(gaps),
        council_quality_score=council.aggregate_score,
        council_quality_status=council.aggregate_outcome,
    )


def _bonferroni(p_value: float, *, trials: int) -> float:
    return min(1.0, p_value * max(trials, 1))


def _candidate_metrics(
    candidate: RubricCandidate,
    result: object,
    corrected_p: float,
) -> dict[str, float | int | str | bool]:
    return {
        "candidate_reason": candidate.reason,
        "p_value": corrected_p,
        "max_drawdown_pct": getattr(result, "max_drawdown_pct", 100.0),
        "alpha_vs_benchmark_pct": getattr(result, "alpha_vs_benchmark_pct", 0.0),
        "deflated_sharpe": getattr(result, "deflated_sharpe", 0.0),
        "deflated_sharpe_p_value": getattr(result, "deflated_sharpe_p_value", 1.0),
        "return_skewness": getattr(result, "return_skewness", 0.0),
        "return_kurtosis": getattr(result, "return_kurtosis", 3.0),
        "probability_backtest_overfit": getattr(result, "probability_backtest_overfit", 1.0),
        "effective_trial_count": getattr(result, "effective_trial_count", 1),
        "capacity_ok": getattr(result, "capacity_ok", False),
        "excluded_error_feature_count": getattr(result, "excluded_error_feature_count", 0),
    }
