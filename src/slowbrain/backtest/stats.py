"""Backtest statistical helpers."""

from __future__ import annotations

from collections.abc import Sequence
from math import e, erfc, sqrt
from statistics import NormalDist, median


def positive_rate(values: Sequence[float]) -> float:
    return sum(1 for value in values if value > 0) / len(values) if values else 0.0


def t_stat(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return 0.0
    standard_error = sqrt(variance) / sqrt(len(values)) if variance > 0 else 0.0
    return average / standard_error if standard_error else 0.0


def sharpe(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return average / sqrt(variance) if variance > 0 else 0.0


def normal_p_value(t_stat: float) -> float:
    return erfc(abs(t_stat) / sqrt(2.0)) if t_stat else 1.0


def sign_flip_p_value(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    observed = abs(sum(values) / len(values))
    if observed == 0:
        return 1.0
    if len({round(value, 10) for value in values}) == 1:
        return 1.0
    total = 2 ** len(values)
    if total > 4096:
        return normal_p_value(t_stat(values))
    extreme = 0
    for mask in range(total):
        signed = [value if mask & (1 << index) else -value for index, value in enumerate(values)]
        if abs(sum(signed) / len(signed)) >= observed:
            extreme += 1
    return float(extreme) / float(total)


def compound_returns(values: Sequence[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value / 100.0
    return (equity - 1.0) * 100.0


def beta(portfolio_returns: Sequence[float], benchmark_returns: Sequence[float]) -> float:
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    pairs = list(zip(portfolio_returns, benchmark_returns, strict=False))
    avg_portfolio = sum(left for left, _ in pairs) / len(pairs)
    avg_benchmark = sum(right for _, right in pairs) / len(pairs)
    covariance = sum((left - avg_portfolio) * (right - avg_benchmark) for left, right in pairs)
    variance = sum((right - avg_benchmark) ** 2 for _, right in pairs)
    return covariance / variance if variance else 0.0


def information_ratio(active_returns: Sequence[float]) -> float:
    return sharpe(active_returns)


def deflated_sharpe(
    returns: Sequence[float],
    *,
    sharpe: float,
    effective_trial_count: int,
) -> tuple[float, float]:
    if len(returns) < 2 or sharpe == 0.0:
        return 0.0, 1.0
    skewness, kurtosis = return_moments(returns)
    standard_error = sharpe_standard_error(
        sharpe,
        sample_count=len(returns),
        skewness=skewness,
        kurtosis=kurtosis,
    )
    if standard_error <= 0.0:
        return 0.0, 1.0
    threshold = standard_error * _expected_maximum_normal(effective_trial_count)
    z_score = (sharpe - threshold) / standard_error
    p_value = 1.0 - NormalDist().cdf(z_score)
    return z_score, max(0.0, min(1.0, p_value))


def sharpe_standard_error(
    sharpe: float,
    *,
    sample_count: int,
    skewness: float,
    kurtosis: float,
) -> float:
    numerator = 1.0 - (skewness * sharpe) + (((kurtosis - 1.0) / 4.0) * sharpe * sharpe)
    return sqrt(max(numerator / max(sample_count - 1, 1), 1e-12))


def _expected_maximum_normal(effective_trial_count: int) -> float:
    trials = max(effective_trial_count, 1)
    if trials <= 1:
        return 0.0
    normal = NormalDist()
    euler_mascheroni = 0.5772156649
    first = normal.inv_cdf(_clipped_probability(1.0 - (1.0 / trials)))
    second = normal.inv_cdf(_clipped_probability(1.0 - (1.0 / (trials * e))))
    return ((1.0 - euler_mascheroni) * first) + (euler_mascheroni * second)


def _clipped_probability(value: float) -> float:
    return min(max(value, 1e-12), 1.0 - 1e-12)


def return_moments(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 3.0
    average = sum(values) / len(values)
    centered = [value - average for value in values]
    variance = sum(value * value for value in centered) / len(centered)
    if variance <= 0.0:
        return 0.0, 3.0
    sigma = sqrt(variance)
    skewness = sum((value / sigma) ** 3 for value in centered) / len(centered)
    kurtosis = sum((value / sigma) ** 4 for value in centered) / len(centered)
    return skewness, kurtosis


def probability_backtest_overfit(train_test_profits: Sequence[tuple[float, float]]) -> float:
    if not train_test_profits:
        return 1.0
    if len(train_test_profits) < 3:
        return 1.0
    in_sample_threshold = median(train for train, _ in train_test_profits)
    out_sample_threshold = median(test for _, test in train_test_profits)
    selected = [test for train, test in train_test_profits if train >= in_sample_threshold]
    if not selected:
        return 1.0
    overfit_count = sum(1 for test in selected if test < out_sample_threshold)
    return overfit_count / len(selected)


def sortino(values: Sequence[float]) -> float:
    downside = [min(value, 0.0) for value in values]
    if not downside or all(value == 0 for value in downside):
        return sharpe(values)
    downside_deviation = sqrt(sum(value * value for value in downside) / len(downside))
    average = sum(values) / len(values) if values else 0.0
    return average / downside_deviation if downside_deviation else 0.0
