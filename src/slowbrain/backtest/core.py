"""Backtest and walk-forward scoring for rubric versions."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from slowbrain.cio import CioPolicy
from slowbrain.costs import CostEstimate, after_cost_return_pct_with_liquidity, estimate_trade_cost
from slowbrain.data_quality import has_error
from slowbrain.market_data import LiquiditySnapshot, MarketDataProvider
from slowbrain.models import BacktestResult, FeatureVector, RubricVersion
from slowbrain.rubrics import decide_feature

from .stats import (
    beta,
    compound_returns,
    deflated_sharpe,
    information_ratio,
    normal_p_value,
    positive_rate,
    probability_backtest_overfit,
    return_moments,
    sharpe,
    sign_flip_p_value,
    sortino,
    t_stat,
)


def evaluate_rubric(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    min_test_trades: int = 3,
    max_drawdown_pct: float = 25.0,
    windows: int = 3,
    effective_trial_count: int = 1,
    market_data_provider: MarketDataProvider | None = None,
) -> BacktestResult:
    ordered = sorted(features, key=lambda feature: (feature.signal_date, feature.idea_id))
    excluded_error_count = sum(1 for feature in ordered if has_error(feature.data_quality_issues))
    clean_ordered = [feature for feature in ordered if not has_error(feature.data_quality_issues)]
    clean_ordered, point_in_time_excluded_count, universe_quality = _apply_point_in_time_universe(
        clean_ordered,
        market_data_provider,
    )
    train, validation, confirmation = purged_embargoed_split(
        clean_ordered,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        embargo_count=min(10, max(0, len(clean_ordered) // 20)),
    )
    train_profit, _, _, _ = _profit_for_features(train, rubric, market_data_provider=market_data_provider)
    validation_profit, validation_trades, _, _ = _profit_for_features(
        validation,
        rubric,
        market_data_provider=market_data_provider,
    )
    total_profit, trade_count, hit_rate, max_drawdown = _profit_for_features(
        clean_ordered,
        rubric,
        market_data_provider=market_data_provider,
    )
    test_profit, test_trades, _, _ = _profit_for_features(
        confirmation,
        rubric,
        market_data_provider=market_data_provider,
    )
    cv_pairs = combinatorial_purged_cv_train_test_profits(
        clean_ordered,
        rubric,
        windows=windows,
        embargo_count=10,
        market_data_provider=market_data_provider,
    )
    window_profits = [test_profit for _, test_profit in cv_pairs]
    worst_window = min(window_profits) if window_profits else 0.0
    positive_window_rate = positive_rate(window_profits)
    confirmation_returns = _buy_returns(confirmation, rubric, market_data_provider=market_data_provider)
    test_return_t_stat = t_stat(confirmation_returns)
    p_value = max(normal_p_value(test_return_t_stat), sign_flip_p_value(confirmation_returns))
    capacity_ok = all(_cost_for_feature(feature, market_data_provider).capacity_ok for feature in clean_ordered)
    portfolio_returns = _portfolio_returns(clean_ordered, rubric, market_data_provider=market_data_provider)
    benchmark_returns, benchmark_quality = _benchmark_returns(clean_ordered, rubric, market_data_provider)
    liquidity_quality = _liquidity_quality(clean_ordered, market_data_provider)
    portfolio_total = compound_returns(portfolio_returns)
    benchmark_total = compound_returns(benchmark_returns)
    active_returns = [left - right for left, right in zip(portfolio_returns, benchmark_returns, strict=False)]
    sharpe_value = sharpe(portfolio_returns)
    return_skewness, return_kurtosis = return_moments(portfolio_returns)
    deflated_sharpe_value, deflated_sharpe_p_value = deflated_sharpe(
        portfolio_returns,
        sharpe=sharpe_value,
        effective_trial_count=effective_trial_count,
    )
    pbo = probability_backtest_overfit(cv_pairs)
    failures: list[str] = []
    if excluded_error_count:
        failures.append("data_quality_errors_excluded")
    if point_in_time_excluded_count:
        failures.append("point_in_time_universe_exclusions")
    if test_trades < min_test_trades:
        failures.append("insufficient_confirmation_trades")
    if test_profit <= 0:
        failures.append("non_positive_confirmation_profit")
    if confirmation_returns and len({round(value, 10) for value in confirmation_returns}) == 1:
        failures.append("degenerate_zero_variance_confirmation_returns")
    if p_value > 0.05:
        failures.append("weak_confirmation_significance")
    if deflated_sharpe_value <= 0.0 or deflated_sharpe_p_value > 0.05:
        failures.append("deflated_sharpe_guard_failed")
    if pbo > 0.34:
        failures.append("probability_backtest_overfit_guard_failed")
    if max_drawdown > max_drawdown_pct:
        failures.append("max_drawdown_exceeded")
    if window_profits and worst_window < -max_drawdown_pct / 2:
        failures.append("worst_window_too_negative")
    if window_profits and positive_window_rate < 0.5:
        failures.append("weak_window_consistency")
    if not capacity_ok:
        failures.append("capacity_guard_failed")
    average = total_profit / trade_count if trade_count else 0.0
    return BacktestResult(
        rubric_version=rubric.version,
        sample_count=len(clean_ordered),
        trade_count=trade_count,
        total_net_profit_pct=round(total_profit, 4),
        average_trade_net_pct=round(average, 4),
        hit_rate=round(hit_rate, 4),
        max_drawdown_pct=round(max_drawdown, 4),
        worst_window_net_pct=round(worst_window, 4),
        positive_window_rate=round(positive_window_rate, 4),
        train_total_net_profit_pct=round(train_profit, 4),
        test_total_net_profit_pct=round(test_profit, 4),
        test_trade_count=test_trades,
        test_return_t_stat=round(test_return_t_stat, 4),
        survived_guards=not failures,
        guard_failures=tuple(failures),
        validation_total_net_profit_pct=round(validation_profit, 4),
        validation_trade_count=validation_trades,
        confirmation_total_net_profit_pct=round(test_profit, 4),
        confirmation_trade_count=test_trades,
        p_value=round(p_value, 6),
        fold_count=len(window_profits),
        turnover=round(trade_count / len(ordered), 4) if ordered else 0.0,
        sharpe=round(sharpe_value, 4),
        sortino=round(sortino(portfolio_returns), 4),
        calmar=round((total_profit / max_drawdown) if max_drawdown > 0 else total_profit, 4),
        capacity_ok=capacity_ok,
        portfolio_total_return_pct=round(portfolio_total, 4),
        benchmark_total_return_pct=round(benchmark_total, 4),
        alpha_vs_benchmark_pct=round(portfolio_total - benchmark_total, 4),
        beta_to_benchmark=round(beta(portfolio_returns, benchmark_returns), 4),
        information_ratio=round(information_ratio(active_returns), 4),
        deflated_sharpe=round(deflated_sharpe_value, 4),
        deflated_sharpe_p_value=round(deflated_sharpe_p_value, 6),
        return_skewness=round(return_skewness, 4),
        return_kurtosis=round(return_kurtosis, 4),
        probability_backtest_overfit=round(pbo, 4),
        effective_trial_count=effective_trial_count,
        excluded_error_feature_count=excluded_error_count,
        point_in_time_excluded_count=point_in_time_excluded_count,
        benchmark_quality=benchmark_quality,
        liquidity_quality=liquidity_quality,
        universe_quality=universe_quality,
    )


def purged_embargoed_split(
    ordered: Sequence[FeatureVector],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_count: int | None = None,
) -> tuple[Sequence[FeatureVector], Sequence[FeatureVector], Sequence[FeatureVector]]:
    count = len(ordered)
    train_end = max(1, int(count * train_fraction))
    validation_end = max(train_end + 1, int(count * (train_fraction + validation_fraction)))
    embargo = embargo_count if embargo_count is not None else (1 if count >= 50 else 0)
    train = ordered[:train_end]
    validation = ordered[min(train_end + embargo, count) : validation_end]
    confirmation = ordered[min(validation_end + embargo, count) :]
    return train, validation, confirmation


def _apply_point_in_time_universe(
    features: Sequence[FeatureVector],
    market_data_provider: MarketDataProvider | None,
) -> tuple[list[FeatureVector], int, str]:
    if market_data_provider is None:
        return list(features), 0, "not_configured"
    included: list[FeatureVector] = []
    excluded_count = 0
    missing_count = 0
    for feature in features:
        membership = market_data_provider.universe_membership(feature)
        if membership is None:
            missing_count += 1
            included.append(feature)
            continue
        if membership.is_member and not membership.is_delisted:
            included.append(feature)
        else:
            excluded_count += 1
    if missing_count:
        return included, excluded_count, "provider_partial_point_in_time_universe"
    return included, excluded_count, "provider_point_in_time_universe"


def _profit_for_features(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    market_data_provider: MarketDataProvider | None = None,
) -> tuple[float, int, float, float]:
    returns = _buy_returns(features, rubric, market_data_provider=market_data_provider)
    if not returns:
        return 0.0, 0, 0.0, 0.0
    total = sum(returns)
    hit_rate = sum(1 for value in returns if value > 0) / len(returns)
    return total, len(returns), hit_rate, _max_drawdown(returns)


def _cost_for_feature(feature: FeatureVector, market_data_provider: MarketDataProvider | None) -> CostEstimate:
    return estimate_trade_cost(feature, liquidity_snapshot=_liquidity_snapshot(feature, market_data_provider))


def _liquidity_snapshot(
    feature: FeatureVector,
    market_data_provider: MarketDataProvider | None,
) -> LiquiditySnapshot | None:
    return market_data_provider.liquidity_snapshot(feature) if market_data_provider else None


def _liquidity_quality(features: Sequence[FeatureVector], market_data_provider: MarketDataProvider | None) -> str:
    if market_data_provider is None:
        return "deterministic_proxy_not_real_liquidity"
    buy_features = [feature for feature in features if market_data_provider.liquidity_snapshot(feature) is not None]
    if len(buy_features) == len(features):
        return "provider_real_per_name_liquidity"
    return "provider_partial_liquidity_with_proxy_gaps"


def _buy_returns(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    market_data_provider: MarketDataProvider | None = None,
) -> list[float]:
    returns: list[float] = []
    for feature in features:
        decision = decide_feature(feature, rubric)
        if decision.action == "BUY":
            returns.append(
                after_cost_return_pct_with_liquidity(
                    feature,
                    notional_gbp=decision.max_notional_gbp or 1_000.0,
                    liquidity_snapshot=_liquidity_snapshot(feature, market_data_provider),
                )
            )
    return returns


def _portfolio_returns(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    market_data_provider: MarketDataProvider | None = None,
) -> list[float]:
    policy = CioPolicy()
    equity = policy.portfolio_value_gbp
    returns: list[float] = []
    for feature in features:
        decision = decide_feature(feature, rubric)
        if decision.action != "BUY":
            continue
        notional = min(
            decision.max_notional_gbp or equity * policy.max_position_pct,
            equity * policy.max_position_pct,
            equity * policy.max_single_ticker_pct,
        )
        trade_return_pct = after_cost_return_pct_with_liquidity(
            feature,
            notional_gbp=notional,
            liquidity_snapshot=_liquidity_snapshot(feature, market_data_provider),
        )
        pnl = notional * trade_return_pct / 100.0
        equity += pnl
        returns.append((pnl / policy.portfolio_value_gbp) * 100.0)
    return returns


def _benchmark_returns(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    market_data_provider: MarketDataProvider | None,
) -> tuple[list[float], str]:
    returns: list[float] = []
    missing_provider_values = 0
    provider_values = 0
    buy_count = 0
    for feature in features:
        if decide_feature(feature, rubric).action == "BUY":
            buy_count += 1
            benchmark = market_data_provider.benchmark_return(feature) if market_data_provider else None
            if benchmark is None:
                missing_provider_values += 1
                returns.append(_spy_proxy_return_pct(feature) * CioPolicy().max_position_pct)
            else:
                provider_values += 1
                returns.append(benchmark.return_pct * CioPolicy().max_position_pct)
    if market_data_provider is None:
        return returns, "proxy_from_feature_momentum_not_real_benchmark"
    if buy_count == 0:
        return returns, "no_buy_trades_no_benchmark_evidence"
    if provider_values == 0:
        return returns, "provider_no_benchmark_with_proxy_fallback"
    if missing_provider_values:
        return returns, "provider_partial_benchmark_with_proxy_gaps"
    return returns, "provider_real_benchmark"


def _spy_proxy_return_pct(feature: FeatureVector) -> float:
    return max(min(feature.momentum_20d_pct * 0.01, 0.5), -0.5)


def _max_drawdown(returns: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in returns:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def _window_profits(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    windows: int,
    market_data_provider: MarketDataProvider | None = None,
) -> list[float]:
    if not features:
        return []
    size = max(1, len(features) // windows)
    profits: list[float] = []
    for start in range(0, len(features), size):
        profit, _, _, _ = _profit_for_features(
            features[start : start + size],
            rubric,
            market_data_provider=market_data_provider,
        )
        profits.append(profit)
    return profits[:windows]


def walk_forward_window_profits(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    windows: int,
    market_data_provider: MarketDataProvider | None = None,
) -> list[float]:
    return _window_profits(features, rubric, windows=windows, market_data_provider=market_data_provider)


def combinatorial_purged_cv_profits(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    windows: int,
    embargo_count: int,
    market_data_provider: MarketDataProvider | None = None,
) -> list[float]:
    return [
        test_profit
        for _, test_profit in combinatorial_purged_cv_train_test_profits(
            features,
            rubric,
            windows=windows,
            embargo_count=embargo_count,
            market_data_provider=market_data_provider,
        )
    ]


def combinatorial_purged_cv_train_test_profits(
    features: Sequence[FeatureVector],
    rubric: RubricVersion,
    *,
    windows: int,
    embargo_count: int,
    market_data_provider: MarketDataProvider | None = None,
) -> list[tuple[float, float]]:
    if not features:
        return []
    chunks = _chunks(features, windows)
    profits: list[tuple[float, float]] = []
    for left, right in combinations(range(len(chunks)), 2):
        test_features = list(chunks[left]) + list(chunks[right])
        purged = _purge_neighbors(chunks, excluded={left, right}, embargo_count=embargo_count)
        if not purged:
            purged = test_features
        profit, _, _, _ = _profit_for_features(test_features, rubric, market_data_provider=market_data_provider)
        train_profit, _, _, _ = _profit_for_features(purged, rubric, market_data_provider=market_data_provider)
        profits.append((train_profit, profit if train_profit >= -25.0 else -abs(profit)))
    if profits:
        return profits
    return [
        (0.0, profit)
        for profit in walk_forward_window_profits(
            features,
            rubric,
            windows=windows,
            market_data_provider=market_data_provider,
        )
    ]


def _chunks(features: Sequence[FeatureVector], windows: int) -> list[Sequence[FeatureVector]]:
    size = max(1, len(features) // max(windows, 1))
    return [features[start : start + size] for start in range(0, len(features), size)][:windows]


def _purge_neighbors(
    chunks: list[Sequence[FeatureVector]],
    *,
    excluded: set[int],
    embargo_count: int,
) -> list[FeatureVector]:
    embargo_windows = max(1, embargo_count // 10)
    result: list[FeatureVector] = []
    for index, chunk in enumerate(chunks):
        if any(abs(index - blocked) <= embargo_windows for blocked in excluded):
            continue
        result.extend(chunk)
    return result

