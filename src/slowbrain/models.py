"""Typed records used across TheSlowBrain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .data_quality import DataQualityIssue

DecisionAction = Literal["BUY", "SELL", "HOLD", "AVOID", "WATCHLIST"]
PromotionAction = Literal["adopt", "reject", "try_variation"]


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    ticker: str
    event_date: str
    source: str
    title: str
    raw_path: str


@dataclass(frozen=True)
class FeatureVector:
    idea_id: str
    ticker: str
    signal_date: str
    sentiment: str
    sentiment_confidence: float
    catalyst_strength: float
    trend: str
    momentum_20d_pct: float
    mean_reversion_z_20d: float
    volume_confirmed: bool
    quality_status: str
    risk_status: str
    net_return_pct: float
    cost_bps: float
    source: str
    data_quality_issues: tuple[DataQualityIssue, ...] = ()
    horizon_days: int = 10
    outcome_future_date: str = ""
    entry_price: float | None = None
    rsi_14: float = 0.0
    macd_signal: str = "unknown"
    atr_pct_14: float = 0.0
    momentum_63d_pct: float = 0.0
    volume_ratio_20d: float = 0.0
    bb_percent_b: float = 0.5
    bb_bandwidth: float = 0.0
    macd_hist_pct: float = 0.0
    ema_trend_pct: float = 0.0
    candle_signal: float = 0.0
    sma_distance_pct: float = 0.0
    value_score: float = 0.0
    fundamental_quality_score: float = 0.0
    size_score: float = 0.0
    liquidity_score: float = 0.0
    pit_enrichment_source: str = ""
    pit_enrichment_available_date: str = ""
    cross_sectional_zscores: Mapping[str, float] = field(default_factory=dict)
    rank_label: DecisionAction | None = None


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    description: str
    expected_direction: Literal["positive", "negative"]
    horizon_days: int


@dataclass(frozen=True)
class RubricVersion:
    version: str
    weights: Mapping[str, float]
    buy_threshold: float
    sell_threshold: float
    max_position_pct: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RubricCandidate:
    candidate_id: str
    parent_version: str
    rubric: RubricVersion
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    rubric_version: str
    sample_count: int
    trade_count: int
    total_net_profit_pct: float
    average_trade_net_pct: float
    hit_rate: float
    max_drawdown_pct: float
    worst_window_net_pct: float
    positive_window_rate: float
    train_total_net_profit_pct: float
    test_total_net_profit_pct: float
    test_trade_count: int
    test_return_t_stat: float
    survived_guards: bool
    guard_failures: tuple[str, ...] = ()
    validation_total_net_profit_pct: float = 0.0
    validation_trade_count: int = 0
    confirmation_total_net_profit_pct: float = 0.0
    confirmation_trade_count: int = 0
    p_value: float = 1.0
    fold_count: int = 0
    turnover: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    capacity_ok: bool = True
    portfolio_total_return_pct: float = 0.0
    benchmark_total_return_pct: float = 0.0
    alpha_vs_benchmark_pct: float = 0.0
    beta_to_benchmark: float = 0.0
    information_ratio: float = 0.0
    deflated_sharpe: float = 0.0
    deflated_sharpe_p_value: float = 1.0
    return_skewness: float = 0.0
    return_kurtosis: float = 3.0
    probability_backtest_overfit: float = 0.0
    effective_trial_count: int = 1
    excluded_error_feature_count: int = 0
    point_in_time_excluded_count: int = 0
    benchmark_quality: str = "not_available"
    liquidity_quality: str = "not_available"
    universe_quality: str = "not_available"


@dataclass(frozen=True)
class PromotionDecision:
    action: PromotionAction
    selected_version: str
    active_version: str
    reason: str
    current_result: BacktestResult
    candidate_result: BacktestResult | None = None
    gaps: tuple[str, ...] = ()
    council_quality_score: float | None = None
    council_quality_status: str = "not_evaluated"


@dataclass(frozen=True)
class TradeDecision:
    ticker: str
    action: DecisionAction
    score: float
    rubric_version: str
    reason: str
    max_notional_gbp: float = 0.0
    acceptable_price_min: float | None = None
    acceptable_price_max: float | None = None


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    action: DecisionAction
    notional_gbp: float
    status: Literal["blocked", "preview_only", "ready"]
    reason: str
    idempotency_key: str = ""


@dataclass(frozen=True)
class PortfolioState:
    holdings: tuple[str, ...] = ()
    holding_market_values_gbp: Mapping[str, float] = field(default_factory=dict)
    profit_since_first_trade_pct: float | None = None
    cash_gbp: float | None = None
    notes: tuple[str, ...] = ()
    profit_quality: str = "not_available"
    realized_profit_gbp: float | None = None
    open_cost_gbp: float | None = None
    unrealized_profit_gbp: float | None = None
    mark_to_market_quality: str = "not_available"
    accounting_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TelegramCioBrief:
    stocks_to_buy: tuple[str, ...] = ()
    stocks_to_sell: tuple[str, ...] = ()
    current_portfolio_stocks: tuple[str, ...] = ()
    profit_since_first_trade_pct: float | None = None
    lines: tuple[str, ...] = field(default_factory=tuple)
