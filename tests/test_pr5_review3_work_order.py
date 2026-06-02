from __future__ import annotations

from pathlib import Path

from slowbrain.backtest import evaluate_rubric
from slowbrain.costs import estimate_trade_cost
from slowbrain.eval_council import (
    JUDGE_PERSONAS,
    JudgeVote,
    OpenAIJudgeClient,
    open_code_failures,
    review_promotion_quality,
)
from slowbrain.market_data import BenchmarkReturn, LiquiditySnapshot, StaticMarketDataProvider, UniverseMembership
from slowbrain.models import FeatureVector, RubricCandidate, RubricVersion
from slowbrain.optimizer import select_rubric
from slowbrain.rubrics import BASE_RUBRIC


class RecordingJudge(OpenAIJudgeClient):
    def __init__(self) -> None:
        super().__init__(api_key="fixture-key", model="fixture-model")
        self.calls = 0

    def review(
        self,
        decision_id: str,
        candidate_metrics: dict[str, float | int | str | bool],
        *,
        cache_dir: Path | None,
    ) -> tuple[JudgeVote, ...]:
        self.calls += 1
        assert decision_id == "active->candidate"
        assert cache_dir == Path("fixture-cache")
        assert candidate_metrics["candidate_reason"] == "fixture reason"
        return (JudgeVote("openai_panel", "pass", "fixture pass", score=1.0, confidence=1.0),)


class PersonaJudge(OpenAIJudgeClient):
    def __init__(self) -> None:
        super().__init__(api_key="fixture-key", model="fixture-model")
        self.personas: list[str] = []

    def _call_openai(self, candidate_metrics: dict[str, float | int | str | bool], *, persona: str) -> object:
        self.personas.append(persona)
        return [
            {
                "dimension": f"openai_panel:{persona}",
                "outcome": "pass",
                "score": 1.0,
                "confidence": 1.0,
                "rationale": f"{persona} fixture pass",
            }
        ]


def test_optimizer_passes_configured_openai_judge_to_adoption_gate() -> None:
    active = RubricVersion("active", BASE_RUBRIC.weights, 0.80, -0.35, 0.05)
    candidate = RubricVersion("candidate", BASE_RUBRIC.weights, 0.40, -0.35, 0.05)
    judge = RecordingJudge()

    decision = select_rubric(
        active=active,
        candidates=(RubricCandidate("candidate", "active", candidate, "fixture reason"),),
        features=tuple(feature(f"feature-{index}", net=1.0) for index in range(80)),
        min_profit_improvement_pct=0.0,
        openai_judge=judge,
        council_cache_dir=Path("fixture-cache"),
    )

    assert judge.calls == 1
    assert decision.action != "adopt"


def test_missing_openai_key_blocks_panel_success_without_crashing() -> None:
    review = review_promotion_quality(
        decision_id="fixture",
        candidate_metrics={
            "candidate_reason": "profitable fixture",
            "p_value": 0.01,
            "max_drawdown_pct": 1.0,
            "alpha_vs_benchmark_pct": 5.0,
            "deflated_sharpe": 2.0,
            "deflated_sharpe_p_value": 0.01,
            "probability_backtest_overfit": 0.1,
            "capacity_ok": True,
            "excluded_error_feature_count": 0,
        },
        openai_judge=OpenAIJudgeClient(api_key=None, model=None),
    )

    assert review.aggregate_outcome == "unknown"
    assert open_code_failures((review,)) == ("openai_panel",)


def test_openai_judge_runs_each_configured_persona() -> None:
    judge = PersonaJudge()

    votes = judge.review(
        "fixture",
        {"candidate_reason": "fixture"},
        cache_dir=None,
    )

    assert tuple(judge.personas) == JUDGE_PERSONAS
    assert {vote.dimension for vote in votes} == {f"openai_panel:{persona}" for persona in JUDGE_PERSONAS}


def test_deflated_sharpe_uses_effective_trial_count_and_return_moments() -> None:
    features = patterned_features((2.0, 3.0, 2.0, 3.0, 2.0, 3.0))

    one_trial = evaluate_rubric(features, BASE_RUBRIC, windows=6, effective_trial_count=1, min_test_trades=0)
    many_trials = evaluate_rubric(features, BASE_RUBRIC, windows=6, effective_trial_count=50, min_test_trades=0)

    assert one_trial.return_kurtosis != 3.0
    assert many_trials.effective_trial_count == 50
    assert many_trials.deflated_sharpe < one_trial.deflated_sharpe
    assert many_trials.deflated_sharpe_p_value >= one_trial.deflated_sharpe_p_value


def test_cscv_pbo_penalizes_rank_degradation() -> None:
    stable = evaluate_rubric(
        patterned_features((1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        BASE_RUBRIC,
        windows=6,
        min_test_trades=0,
    )
    unstable = evaluate_rubric(
        patterned_features((10.0, -10.0, -10.0, 10.0, -10.0, -10.0)),
        BASE_RUBRIC,
        windows=6,
        min_test_trades=0,
    )

    assert stable.probability_backtest_overfit == 0.0
    assert unstable.probability_backtest_overfit > stable.probability_backtest_overfit
    assert "probability_backtest_overfit_guard_failed" in unstable.guard_failures


def test_market_data_provider_supplies_real_benchmark_and_liquidity() -> None:
    features = patterned_features((2.0, 3.0, 2.0, 3.0, 2.0, 3.0))
    benchmark_returns = {
        (feature.ticker, feature.signal_date): BenchmarkReturn(
            feature.ticker,
            feature.signal_date,
            0.5,
            "fixture_vendor",
        )
        for feature in features
    }
    provider = StaticMarketDataProvider(
        benchmark_returns=benchmark_returns,
        liquidity_snapshots={
            "AAPL": LiquiditySnapshot("AAPL", 100_000_000.0, 0.8, 2.0, "fixture_vendor"),
        },
        universe_memberships={
            (feature.ticker, feature.signal_date): UniverseMembership(
                feature.ticker,
                feature.signal_date,
                True,
                False,
                "fixture_vendor",
            )
            for feature in features
        },
    )

    result = evaluate_rubric(features, BASE_RUBRIC, windows=6, min_test_trades=0, market_data_provider=provider)
    cost = estimate_trade_cost(features[0], liquidity_snapshot=provider.liquidity_snapshot(features[0]))

    assert result.benchmark_quality == "provider_real_benchmark"
    assert result.liquidity_quality == "provider_real_per_name_liquidity"
    assert result.universe_quality == "provider_point_in_time_universe"
    assert cost.spread_bps == 2.0


def test_market_data_provider_excludes_non_point_in_time_universe_members() -> None:
    kept = feature("kept", net=1.0)
    removed = feature("removed", ticker="ZZZZ", net=1.0)
    provider = StaticMarketDataProvider(
        benchmark_returns={},
        liquidity_snapshots={},
        universe_memberships={
            (kept.ticker, kept.signal_date): UniverseMembership(kept.ticker, kept.signal_date, True, False, "fixture"),
            (removed.ticker, removed.signal_date): UniverseMembership(
                removed.ticker,
                removed.signal_date,
                False,
                True,
                "fixture",
            ),
        },
    )

    result = evaluate_rubric((kept, removed), BASE_RUBRIC, min_test_trades=0, market_data_provider=provider)

    assert result.sample_count == 1
    assert result.point_in_time_excluded_count == 1
    assert "point_in_time_universe_exclusions" in result.guard_failures


def feature(idea_id: str, *, ticker: str = "AAPL", net: float) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker=ticker,
        signal_date="2026-01-01",
        sentiment="positive",
        sentiment_confidence=0.9,
        catalyst_strength=0.9,
        trend="uptrend",
        momentum_20d_pct=8.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=10.0,
        source="fixture",
    )


def patterned_features(pattern: tuple[float, ...]) -> tuple[FeatureVector, ...]:
    features: list[FeatureVector] = []
    index = 0
    for chunk, net in enumerate(pattern):
        for offset in range(10):
            features.append(
                FeatureVector(
                    idea_id=f"feature-{index}",
                    ticker="AAPL",
                    signal_date=f"2026-{chunk + 1:02d}-{offset + 1:02d}",
                    sentiment="positive",
                    sentiment_confidence=0.9,
                    catalyst_strength=0.9,
                    trend="uptrend",
                    momentum_20d_pct=8.0,
                    mean_reversion_z_20d=0.0,
                    volume_confirmed=True,
                    quality_status="pass",
                    risk_status="pass",
                    net_return_pct=net,
                    cost_bps=10.0,
                    source="fixture",
                )
            )
            index += 1
    return tuple(features)
