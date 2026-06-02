from __future__ import annotations

from pathlib import Path

from slowbrain.grader_council import propose_rubric_candidates
from slowbrain.models import FeatureVector, RubricCandidate, RubricVersion
from slowbrain.optimizer import select_rubric
from slowbrain.reporting import build_eric_brief
from slowbrain.rubrics import BASE_RUBRIC, decide_feature
from slowbrain.trading_flow import build_blocked_order_intents, build_trade_decisions, load_portfolio_state


def feature(
    idea_id: str,
    *,
    ticker: str = "AAPL",
    signal_date: str = "2026-01-01",
    sentiment: str = "positive",
    confidence: float = 0.8,
    catalyst: float = 0.7,
    trend: str = "uptrend",
    momentum: float = 8.0,
    mean_reversion: float = 0.0,
    volume: bool = True,
    net: float = 1.0,
) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker=ticker,
        signal_date=signal_date,
        sentiment=sentiment,
        sentiment_confidence=confidence,
        catalyst_strength=catalyst,
        trend=trend,
        momentum_20d_pct=momentum,
        mean_reversion_z_20d=mean_reversion,
        volume_confirmed=volume,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=45.0,
        source="fixture",
    )


def test_decision_uses_rubric_thresholds() -> None:
    buy = decide_feature(feature("buy"), BASE_RUBRIC)
    weak = decide_feature(
        feature("weak", confidence=0.2, catalyst=0.0, trend="unknown", momentum=0.0, volume=False),
        BASE_RUBRIC,
    )
    sell = decide_feature(
        feature(
            "sell",
            sentiment="negative",
            confidence=0.9,
            catalyst=0.0,
            trend="downtrend",
            momentum=-5.0,
            volume=False,
        ),
        BASE_RUBRIC,
    )

    assert buy.action == "BUY"
    assert weak.action == "HOLD"
    assert sell.action == "SELL"


def test_slowbrain_adopts_profitable_candidate() -> None:
    active = RubricVersion(
        version="active",
        weights=BASE_RUBRIC.weights,
        buy_threshold=0.58,
        sell_threshold=-0.35,
        max_position_pct=0.05,
    )
    candidate = RubricVersion(
        version="candidate_lower_threshold",
        weights=BASE_RUBRIC.weights,
        buy_threshold=0.40,
        sell_threshold=-0.35,
        max_position_pct=0.05,
    )
    features = tuple(
        [feature(f"strong_train_{index}", signal_date=f"2026-01-{index + 1:02d}", net=1.0) for index in range(30)]
        + [
            feature(
                f"weak_test_{index}",
                ticker=f"T{index:03d}",
                signal_date=f"2026-02-{(index % 28) + 1:02d}",
                confidence=0.5,
                catalyst=0.3,
                trend="uptrend",
                momentum=3.0,
                volume=False,
                net=2.0 + (index % 5) * 0.1,
            )
            for index in range(40)
        ]
    )
    decision = select_rubric(
        active=active,
        candidates=(
            RubricCandidate(
                candidate_id="candidate",
                parent_version="active",
                rubric=candidate,
                reason="fixture",
            ),
        ),
        features=features,
        min_profit_improvement_pct=0.1,
    )

    assert decision.action == "adopt"
    assert decision.selected_version == "candidate_lower_threshold"
    assert decision.candidate_result is not None
    assert decision.candidate_result.positive_window_rate >= 0.5
    assert decision.candidate_result.test_return_t_stat >= 0.0


def test_grader_council_proposes_bounded_candidates() -> None:
    candidates = propose_rubric_candidates(
        BASE_RUBRIC,
        [feature("one"), feature("two", sentiment="negative", net=-1.0)],
    )

    assert candidates
    assert all(candidate.parent_version == BASE_RUBRIC.version for candidate in candidates)
    assert all(0.0 <= weight <= 1.0 for candidate in candidates for weight in candidate.rubric.weights.values())


def test_cio_report_and_broker_intents_are_blocked(tmp_path: Path) -> None:
    import_root = tmp_path / "import"
    paper = import_root / "paper_trading"
    paper.mkdir(parents=True)
    (paper / "pead_positions.csv").write_text(
        "position_id,strategy_id,ticker,entry_date,entry_price,quantity,target_exit_date,status,exit_date,exit_price,gross_return_pct,net_return_pct,cost_bps,pnl_currency,updated_at\n"
        "p1,s,MSFT,2026-01-01,10,1,2026-01-10,open,,,,,45,,now\n",
        encoding="utf-8",
    )
    (paper / "live_fills.csv").write_text(
        "execution_id,order_id,ticker,side,order_type,filled_quantity,average_fill_price,limit_price,currency,submitted_at,filled_at,status,source,net_value,fx_rate,taxes,raw_order_status,error\n"
        "b,1,AVGO,buy,limit,1,100,100,GBP,now,now,FILLED,test,100,1,[],FILLED,\n"
        "s,2,AVGO,sell,limit,-1,110,110,GBP,now,now,FILLED,test,110,1,[],FILLED,\n",
        encoding="utf-8",
    )
    decisions = build_trade_decisions([feature("buy", ticker="AAPL")], BASE_RUBRIC)
    intents = build_blocked_order_intents(decisions)
    portfolio = load_portfolio_state(import_root)
    brief = build_eric_brief(decisions, portfolio)

    assert intents
    assert all(intent.status == "blocked" for intent in intents)
    assert "AAPL" in brief.stocks_to_buy
    assert "MSFT" in brief.current_portfolio_stocks
    assert brief.profit_since_first_trade_pct == 10.0
    assert brief.lines[0] == "Eric - TheSlowBrain"
