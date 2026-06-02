from __future__ import annotations

from pathlib import Path

from slowbrain.accounting import calculate_fifo_pnl_from_csv, load_open_position_market_values
from slowbrain.cio import CioPolicy, blocked_order_intents_from_policy
from slowbrain.costs import estimate_trade_cost
from slowbrain.models import FeatureVector, PortfolioState
from slowbrain.rubrics import BASE_RUBRIC, decide_feature
from slowbrain.trading_flow import load_portfolio_state


def test_fifo_pnl_matches_lots_per_ticker(tmp_path: Path) -> None:
    fills = tmp_path / "live_fills.csv"
    fills.write_text(
        "ticker,side,filled_quantity,net_value,fx_rate,taxes,status\n"
        "AAA,buy,2,200,1,[],FILLED\n"
        "BBB,buy,1,50,1,[],FILLED\n"
        "AAA,sell,-1,120,1,[],FILLED\n",
        encoding="utf-8",
    )

    summary = calculate_fifo_pnl_from_csv(fills)

    assert summary.realized_profit_gbp == 20.0
    assert summary.matched_cost_gbp == 100.0
    assert summary.realized_profit_pct == 20.0
    assert summary.open_cost_gbp == 150.0


def test_open_position_market_values_are_loaded_from_prices(tmp_path: Path) -> None:
    positions = tmp_path / "positions.csv"
    positions.write_text(
        "ticker,status,entry_price,quantity,current_price\n"
        "AAA,open,10,2,12\n"
        "BBB,closed,10,1,99\n"
        "CCC,open,5,3,\n",
        encoding="utf-8",
    )

    values = load_open_position_market_values(positions)

    assert values == {"AAA": 24.0, "CCC": 15.0}


def test_portfolio_state_profit_combines_realized_and_mark_to_market(tmp_path: Path) -> None:
    paper = tmp_path / "paper_trading"
    paper.mkdir()
    (paper / "live_fills.csv").write_text(
        "ticker,side,filled_quantity,net_value,fx_rate,taxes,status\n"
        "AAA,buy,2,200,1,[],FILLED\n"
        "AAA,sell,-1,120,1,[],FILLED\n"
        "BBB,buy,1,50,1,[],FILLED\n",
        encoding="utf-8",
    )
    (paper / "pead_positions.csv").write_text(
        "ticker,status,entry_price,quantity,current_price\n"
        "AAA,open,100,1,110\n"
        "BBB,open,50,1,60\n",
        encoding="utf-8",
    )

    portfolio = load_portfolio_state(tmp_path)

    assert portfolio.holding_market_values_gbp == {"AAA": 110.0, "BBB": 60.0}
    assert portfolio.profit_since_first_trade_pct == 16.0
    assert portfolio.profit_quality == "realized_fifo_plus_mark_to_market"


def test_cost_model_includes_explicit_cost_and_market_impact() -> None:
    estimate = estimate_trade_cost(feature("cost", net=5.0, volume=True), notional_gbp=10_000)

    assert estimate.total_cost_bps > estimate.explicit_cost_bps
    assert estimate.market_impact_bps > 0
    assert estimate.capacity_ok


def test_cio_policy_blocks_live_orders_with_idempotency_key() -> None:
    decision = decide_feature(feature("buy", ticker="AAPL"), BASE_RUBRIC)
    intents = blocked_order_intents_from_policy(
        (decision,),
        PortfolioState(),
        policy=CioPolicy(portfolio_value_gbp=20_000, cash_reserve_pct=0.25),
    )

    assert len(intents) == 1
    assert intents[0].status == "blocked"
    assert intents[0].notional_gbp <= 1_000
    assert intents[0].idempotency_key


def test_cio_sector_cap_counts_existing_portfolio_exposure() -> None:
    decision = decide_feature(feature("buy", ticker="BBBB"), BASE_RUBRIC)

    intents = blocked_order_intents_from_policy(
        (decision,),
        PortfolioState(holdings=("AAPL",), holding_market_values_gbp={"AAPL": 3_000.0}),
        policy=CioPolicy(portfolio_value_gbp=10_000, max_sector_pct=0.30),
    )

    assert intents[0].notional_gbp == 0.0
    assert intents[0].reason == "CIO exposure cap blocked this order preview."


def feature(idea_id: str, *, ticker: str = "AAPL", net: float = 1.0, volume: bool = True) -> FeatureVector:
    return FeatureVector(
        idea_id=idea_id,
        ticker=ticker,
        signal_date="2026-01-01",
        sentiment="positive",
        sentiment_confidence=0.8,
        catalyst_strength=0.7,
        trend="uptrend",
        momentum_20d_pct=8.0,
        mean_reversion_z_20d=0.0,
        volume_confirmed=volume,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=45.0,
        source="fixture",
    )
