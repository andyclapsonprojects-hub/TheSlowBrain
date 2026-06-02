from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from slowbrain.accounting import Fill, calculate_fifo_pnl, calculate_fifo_pnl_from_csv, load_fills
from slowbrain.cio import CioPolicy, blocked_order_intents_from_policy
from slowbrain.config import load_config
from slowbrain.costs import estimate_trade_cost
from slowbrain.data_import import default_legacy_root
from slowbrain.data_quality import DataQualityIssue, has_error, parse_json_object
from slowbrain.eval_council import calibrate_against_humans, load_human_examples
from slowbrain.hypotheses import matching_hypotheses
from slowbrain.models import FeatureVector, PortfolioState
from slowbrain.optimizer import select_rubric
from slowbrain.rubrics import BASE_RUBRIC, decide_feature


def test_config_uses_environment_root(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SLOWBRAIN_LEGACY_STOCK_PROJECT", "C:/tmp/legacy")

    assert load_config().legacy_stock_project_root == Path("C:/tmp/legacy")
    assert default_legacy_root() == Path("C:/tmp/legacy")


def test_config_loads_local_dotenv_without_overriding_environment(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADING212_ENV", raising=False)
    monkeypatch.delenv("TRADING_MAX_DAILY_ORDERS", raising=False)
    monkeypatch.delenv("SLOWBRAIN_MARKET_DATA_ENABLED", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    (tmp_path / ".env").write_text(
        "TRADING212_ENV=demo\nOPENAI_MODEL=dotenv-model\nTRADING_MAX_DAILY_ORDERS=3\n",
        encoding="utf-8",
    )

    config = load_config(project_root=tmp_path)

    assert config.trading212_env == "demo"
    assert config.openai_model == "env-model"
    assert config.trading_max_daily_orders == 3
    assert config.market_data_enabled is True


def test_data_quality_missing_and_non_object_json_paths() -> None:
    issues: list[DataQualityIssue] = []

    assert parse_json_object("", field="payload", issues=issues) == {}
    assert parse_json_object("[]", field="payload", issues=issues) == {}
    assert not has_error((issues[0],))
    assert has_error((issues[1],))


def test_accounting_handles_unmatched_sell_unknown_side_and_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    assert load_fills(missing) == ()
    summary = calculate_fifo_pnl(
        (
            Fill("AAA", "sell", 1.0, 10.0, 0.0),
            Fill("BBB", "dividend", 1.0, 1.0, 0.0),
        )
    )

    assert summary.realized_profit_pct is None
    assert "unmatched_sell:AAA:1.000000" in summary.warnings
    assert "unknown_side:BBB:dividend" in summary.warnings


def test_accounting_parses_raw_order_status_and_tax_quantity(tmp_path: Path) -> None:
    fills = tmp_path / "fills.csv"
    fills.write_text(
        "ticker,side,filled_quantity,net_value,fx_rate,taxes,status,raw_order_status,currency\n"
        "AAA,buy,1,100,2,\"[{\"\"quantity\"\": -1}]\",broker_imported_live_fill,FILLED,USD\n"
        "AAA,sell,-1,120,2,not-json,broker_imported_live_fill,FILLED,USD\n",
        encoding="utf-8",
    )

    summary = calculate_fifo_pnl_from_csv(fills)

    assert summary.realized_profit_gbp == 38.0


def test_cost_model_capacity_warning() -> None:
    estimate = estimate_trade_cost(feature("capacity"), notional_gbp=1_000_000, avg_daily_volume_gbp=1_000_000)

    assert not estimate.capacity_ok
    assert "capacity_participation_above_one_percent_adv" in estimate.warnings


def test_cio_blocks_existing_holding_reason() -> None:
    decision = decide_feature(feature("buy", ticker="AAPL"), BASE_RUBRIC)
    intents = blocked_order_intents_from_policy(
        (decision,),
        PortfolioState(holdings=("AAPL",)),
        policy=CioPolicy(),
    )

    assert "already held" in intents[0].reason


def test_hypothesis_matching_and_negative_branch() -> None:
    positive = feature("positive", net=1.0)
    negative = feature("negative", net=-1.0)
    negative = type(negative)(**{**negative.__dict__, "sentiment": "negative"})

    assert "H1_POSITIVE_SENTIMENT_10D" in matching_hypotheses(positive)
    assert matching_hypotheses(negative) == ("H4_NEGATIVE_AVOID_10D",)


def test_eval_loader_and_non_perfect_kappa(tmp_path: Path) -> None:
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps(
            [
                {
                    "example_id": "one",
                    "ticker": "AAPL",
                    "decision_date": "2026-01-01",
                    "human_label": "BUY",
                    "rationale": "ok",
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = load_human_examples(path)
    report = calibrate_against_humans(examples, {"one": "HOLD"})

    assert len(examples) == 1
    assert report.kappa == 0.0


def test_optimizer_rejects_when_no_candidates() -> None:
    decision = select_rubric(active=BASE_RUBRIC, candidates=(), features=(feature("one"),))

    assert decision.action == "reject"
    assert "candidate_generation_empty" in decision.gaps


def feature(idea_id: str, *, ticker: str = "AAPL", net: float = 1.0) -> FeatureVector:
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
        volume_confirmed=True,
        quality_status="pass",
        risk_status="pass",
        net_return_pct=net,
        cost_bps=45.0,
        source="fixture",
    )
