from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def test_health_script_missing_credentials_writes_blocked_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_trading_env(monkeypatch)
    module = _script_module("scripts/build_broker_health_report.py", "build_broker_health_report_script")

    exit_code = module.main(["--project-root", str(tmp_path)])

    report = json.loads((tmp_path / "reports" / "live-execution" / "broker-health.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "blocked"
    assert report["orders_submitted"] is False


def test_preview_script_missing_credentials_writes_blocked_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_trading_env(monkeypatch)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "first-slowbrain-report.json").write_text(
        '{"trade_decisions": [{"ticker": "AVGO", "action": "BUY"}]}',
        encoding="utf-8",
    )
    module = _script_module("scripts/build_live_execution_preview.py", "build_live_execution_preview_script")

    exit_code = module.main(["--project-root", str(tmp_path)])

    preview = json.loads((tmp_path / "reports" / "live-execution" / "latest-preview.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert preview["status"] == "blocked"
    assert preview["reason"] == "missing_trading212_credentials"
    assert preview["orders_submitted"] is False


def test_submit_script_without_execute_writes_blocked_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_trading_env(monkeypatch)
    preview_dir = tmp_path / "reports" / "live-execution"
    preview_dir.mkdir(parents=True)
    (preview_dir / "latest-preview.json").write_text(
        json.dumps(
            {
                "preview_id": "preview-1",
                "expires_at": "2026-06-02T10:30:00+00:00",
                "orders": [
                    {
                        "preview_id": "preview-1",
                        "intent_id": "intent-1",
                        "status": "ready",
                        "broker_ticker": "AVGO_US_EQ",
                        "side": "BUY",
                        "quantity": 0.5,
                        "estimated_notional_gbp": 10.0,
                        "order_payload": {"ticker": "AVGO_US_EQ", "quantity": 0.5},
                    }
                ],
                "approval_token": "token",
            }
        ),
        encoding="utf-8",
    )
    module = _script_module("scripts/submit_live_orders.py", "submit_live_orders_script")

    exit_code = module.main(["--project-root", str(tmp_path), "--approval-token", "token"])

    result = json.loads((preview_dir / "latest-submission.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["status"] == "blocked"
    assert result["reason"] == "execute_flag_not_set"
    assert result["orders_submitted"] is False
    assert not (preview_dir / "execution-ledger.jsonl").exists()


def _script_module(path: str, name: str) -> ModuleType:
    script_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"{path} spec could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_trading_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TRADING212_API_KEY", "TRADING212_API_SECRET", "TRADING_LIVE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
