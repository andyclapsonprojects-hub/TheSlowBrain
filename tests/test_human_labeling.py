from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from slowbrain.human_labeling import (
    build_human_labeling_pack,
    load_decision_capture_records,
    load_label_csv,
    merge_completed_human_labels,
    write_decision_capture_records,
    write_human_labeling_pack,
)
from slowbrain.market_data import DailyPrice


class FixturePriceProvider:
    def daily_prices(self, symbol: str) -> tuple[DailyPrice, ...]:
        return _prices() if symbol == "ABCD" else ()


def test_human_labeling_pack_writes_usable_csv_and_html(tmp_path: Path) -> None:
    pack = build_human_labeling_pack(
        capture_path=tmp_path / "capture.jsonl",
        records=(_capture_record(),),
        price_provider=FixturePriceProvider(),
        generated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    outputs = write_human_labeling_pack(pack, project_root=tmp_path)

    with outputs.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    html = outputs.html_path.read_text(encoding="utf-8")

    assert rows[0]["example_id"] == "case-1"
    assert rows[0]["ticker"] == "ABCD"
    assert rows[0]["close"] == "106.0"
    assert rows[0]["volume_signal"] == "high_volume_confirmation"
    assert "bullish_engulfing" in rows[0]["candlestick_patterns"]
    assert rows[0]["human_label"] == ""
    assert rows[0]["human_rationale"] == ""
    assert "Last Available Candles" in html
    assert "TheSlowBrain Human Label Pack" in html


def test_merge_completed_human_labels_updates_matching_capture_records() -> None:
    merged = merge_completed_human_labels(
        capture_records=(_capture_record(),),
        label_rows=(
            {"example_id": "case-1", "human_label": "BUY", "human_rationale": "Good volume and bullish candle."},
        ),
    )

    assert merged[0]["human_label"] == "BUY"
    assert merged[0]["human_rationale"] == "Good volume and bullish candle."


def test_human_labeling_pack_prioritizes_available_market_context() -> None:
    missing = _capture_record()
    missing["feature"] = {
        **dict(missing["feature"] if isinstance(missing["feature"], dict) else {}),
        "idea_id": "missing-case",
        "ticker": "MISS",
    }

    pack = build_human_labeling_pack(
        capture_path=Path("capture.jsonl"),
        records=(missing, _capture_record()),
        price_provider=FixturePriceProvider(),
        limit=1,
        generated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert pack.case_count == 1
    assert pack.cases[0].example_id == "case-1"
    assert pack.cases[0].technical_context.status == "available"


def test_merge_completed_human_labels_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="invalid human_label"):
        merge_completed_human_labels(
            capture_records=(_capture_record(),),
            label_rows=({"example_id": "case-1", "human_label": "MAYBE", "human_rationale": ""},),
        )


def test_load_label_csv_returns_rows(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("example_id,human_label,human_rationale\ncase-1,HOLD,Not enough edge\n", encoding="utf-8")

    rows = load_label_csv(path)

    assert rows == ({"example_id": "case-1", "human_label": "HOLD", "human_rationale": "Not enough edge"},)


def test_load_decision_capture_records_handles_missing_blank_and_non_object(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    assert load_decision_capture_records(missing) == ()

    path = tmp_path / "capture.jsonl"
    path.write_text('\n{"feature":{"idea_id":"one"}}\n[]\n', encoding="utf-8")

    assert load_decision_capture_records(path) == ({"feature": {"idea_id": "one"}},)


def test_load_decision_capture_records_reads_mixed_pretty_and_jsonl(tmp_path: Path) -> None:
    # One JSONL machine row followed by a hand-added pretty-printed object: the old
    # per-line loader raised on the pretty fragments; the tolerant loader must read both.
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"feature": {"idea_id": "machine"}, "human_label": null}\n'
        "{\n"
        '  "feature": {"idea_id": "hand"},\n'
        '  "human_label": "BUY"\n'
        "}\n",
        encoding="utf-8",
    )

    records = load_decision_capture_records(path)

    assert len(records) == 2
    assert records[0]["human_label"] is None
    assert records[1]["human_label"] == "BUY"


def test_write_decision_capture_records_round_trips_empty_and_labelled(tmp_path: Path) -> None:
    empty = write_decision_capture_records(tmp_path / "empty.jsonl", ())
    labelled = write_decision_capture_records(tmp_path / "labelled.jsonl", ({"human_label": "BUY"},))

    assert empty.read_text(encoding="utf-8") == ""
    assert '"human_label": "BUY"' in labelled.read_text(encoding="utf-8")


def test_pack_without_price_provider_is_still_honest_and_renderable(tmp_path: Path) -> None:
    pack = build_human_labeling_pack(
        capture_path=tmp_path / "capture.jsonl",
        records=(_capture_record_without_idea_id(),),
        price_provider=None,
        generated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    outputs = write_human_labeling_pack(pack, project_root=tmp_path)
    html = outputs.html_path.read_text(encoding="utf-8")

    assert pack.cases[0].example_id == "run-1:1"
    assert pack.cases[0].technical_context.status == "unavailable"
    assert "No chart available" in html
    assert "No recent OHLCV bars were available" in html


def test_load_label_csv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_label_csv(tmp_path / "missing.csv") == ()


def _capture_record() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "feature": {
            "idea_id": "case-1",
            "ticker": "ABCD",
            "signal_date": "2026-01-20",
            "sentiment": "positive",
            "sentiment_confidence": 0.81,
            "catalyst_strength": 0.7,
            "momentum_20d_pct": 6.2,
            "mean_reversion_z_20d": 0.4,
            "source": "fixture",
        },
        "decision": {"action": "HOLD", "score": 0.48, "reason": "insufficient_edge_for_buy_or_sell"},
        "outcome": {"realized_net_return_pct": 4.5},
        "human_label": None,
        "human_rationale": None,
    }


def _capture_record_without_idea_id() -> dict[str, object]:
    record = _capture_record()
    feature = dict(record["feature"] if isinstance(record["feature"], dict) else {})
    feature.pop("idea_id", None)
    feature["ticker"] = "MISS"
    record["feature"] = feature
    record["human_label"] = "invalid"
    record["outcome"] = {"realized_net_return_pct": "not-a-number"}
    return record


def _prices() -> tuple[DailyPrice, ...]:
    prices: list[DailyPrice] = []
    for day in range(1, 21):
        close = 100.0 + day * 0.2
        prices.append(
            DailyPrice(
                "ABCD",
                f"2026-01-{day:02d}",
                close - 0.1,
                close + 0.5,
                close - 0.5,
                close,
                close,
                1_000_000.0,
                "fixture",
            )
        )
    prices[-2] = DailyPrice("ABCD", "2026-01-19", 105.0, 106.0, 101.0, 102.0, 102.0, 1_000_000.0, "fixture")
    prices[-1] = DailyPrice("ABCD", "2026-01-20", 101.5, 107.0, 101.0, 106.0, 106.0, 2_000_000.0, "fixture")
    return tuple(prices)
