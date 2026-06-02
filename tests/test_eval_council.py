from __future__ import annotations

from pathlib import Path

from slowbrain.eval_council import (
    HumanExample,
    JudgeVote,
    calibrate_against_humans,
    load_human_examples_from_decision_capture,
    load_human_examples_from_labeling_csv,
    open_code_failures,
    review_decision,
)


def test_human_examples_are_required_before_calibration_claim() -> None:
    report = calibrate_against_humans((), {})

    assert report.status == "not_available"
    assert report.human_examples_required
    assert report.kappa is None


def test_calibration_computes_kappa_when_examples_exist() -> None:
    examples = (
        HumanExample("one", "AAPL", "2026-01-01", "BUY", "good setup"),
        HumanExample("two", "MSFT", "2026-01-02", "HOLD", "unclear setup"),
    )

    report = calibrate_against_humans(examples, {"one": "BUY", "two": "HOLD"})

    assert report.status == "calibrated"
    assert report.kappa == 1.0


def test_calibration_fails_below_required_kappa() -> None:
    examples = (
        HumanExample("one", "AAPL", "2026-01-01", "BUY", "good setup"),
        HumanExample("two", "MSFT", "2026-01-02", "HOLD", "unclear setup"),
    )

    report = calibrate_against_humans(examples, {"one": "HOLD", "two": "BUY"})

    assert report.status == "failed"
    assert report.human_examples_required
    assert report.kappa is not None and report.kappa < 0.80


def test_human_examples_load_from_labelled_decision_capture(tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text(
        '{"run_id":"run","feature":{"idea_id":"one","ticker":"AAPL","signal_date":"2026-01-01"},'
        '"human_label":"BUY","human_rationale":"fixture"}\n'
        '{"run_id":"run","feature":{"idea_id":"two","ticker":"MSFT","signal_date":"2026-01-02"},'
        '"human_label":null,"human_rationale":null}\n',
        encoding="utf-8",
    )

    examples = load_human_examples_from_decision_capture(path)

    assert examples == (HumanExample("one", "AAPL", "2026-01-01", "BUY", "fixture"),)


def test_human_examples_load_from_labeling_csv(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "example_id,ticker,signal_date,human_label,human_rationale\n"
        "one,AAPL,2026-01-01,BUY,clear setup\n"
        "two,MSFT,2026-01-02,,blank ignored\n",
        encoding="utf-8",
    )

    examples = load_human_examples_from_labeling_csv(path)

    assert examples == (HumanExample("one", "AAPL", "2026-01-01", "BUY", "clear setup"),)


def test_council_unknown_output_blocks_aggregate_pass() -> None:
    review = review_decision(
        "decision-1",
        (
            JudgeVote("profit_evidence", "pass", "profitable"),
            JudgeVote("data_quality", "unknown", "missing human example"),
        ),
    )

    assert review.aggregate_outcome == "unknown"
    assert open_code_failures((review,)) == ("data_quality",)
