"""Evaluation council scaffolding and human-example calibration."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

JudgeOutcome = Literal["pass", "fail", "unknown"]
HumanLabel = Literal["BUY", "SELL", "HOLD", "AVOID", "WATCHLIST", "UNKNOWN"]
CalibrationStatus = Literal["not_available", "calibrated", "failed"]

VALID_HUMAN_LABELS = {"BUY", "SELL", "HOLD", "AVOID", "WATCHLIST", "UNKNOWN"}

# The machine can act in three ways; humans label in five. Kappa needs a shared space,
# so both sides are projected to {BUY, SELL, HOLD, UNKNOWN} before agreement is scored.
# This is the single documented bridge between the human vocabulary and the machine action.
_KAPPA_PROJECTION: dict[str, str] = {
    "WATCHLIST": "HOLD",  # monitoring, no position taken now
    "AVOID": "SELL",  # active negative screen == do-not-hold side
}


def project_label_for_kappa(label: str) -> HumanLabel:
    """Project a 5-label vocabulary value onto the 3-way machine action space for kappa."""
    projected = _KAPPA_PROJECTION.get(str(label).upper(), str(label).upper())
    return cast("HumanLabel", projected if projected in {"BUY", "SELL", "HOLD", "UNKNOWN"} else "UNKNOWN")


@dataclass(frozen=True)
class HumanExample:
    example_id: str
    ticker: str
    decision_date: str
    human_label: HumanLabel
    rationale: str


@dataclass(frozen=True)
class JudgeVote:
    dimension: str
    outcome: JudgeOutcome
    rationale: str
    score: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class CouncilReview:
    decision_id: str
    votes: tuple[JudgeVote, ...]
    aggregate_outcome: JudgeOutcome
    aggregate_score: float = 0.0
    aggregate_confidence: float = 0.0


@dataclass(frozen=True)
class CalibrationReport:
    status: CalibrationStatus
    human_examples_required: bool
    example_count: int
    kappa: float | None
    warnings: tuple[str, ...]


JUDGE_DIMENSIONS = (
    "profit_evidence",
    "risk_control",
    "data_quality",
    "execution_safety",
    "report_honesty",
    "overfitting_robustness",
    "economic_rationale",
)

JUDGE_PERSONAS = (
    "profit_skeptic",
    "risk_officer",
    "data_auditor",
)

CRITICAL_DIMENSIONS = {"profit_evidence", "risk_control", "data_quality", "overfitting_robustness"}


def load_human_examples(path: Path) -> tuple[HumanExample, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("human example file must be a JSON list")
    return tuple(_human_example_from_mapping(item) for item in payload if isinstance(item, dict))


def load_human_examples_from_decision_capture(path: Path) -> tuple[HumanExample, ...]:
    if not path.exists():
        return ()
    examples: list[HumanExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("human_label") is None:
            continue
        examples.append(_human_example_from_capture(value))
    return tuple(examples)


def load_human_examples_from_labeling_csv(path: Path) -> tuple[HumanExample, ...]:
    if not path.exists():
        return ()
    examples: list[HumanExample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("human_label") or "").strip()
            if not label:
                continue
            examples.append(
                HumanExample(
                    example_id=str(row.get("example_id") or ""),
                    ticker=str(row.get("ticker") or ""),
                    decision_date=str(row.get("signal_date") or ""),
                    human_label=_validated_human_label(label),
                    rationale=str(row.get("human_rationale") or ""),
                )
            )
    return tuple(examples)


def _human_example_from_mapping(item: dict[str, object]) -> HumanExample:
    return HumanExample(
        example_id=str(item.get("example_id") or ""),
        ticker=str(item.get("ticker") or ""),
        decision_date=str(item.get("decision_date") or item.get("signal_date") or ""),
        human_label=_validated_human_label(item.get("human_label")),
        rationale=str(item.get("rationale") or item.get("human_rationale") or ""),
    )


def _human_example_from_capture(item: dict[str, object]) -> HumanExample:
    feature = item.get("feature")
    feature_map = feature if isinstance(feature, dict) else {}
    return HumanExample(
        example_id=str(feature_map.get("idea_id") or item.get("run_id") or ""),
        ticker=str(feature_map.get("ticker") or ""),
        decision_date=str(feature_map.get("signal_date") or ""),
        human_label=_validated_human_label(item.get("human_label")),
        rationale=str(item.get("human_rationale") or ""),
    )


def _validated_human_label(value: object) -> HumanLabel:
    label = str(value or "UNKNOWN").upper()
    if label not in VALID_HUMAN_LABELS:
        raise ValueError(f"invalid human label: {label}")
    return cast("HumanLabel", label)


def calibrate_against_humans(
    examples: tuple[HumanExample, ...],
    automated_labels: dict[str, HumanLabel],
    *,
    min_kappa: float = 0.80,
) -> CalibrationReport:
    if not examples:
        return CalibrationReport(
            status="not_available",
            human_examples_required=True,
            example_count=0,
            kappa=None,
            warnings=("Andy human examples are required before calibration can be claimed.",),
        )
    paired = [
        (
            project_label_for_kappa(example.human_label),
            project_label_for_kappa(automated_labels.get(example.example_id, "UNKNOWN")),
        )
        for example in examples
    ]
    kappa = round(_cohen_kappa(paired), 4)
    if kappa < min_kappa:
        return CalibrationReport(
            status="failed",
            human_examples_required=True,
            example_count=len(paired),
            kappa=kappa,
            warnings=(f"Human calibration kappa {kappa:.4f} is below required {min_kappa:.2f}.",),
        )
    return CalibrationReport(
        status="calibrated",
        human_examples_required=False,
        example_count=len(paired),
        kappa=kappa,
        warnings=(),
    )


def review_decision(decision_id: str, votes: tuple[JudgeVote, ...]) -> CouncilReview:
    if any(vote.outcome == "unknown" for vote in votes):
        aggregate: JudgeOutcome = "unknown"
    elif any(vote.dimension in CRITICAL_DIMENSIONS and vote.outcome == "fail" for vote in votes):
        aggregate = "fail"
    else:
        aggregate = "pass" if _weighted_score(votes) >= 0.70 else "fail"
    return CouncilReview(
        decision_id=decision_id,
        votes=votes,
        aggregate_outcome=aggregate,
        aggregate_score=round(_weighted_score(votes), 4),
        aggregate_confidence=round(_mean_confidence(votes), 4),
    )


def review_promotion_quality(
    *,
    decision_id: str,
    candidate_metrics: dict[str, float | int | str | bool],
    cache_dir: Path | None = None,
    openai_judge: OpenAIJudgeClient | None = None,
) -> CouncilReview:
    """Score a rubric candidate with deterministic judges plus optional cached OpenAI input."""
    votes = list(_heuristic_votes(candidate_metrics))
    if openai_judge is not None:
        votes.extend(openai_judge.review(decision_id, candidate_metrics, cache_dir=cache_dir))
    return review_decision(decision_id, tuple(votes))


class OpenAIJudgeClient:
    """Small cached OpenAI judge adapter that degrades to unknown on any failure."""

    def __init__(self, *, api_key: str | None, model: str | None) -> None:
        self.api_key = api_key
        self.model = model or "gpt-5-mini"

    def review(
        self,
        decision_id: str,
        candidate_metrics: dict[str, float | int | str | bool],
        *,
        cache_dir: Path | None,
    ) -> tuple[JudgeVote, ...]:
        if not self.api_key:
            return (
                JudgeVote(
                    "openai_panel",
                    "unknown",
                    "OpenAI judge unavailable because OPENAI_API_KEY is not configured.",
                ),
            )
        votes: list[JudgeVote] = []
        for persona in JUDGE_PERSONAS:
            cache_path = _cache_path(cache_dir, self.model, decision_id, persona, candidate_metrics)
            if cache_path is not None and cache_path.exists():
                votes.extend(_votes_from_cached_json(cache_path))
                continue
            try:
                payload = self._call_openai(candidate_metrics, persona=persona)
            except Exception:
                votes.append(JudgeVote(f"openai_panel:{persona}", "unknown", "OpenAI judge call failed safely."))
                continue
            persona_votes = _votes_from_payload(payload, persona=persona)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps([asdict(vote) for vote in persona_votes], indent=2),
                    encoding="utf-8",
                )
            votes.extend(persona_votes)
        return tuple(votes)

    def _call_openai(self, candidate_metrics: dict[str, float | int | str | bool], *, persona: str) -> object:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        prompt = (
            f"You are the {persona} judge on TheSlowBrain rubric promotion panel. "
            "Score this Slow Brain rubric candidate as JSON list of objects with "
            "dimension,outcome,score,confidence,rationale. Use pass/fail/unknown outcomes. "
            f"Dimensions: {', '.join(JUDGE_DIMENSIONS)}. Metrics: {json.dumps(candidate_metrics, sort_keys=True)}"
        )
        response = client.responses.create(model=self.model, input=prompt)
        return json.loads(str(response.output_text))


def open_code_failures(reviews: tuple[CouncilReview, ...]) -> tuple[str, ...]:
    failures = [
        vote.dimension
        for review in reviews
        for vote in review.votes
        if vote.outcome in {"fail", "unknown"}
    ]
    return tuple(sorted(set(failures)))


def _heuristic_votes(metrics: dict[str, float | int | str | bool]) -> tuple[JudgeVote, ...]:
    p_value = _float_metric(metrics, "p_value", 1.0)
    drawdown = _float_metric(metrics, "max_drawdown_pct", 100.0)
    alpha = _float_metric(metrics, "alpha_vs_benchmark_pct", 0.0)
    deflated_sharpe = _float_metric(metrics, "deflated_sharpe", -99.0)
    deflated_sharpe_p_value = _float_metric(metrics, "deflated_sharpe_p_value", 1.0)
    pbo = _float_metric(metrics, "probability_backtest_overfit", 1.0)
    capacity_ok = bool(metrics.get("capacity_ok"))
    excluded_errors = _float_metric(metrics, "excluded_error_feature_count", 0.0)
    return (
        _vote("profit_evidence", alpha > 0.0, 0.78 if alpha > 0.0 else 0.35, "Candidate must beat benchmark."),
        _vote("risk_control", drawdown <= 25.0 and capacity_ok, 0.76, "Drawdown and capacity must clear guards."),
        _vote("data_quality", excluded_errors == 0.0, 0.72, "Error-severity data-quality rows must be excluded."),
        _vote("execution_safety", True, 0.85, "Live broker execution remains blocked."),
        _vote("report_honesty", True, 0.80, "Report carries safety and confidence caveats."),
        _vote(
            "overfitting_robustness",
            p_value <= 0.05 and deflated_sharpe > 0.0 and deflated_sharpe_p_value <= 0.05 and pbo <= 0.34,
            0.82,
            "Candidate needs corrected significance, positive deflated Sharpe, and low PBO.",
        ),
        _vote(
            "economic_rationale",
            str(metrics.get("candidate_reason") or "").strip() != "",
            0.70,
            "Candidate must have a reason beyond score mining.",
        ),
    )


def _vote(dimension: str, passed: bool, confidence: float, rationale: str) -> JudgeVote:
    return JudgeVote(
        dimension=dimension,
        outcome="pass" if passed else "fail",
        rationale=rationale,
        score=1.0 if passed else 0.0,
        confidence=confidence,
    )


def _weighted_score(votes: tuple[JudgeVote, ...]) -> float:
    known = [vote for vote in votes if vote.outcome != "unknown" and vote.confidence > 0.0]
    if not known:
        return 0.0
    return sum(vote.score * vote.confidence for vote in known) / sum(vote.confidence for vote in known)


def _mean_confidence(votes: tuple[JudgeVote, ...]) -> float:
    return sum(vote.confidence for vote in votes) / len(votes) if votes else 0.0


def _float_metric(metrics: dict[str, float | int | str | bool], key: str, default: float) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _cache_path(
    cache_dir: Path | None,
    model: str,
    decision_id: str,
    persona: str,
    metrics: dict[str, float | int | str | bool],
) -> Path | None:
    if cache_dir is None:
        return None
    raw = f"{model}:{decision_id}:{persona}:{json.dumps(metrics, sort_keys=True)}"
    return cache_dir / f"{sha256(raw.encode('utf-8')).hexdigest()}.json"


def _votes_from_cached_json(path: Path) -> tuple[JudgeVote, ...]:
    return _votes_from_payload(json.loads(path.read_text(encoding="utf-8")))


def _votes_from_payload(payload: object, *, persona: str = "openai_panel") -> tuple[JudgeVote, ...]:
    if not isinstance(payload, list):
        return (JudgeVote(f"openai_panel:{persona}", "unknown", "OpenAI judge returned non-list JSON."),)
    votes: list[JudgeVote] = []
    for item in payload:
        if isinstance(item, dict):
            votes.append(_vote_from_mapping(item))
    return tuple(votes) or (JudgeVote(f"openai_panel:{persona}", "unknown", "OpenAI judge returned no votes."),)


def _vote_from_mapping(item: dict[str, Any]) -> JudgeVote:
    outcome = item.get("outcome")
    if outcome not in {"pass", "fail", "unknown"}:
        outcome = "unknown"
    typed_outcome = cast("JudgeOutcome", outcome)
    return JudgeVote(
        dimension=str(item.get("dimension") or "openai_panel"),
        outcome=typed_outcome,
        score=max(0.0, min(1.0, _raw_float(item.get("score"), 0.0))),
        confidence=max(0.0, min(1.0, _raw_float(item.get("confidence"), 0.0))),
        rationale=str(item.get("rationale") or "No rationale supplied."),
    )


def _raw_float(value: object, default: float) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cohen_kappa(paired: list[tuple[HumanLabel, HumanLabel]]) -> float:
    if not paired:
        return 0.0
    labels = {"BUY", "SELL", "HOLD", "UNKNOWN"}
    observed = sum(1 for human, automatic in paired if human == automatic) / len(paired)
    expected = 0.0
    for label in labels:
        human_rate = sum(1 for human, _ in paired if human == label) / len(paired)
        auto_rate = sum(1 for _, automatic in paired if automatic == label) / len(paired)
        expected += human_rate * auto_rate
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
