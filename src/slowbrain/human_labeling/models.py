"""Human-labeling pack constants and data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..technical_context import TechnicalContext

HUMAN_LABELING_DIR = Path("reports/human-labeling")
HUMAN_LABELING_JSON = HUMAN_LABELING_DIR / "latest-human-labeling-pack.json"
HUMAN_LABELING_CSV = HUMAN_LABELING_DIR / "latest-human-labeling-pack.csv"
HUMAN_LABELING_HTML = HUMAN_LABELING_DIR / "latest-human-labeling-pack.html"

CSV_COLUMNS = (
    "example_id",
    "ticker",
    "signal_date",
    "slowbrain_action",
    "slowbrain_score",
    "slowbrain_reason",
    "sentiment",
    "sentiment_confidence",
    "catalyst_strength",
    "feature_momentum_20d_pct",
    "feature_mean_reversion_z_20d",
    "outcome_10d_net_return_pct",
    "market_context_status",
    "market_context_reason",
    "price_asof_date",
    "price_source",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "previous_close",
    "day_change_pct",
    "gap_pct",
    "intraday_return_pct",
    "candle_range_pct",
    "candle_body_pct",
    "close_location_pct",
    "volume_ratio_20d",
    "volume_signal",
    "sma_5",
    "sma_20",
    "technical_trend",
    "technical_momentum_5d_pct",
    "technical_momentum_20d_pct",
    "distance_from_20d_high_pct",
    "distance_from_20d_low_pct",
    "candlestick_patterns",
    "candlestick_summary",
    "human_label",
    "human_rationale",
)


@dataclass(frozen=True)
class HumanLabelingCase:
    example_id: str
    ticker: str
    signal_date: str
    line_number: int
    slowbrain_action: str
    slowbrain_score: float
    slowbrain_reason: str
    sentiment: str
    sentiment_confidence: float
    catalyst_strength: float
    feature_momentum_20d_pct: float
    feature_mean_reversion_z_20d: float
    outcome_10d_net_return_pct: float | None
    source: str
    technical_context: TechnicalContext
    human_label: str
    human_rationale: str


@dataclass(frozen=True)
class HumanLabelingPack:
    schema: str
    generated_at: str
    source_capture_path: str
    case_count: int
    mode: str
    rows_are_human_labels: bool
    label_values_allowed: tuple[str, ...]
    notes: tuple[str, ...]
    cases: tuple[HumanLabelingCase, ...]


@dataclass(frozen=True)
class HumanLabelingOutputs:
    json_path: Path
    csv_path: Path
    html_path: Path
