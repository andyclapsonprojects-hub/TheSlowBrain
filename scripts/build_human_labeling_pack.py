"""Build a human-usable Slow Brain labelling pack with OHLCV context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import load_config
from slowbrain.decision_capture import DECISION_CAPTURE_JSONL
from slowbrain.human_labeling import build_human_labeling_pack, load_decision_capture_records, write_human_labeling_pack
from slowbrain.legacy_price_cache import LegacyN8nPriceCacheProvider
from slowbrain.market_data import FallbackPriceHistoryProvider, PriceHistoryProvider
from slowbrain.market_data_vendors import build_price_history_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture", type=Path, default=DECISION_CAPTURE_JSONL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    capture_path = args.capture if args.capture.is_absolute() else args.project_root / args.capture
    config = load_config(args.project_root)
    records = load_decision_capture_records(capture_path)
    provider = _build_review_price_provider(config.legacy_stock_project_root, args.project_root)
    pack = build_human_labeling_pack(
        capture_path=capture_path,
        records=records,
        price_provider=provider,
        limit=args.limit,
    )
    outputs = write_human_labeling_pack(pack, project_root=args.project_root)

    print("Human labelling pack")
    print(f"- cases: {pack.case_count}")
    print(f"- CSV: {outputs.csv_path}")
    print(f"- HTML: {outputs.html_path}")
    print(f"- JSON: {outputs.json_path}")
    return 0


def _build_review_price_provider(legacy_root: Path, project_root: Path) -> PriceHistoryProvider:
    providers: list[PriceHistoryProvider] = []
    legacy_provider = LegacyN8nPriceCacheProvider.from_legacy_project_root(legacy_root)
    if legacy_provider.cache_dir.exists():
        providers.append(legacy_provider)
    config = load_config(project_root)
    vendor_provider = build_price_history_provider(config, project_root=project_root)
    if vendor_provider is not None:
        providers.append(vendor_provider)
    return FallbackPriceHistoryProvider(tuple(providers))


if __name__ == "__main__":
    raise SystemExit(main())
