"""Submit a fresh, approved Trading 212 execution preview."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import AppConfig, load_config
from slowbrain.live_execution import (
    EXECUTION_LEDGER_JSONL,
    LATEST_PREVIEW_JSON,
    LATEST_SUBMISSION_JSON,
    load_json_object,
    submit_execution_preview,
    write_json,
)
from slowbrain.trading212 import Trading212Gateway, build_trading212_client, credentials_available

ClientFactory = Callable[[AppConfig], Trading212Gateway]


def main(argv: Sequence[str] | None = None, *, client_factory: ClientFactory | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preview", type=Path, default=LATEST_PREVIEW_JSON)
    parser.add_argument("--out", type=Path, default=LATEST_SUBMISSION_JSON)
    parser.add_argument("--ledger", type=Path, default=EXECUTION_LEDGER_JSONL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(project_root=args.project_root)
    preview = load_json_object(args.project_root / args.preview)
    result = submit_execution_preview(
        preview=preview,
        config=config,
        client=_build_client(config, client_factory),
        ledger_path=args.project_root / args.ledger,
        execute=args.execute,
        approval_token=args.approval_token,
    )
    out_path = args.project_root / args.out
    write_json(out_path, result)
    print("TheSlowBrain live execution submit complete.")
    print(f"Status: {result.get('status')}")
    print(f"Reason: {result.get('reason', 'none')}")
    print(f"Orders attempted: {result.get('orders_attempted', 0)}")
    print(f"Orders submitted: {str(result.get('orders_submitted')).lower()}")
    print(f"Submission: {out_path}")
    if result.get("status") == "submitted":
        return 0
    if not args.execute and result.get("reason") == "execute_flag_not_set":
        return 0
    return 2


def _build_client(config: AppConfig, client_factory: ClientFactory | None) -> Trading212Gateway | None:
    if not credentials_available(config):
        return None
    factory = client_factory or build_trading212_client
    return factory(config)


if __name__ == "__main__":
    raise SystemExit(main())
