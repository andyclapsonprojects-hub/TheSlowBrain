"""Build a read-only Trading 212 broker health report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import AppConfig, load_config
from slowbrain.live_execution import BROKER_HEALTH_JSON, build_broker_health_report, write_json
from slowbrain.trading212 import Trading212Gateway, build_trading212_client, credentials_available

ClientFactory = Callable[[AppConfig], Trading212Gateway]


def main(argv: Sequence[str] | None = None, *, client_factory: ClientFactory | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=BROKER_HEALTH_JSON)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(project_root=args.project_root)
    client = _build_client(config, client_factory)
    report = build_broker_health_report(config=config, client=client)
    out_path = args.project_root / args.out
    write_json(out_path, report)
    print("TheSlowBrain Trading 212 broker health complete.")
    print(f"Status: {report.get('status')}")
    print(f"Environment: {report.get('environment')}")
    print(f"Orders submitted: {str(report.get('orders_submitted')).lower()}")
    print(f"Report: {out_path}")
    return 0 if report.get("status") in {"ok", "blocked"} else 2


def _build_client(config: AppConfig, client_factory: ClientFactory | None) -> Trading212Gateway | None:
    if not credentials_available(config):
        return None
    factory = client_factory or build_trading212_client
    return factory(config)


if __name__ == "__main__":
    raise SystemExit(main())
