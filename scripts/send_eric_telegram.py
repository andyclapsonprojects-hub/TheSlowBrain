"""Preview or send the latest Eric brief to Telegram."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.config import load_config
from slowbrain.telegram import TelegramDeliveryResult, extract_eric_message, send_telegram_message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=Path("reports/first-slowbrain-report.json"))
    parser.add_argument("--send", action="store_true", help="Actually call Telegram. Default is dry-run preview.")
    args = parser.parse_args()

    report_path = args.report if args.report.is_absolute() else args.project_root / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    message = extract_eric_message(report)
    result = send_telegram_message(load_config(args.project_root), message, send=args.send)
    delivery_path = _write_delivery_record(args.project_root, report_path, message, args.send, result)

    print(f"Telegram status: {result.status}")
    print(f"Reason: {result.reason}")
    print(f"Delivery record: {delivery_path}")
    if not args.send:
        print(message)
    return 0 if result.status in {"preview_only", "sent"} else 2


def _write_delivery_record(
    project_root: Path,
    report_path: Path,
    message: str,
    send: bool,
    result: TelegramDeliveryResult,
) -> Path:
    output_dir = project_root / "reports" / "telegram-deliveries"
    output_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC)
    created_at = created.isoformat()
    output_path = output_dir / f"telegram-{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    payload = {
        "schema": "theslowbrain.telegram_delivery.v1",
        "created_at": created_at,
        "report_path": str(report_path),
        "mode": "send" if send else "dry_run",
        "message_characters": len(message),
        "message_preview": message,
        "result": asdict(result),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
