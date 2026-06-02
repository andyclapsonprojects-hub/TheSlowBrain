"""Import legacy n8n Stock Trader raw data and ledgers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slowbrain.data_import import (
    build_import_manifest,
    default_import_root,
    default_legacy_root,
    latest_manifest,
    validate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=default_legacy_root())
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing import manifest without copying files.",
    )
    args = parser.parse_args()

    if args.check:
        manifest_path = latest_manifest(args.project_root)
        errors = validate_manifest(manifest_path)
        if errors:
            print("Import manifest invalid:")
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"Import manifest valid: {manifest_path}")
        return 0

    manifest = build_import_manifest(legacy_root=args.legacy_root, project_root=args.project_root, copy_files=True)
    print(f"Imported records: {len(manifest.records)}")
    print(f"Import root: {default_import_root(args.project_root)}")
    print("Old reports imported: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
