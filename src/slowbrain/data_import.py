"""Import legacy raw data and ledgers without importing old reports."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config

IMPORT_NAME = "n8n_original_stock_trader_2026-05-31"
MANIFEST_NAME = "import_manifest.json"
EXCLUDED_TOP_LEVEL = {"reports", "reviews", "memory", ".git", ".venv", "__pycache__"}
EXCLUDED_PAPER_FILES = {"README.md"}
BROKER_AUDIT_FILES = {"live_fills.csv", "live_orders.sqlite", "trading212_idempotency.sqlite"}


@dataclass(frozen=True)
class ImportRecord:
    source_path: str
    destination_path: str
    classification: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ImportManifest:
    schema: str
    generated_at: str
    source_project: str
    import_root: str
    records: tuple[ImportRecord, ...]


def default_legacy_root() -> Path:
    return load_config().legacy_stock_project_root


def default_import_root(project_root: Path) -> Path:
    return project_root / "data" / "imports" / IMPORT_NAME


def iter_import_sources(legacy_root: Path) -> Iterable[tuple[Path, str]]:
    """Yield importable legacy files and their classification."""
    data_root = legacy_root / "data"
    paper_root = legacy_root / "paper_trading"
    if data_root.exists():
        for path in sorted(data_root.rglob("*")):
            if path.is_file():
                yield path, "raw"
    if paper_root.exists():
        for path in sorted(paper_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in EXCLUDED_PAPER_FILES or path.name.endswith(".template.csv"):
                continue
            classification = "broker_audit" if path.name in BROKER_AUDIT_FILES else "ledger"
            if path.name == "andy_hypotheses.txt":
                classification = "raw"
            yield path, classification


def assert_allowed_source(path: Path, legacy_root: Path) -> None:
    relative = path.relative_to(legacy_root)
    first_part = relative.parts[0]
    if first_part in EXCLUDED_TOP_LEVEL:
        raise ValueError(f"Refusing to import excluded legacy path: {relative}")
    if first_part not in {"data", "paper_trading"}:
        raise ValueError(f"Refusing to import non-data legacy path: {relative}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_import_manifest(*, legacy_root: Path, project_root: Path, copy_files: bool) -> ImportManifest:
    import_root = default_import_root(project_root)
    records: list[ImportRecord] = []
    for source, classification in iter_import_sources(legacy_root):
        assert_allowed_source(source, legacy_root)
        relative = source.relative_to(legacy_root)
        destination = import_root / relative
        if copy_files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        records.append(
            ImportRecord(
                source_path=str(source),
                destination_path=str(destination),
                classification=classification,
                size_bytes=source.stat().st_size,
                sha256=sha256_file(source),
            )
        )
    manifest = ImportManifest(
        schema="theslowbrain.import_manifest.v1",
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        source_project=str(legacy_root),
        import_root=str(import_root),
        records=tuple(records),
    )
    if copy_files:
        write_manifest(manifest, import_root / MANIFEST_NAME)
    return manifest


def write_manifest(manifest: ImportManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> ImportManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(ImportRecord(**record) for record in payload["records"])
    return ImportManifest(
        schema=str(payload["schema"]),
        generated_at=str(payload["generated_at"]),
        source_project=str(payload["source_project"]),
        import_root=str(payload["import_root"]),
        records=records,
    )


def latest_manifest(project_root: Path) -> Path:
    manifest = default_import_root(project_root) / MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(f"Import manifest not found: {manifest}")
    return manifest


def validate_manifest(path: Path) -> list[str]:
    manifest = load_manifest(path)
    errors: list[str] = []
    if manifest.schema != "theslowbrain.import_manifest.v1":
        errors.append(f"unexpected_schema:{manifest.schema}")
    if not manifest.records:
        errors.append("empty_manifest")
    for record in manifest.records:
        source_parts = Path(record.source_path).parts
        if "reports" in source_parts or "reviews" in source_parts:
            errors.append(f"excluded_legacy_path_imported:{record.source_path}")
        destination = Path(record.destination_path)
        if not destination.exists():
            errors.append(f"missing_destination:{destination}")
            continue
        if destination.stat().st_size != record.size_bytes:
            errors.append(f"size_mismatch:{destination}")
        if sha256_file(destination) != record.sha256:
            errors.append(f"checksum_mismatch:{destination}")
    return errors
