from __future__ import annotations

from pathlib import Path

from slowbrain.data_import import build_import_manifest, validate_manifest


def test_import_excludes_old_reports(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    project = tmp_path / "project"
    (legacy / "data").mkdir(parents=True)
    (legacy / "paper_trading").mkdir()
    (legacy / "reports").mkdir()
    (legacy / "data" / "raw.json").write_text('{"ok": true}', encoding="utf-8")
    (legacy / "paper_trading" / "orders.csv").write_text("ticker\nAAPL\n", encoding="utf-8")
    (legacy / "paper_trading" / "orders.template.csv").write_text("template\n", encoding="utf-8")
    (legacy / "reports" / "old-report.json").write_text('{"verdict": "messy"}', encoding="utf-8")

    manifest = build_import_manifest(legacy_root=legacy, project_root=project, copy_files=True)

    source_paths = {Path(record.source_path).name for record in manifest.records}
    destination_paths = {record.destination_path for record in manifest.records}
    assert "raw.json" in source_paths
    assert "orders.csv" in source_paths
    assert "old-report.json" not in source_paths
    assert "orders.template.csv" not in source_paths
    assert not any("reports" in path for path in destination_paths)
    assert validate_manifest(Path(manifest.import_root) / "import_manifest.json") == []


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    project = tmp_path / "project"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "raw.json").write_text('{"ok": true}', encoding="utf-8")
    manifest = build_import_manifest(legacy_root=legacy, project_root=project, copy_files=True)
    imported = Path(manifest.records[0].destination_path)
    imported.write_text("changed", encoding="utf-8")

    errors = validate_manifest(Path(manifest.import_root) / "import_manifest.json")

    assert any(error.startswith("size_mismatch") or error.startswith("checksum_mismatch") for error in errors)
