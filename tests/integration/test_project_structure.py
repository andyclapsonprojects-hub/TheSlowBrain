from __future__ import annotations

from pathlib import Path


def test_slowbrain_source_modules_stay_under_reviewable_line_limit() -> None:
    source_root = Path("src/slowbrain")
    oversized = {
        path.as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts and len(path.read_text(encoding="utf-8").splitlines()) > 500
    }

    assert oversized == {}
