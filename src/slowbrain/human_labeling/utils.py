"""Small coercion and formatting helpers for human-labeling artifacts."""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path

from ..eval_council import VALID_HUMAN_LABELS
from ..numeric import optional_float


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return str(value or "").strip()


def _float(value: object) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else 0.0


def _optional_float(value: object) -> float | None:
    return optional_float(value, ndigits=6)


def _existing_label(value: object) -> str:
    label = _text(value).upper()
    return label if label in VALID_HUMAN_LABELS else ""


def _blank_none(value: object) -> object:
    return "" if value is None else value


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _fmt(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)
