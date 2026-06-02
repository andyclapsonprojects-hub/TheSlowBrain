"""Decision-capture loading helpers for human-labeling packs."""

from __future__ import annotations

import json
from pathlib import Path


def load_decision_capture_records(path: Path) -> tuple[dict[str, object], ...]:
    """Load captured decisions tolerantly.

    Handles both one-object-per-line JSONL and pretty-printed/concatenated JSON objects
    (and a top-level JSON array), so a hand-edited or mixed-format file is read rather
    than crashing the loader.
    """
    if not path.exists():
        return ()
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, object]] = []
    decoder = json.JSONDecoder()
    position = 0
    length = len(text)
    while position < length:
        while position < length and text[position] in " \t\r\n,[]":
            position += 1
        if position >= length:
            break
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            records.append({str(key): item for key, item in value.items()})
        position = end
    return tuple(records)

