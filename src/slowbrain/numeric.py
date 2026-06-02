"""Shared numeric coercion helpers."""

from __future__ import annotations


def optional_float(value: object, *, allow_bool: bool = False, ndigits: int | None = None) -> float | None:
    """Parse a numeric value without treating blank or malformed text as zero."""
    if isinstance(value, bool) and not allow_bool:
        return None
    if isinstance(value, int | float):
        parsed = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            return None
    else:
        return None
    return round(parsed, ndigits) if ndigits is not None else parsed


def float_or_default(value: object, *, default: float = 0.0, allow_bool: bool = True) -> float:
    parsed = optional_float(value, allow_bool=allow_bool)
    return parsed if parsed is not None else default
