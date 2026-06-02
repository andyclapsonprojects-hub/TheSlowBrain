from __future__ import annotations

from slowbrain.numeric import float_or_default, optional_float


def test_optional_float_rejects_missing_malformed_and_bool_by_default() -> None:
    assert optional_float(None) is None
    assert optional_float("") is None
    assert optional_float("not-a-number") is None
    assert optional_float(True) is None


def test_optional_float_can_preserve_legacy_bool_and_rounding_behaviour() -> None:
    assert optional_float(True, allow_bool=True) == 1.0
    assert optional_float("1.2345678", ndigits=4) == 1.2346


def test_float_or_default_returns_default_for_bad_values() -> None:
    assert float_or_default("bad", default=-1.0) == -1.0
    assert float_or_default("2.5", default=-1.0) == 2.5
