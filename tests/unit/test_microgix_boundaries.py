from __future__ import annotations

from slowbrain.microgix import Value, zero_grad


def test_microgix_reverse_ops_and_zero_grad() -> None:
    x = Value(4.0)
    y = 10.0 - x + 8.0 / x + x / 2.0

    y.backward()

    assert round(y.data, 6) == 10.0
    assert round(x.grad, 6) == -1.0

    zero_grad((x,))
    assert x.grad == 0.0


def test_microgix_nonlinear_ops_are_stable_at_extremes() -> None:
    x = Value(100.0)
    y = x.exp().log() + x.sigmoid() + x.tanh()

    y.backward()

    assert y.data > 0.0
    assert x.grad > 0.0
    assert "Value(data=" in repr(y)
    assert x == x
    assert x != Value(100.0)
