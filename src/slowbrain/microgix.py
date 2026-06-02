"""Tiny stdlib scalar autograd core for PR9 gating experiments."""

from __future__ import annotations

from collections.abc import Callable
from math import exp, log, tanh
from typing import Any, SupportsFloat


class Value:
    """A scalar value with reverse-mode autodiff."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Value] = set(_children)
        self._op = _op

    def __add__(self, other: Value | SupportsFloat) -> Value:
        right = _as_value(other)
        out = Value(self.data + right.data, (self, right), "+")

        def _backward() -> None:
            self.grad += out.grad
            right.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | SupportsFloat) -> Value:
        return self + other

    def __sub__(self, other: Value | SupportsFloat) -> Value:
        return self + (-_as_value(other))

    def __rsub__(self, other: Value | SupportsFloat) -> Value:
        return _as_value(other) + (-self)

    def __mul__(self, other: Value | SupportsFloat) -> Value:
        right = _as_value(other)
        out = Value(self.data * right.data, (self, right), "*")

        def _backward() -> None:
            self.grad += right.data * out.grad
            right.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | SupportsFloat) -> Value:
        return self * other

    def __truediv__(self, other: Value | SupportsFloat) -> Value:
        return self * (_as_value(other) ** -1.0)

    def __rtruediv__(self, other: Value | SupportsFloat) -> Value:
        return _as_value(other) * (self**-1.0)

    def __pow__(self, other: float) -> Value:
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1.0)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def tanh(self) -> Value:
        value = tanh(self.data)
        out = Value(value, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - value**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        value = exp(_clamp(self.data, -50.0, 50.0))
        out = Value(value, (self,), "exp")

        def _backward() -> None:
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        value = log(max(self.data, 1e-12))
        out = Value(value, (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / max(self.data, 1e-12)) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        return (1.0 + (-self).exp()) ** -1.0

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(value: Value) -> None:
            if value in visited:
                return
            visited.add(value)
            for child in value._prev:
                build(child)
            topo.append(value)

        build(self)
        self.grad = 1.0
        for value in reversed(topo):
            value._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other


def zero_grad(values: tuple[Value, ...]) -> None:
    for value in values:
        value.grad = 0.0


def _as_value(value: Value | SupportsFloat) -> Value:
    return value if isinstance(value, Value) else Value(float(value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
