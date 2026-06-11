"""BUILD-IT-YOURSELF worksheet. YOU write the 5 ideas; the boring scaffolding is done for you.

How to use:
  1. Find each "TODO (n)" below and replace the `raise NotImplementedError(...)` with real code.
     Each one is 1-3 lines. The hint above it points back to what we talked about.
  2. Run it:  uv run python learn/build_it.py
  3. It will fail at the first unfilled TODO with a clear message -- fill that one, run again, repeat.
  4. When all 5 are done: the flat net stays stuck (~4.0) and the deep net's error falls near 0.

Stuck on one? The finished version is in learn/tiny_nn.py -- but try first, that's where the learning is.
There are 5 TODOs:  (1) add-backward  (2) mul-backward  (3) tanh-backward  (4) neuron forward  (5) the train step.
"""

from __future__ import annotations

import math
import random


class Value:
    """One number that remembers how it was made, so blame (`grad`) can flow backward."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # TODO (1): addition passes blame STRAIGHT THROUGH, unchanged, to BOTH parents.
            # Bump each parent's .grad up by out.grad. (Two lines: self.grad += ... ; other.grad += ...)
            raise NotImplementedError("TODO (1): write the add-backward (see hint above)")

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # TODO (2): multiplication scales each parent's blame by the OTHER parent's value.
            # self.grad += (other's value) * out.grad ;  other.grad += (self's value) * out.grad
            raise NotImplementedError("TODO (2): write the mul-backward (use other.data, self.data, out.grad)")

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> Value:
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward() -> None:
            self.grad += (power * self.data ** (power - 1)) * out.grad  # given (power rule)

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # TODO (3): the slope of tanh is (1 - t*t). Add (1 - t*t) * out.grad to self.grad.
            raise NotImplementedError("TODO (3): write the tanh-backward")

        out._backward = _backward
        return out

    def backward(self) -> None:
        # GIVEN: list every Value in build order, then hand blame backward parent-by-parent.
        order: list[Value] = []
        seen: set[Value] = set()

        def build(node: Value) -> None:
            if node not in seen:
                seen.add(node)
                for child in node._prev:
                    build(child)
                order.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other


class Neuron:
    def __init__(self, n_inputs: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x: list[Value | float]) -> Value:
        # TODO (4): a neuron = (each wire * its input), all summed, + the bias, then "lit up" with tanh.
        # hint: total = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b) ; return total.tanh()
        raise NotImplementedError("TODO (4): write the neuron forward pass")

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, n_inputs: int, sizes: list[int]) -> None:
        widths = [n_inputs, *sizes]
        self.layers = [Layer(widths[i], widths[i + 1]) for i in range(len(sizes))]

    def __call__(self, x: list[Value | float]) -> Value:
        values: list[Value] = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            values = layer(values)
        return values[0]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TARGETS = [-1.0, 1.0, 1.0, -1.0]  # XOR: +1 when the inputs differ


def train(model: MLP, epochs: int, learning_rate: float, label: str) -> None:
    for epoch in range(epochs):
        predictions = [model(x) for x in INPUTS]

        # TODO (5a): error = sum of (prediction - target)**2 over the 4 examples.
        # hint: loss = sum(((p - y) ** 2 for p, y in zip(predictions, TARGETS, strict=True)), Value(0.0))
        loss = None  # replace this
        if loss is None:
            raise NotImplementedError("TODO (5a): compute the error (loss)")

        for p in model.parameters():
            p.grad = 0.0
        # TODO (5b): hand blame back to every wire -- call .backward() on the loss.
        raise NotImplementedError("TODO (5b): call loss.backward(), then delete this line")

        for p in model.parameters():  # noqa: F811 -- you'll reach this once 5b is filled
            # TODO (5c): nudge this wire a small step DOWNHILL: p.data -= learning_rate * p.grad
            raise NotImplementedError("TODO (5c): take the downhill step")

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  [{label}] epoch {epoch:4d}   error {loss.data:.4f}")

    guesses = [round(model(x).data, 2) for x in INPUTS]
    print(f"  [{label}] final guesses {guesses}  (target {TARGETS})\n")


def main() -> None:
    random.seed(1)
    print("\n--- NO hidden layer: should get STUCK around error 4.0 (a line can't split XOR) ---")
    train(MLP(2, [1]), epochs=200, learning_rate=0.1, label="flat ")

    random.seed(1)
    print("--- WITH a hidden layer: error should FALL near 0 ---")
    train(MLP(2, [8, 1]), epochs=200, learning_rate=0.1, label="deep ")


if __name__ == "__main__":
    main()
