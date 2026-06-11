"""A neural network from NOTHING -- ~120 lines, pure Python, no libraries.

This is a learning sandbox (not part of the trading project). It builds the three things we talked
about, from scratch:

    PART 1  -- one number that remembers how it was made, so blame can flow backward (the engine)
    PART 2  -- neurons, layers, a network (the "scouts -> verdicts" wiring)
    PART 3  -- training: walk every wire a little downhill to shrink the error

Then it proves the big lesson live: a network with NO hidden layer canNOT learn XOR
("this OR that, but not both"), while one WITH a hidden layer can.

Run it:   uv run python learn/tiny_nn.py     (or just: python learn/tiny_nn.py)
"""

from __future__ import annotations

import math
import random

# ======================================================================================
# PART 1 -- THE ENGINE: a number that remembers how it was made.
# Every Value holds its number (`data`) and its blame (`grad`). When you add or multiply
# Values, the result remembers its parents, so `.backward()` can trace blame back to every wire.
# ======================================================================================


class Value:
    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = data
        self.grad = 0.0  # "how much would the error change if I nudged this number up?" -- filled in by backward()
        self._backward = lambda: None  # how to hand blame to my parents
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # adding passes blame straight through to both parents, unchanged
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # multiplying: each parent's blame is scaled by the OTHER parent's value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> Value:
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward() -> None:
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        # the "light-up" function: squashes any number into (-1, 1)
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        # 1) list every Value in the order it was built (topological sort)
        order: list[Value] = []
        seen: set[Value] = set()

        def build(node: Value) -> None:
            if node not in seen:
                seen.add(node)
                for child in node._prev:
                    build(child)
                order.append(node)

        build(self)
        # 2) the final answer is 100% to blame for itself, then hand blame backward, parent by parent
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    # let Python's -, sum(), etc. work on Values
    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other


# ======================================================================================
# PART 2 -- THE NETWORK: neurons (scouts), layers, and a stack of layers.
# A neuron = (each input x its wire) all added up, + a bias, then "lit up" by tanh.
# ======================================================================================


class Neuron:
    def __init__(self, n_inputs: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]  # the wires
        self.b = Value(0.0)  # the bias (a starting nudge)

    def __call__(self, x: list[Value | float]) -> Value:
        total = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return total.tanh()  # light up

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
    """A stack of layers. sizes=[4, 1] means: hidden layer of 4 scouts, then 1 output."""

    def __init__(self, n_inputs: int, sizes: list[int]) -> None:
        widths = [n_inputs, *sizes]
        self.layers = [Layer(widths[i], widths[i + 1]) for i in range(len(sizes))]

    def __call__(self, x: list[Value | float]) -> Value:
        values: list[Value] = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            values = layer(values)
        return values[0]  # our toy nets end in a single output

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


# ======================================================================================
# PART 3 -- TRAINING: show examples, measure the error, walk every wire downhill, repeat.
# ======================================================================================

# XOR: the classic "you NEED a hidden layer" problem. Output is +1 when the two inputs DIFFER.
INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TARGETS = [-1.0, 1.0, 1.0, -1.0]


def train(model: MLP, epochs: int, learning_rate: float, label: str) -> None:
    for epoch in range(epochs):
        predictions = [model(x) for x in INPUTS]  # forward: make a guess for each example
        # error = how far the guesses are from the truth (squared so big misses hurt more)
        loss = sum(((p - y) ** 2 for p, y in zip(predictions, TARGETS, strict=True)), Value(0.0))

        for p in model.parameters():  # wipe last round's blame
            p.grad = 0.0
        loss.backward()  # backward: hand blame back to every wire
        for p in model.parameters():  # nudge every wire a small step downhill
            p.data -= learning_rate * p.grad

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  [{label}] epoch {epoch:4d}   error {loss.data:.4f}")

    guesses = [round(model(x).data, 2) for x in INPUTS]
    print(f"  [{label}] final guesses {guesses}  (target {TARGETS})\n")


def main() -> None:
    random.seed(1)
    print("\n--- NO hidden layer (just inputs -> 1 output): a straight line, CANNOT solve XOR ---")
    train(MLP(2, [1]), epochs=200, learning_rate=0.1, label="flat ")

    random.seed(1)
    print("--- WITH a hidden layer (inputs -> 8 scouts -> 1 output): learns the AND/OR combo ---")
    train(MLP(2, [8, 1]), epochs=200, learning_rate=0.1, label="deep ")

    print("Read it bottom-up: the flat net's error gets stuck; the deep net's error falls near 0.")
    print("That gap is the whole point of a hidden layer.\n")
    print("TRY THIS (edit the file and re-run):")
    print("  * change tanh() to a ReLU-style light-up and see what happens")
    print("  * shrink the hidden layer to [2, 1], or grow it to [16, 16, 1]")
    print("  * set learning_rate = 1.5 and watch the error explode (step too big)")
    print("  * set learning_rate = 0.001 and watch it crawl (step too small)")


if __name__ == "__main__":
    main()
