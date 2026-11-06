# %% [markdown]
# # C1.3 An autograd engine — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/C/c1_3_autograd_engine_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# The engine, copied from labs/C/c1_3_autograd_engine.py; the four new operations are added below it.
def topological_order(root: Tensor) -> list[Tensor]:
    """Every node reachable from root, each after all of its inputs (iterative DFS post-order)."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, inputs_done = stack.pop()
        if inputs_done:
            order.append(node)  # all of its inputs are already in `order`
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))
    return order



def unbroadcast(grad: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Sum a gradient down to `shape`, undoing NumPy broadcasting. grad (broadcast shape) -> shape."""
    while grad.ndim > len(shape):  # axes that broadcasting added on the left
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):  # axes that were size 1 and got stretched
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy array that records the operations producing it, for reverse-mode autodiff."""

    def __init__(self, data: FloatArray | float, _prev: tuple[Tensor, ...] = (), op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = _prev
        self._backward: Callable[[], None] = lambda: None
        self.op = op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data @ other.data, (self, other), "@")  # (n, d) @ (d, k) -> (n, k)

        def _backward() -> None:
            self.grad += out.grad @ other.data.T  # (n, k) @ (k, d) -> (n, d)
            other.grad += self.data.T @ out.grad  # (d, n) @ (n, k) -> (d, k)

        out._backward = _backward
        return out

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # put the summed axis back as size 1
            self.grad += np.broadcast_to(g, self.shape)  # every input entry gets the gradient of its sum

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self + (-other if isinstance(other, Tensor) else -float(other))

    def __radd__(self, other: float) -> Tensor:
        return self + other

    def __rmul__(self, other: float) -> Tensor:
        return self * other

    def backward(self) -> None:
        """Backpropagate from a scalar (size-1) tensor."""
        if self.data.size != 1:
            raise ValueError(f"backward() needs a scalar output, got shape {self.shape}")
        order = topological_order(self)  # works on any object with ._prev
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()




# %%
# Exercise 2: four new operations, attached to the class.
def _pow(self: Tensor, k: float) -> Tensor:
    out = Tensor(self.data**k, (self,), f"**{k}")

    def _backward() -> None:
        self.grad += k * self.data ** (k - 1) * out.grad  # d(a^k)/da = k a^(k-1)

    out._backward = _backward
    return out


def _tanh(self: Tensor) -> Tensor:
    out = Tensor(np.tanh(self.data), (self,), "tanh")

    def _backward() -> None:
        self.grad += (1.0 - out.data**2) * out.grad

    out._backward = _backward
    return out


def _transpose(self: Tensor) -> Tensor:
    if self.data.ndim != 2:
        raise ValueError(f".T needs a 2-D tensor, got shape {self.shape}")
    out = Tensor(self.data.T, (self,), "T")

    def _backward() -> None:
        self.grad += out.grad.T  # transposing is its own inverse

    out._backward = _backward
    return out


def _max(self: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    idx = np.argmax(self.data, axis=axis)  # first maximum on ties
    out = Tensor(self.data.max(axis=axis, keepdims=keepdims), (self,), "max")

    def _backward() -> None:
        g = out.grad if keepdims else np.expand_dims(out.grad, axis)
        mask = np.zeros_like(self.data)
        np.put_along_axis(mask, np.expand_dims(idx, axis), 1.0, axis=axis)  # 1 at the chosen max
        # Near a tie the max switches between two entries when one is nudged by h, so central differences
        # see half the gradient on each: a correct implementation "fails" the check there.
        self.grad += mask * g

    out._backward = _backward
    return out


Tensor.__pow__ = _pow  # type: ignore[attr-defined]
Tensor.tanh = _tanh  # type: ignore[attr-defined]
Tensor.T = property(_transpose)  # type: ignore[attr-defined]
Tensor.max = _max  # type: ignore[attr-defined]


def tensor_gradcheck(f: Callable[..., Tensor], *shapes: tuple[int, ...]) -> float:
    """Worst relative error between Tensor gradients and central differences, with a random output weighting."""
    gen = np.random.default_rng(1)
    arrays = [gen.normal(size=s) for s in shapes]
    R = gen.normal(size=f(*[Tensor(a) for a in arrays]).shape)

    def loss_of(arrs: list[FloatArray]) -> float:
        return float(np.sum(R * f(*[Tensor(a) for a in arrs]).data))

    ts = [Tensor(a) for a in arrays]
    (f(*ts) * Tensor(R)).sum().backward()
    worst = 0.0
    for k, t in enumerate(ts):
        num = np.zeros_like(arrays[k])
        for idx in np.ndindex(arrays[k].shape):
            up = [a.copy() for a in arrays]
            dn = [a.copy() for a in arrays]
            up[k][idx] += 1e-6
            dn[k][idx] -= 1e-6
            num[idx] = (loss_of(up) - loss_of(dn)) / 2e-6
        assert t.grad.shape == arrays[k].shape
        worst = max(worst, float(np.linalg.norm(t.grad - num) / max(np.linalg.norm(t.grad) + np.linalg.norm(num), 1e-12)))
    return worst


section("exercise 2: gradient checks of the new operations")
for label, fn, shapes in [
    ("pow 3", lambda p: p**3, ((3, 4),)),
    ("pow 2 with broadcast add", lambda p, q: (p + q) ** 2, ((3, 4), (4,))),
    ("tanh", lambda p: p.tanh(), ((3, 4),)),
    ("transpose then matmul", lambda p, q: p.T @ q, ((3, 4), (3, 2))),
    ("max axis=1", lambda p: p.max(axis=1), ((3, 4),)),
    ("max axis=0 keepdims", lambda p: p.max(axis=0, keepdims=True), ((3, 4),)),
]:
    err = tensor_gradcheck(fn, *shapes)
    print(f"{label:<28} rel error {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}  {'PASS' if err < 1e-6 else 'FAIL'}")

section("exercise 2: a tie in max")
tie = Tensor(np.array([[1.0, 3.0, 3.0]]))
tie.max(axis=1).sum().backward()
print(f"x = {show(tie.data)}  our gradient {show(tie.grad)}  (all to the first maximum)")
num = []
for j in range(3):
    up, dn = tie.data.copy(), tie.data.copy()
    up[0, j] += 1e-6
    dn[0, j] -= 1e-6
    num.append((up.max() - dn.max()) / 2e-6)
print(f"central differences {show(np.array(num))}  (each tied entry looks like half)")

section("exercise 2: shifting by the row max")
x = Tensor(np.random.default_rng(0).normal(size=(2, 4)))
(x - x.max(axis=1, keepdims=True)).sum().backward()
print(f"gradient {show(x.grad)}")
print(f"row sums of the gradient {show(x.grad.sum(axis=1))}")

# %%
section("exercise 1 check")
xv, yv = Tensor(0.0), Tensor(2.0)
f = (xv * yv + xv.exp()).log()
f.backward()
print(f"(a) f = {float(f.data)}  df/dx = {float(xv.grad)}  df/dy = {float(yv.grad)}   formula: (y + e^x)/(xy + e^x) = 3, x/(xy + e^x) = 0")
Xc = Tensor(np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0]]))
cc = Tensor(np.array([[2.0], [-1.0]]))
(Xc * cc).sum().backward()
print(f"(c) dL/dX {show(Xc.grad)} shape {Xc.grad.shape}   dL/dc {show(cc.grad)} shape {cc.grad.shape}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
