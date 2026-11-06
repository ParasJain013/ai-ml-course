# %% [markdown]
# # C1.3 An autograd engine from scratch — lab
# Lesson: lessons/C/c1_3_autograd_engine.md
# Run: uv run python labs/C/c1_3_autograd_engine.py   (CPU, defaults: ~60 s; the pure-Python scalar engine is the slow part)
# GPU: no

# %%
from __future__ import annotations

import math
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from sklearn.datasets import make_moons

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "c1_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "C" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here uses a GPU: the scalar engine is pure Python. GPU_SCALE only trains longer on more points.
GPU_SCALE = False
N_TRAIN = 400 if GPU_SCALE else 100  # make_moons training points
N_TEST = 1000  # held-out points for accuracy
HIDDEN = 8  # units in each of the two hidden layers
STEPS = 300 if GPU_SCALE else 100  # full-batch gradient-descent steps for the slow scalar engine
TENSOR_STEPS = 1000  # the tensor engine repeats those steps, then keeps going
LR = 0.5

# %%
# Data: scikit-learn's two interleaving half-moons, labels -1 / +1.
section("data")
X_tr, y01_tr = make_moons(n_samples=N_TRAIN, noise=0.1, random_state=SEED)
X_te, y01_te = make_moons(n_samples=N_TEST, noise=0.1, random_state=SEED + 1)
y_tr = 2.0 * y01_tr - 1.0  # (n,) in {-1, +1}
y_te = 2.0 * y01_te - 1.0
print(f"train X {X_tr.shape}  test X {X_te.shape}  classes {np.unique(y_tr).tolist()}")
print(f"first 3 rows {show(X_tr[:3])}  labels {show(y_tr[:3])}")

# %%
# The scalar engine. Each Value remembers its inputs (_prev) and a closure (_backward) that pushes its
# gradient into those inputs. backward() orders the graph topologically and calls the closures in reverse.
section("scalar engine: the Value class")


class Value:
    """A scalar that records the operations producing it, for reverse-mode autodiff."""

    ACCUMULATE = True  # the lab flips this to False once, to show the "=" bug

    def __init__(self, data: float, _prev: tuple[Value, ...] = (), op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = _prev
        self._backward: Callable[[], None] = lambda: None
        self.op = op

    def _add_grad(self, g: float) -> None:
        if Value.ACCUMULATE:
            self.grad += g  # correct: every use of a value adds its share
        else:
            self.grad = g  # the bug: a later use overwrites an earlier one

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self._add_grad(out.grad)  # d(a + b)/da = 1
            other._add_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self._add_grad(other.data * out.grad)  # d(a b)/da = b
            other._add_grad(self.data * out.grad)

        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            self._add_grad(k * self.data ** (k - 1) * out.grad)  # d(a^k)/da = k a^(k-1)

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self._add_grad(out.data * out.grad)  # d(e^a)/da = e^a, the output itself

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self._add_grad(out.grad / self.data)  # d(log a)/da = 1/a

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self._add_grad((1.0 - out.data**2) * out.grad)

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self._add_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    # Everything else is built from the operations above.
    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def backward(self) -> None:
        """Set d(self)/d(self) = 1 and push gradients to every node, in reverse topological order."""
        order = topological_order(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def topological_order(root: Value) -> list[Value]:
    """Every node reachable from root, each after all of its inputs (iterative DFS post-order)."""
    order: list[Value] = []
    visited: set[int] = set()
    stack: list[tuple[Value, bool]] = [(root, False)]
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


print("Value supports: + * ** exp log tanh relu, and - / built from them")

# %%
# Worked example by hand: e = (a b + b)^2 at a = 2, b = -3. b is used twice.
section("worked example: e = (a b + b)^2")
a, b = Value(2.0), Value(-3.0)
c = a * b
d = c + b
e = d**2
names = {id(a): "a", id(b): "b", id(c): "c = a*b", id(d): "d = c+b", id(e): "e = d**2"}
print("topological order:", [names[id(v)] for v in topological_order(e)])
e.backward()
print(f"forward: c = {c.data}  d = {d.data}  e = {e.data}")
print(f"backward: de/dd = {d.grad}  de/dc = {c.grad}  de/da = {a.grad}  de/db = {b.grad}")
print(f"check: de/da = 2 b^2 (a + 1) = {2 * 9 * 3}   de/db = 2 b (a + 1)^2 = {2 * -3 * 9}")

# %%
# The bug: assign gradients with = instead of +=. y = x * x at x = 3 should give dy/dx = 6.
section("the = versus += bug")
for accumulate in (False, True):
    Value.ACCUMULATE = accumulate
    x = Value(3.0)
    y = x * x
    y.backward()
    print(f"{'+=' if accumulate else '= ':<2}  y = x * x at x = 3: dy/dx = {x.grad}  (true value 6.0)")
    x2, y2 = Value(2.0), Value(-3.0)
    e2 = (x2 * y2 + y2) ** 2
    e2.backward()
    print(f"{'+=' if accumulate else '= ':<2}  e = (a b + b)^2: de/db = {y2.grad}  (true value -54.0)")
Value.ACCUMULATE = True

# %%
# Check the scalar engine against central differences on an expression that uses every operation.
section("scalar engine: gradient check of every operation")


def scalar_expr(p: list[float]) -> tuple[Value, list[Value]]:
    """An expression using every op, with p[0] used three times. -> (output, leaf Values)."""
    u, v, w = (Value(t) for t in p)
    out = (u * v + u**3).tanh() + (w.exp() + u).log() - (v - w).relu() / (u**2 + 1.0)
    return out, [u, v, w]


point = [0.7, -1.3, 0.4]
out, leaves = scalar_expr(point)
out.backward()
analytic = np.array([lv.grad for lv in leaves])
numeric = np.zeros(3)
for j in range(3):
    up, dn = list(point), list(point)
    up[j] += 1e-6
    dn[j] -= 1e-6
    numeric[j] = (scalar_expr(up)[0].data - scalar_expr(dn)[0].data) / 2e-6
err = float(np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric)))
print(f"analytic {show(analytic)}")
print(f"numeric  {show(numeric)}")
print(f"relative error {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}: {'PASS' if err < 1e-6 else 'FAIL'}")

# %%
# Train a 2-8-8-1 MLP on the moons with the scalar engine. Weights are drawn in NumPy first so the
# tensor engine (below) can start from exactly the same numbers.
section("scalar engine: training an MLP on make_moons")
widths = [2, HIDDEN, HIDDEN, 1]
init_W = [rng.normal(0, math.sqrt(2 / a_), size=(a_, b_)) for a_, b_ in zip(widths[:-1], widths[1:])]
init_b = [np.zeros(b_) for b_ in widths[1:]]


class ScalarMLP:
    """An MLP of Values: ReLU hidden layers, linear output. Built from NumPy weights W (d_in, d_out)."""

    def __init__(self, Ws: list[FloatArray], bs: list[FloatArray]) -> None:
        self.W = [[[Value(w) for w in row] for row in W] for W in Ws]  # W[l][i][j]
        self.b = [[Value(v) for v in b_] for b_ in bs]

    def parameters(self) -> list[Value]:
        return [w for W in self.W for row in W for w in row] + [v for b_ in self.b for v in b_]

    def __call__(self, x: list[Value | float]) -> Value:
        h: list[Value | float] = list(x)
        for l, (W, b_) in enumerate(zip(self.W, self.b)):
            z = [sum((h[i] * W[i][j] for i in range(len(h))), b_[j]) for j in range(len(b_))]
            h = z if l == len(self.W) - 1 else [zj.relu() for zj in z]
        return h[0]  # type: ignore[return-value]


def scalar_loss(model: ScalarMLP, X: FloatArray, y: FloatArray) -> Value:
    """Mean logistic loss log(1 + exp(-y z)) over the batch, built from Value ops. X (n, 2), y (n,) -> Value."""
    total = Value(0.0)
    for xi, yi in zip(X, y):
        z = model([float(xi[0]), float(xi[1])])
        total = total + (1.0 + (-yi * z).exp()).log()
    return total * (1.0 / len(y))


smlp = ScalarMLP(init_W, init_b)
params = smlp.parameters()
print(f"parameters: {len(params)}")
probe = scalar_loss(smlp, X_tr, y_tr)
print(f"graph size for one loss: {len(topological_order(probe))} nodes")
t_start = time.perf_counter()
scalar_losses = []
for step in range(STEPS):
    for p in params:
        p.grad = 0.0  # gradients accumulate across backward() calls, so zero them every step
    loss = scalar_loss(smlp, X_tr, y_tr)
    loss.backward()
    for p in params:
        p.data -= LR * p.grad
    scalar_losses.append(loss.data)
t_scalar = time.perf_counter() - t_start
print(f"loss: step 0 {scalar_losses[0]:.4f}  step {STEPS // 2} {scalar_losses[STEPS // 2]:.4f}  step {STEPS - 1} {scalar_losses[-1]:.4f}")
print(f"~ scalar engine: {t_scalar:.1f} s for {STEPS} steps")

# %%
# The tensor engine: the same idea, but each node holds a NumPy array. The new problem is broadcasting:
# when a (d,) bias is added to an (n, d) matrix, its gradient must be summed back to shape (d,).
section("tensor engine: the Tensor class")


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


print("Tensor supports: + * @ sum mean reshape relu exp log, with broadcasting in + and *")

# %%
# Gradient-check every tensor op on random inputs, including the broadcast cases.
section("tensor engine: gradient check of every operation")


def tensor_gradcheck(f: Callable[..., Tensor], *shapes: tuple[int, ...], positive: bool = False) -> float:
    """Worst relative error between Tensor gradients and central differences for f(*inputs)."""
    gen = np.random.default_rng(1)
    arrays = [gen.uniform(0.5, 2.0, s) if positive else gen.normal(size=s) for s in shapes]
    out_shape = f(*[Tensor(a_) for a_ in arrays]).shape
    R = gen.normal(size=out_shape)  # random weights: loss = sum(R * f(...)) tests every output entry

    def loss_of(arrs: list[FloatArray]) -> float:
        return float(np.sum(R * f(*[Tensor(a_) for a_ in arrs]).data))

    ts = [Tensor(a_) for a_ in arrays]
    (f(*ts) * Tensor(R)).sum().backward()
    worst = 0.0
    for k, t in enumerate(ts):
        num = np.zeros_like(arrays[k])
        for idx in np.ndindex(arrays[k].shape):
            up = [a_.copy() for a_ in arrays]
            dn = [a_.copy() for a_ in arrays]
            up[k][idx] += 1e-6
            dn[k][idx] -= 1e-6
            num[idx] = (loss_of(up) - loss_of(dn)) / 2e-6
        assert t.grad.shape == arrays[k].shape, (t.grad.shape, arrays[k].shape)
        err = float(np.linalg.norm(t.grad - num) / max(np.linalg.norm(t.grad) + np.linalg.norm(num), 1e-12))
        worst = max(worst, err)
    return worst


checks: list[tuple[str, Callable[..., Tensor], tuple[tuple[int, ...], ...], bool]] = [
    ("add (3, 4) + (3, 4)", lambda p, q: p + q, ((3, 4), (3, 4)), False),
    ("add (3, 4) + (4,)      [broadcast]", lambda p, q: p + q, ((3, 4), (4,)), False),
    ("add (3, 1) + (1, 4)    [broadcast]", lambda p, q: p + q, ((3, 1), (1, 4)), False),
    ("add (3, 4) + scalar ()  [broadcast]", lambda p, q: p + q, ((3, 4), ()), False),
    ("mul (3, 4) * (3, 1)    [broadcast]", lambda p, q: p * q, ((3, 4), (3, 1)), False),
    ("mul (3, 4) * (4,)      [broadcast]", lambda p, q: p * q, ((3, 4), (4,)), False),
    ("matmul (3, 4) @ (4, 2)", lambda p, q: p @ q, ((3, 4), (4, 2)), False),
    ("sum all", lambda p: p.sum(), ((3, 4),), False),
    ("sum axis=0", lambda p: p.sum(axis=0), ((3, 4),), False),
    ("sum axis=1 keepdims", lambda p: p.sum(axis=1, keepdims=True), ((3, 4),), False),
    ("mean all", lambda p: p.mean(), ((3, 4),), False),
    ("mean axis=0", lambda p: p.mean(axis=0), ((3, 4),), False),
    ("reshape (3, 4) -> (2, 6)", lambda p: p.reshape(2, 6), ((3, 4),), False),
    ("relu", lambda p: p.relu(), ((3, 4),), False),
    ("exp", lambda p: p.exp(), ((3, 4),), False),
    ("log (positive inputs)", lambda p: p.log(), ((3, 4),), True),
    ("x used twice: x * x + x", lambda p: p * p + p, ((3, 4),), False),
    ("dense layer relu(X W + b)", lambda p, q, r: (p @ q + r).relu(), ((5, 3), (3, 4), (4,)), False),
]
all_pass = True
for label, fn, shapes, positive in checks:
    err = tensor_gradcheck(fn, *shapes, positive=positive)
    ok = err < 1e-6
    all_pass &= ok
    print(f"{label:<38} rel error {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}  {'PASS' if ok else 'FAIL'}")
print("all tensor ops pass:", "OK" if all_pass else "FAIL")

# %%
# Worked example by hand: a broadcast bias. L = sum((X + b)^2) with X (2, 2) and b (2,).
section("worked example: a broadcast bias")
Xh = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
bh = Tensor(np.array([10.0, 20.0]))
Zh = Xh + bh  # (2, 2): b is added to both rows
Lh = (Zh * Zh).sum()
Lh.backward()
print(f"Z = X + b = {show(Zh.data)}   L = {float(Lh.data)}")
print(f"dL/dZ = 2 Z = {show(Zh.grad)}")
print(f"dL/dX = {show(Xh.grad)}  (same shape as X)")
print(f"dL/db = column sums of dL/dZ = {show(bh.grad)}  (shape {bh.grad.shape}, same as b)")

# %%
# What goes wrong without unbroadcast: the bias gradient comes back with the batch's shape.
section("without unbroadcast")
Xb, bb = Tensor(rng.normal(size=(5, 4))), Tensor(np.zeros(4))
(Xb + bb).sum().backward()
print(f"bias shape {bb.shape}, correct grad {show(bb.grad)} (each bias entry was used 5 times)")
raw = np.ones((5, 4))  # what `other.grad += out.grad` would try to add
try:
    wrong = np.zeros(4)
    wrong += raw
except ValueError as exc:
    print(f"adding the raw (5, 4) gradient to a (4,) bias fails: {exc}")
wrong_keep = np.zeros((1, 4))
wrong_keep = wrong_keep + raw  # with `=` instead of `+=` it silently becomes (5, 4)
print(f"with b of shape (1, 4) and `b.grad = b.grad + g`, b.grad silently becomes shape {wrong_keep.shape}")

# %%
# Train the same 2-8-8-1 MLP from the same initial weights with the tensor engine.
section("tensor engine: the same MLP, same start")


def tensor_loss(Ws: list[Tensor], bs: list[Tensor], X: FloatArray, y: FloatArray) -> Tensor:
    """Mean logistic loss of the ReLU MLP. X (n, 2), y (n,) -> scalar Tensor."""
    H = Tensor(X)
    for l, (W, b_) in enumerate(zip(Ws, bs)):
        H = H @ W + b_  # (n, d_out); the (d_out,) bias broadcasts over rows
        if l < len(Ws) - 1:
            H = H.relu()
    z = H.reshape(-1)  # (n,)
    return ((Tensor(-y) * z).exp() + 1.0).log().mean()


def predict(Ws: list[Tensor], bs: list[Tensor], X: FloatArray) -> FloatArray:
    """Class predictions in {-1, +1}. X (n, 2) -> (n,)."""
    H = X
    for l, (W, b_) in enumerate(zip(Ws, bs)):
        H = H @ W.data + b_.data
        if l < len(Ws) - 1:
            H = np.maximum(H, 0)
    return np.where(H[:, 0] > 0, 1.0, -1.0)


tW = [Tensor(W.copy()) for W in init_W]
tb = [Tensor(b_.copy()) for b_ in init_b]
t_start = time.perf_counter()
tensor_losses = []
for step in range(TENSOR_STEPS):
    if step == STEPS:
        t_tensor = time.perf_counter() - t_start  # time for the same number of steps as the scalar engine
        acc_at_steps = (np.mean(predict(tW, tb, X_tr) == y_tr), np.mean(predict(tW, tb, X_te) == y_te))
        W_at_steps = [W.data.copy() for W in tW]
    for p in tW + tb:
        p.grad = np.zeros_like(p.data)
    loss_t = tensor_loss(tW, tb, X_tr, y_tr)
    loss_t.backward()
    for p in tW + tb:
        p.data -= LR * p.grad
    tensor_losses.append(float(loss_t.data))
gap = float(np.max(np.abs(np.array(scalar_losses) - np.array(tensor_losses[:STEPS]))))
print(f"loss: step 0 {tensor_losses[0]:.4f}  step {STEPS // 2} {tensor_losses[STEPS // 2]:.4f}  step {STEPS - 1} {tensor_losses[STEPS - 1]:.4f}")
print(f"max |scalar loss - tensor loss| over {STEPS} steps: {'< 1e-10' if gap < 1e-10 else f'{gap:.1e}'}")
smlp_W = [np.array([[w.data for w in row] for row in W]) for W in smlp.W]
w_gap = max(float(np.max(np.abs(sw - tw))) for sw, tw in zip(smlp_W, W_at_steps))
print(f"max |scalar weights - tensor weights| after {STEPS} steps: {'< 1e-10' if w_gap < 1e-10 else f'{w_gap:.1e}'}")
print(f"~ tensor engine: {t_tensor:.3f} s for {STEPS} steps ({t_scalar / t_tensor:.0f}x faster)")
print(f"after {STEPS} steps:  train accuracy {acc_at_steps[0]:.4f}   test accuracy {acc_at_steps[1]:.4f}")
print(f"after {TENSOR_STEPS} steps: loss {tensor_losses[-1]:.4f}  train accuracy {np.mean(predict(tW, tb, X_tr) == y_tr):.4f}   test accuracy {np.mean(predict(tW, tb, X_te) == y_te):.4f}")





# %%
# With PyTorch: the same MLP, weights and batch in float64; compare every gradient with ours.
section("with PyTorch")
pW = [torch.tensor(W, dtype=torch.float64, requires_grad=True) for W in init_W]
pb = [torch.tensor(b_, dtype=torch.float64, requires_grad=True) for b_ in init_b]
Hp = torch.tensor(X_tr, dtype=torch.float64)
for l, (W, b_) in enumerate(zip(pW, pb)):
    Hp = Hp @ W + b_
    if l < len(pW) - 1:
        Hp = torch.relu(Hp)
loss_p = torch.nn.functional.softplus(-torch.tensor(y_tr) * Hp[:, 0]).mean()  # log(1 + e^(-y z))
loss_p.backward()
gW = [Tensor(W.copy()) for W in init_W]
gb = [Tensor(b_.copy()) for b_ in init_b]
tensor_loss(gW, gb, X_tr, y_tr).backward()
worst = max(
    max(float(np.max(np.abs(p.grad.numpy() - q.grad))) for p, q in zip(pW, gW)),
    max(float(np.max(np.abs(p.grad.numpy() - q.grad))) for p, q in zip(pb, gb)),
)
print(f"initial loss: torch {loss_p.item():.6f}   ours {tensor_losses[0]:.6f}")
print(f"max |torch grad - our grad| over all 6 parameter arrays: {'< 1e-12' if worst < 1e-12 else f'{worst:.1e}'}")
for p, q in zip(pW + pb, gW + gb):
    np.testing.assert_allclose(p.grad.numpy(), q.grad, rtol=1e-6, atol=1e-10)
print("matches PyTorch autograd: OK")
xs = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
(xs * xs).backward()
print(f"PyTorch, y = x * x at x = 3: dy/dx = {xs.grad.item()} (it accumulates too)")

# %%
# Key plot: the two engines' loss curves coincide; the decision boundary the MLP learned.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
axes[0].plot(scalar_losses, "-", lw=4, alpha=0.5, label=f"scalar engine (Value), {STEPS} steps")
axes[0].plot(tensor_losses, "k--", lw=1.2, label="tensor engine (Tensor)")
axes[0].set(title="Same start: the curves coincide; Tensor runs on", xlabel="gradient-descent step",
            ylabel="mean logistic loss (log scale)", yscale="log")
axes[0].legend(loc="best")
g1, g2 = np.meshgrid(np.linspace(-1.5, 2.5, 300), np.linspace(-1.0, 1.5, 300))
region = predict(tW, tb, np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
axes[1].contourf(g1, g2, region, levels=[-2, 0, 2], colors=["tab:blue", "tab:orange"], alpha=0.2)
axes[1].scatter(*X_tr[y_tr > 0].T, marker="^", c="tab:orange", edgecolor="k", label="y = +1")
axes[1].scatter(*X_tr[y_tr < 0].T, marker="o", c="tab:blue", edgecolor="k", label="y = -1")
test_acc = np.mean(predict(tW, tb, X_te) == y_te)
axes[1].set(title=f"2-8-8-1 ReLU MLP: curved boundary, test accuracy {test_acc:.1%}", xlabel="x1", ylabel="x2")
axes[1].legend(loc="lower left")
fig.tight_layout()
key_path = IMG / "c1_3_engines_moons.png"
fig.savefig(key_path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {key_path.relative_to(ROOT).as_posix()}")

# %%
# Try this (not graded):
# - Remove the `p.grad = 0.0` line in the scalar training loop: gradients pile up across steps and the
#   loss soon explodes. PyTorch behaves the same way, which is why every training loop calls zero_grad().
# - Replace the logistic loss with log(1 + exp(-y z)) on logits scaled by 1000 (multiply z by 1000):
#   exp overflows to inf and the loss becomes inf or nan. C1.4 fixes this with a stable fused op.
# - Add a `tanh` op to Tensor, gradient-check it, and train with tanh hidden units instead of ReLU.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
