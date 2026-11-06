# %% [markdown]
# # C1.1 Perceptron and MLP — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/C/c1_1_perceptron_mlp_ex2.py   (CPU, ~1 s)

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


def relu(z: FloatArray) -> FloatArray:
    """max(0, z), elementwise."""
    return np.maximum(z, 0.0)


def step(z: FloatArray) -> FloatArray:
    """1 where z > 0, else 0, elementwise."""
    return (z > 0).astype(np.float64)


# %%
section("exercise 1 check")
# (a) the perceptron rule, eta = 1, from zero
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, -1.0, 1.0])
w, b = np.zeros(2), 0.0
for epoch in range(1, 10):
    mistakes = 0
    for i in range(3):
        z = float(X[i] @ w + b)
        if y[i] * z <= 0:
            w, b = w + y[i] * X[i], b + y[i]
            mistakes += 1
            print(f"(a) epoch {epoch} point {i + 1}: z={z:+.0f} wrong -> w={show(w)} b={b:+.0f}")
        else:
            print(f"(a) epoch {epoch} point {i + 1}: z={z:+.0f} correct")
    if mistakes == 0:
        print(f"(a) converged: w={show(w)} b={b:+.0f}")
        break
# (b) XNOR from OR and AND hidden units
W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([-0.5, -1.5])
W2 = np.array([[-1.0], [1.0]])  # -OR + AND
b2 = np.array([0.5])
X4 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
out = step(step(X4 @ W1 + b1) @ W2 + b2)[:, 0]
print(f"(b) XNOR outputs {out.tolist()}  expected [1.0, 0.0, 0.0, 1.0]")
# (c) two linear layers are one
A1 = np.array([[2.0, -1.0], [1.0, 1.0]])
A2 = np.array([[1.0], [2.0]])
x = np.array([1.0, -1.0])
print(f"(c) W1 W2 = {show(A1 @ A2)}  layer by layer {show(x @ A1 @ A2)}  one matrix {show(x @ (A1 @ A2))}")
print(f"(c) with ReLU: hidden {show(relu(x @ A1))}  output {show(relu(x @ A1) @ A2)}")

# %%
section("mlp_forward")


def mlp_forward(
    X: FloatArray,
    Ws: list[FloatArray],
    bs: list[FloatArray],
    activation: Callable[[FloatArray], FloatArray] = relu,
) -> FloatArray:
    """MLP forward pass, no activation on the last layer. X (n, d_0), Ws[l] (d_l, d_{l+1}) -> (n, d_L)."""
    if len(Ws) != len(bs):
        raise ValueError(f"{len(Ws)} weight matrices but {len(bs)} bias vectors")
    H = X
    for l, (W, b) in enumerate(zip(Ws, bs)):
        if H.shape[1] != W.shape[0]:
            raise ValueError(f"layer {l + 1}: input width {H.shape[1]} but W has {W.shape[0]} rows")
        Z = H @ W + b  # (n, d_{l+1})
        H = Z if l == len(Ws) - 1 else activation(Z)
    return H


Xs = np.random.default_rng(0).normal(size=(5, 3))
Ws = [np.ones((3, 4)), np.ones((4, 2))]
bs = [np.zeros(4), np.zeros(2)]
print("output shape for (5, 3) through 3 -> 4 -> 2:", mlp_forward(Xs, Ws, bs).shape)

# %%
section("|x| with two ReLUs")
# |x| = relu(x) + relu(-x)
W_abs = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]
b_abs = [np.zeros(2), np.zeros(1)]
xs = np.linspace(-3, 3, 13)[:, None]  # (13, 1)
err = np.max(np.abs(mlp_forward(xs, W_abs, b_abs)[:, 0] - np.abs(xs[:, 0])))
print(f"max | net(x) - |x| | on [-3, 3]: {err:.1e}")

# %%
section("piecewise-linear sin with k hidden ReLUs")


def interpolating_net(f: Callable[[FloatArray], FloatArray], lo: float, hi: float, k: int) -> tuple[list[FloatArray], list[FloatArray]]:
    """One-hidden-layer ReLU net that linearly interpolates f at k + 1 equal knots on [lo, hi]."""
    knots = np.linspace(lo, hi, k + 1)  # (k + 1,)
    vals = f(knots)
    slopes = np.diff(vals) / np.diff(knots)  # (k,) slope on each segment
    c = np.concatenate([[slopes[0]], np.diff(slopes)])  # (k,) change of slope at each knot
    W1 = np.ones((1, k))  # every unit reads x
    b1 = -knots[:-1]  # unit j is relu(x - t_j)
    W2 = c[:, None]  # (k, 1)
    b2 = np.array([vals[0]])  # f(t_0): on [t_0, t_1] only unit 0 is on, giving f(t_0) + slope_0 (x - t_0)
    return [W1, W2], [b1, b2]


grid = np.linspace(-np.pi, np.pi, 100_001)[:, None]
prev = None
for k in [4, 8, 16, 32]:
    Wk, bk = interpolating_net(np.sin, -np.pi, np.pi, k)
    e = float(np.max(np.abs(mlp_forward(grid, Wk, bk)[:, 0] - np.sin(grid[:, 0]))))
    bound = (2 * np.pi / k) ** 2 / 8
    ratio = "" if prev is None else f"  error ratio to k/2: {prev / e:.2f}"
    print(f"k = {k:>2} hidden units: max error {e:.5f}  (bound h^2/8 = {bound:.5f}){ratio}")
    prev = e

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
