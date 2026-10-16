# %% [markdown]
# # A4.3 The chain rule as a computational graph — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_3_chain_rule_graphs_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
H = 1e-6


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = H) -> FloatArray:
    """Central-difference gradient of a scalar function of an array of any shape. x (...) -> (...)."""
    # from labs/A/a4_2_gradients_jacobians.py
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Relative error |a - b| / (|a| + |b|), safe when both are zero."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


def sigmoid(z: FloatArray | float) -> FloatArray | float:
    """Logistic sigmoid, elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


# %%
section("exercise 1 check")
x1, w1, b1, y1 = np.array([1.0, 2.0]), np.array([0.5, -0.25]), 0.0, 1.0
p1 = float(sigmoid(x1 @ w1 + b1))
dz = p1 - y1
print(f"(a) p = {p1}   dL/dz = p - y = {dz}   dL/dw = {show(dz * x1)}   dL/db = {dz}")


def log_loss(params: FloatArray) -> float:
    """Log loss of the single neuron; params = (w1, w2, b)."""
    p = float(sigmoid(x1 @ params[:2] + params[2]))
    return -(y1 * np.log(p) + (1 - y1) * np.log(1 - p))


print("    numerical (dL/dw1, dL/dw2, dL/db) =", show(numerical_gradient(log_loss, np.array([*w1, b1]))))
for p in (0.5, 0.1, 0.001):
    print(f"    p = {p:<6} y = 1: log loss dL/dz = {p - 1:.4f}   squared error dL/dz = {2 * (p - 1) * p * (1 - p):.4f}")
x, y, z = 1.0, 2.0, 3.0
a, c = x + y, y + z
print(f"(b) a = x + y = {a}  c = y + z = {c}  h = {a * c}   dh/dx = {c}  dh/dy = {c} + {a} = {c + a}  dh/dz = {a}")

# %%
section("exercise 2: gradient check of the two-layer network")


def forward(params: dict[str, FloatArray], X: FloatArray, y: FloatArray) -> tuple[float, dict[str, FloatArray]]:
    """Forward pass. X (n, 2), y (n,) -> (loss, cache)."""
    A = X @ params["W1"] + params["b1"]  # (n, 3) pre-activations; b1 (3,) broadcasts over rows
    Hid = np.tanh(A)  # (n, 3)
    y_hat = Hid @ params["w2"] + params["b2"]  # (n,)
    loss = float(np.mean((y_hat - y) ** 2))
    return loss, {"X": X, "H": Hid, "y_hat": y_hat, "y": y}


def backward(params: dict[str, FloatArray], cache: dict[str, FloatArray]) -> dict[str, FloatArray]:
    """Backward pass. -> gradients with the same keys and shapes as params."""
    X, Hid, y_hat, y = cache["X"], cache["H"], cache["y_hat"], cache["y"]
    n = len(y)
    d_yhat = 2.0 / n * (y_hat - y)  # (n,)
    d_w2 = Hid.T @ d_yhat  # (3,)   y_hat = H w2: X^T G pattern with H as the input
    d_b2 = np.array(d_yhat.sum())  # ()     b2 is added to every row: sum over rows
    d_H = np.outer(d_yhat, params["w2"])  # (n, 3) the other matmul rule: G W^T
    d_A = d_H * (1.0 - Hid**2)  # (n, 3) tanh' = 1 - tanh^2, elementwise
    d_W1 = X.T @ d_A  # (2, 3)
    d_b1 = d_A.sum(axis=0)  # (3,)   broadcast over rows: sum over rows
    return {"W1": d_W1, "b1": d_b1, "w2": d_w2, "b2": d_b2}


def init(rng: np.random.Generator) -> dict[str, FloatArray]:
    """Small random parameters."""
    return {"W1": rng.normal(size=(2, 3)), "b1": rng.normal(size=3) * 0.1,
            "w2": rng.normal(size=3), "b2": np.array(0.0)}


n = 6
X = rng.normal(size=(n, 2))
y_t = rng.normal(size=n)
params = init(rng)
_, cache = forward(params, X, y_t)
grads = backward(params, cache)
for key in params:
    assert grads[key].shape == params[key].shape, key

    def f_key(v: FloatArray, key: str = key) -> float:
        return forward({**params, key: v}, X, y_t)[0]

    num = numerical_gradient(f_key, params[key].copy())
    print(f"{key:<3} shape {str(params[key].shape):<7} rel error < 1e-8: {rel_error(grads[key], num) < 1e-8}")

# %%
section("exercise 2: training with these gradients")
n = 64
X = rng.uniform(-2, 2, size=(n, 2))
y_t = np.sin(X[:, 0]) + 0.5 * X[:, 1]
params = init(rng)
for step in range(201):
    loss, cache = forward(params, X, y_t)
    if step % 50 == 0:
        print(f"step {step:>3}   loss = {loss:.4f}")
    grads = backward(params, cache)
    params = {k: params[k] - 0.1 * grads[k] for k in params}

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
