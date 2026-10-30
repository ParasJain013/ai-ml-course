# %% [markdown]
# # A4.3 The chain rule as a computational graph — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_3_chain_rule_graphs_ex2.py   (CPU, ~5 s, mostly importing PyTorch)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# from labs/A/a4_2_gradients_jacobians.py
def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-5) -> FloatArray:
    """Central-difference gradient of a scalar f at x of any shape. Returns the same shape as x."""
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * h)
    return grad


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Largest relative error max |a - b| / max(|a|, |b|, 1e-12), elementwise."""
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)))


def sigmoid(z: float) -> float:
    """Logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


# %%
section("exercise 1 check")
x, y, z = 1.0, 2.0, 4.0
q1, q2 = x + y, y - z
print(f"(a) q1 = x + y = {q1}  q2 = y - z = {q2}  f = {q1 * q2}")
print(f"    df/dx = q2 = {q2}  df/dy = q2 + q1 = {q2 + q1}  df/dz = -q1 = {-q1}")
print("    numerical:", show(numerical_gradient(lambda v: (v[0] + v[1]) * (v[1] - v[2]), np.array([x, y, z]))))
xv, wv, bv, yv = np.array([2.0, 1.0]), np.array([1.0, -1.0]), -1.0, 0.0
zz = wv @ xv + bv
p = sigmoid(zz)
dz = 2 * (p - yv) * p * (1 - p)
print(f"(b) z = {zz}  p = {p}  loss = {(p - yv) ** 2}  dl/dz = {dz}  dl/dw = {show(dz * xv)}  dl/db = {dz}")
w_new, b_new = wv - 1.0 * dz * xv, bv - 1.0 * dz
z_new = w_new @ xv + b_new
print(f"    after one step (lr = 1): w = {show(w_new)}  b = {b_new}  z = {z_new}  p = {sigmoid(z_new):.5f}  loss = {(sigmoid(z_new) - yv) ** 2:.5f}")
ts = np.array([-2.0, 0.0, 3.0])
softplus_slope = np.array([(np.log1p(np.exp(t + 1e-5)) - np.log1p(np.exp(t - 1e-5))) / 2e-5 for t in ts])
print("(c) numerical softplus' at [-2, 0, 3] =", show(softplus_slope), "  sigmoid =", show(1 / (1 + np.exp(-ts))))

# %%
section("exercise 2: a one-hidden-unit network")
X = np.linspace(-2.0, 2.0, 8)  # (n,)
Y = np.sin(X)  # (n,)


def forward(params: FloatArray, x_: FloatArray, y_: FloatArray) -> tuple[float, dict[str, FloatArray]]:
    """Loss of y_hat = w2 tanh(w1 x + b1) + b2. params = (w1, b1, w2, b2); x, y (n,)."""
    w1, b1, w2, b2 = params
    u = w1 * x_ + b1  # (n,)
    h = np.tanh(u)  # (n,)
    y_hat = w2 * h + b2  # (n,)
    loss = float(np.mean((y_hat - y_) ** 2))
    return loss, {"x": x_, "y": y_, "h": h, "y_hat": y_hat, "w2": np.array(w2)}


def backward(cache: dict[str, FloatArray]) -> FloatArray:
    """Gradients (dw1, db1, dw2, db2) of the mean squared error."""
    x_, y_, h, y_hat, w2 = cache["x"], cache["y"], cache["h"], cache["y_hat"], float(cache["w2"])
    n = len(y_)
    dy_hat = 2.0 * (y_hat - y_) / n  # (n,)
    dw2 = float(np.sum(dy_hat * h))  # y_hat = w2 * h: local derivative h; w2 is shared by all n -> sum
    db2 = float(np.sum(dy_hat))  # broadcast scalar -> sum
    dh = dy_hat * w2  # (n,)
    du = dh * (1.0 - h**2)  # (n,): tanh'(u) = 1 - tanh(u)^2, reusing the forward value h
    dw1 = float(np.sum(du * x_))
    db1 = float(np.sum(du))
    return np.array([dw1, db1, dw2, db2])


def torch_grads(params: FloatArray) -> FloatArray:
    """The same gradients from PyTorch autograd, in float64."""
    t = torch.tensor(params, requires_grad=True)
    xt, yt = torch.tensor(X), torch.tensor(Y)
    loss = torch.mean((t[2] * torch.tanh(t[0] * xt + t[1]) + t[3] - yt) ** 2)
    loss.backward()
    return t.grad.numpy()


for params in (np.array([0.5, 0.1, -0.3, 0.2]), np.array([10.0, 0.1, -0.3, 0.2])):
    loss, cache = forward(params, X, Y)
    g = backward(cache)
    g_num = numerical_gradient(lambda p_: forward(p_, X, Y)[0], params.copy())
    print(f"params (w1, b1, w2, b2) = {show(params)}  loss = {loss:.6f}")
    print(f"  backward (dw1, db1, dw2, db2) = {show(g)}")
    print(f"  rel error vs numerical {rel_error(g, g_num):.1e}   vs torch {rel_error(g, torch_grads(params)):.1e}")
    print(f"  tanh'(u) = 1 - h^2 at the 8 points: {show(1 - cache['h'] ** 2)}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
