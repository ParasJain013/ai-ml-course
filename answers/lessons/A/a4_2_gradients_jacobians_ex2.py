# %% [markdown]
# # A4.2 Gradients and Jacobians — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_2_gradients_jacobians_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge

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


# %%
section("exercise 1 check")
x = np.array([3.0, 1.0])
grad_a = np.array([2 * (x[0] - 2 * x[1]) + x[1], -4 * (x[0] - 2 * x[1]) + x[0]])
print("(a) gradient at (3, 1) =", show(grad_a))
X_s = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y_s = np.array([1.0, 2.0, 2.0])
w_s = np.array([0.0, 1.0])
lam = 0.5
grad_b = 2 / 3 * X_s.T @ (X_s @ w_s - y_s) + 2 * lam * w_s
print("(b) ridge gradient =", show(grad_b))
xc = np.array([1.0, 2.0, 3.0])
J = np.array([[xc[2], 0.0, xc[0]], [-1.0, 2 * xc[1], 0.0]])
print("(c) Jacobian =", show(J), " shape", J.shape)

# %%
section("exercise 2: multi-output ridge gradient")


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


def loss(W: FloatArray, X: FloatArray, Y: FloatArray, lam: float) -> float:
    """(1/n) |XW - Y|_F^2 + lam |W|_F^2. X (n, d), W (d, k), Y (n, k) -> scalar."""
    R = X @ W - Y  # (n, k)
    return float(np.sum(R * R) / X.shape[0] + lam * np.sum(W * W))


def loss_grad(W: FloatArray, X: FloatArray, Y: FloatArray, lam: float) -> FloatArray:
    """Gradient X^T G + 2 lam W with G = (2/n)(XW - Y). -> (d, k), the shape of W."""
    G = 2.0 / X.shape[0] * (X @ W - Y)  # (n, k): dL/dZ for Z = XW
    return X.T @ G + 2.0 * lam * W  # (d, n) @ (n, k) -> (d, k)


def wrong_grad(W: FloatArray, X: FloatArray, Y: FloatArray, lam: float) -> FloatArray:
    """The transposed bug: G^T X has shape (k, d)."""
    G = 2.0 / X.shape[0] * (X @ W - Y)
    return G.T @ X + 2.0 * lam * W.T


LAM = 0.1
for n, d, k in ((8, 3, 2), (8, 3, 3)):
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=(n, k))
    W = rng.normal(size=(d, k))
    num = numerical_gradient(lambda V: loss(V, X, Y, LAM), W)
    ok = loss_grad(W, X, Y, LAM)
    print(f"n={n} d={d} k={k}: X^T G + 2 lam W   shape {ok.shape}  rel error < 1e-8: {rel_error(ok, num) < 1e-8}")
    if d == k:
        bad = wrong_grad(W, X, Y, LAM)
        assert bad.shape == W.shape  # the shape check passes...
        print(f"n={n} d={d} k={k}: transposed formula shape {bad.shape}  rel error {rel_error(bad, num):.2f}"
              "  (shape matches, gradient wrong)")

# %%
section("exercise 2: closed form versus scikit-learn")
n, d, k = 8, 3, 2
X = rng.normal(size=(n, d))
Y = rng.normal(size=(n, k))
# (2/n) X^T X W - (2/n) X^T Y + 2 lam W = 0  ->  (X^T X + n lam I) W = X^T Y
W_star = np.linalg.solve(X.T @ X + n * LAM * np.eye(d), X.T @ Y)  # (d, k)
print("gradient at the solution is zero:", np.allclose(loss_grad(W_star, X, Y, LAM), 0.0, atol=1e-12))
ridge = Ridge(alpha=n * LAM, fit_intercept=False).fit(X, Y)  # sklearn: |Y - XW|^2 + alpha |W|^2
np.testing.assert_allclose(ridge.coef_.T, W_star, rtol=1e-6, atol=1e-8)  # coef_ is (k, d)
print(f"matches sklearn Ridge(alpha = n * lam = {n * LAM:g}, fit_intercept=False): OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
