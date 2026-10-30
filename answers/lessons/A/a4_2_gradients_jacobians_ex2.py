# %% [markdown]
# # A4.2 Gradients and Jacobians — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_2_gradients_jacobians_ex2.py   (CPU, ~1 s)

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


# %%
section("exercise 1 check")
X4 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])  # (4, 2)
y4 = np.array([1.0, 3.0, 4.0, 4.0])
lam, n = 0.5, 4


def ridge(w: FloatArray) -> float:
    """(1/n) ||X w - y||^2 + lam ||w||^2. w (2,) -> scalar."""
    r = X4 @ w - y4
    return float(r @ r) / n + lam * float(w @ w)


w = np.array([1.0, 1.0])
grad = 2 / n * X4.T @ (X4 @ w - y4) + 2 * lam * w  # (2,)
print("residual at w = (1, 1):", show(X4 @ w - y4), "  gradient =", show(grad),
      "  numerical =", show(numerical_gradient(ridge, w.copy())))
lhs = X4.T @ X4 + n * lam * np.eye(2)
w_ridge = np.linalg.solve(lhs, X4.T @ y4)
print("(X^T X + n lam I) =", show(lhs), "  X^T y =", show(X4.T @ y4), "  w_ridge =", show(w_ridge))
print("unpenalized w =", show(np.linalg.solve(X4.T @ X4, X4.T @ y4)),
      "  eigenvalues of X^T X =", show(np.linalg.eigvalsh(X4.T @ X4)))
x0 = np.array([0.0, 1.0])
J = np.array([[x0[1] ** 2, 2 * x0[0] * x0[1]], [np.exp(x0[0]), -1.0]])
print("Jacobian of (x1 x2^2, e^x1 - x2) at (0, 1) =", show(J), " shape", J.shape)

# %%
section("exercise 2a: bias by broadcasting")
n, d, k = 5, 3, 2
X = rng.normal(size=(n, d))  # (n, d)
Y = rng.normal(size=(n, k))  # (n, k)
W = rng.normal(size=(d, k))  # (d, k)
b = rng.normal(size=k)  # (k,), added to every row


def loss_dense(W_: FloatArray, b_: FloatArray) -> float:
    """(1/n) ||X W + b - Y||_F^2. W (d, k), b (k,) -> scalar."""
    R = X @ W_ + b_ - Y  # (n, k); b_ broadcasts over the n rows
    return float(np.sum(R * R)) / n


R = X @ W + b - Y  # (n, k)
dW = 2.0 / n * X.T @ R  # (d, k)
db = 2.0 / n * R.sum(axis=0)  # (k,): b was copied to every row, so its gradient sums over rows
assert dW.shape == W.shape and db.shape == b.shape
dW_num = numerical_gradient(lambda W_: loss_dense(W_, b), W.copy())
db_num = numerical_gradient(lambda b_: loss_dense(W, b_), b.copy())
print(f"dW shape {dW.shape}  rel error {rel_error(dW, dW_num):.1e}")
print(f"db shape {db.shape}  rel error {rel_error(db, db_num):.1e}")

# %%
section("exercise 2b: the (n, 1) broadcasting bug")
w = rng.normal(size=d)  # (d,)
y = rng.normal(size=n)  # (n,)
y_col = y.reshape(-1, 1)  # (n, 1), e.g. df[["target"]].to_numpy()
r_bad = X @ w - y_col  # (n,) - (n, 1) broadcasts to (n, n)!
g_bad = 2.0 / n * X.T @ r_bad  # (d, n): wrong shape, no error raised
g_num = numerical_gradient(lambda w_: float(np.sum((X @ w_ - y) ** 2)) / n, w.copy())
print("residual shape", r_bad.shape, "  gradient shape", g_bad.shape, "  expected", w.shape)
print(f"rel error of the mean over the extra axis vs the correct gradient: {rel_error(g_bad.mean(axis=1), g_num):.1e}")
r_ok = X @ w - y_col.ravel()  # (n,)
print("with y.ravel(): residual shape", r_ok.shape, f"  rel error {rel_error(2.0 / n * X.T @ r_ok, g_num):.1e}")
try:
    assert r_bad.shape == (n,), f"residual has shape {r_bad.shape}, expected {(n,)}"
except AssertionError as e:
    print("the assertion that catches it: AssertionError:", e)

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
