# %% [markdown]
# # A4.5 Convexity and Lagrange multipliers — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_5_convexity_lagrange_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
for c in (10.0, 10.1):
    x = y = c / 2
    print(f"max xy s.t. x + y = {c}: x = y = {x}  lambda = {x}  f* = {x * y:.4f}")
print(f"(f*(10.1) - f*(10)) / 0.1 = {(10.1**2 / 4 - 25) / 0.1:.4f}  -> lambda = 5 (exactly df*/dc = c / 2 = 5 at c = 10)")
for name, H in (("x^2 + xy + 2y^2", np.array([[2.0, 1.0], [1.0, 4.0]])), ("x^2 - y^2", np.array([[2.0, 0.0], [0.0, -2.0]]))):
    print(f"{name:<16} Hessian eigenvalues = {show(np.linalg.eigvalsh(H))}")
for c in (1.0, 5.0):
    res = minimize(lambda v: (v[0] - 3.0) ** 2, x0=np.array([0.0]), method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda v, c=c: c - v[0]}])
    print(f"min (x - 3)^2 s.t. x <= {c}: x* = {res.x[0]:.4f}  mu = {res.multipliers[0]:.4f}")

# %%
section("exercise 2: minimum-norm solution")
n, d = 3, 6
X = rng.normal(size=(n, d))  # (n, d), fewer equations than unknowns
y = rng.normal(size=n)  # (n,)


def min_norm_solution(X_: FloatArray, y_: FloatArray) -> FloatArray:
    """Minimize ||w||^2 s.t. X w = y via Lagrange: w = X^T (X X^T)^-1 y. X (n, d), y (n,) -> (d,)."""
    nu_half = np.linalg.solve(X_ @ X_.T, y_)  # (n,): solves (X X^T) (nu / 2) = y, never forming an inverse
    return X_.T @ nu_half  # (d,): a combination of the rows of X


w = min_norm_solution(X, y)
print("w =", show(w), f"  ||w|| = {np.linalg.norm(w):.4f}   X w - y = {show(X @ w - y)}")
w_pinv = np.linalg.pinv(X) @ y
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
res = minimize(lambda v: v @ v, x0=np.zeros(d), jac=lambda v: 2 * v, method="SLSQP",
               constraints=[{"type": "eq", "fun": lambda v: X @ v - y, "jac": lambda v: X}])
np.testing.assert_allclose(w, w_pinv, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(w, w_lstsq, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(w, res.x, rtol=1e-5, atol=1e-6)  # SLSQP converges to ~1e-7, hence the looser check
print("matches pinv, lstsq and SLSQP: OK")

_, _, Vt = np.linalg.svd(X)  # full SVD: the last d - n rows of Vt span the null space of X
z = Vt[n]  # (d,), unit vector with X z = 0
w2 = w + 0.5 * z
print(f"another solution w + 0.5 z: X w2 - y = {show(X @ w2 - y)}   ||w2|| = {np.linalg.norm(w2):.4f} > ||w|| = {np.linalg.norm(w):.4f}")
print(f"|w . z| < 1e-12 (orthogonal): {abs(w @ z) < 1e-12}   ||w2||^2 = ||w||^2 + 0.25 ||z||^2: "
      f"{np.isclose(w2 @ w2, w @ w + 0.25)}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
