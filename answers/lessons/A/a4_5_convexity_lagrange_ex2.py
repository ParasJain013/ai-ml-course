# %% [markdown]
# # A4.5 Convexity and Lagrange multipliers — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_5_convexity_lagrange_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy import optimize
from sklearn.linear_model import Ridge

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
r = optimize.minimize(lambda v: v[0] ** 2 + 2 * v[1] ** 2, np.zeros(2), method="SLSQP",
                      constraints=[{"type": "eq", "fun": lambda v: v[0] + v[1] - 3.0}])
print(f"(a) SLSQP: (x, y) = {show(r.x)}   f* = {r.fun:.4f}   multiplier = {r.multipliers[0]:.4f}")
for c in (3.0, 3.01):
    print(f"    c = {c}: f*(c) = 2c^2/3 = {2 * c * c / 3:.5f}")
print(f"    (f*(3.01) - f*(3)) / 0.01 = {(2 * 3.01**2 / 3 - 6) / 0.01:.4f}")


def chord_share(f, lo: float, hi: float, n: int = 20_000) -> float:
    """Share of random chords on [lo, hi] that dip below f (0 = no counterexample)."""
    gen = np.random.default_rng(SEED)
    x1, x2, t = gen.uniform(lo, hi, n), gen.uniform(lo, hi, n), gen.uniform(0, 1, n)
    return float(np.mean(f(t * x1 + (1 - t) * x2) > t * f(x1) + (1 - t) * f(x2) + 1e-12))


candidates = {
    "exp(-x)": (lambda x: np.exp(-x), -3, 3),
    "x^3": (lambda x: x**3, -3, 3),
    "max(0, x)": (lambda x: np.maximum(0, x), -3, 3),
    "-log(x), x > 0": (lambda x: -np.log(x), 0.01, 5),
    "x^4 - x^2": (lambda x: x**4 - x**2, -3, 3),
    "|x - 1| + x^2": (lambda x: np.abs(x - 1) + x**2, -3, 3),
}
for name, (f, lo, hi) in candidates.items():
    print(f"(b) {name:<16} chords below the curve: {chord_share(f, lo, hi):6.1%}")

# %%
section("exercise 2: data and the unconstrained fit")
N, D = 60, 3
X = rng.normal(size=(N, D))  # (n, d)
w_true = np.array([3.0, -2.0, 1.0])
y = X @ w_true + rng.normal(0.0, 0.5, size=N)  # (n,)


def loss(w: FloatArray) -> float:
    """(1/n) |X w - y|^2."""
    r_ = X @ w - y
    return float(r_ @ r_ / N)


def grad(w: FloatArray) -> FloatArray:
    """(2/n) X^T (X w - y). -> (d,)."""
    return 2.0 / N * X.T @ (X @ w - y)


w_ls = np.linalg.solve(X.T @ X, X.T @ y)
radius = 0.5 * np.linalg.norm(w_ls)
print(f"least squares w = {show(w_ls)}   |w| = {np.linalg.norm(w_ls):.4f}   radius r = {radius:.4f}")

# %%
section("exercise 2: projected gradient descent")


def project(w: FloatArray, r: float) -> FloatArray:
    """Closest point to w in the ball |w| <= r: shrink w onto the sphere if it is outside."""
    norm = np.linalg.norm(w)
    return w if norm <= r else w * (r / norm)


lr = 1.0 / np.linalg.eigvalsh(2.0 / N * X.T @ X)[-1]  # safe step: 1 / lambda_max (A4.4)
w = np.zeros(D)
for k in range(1, 100_001):
    w_new = project(w - lr * grad(w), radius)
    if np.linalg.norm(w_new - w) < 1e-12:
        break
    w = w_new
w_pgd = w_new
print(f"projected GD: {k} steps   w = {show(w_pgd)}   |w| = {np.linalg.norm(w_pgd):.4f}")

# %%
section("exercise 2: SLSQP, the multiplier and ridge")
res = optimize.minimize(loss, np.zeros(D), jac=grad, method="SLSQP",
                        constraints=[{"type": "ineq", "fun": lambda v: radius**2 - v @ v,
                                      "jac": lambda v: -2.0 * v}], options={"ftol": 1e-12})
mu = float(res.multipliers[0])
print(f"SLSQP: w = {show(res.x)}   multiplier mu = {mu:.4f}")
np.testing.assert_allclose(res.x, w_pgd, atol=1e-6)
print("projected GD matches SLSQP (atol 1e-6): OK")
kkt = grad(w_pgd) + 2 * mu * w_pgd  # stationarity: grad L + mu grad(|w|^2 - r^2) = 0
print(f"KKT stationarity: grad L + 2 mu w = {show(kkt)}")
w_ridge = np.linalg.solve(X.T @ X + N * mu * np.eye(D), X.T @ y)  # minimizes L + mu |w|^2 (A4.2 ex. 1)
print(f"ridge with lambda = mu: w = {show(w_ridge)}   |w| = {np.linalg.norm(w_ridge):.4f}")
np.testing.assert_allclose(w_ridge, w_pgd, atol=1e-6)
sk = Ridge(alpha=N * mu, fit_intercept=False).fit(X, y)
np.testing.assert_allclose(sk.coef_, w_pgd, atol=1e-6)
print(f"sklearn Ridge(alpha = n mu = {N * mu:.4f}) gives the same w: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
