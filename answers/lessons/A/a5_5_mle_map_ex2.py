# %% [markdown]
# # A5.5 MLE and MAP — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a5_5_mle_map_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
N = 200
TRUE_W, TRUE_B = 2.0, -0.5
TAU2 = 0.05  # prior variance of the weight
LR = 0.5
MAX_ITERS = 200_000
GRAD_TOL = 1e-10


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
counts = np.array([2.0, 0.0, 3.0, 1.0])
print(f"(a) Poisson rate MLE = mean count = {counts.mean():.4f}")
xs, sigma2 = np.array([2.0, 4.0, 6.0]), 4.0
for tau2 in (1.0, 100.0):
    mu_map = (xs.sum() / sigma2) / (len(xs) / sigma2 + 1 / tau2)
    print(f"(b) tau^2 = {tau2:>5}: mu_MAP = (sum x / sigma^2) / (n / sigma^2 + 1 / tau^2) = {mu_map:.4f}"
          f"   (MLE {xs.mean():.1f})")


# %%
def sigmoid(z: FloatArray) -> FloatArray:
    """Logistic sigmoid, elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: FloatArray, y: FloatArray, tau2: float | None) -> tuple[FloatArray, int]:
    """GD on mean NLL (+ w^2 / (2 tau^2 n) if tau2). X (n, 2) = [1, x], y (n,) -> ((b, w), steps)."""
    n = len(y)
    theta = np.zeros(2)  # (b, w)
    for step in range(MAX_ITERS):
        p = sigmoid(X @ theta)  # (n,)
        g = X.T @ (p - y) / n  # (2,) gradient of the mean NLL
        if tau2 is not None:
            g = g + np.array([0.0, theta[1] / (tau2 * n)])  # prior on the weight only
        if np.linalg.norm(g) < GRAD_TOL:
            return theta, step
        theta = theta - LR * g
    raise RuntimeError("no convergence")


section("exercise 2: data")
x = rng.normal(0.0, 1.0, size=N)  # (n,) draw x first
y = (rng.random(N) < sigmoid(TRUE_W * x + TRUE_B)).astype(np.float64)  # (n,) then one uniform each
X = np.column_stack([np.ones(N), x])  # (n, 2)
print(f"n = {N}   positives = {int(y.sum())}")

for n_use in (200, 20):
    Xn, yn, xn = X[:n_use], y[:n_use], x[:n_use].reshape(-1, 1)
    section(f"exercise 2 ({'a, b' if n_use == 200 else 'c'}): first {n_use} examples")
    th_mle, steps_mle = fit_logistic(Xn, yn, None)
    sk_mle = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(xn, yn)
    sk_mle_th = np.array([sk_mle.intercept_[0], sk_mle.coef_[0, 0]])
    print(f"MLE by GD ({steps_mle} steps): (b, w) = {show(th_mle)}   LogisticRegression(C=inf): {show(sk_mle_th)}")
    np.testing.assert_allclose(th_mle, sk_mle_th, atol=1e-4)
    th_map, steps_map = fit_logistic(Xn, yn, TAU2)
    sk_map = LogisticRegression(C=TAU2, tol=1e-10, max_iter=10_000).fit(xn, yn)
    sk_map_th = np.array([sk_map.intercept_[0], sk_map.coef_[0, 0]])
    print(f"MAP by GD ({steps_map} steps), tau^2 = {TAU2}: (b, w) = {show(th_map)}   "
          f"LogisticRegression(C={TAU2}): {show(sk_map_th)}")
    np.testing.assert_allclose(th_map, sk_map_th, atol=1e-4)
    print(f"matches scikit-learn: OK   weight shrank from {th_mle[1]:.4f} to {th_map[1]:.4f} "
          f"({100 * (1 - th_map[1] / th_mle[1]):.0f}%)")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
