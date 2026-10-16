# %% [markdown]
# # A4.4 Gradient descent — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_4_gradient_descent_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time

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


# %%
section("exercise 1 check")
for lr in (0.1, 0.4):
    w, path = 1.0, []
    for _ in range(3):
        w = w - lr * 4 * (w + 1)  # gradient of 2 (w + 1)^2 is 4 (w + 1)
        path.append(round(w, 4))
    print(f"(a) lr = {lr}: factor {1 - 4 * lr:.1f}   w = {path}")
lam = np.array([4.0, 1.0])
lr_best = 2 / (lam.max() + lam.min())
rate = (lam.max() / lam.min() - 1) / (lam.max() / lam.min() + 1)
print(f"(b) lr_max = {2 / lam.max()}   lr_best = {lr_best}   factor = {rate}"
      f"   steps for 1e-6 = {int(np.ceil(np.log(1e-6) / np.log(rate)))}")
for big in (40.0, 400.0):
    r = (big - 1) / (big + 1)
    print(f"(c) H = diag({big:g}, 1): factor {r:.4f}   steps for 1e-6 = {int(np.ceil(np.log(1e-6) / np.log(r)))}")

# %%
section("exercise 2: data")
N = 40
x = rng.uniform(-3.0, 4.0, size=N)  # (n,) drawn first
y = 1.0 - 2.0 * x + 0.5 * x**2 + rng.normal(0.0, 1.0, size=N)  # (n,) then the noise
X_raw = np.column_stack([np.ones(N), x, x**2])  # (n, 3)
w_ls, *_ = np.linalg.lstsq(X_raw, y, rcond=None)
print("X_raw", X_raw.shape, "  lstsq w =", show(w_ls))


def mse_grad(w: FloatArray, X: FloatArray, y: FloatArray) -> FloatArray:
    """Gradient (2/n) X^T (X w - y). -> (d,)."""
    return 2.0 / len(y) * (X.T @ (X @ w - y))


def gd(X: FloatArray, y: FloatArray, lr: float, tol: float = 1e-8, max_iters: int = 2_000_000) -> tuple[FloatArray, int]:
    """Batch GD from zero until |grad| < tol. -> (w, steps)."""
    w = np.zeros(X.shape[1])
    for k in range(max_iters):
        g = mse_grad(w, X, y)
        if np.linalg.norm(g) < tol:
            return w, k
        w = w - lr * g
    raise RuntimeError(f"no convergence in {max_iters} steps")


def best_lr(X: FloatArray) -> tuple[float, float]:
    """Best fixed learning rate and condition number from the Hessian (2/n) X^T X."""
    ev = np.linalg.eigvalsh(2.0 / X.shape[0] * X.T @ X)
    return 2.0 / (ev[0] + ev[-1]), ev[-1] / ev[0]


# %%
section("exercise 2: raw features")
lr_r, kappa_r = best_lr(X_raw)
w_r, steps_r = gd(X_raw, y, lr_r)
print(f"condition number = {kappa_r:.1f}   lr = {lr_r:.5f}   steps = {steps_r}   w = {show(w_r)}")
np.testing.assert_allclose(w_r, w_ls, rtol=0, atol=5e-5)
print("matches lstsq to 4 decimals: OK")

# %%
section("exercise 2: standardized features")
mu = X_raw[:, 1:].mean(axis=0)  # (2,) means of x and x^2
sd = X_raw[:, 1:].std(axis=0)  # (2,)
X_std = np.column_stack([np.ones(N), (X_raw[:, 1:] - mu) / sd])  # (n, 3)
lr_s, kappa_s = best_lr(X_std)
w_s, steps_s = gd(X_std, y, lr_s)
w_back = np.concatenate([[w_s[0] - np.sum(w_s[1:] * mu / sd)], w_s[1:] / sd])
print(f"condition number = {kappa_s:.1f}   lr = {lr_s:.5f}   steps = {steps_s}   w (raw scale) = {show(w_back)}")
np.testing.assert_allclose(w_back, w_ls, rtol=0, atol=5e-5)
print("matches lstsq to 4 decimals: OK")
print(f"correlation of x and x^2 on this data = {np.corrcoef(x, x**2)[0, 1]:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
