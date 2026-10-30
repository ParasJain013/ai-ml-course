# %% [markdown]
# # A4.4 Gradient descent — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_4_gradient_descent_ex2.py   (CPU, ~2 s)

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
for lr in (0.1, 0.25, 0.4, 0.6):
    w, path = 0.0, []
    for _ in range(3):
        w -= lr * 4 * (w - 3)  # L = 2 (w - 3)^2, L' = 4 (w - 3), curvature a = 4
        path.append(round(w, 4))
    print(f"lr = {lr:<4} factor 1 - 4 lr = {1 - 4 * lr:+.1f}   w1, w2, w3 = {path}")
for lam in (4.0, 1.0):
    f = 1 - 0.4 * lam
    print(f"curvature {lam}: factor {f:+.1f}  steps for a 1000-fold shrink = {np.log(1e-3) / np.log(abs(f)):.1f}")

# %%
# The A2.4 data, regenerated exactly (same seed, same draws, same order).
x_data = rng.uniform(0.0, 10.0, size=50)
y_data = 2.0 + 0.5 * x_data + rng.normal(0.0, 1.0, size=50)
X = np.column_stack([np.ones_like(x_data), x_data])  # (50, 2)
w_normal = np.linalg.solve(X.T @ X, X.T @ y_data)
n_evals = 0  # counts loss evaluations


def mse(w: FloatArray) -> float:
    """Mean squared error on the A2.4 data, counting calls. w (2,) -> scalar."""
    global n_evals
    n_evals += 1
    r = X @ w - y_data
    return float(r @ r) / len(y_data)


def grad(w: FloatArray) -> FloatArray:
    """MSE gradient (2/n) X^T (X w - y). w (2,) -> (2,)."""
    return 2.0 / len(y_data) * X.T @ (X @ w - y_data)


def converges(lr: float, iters: int = 20_000) -> bool:
    """True if fixed-rate GD from 0 reaches a gradient norm below 1e-6 without blowing up."""
    w = np.zeros(2)
    for _ in range(iters):
        g = grad(w)
        if np.linalg.norm(g) < 1e-6:
            return True
        if not np.isfinite(g).all() or np.linalg.norm(g) > 1e8:
            return False
        w -= lr * g
    return False


section("exercise 2a: the largest stable learning rate")
lo, hi = 0.02, 0.04
assert converges(lo) and not converges(hi)
while hi - lo > 1e-6:
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if converges(mid) else (lo, mid)
lam_max = np.linalg.eigvalsh(2 / 50 * X.T @ X)[-1]
print(f"bisection: largest converging lr ~ {lo:.5f}   theory 2 / lambda_max = {2 / lam_max:.5f}")

# %%
section("exercise 2b: backtracking line search")


def gd_fixed(lr: float, tol: float = 1e-10) -> tuple[FloatArray, int]:
    """Fixed-rate GD; one loss evaluation per iteration is counted for a fair comparison."""
    w = np.zeros(2)
    for k in range(1, 100_000):
        g = grad(w)
        if np.linalg.norm(g) < tol:
            return w, k
        w -= lr * g
        mse(w)  # monitoring the loss each step, as any real loop does
    raise RuntimeError("no convergence")


def gd_backtracking(tol: float = 1e-6, max_halvings: int = 60, max_iters: int = 20_000) -> tuple[FloatArray, int]:
    """GD with Armijo backtracking: halve lr from 1 until L(w - lr g) <= L(w) - 0.5 lr ||g||^2."""
    w = np.zeros(2)
    for k in range(1, max_iters + 1):
        g = grad(w)
        gg = float(g @ g)
        if np.sqrt(gg) < tol:
            return w, k
        loss_w, lr = mse(w), 1.0
        for _ in range(max_halvings):
            if mse(w - lr * g) <= loss_w - 0.5 * lr * gg:
                break
            lr /= 2
        else:
            raise RuntimeError(f"line search failed at iteration {k}: gradient norm {np.sqrt(gg):.1e}, loss {loss_w!r}")
        w = w - lr * g
    raise RuntimeError(f"no convergence in {max_iters} iterations: gradient norm still {np.linalg.norm(grad(w)):.1e}")


for name, fn in (("fixed lr = 0.025", lambda: gd_fixed(0.025, tol=1e-6)), ("backtracking", gd_backtracking)):
    n_evals = 0
    w, iters = fn()
    np.testing.assert_allclose(w, w_normal, rtol=0, atol=1e-4)  # ||g|| < 1e-6 leaves an error ~ 1e-6 / lambda_min
    print(f"{name:<17} iterations = {iters:>5}   loss evaluations = {n_evals:>6}   w = {show(w)}  matches to 4 decimals: OK")

section("exercise 2c: asking for ||g|| < 1e-10")
try:
    gd_backtracking(tol=1e-10)
except RuntimeError as e:
    print(e)
print(f"loss values near the minimum are spaced {np.spacing(mse(w_normal)):.1e} apart; "
      f"the required decrease 0.5 lr ||g||^2 at ||g|| = 1e-8 is at most {0.5 * 1e-16:.1e}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
