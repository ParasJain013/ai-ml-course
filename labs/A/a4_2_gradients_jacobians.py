# %% [markdown]
# # A4.2 Gradients, Jacobians and the shapes of matrix derivatives — lab
# Lesson: lessons/A/a4_2_gradients_jacobians.md
# Run: uv run python labs/A/a4_2_gradients_jacobians.py   (CPU, defaults: ~2 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import differentiate, optimize

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "a4_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU. A numerical gradient costs 2 evaluations per parameter, so the random
# test problems stay small on purpose; bigger ones only make the check slower, not better.
GPU_SCALE = False
N_ROWS = 200 if GPU_SCALE else 20  # examples in the random regression problem
N_FEATURES = 10 if GPU_SCALE else 3  # features (d)
H = 1e-6  # central-difference step (A4.1)

# %%
# Data: the small examples of the worked example, and a random regression problem.
section("data")
X_small = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])  # (3, 2): a ones column and x = 1, 2, 3
y_small = np.array([1.0, 2.0, 2.0])  # (3,)
X = rng.normal(size=(N_ROWS, N_FEATURES))  # (n, d)
w_true = rng.normal(size=N_FEATURES)  # (d,)
y = X @ w_true + 0.1 * rng.normal(size=N_ROWS)  # (n,)
print("X_small", X_small.shape, " y_small", y_small.shape, " X", X.shape, " y", y.shape)

# %%
# Worked example by hand, part 1: a gradient, a directional derivative, a linear approximation.
section("worked example: gradient of x1^2 + 3 x1 x2")


def f_quad(x: FloatArray) -> float:
    """f(x) = x1^2 + 3 x1 x2. x (2,) -> scalar."""
    return float(x[0] ** 2 + 3 * x[0] * x[1])


x0 = np.array([1.0, 2.0])
grad0 = np.array([2 * x0[0] + 3 * x0[1], 3 * x0[0]])  # (2,), by hand
print("gradient at (1, 2) by hand =", show(grad0), "  length =", round(float(np.linalg.norm(grad0)), 4))
u = np.array([0.6, 0.8])  # (2,), a unit direction
num_dir = (f_quad(x0 + H * u) - f_quad(x0 - H * u)) / (2 * H)
print(f"directional derivative along (0.6, 0.8): grad . u = {grad0 @ u:.4f}   numerical = {num_dir:.4f}")
u_best = grad0 / np.linalg.norm(grad0)
print(f"along the gradient's own direction: {grad0 @ u_best:.4f} (= its length, the steepest)")
delta = np.array([0.01, -0.02])
print(f"f(x0 + delta) = {f_quad(x0 + delta):.6f}   f(x0) + grad . delta = {f_quad(x0) + grad0 @ delta:.6f}")

# %%
# Worked example by hand, part 2: gradient of the mean squared error.
section("worked example: gradient of MSE")


def mse(w: FloatArray, X: FloatArray, y: FloatArray) -> float:
    """Mean squared error (1/n) |X w - y|^2. X (n, d), w (d,), y (n,) -> scalar."""
    r = X @ w - y  # (n,) residuals
    return float(r @ r / len(y))


def mse_grad(w: FloatArray, X: FloatArray, y: FloatArray) -> FloatArray:
    """Gradient of the MSE: (2/n) X^T (X w - y). X (n, d), w (d,), y (n,) -> (d,)."""
    r = X @ w - y  # (n,)
    return 2.0 / len(y) * (X.T @ r)  # (d, n) @ (n,) -> (d,)


w_s = np.array([0.0, 1.0])
r_s = X_small @ w_s - y_small
print("w =", show(w_s), "  predictions =", show(X_small @ w_s), "  residuals =", show(r_s))
print(f"MSE = {mse(w_s, X_small, y_small):.4f}   X^T r = {show(X_small.T @ r_s)}"
      f"   gradient = (2/3) X^T r = {show(mse_grad(w_s, X_small, y_small))}")
w_ls = np.linalg.solve(X_small.T @ X_small, X_small.T @ y_small)  # normal equations (A2.4)
print("at the least-squares w =", show(w_ls), " the gradient is", show(mse_grad(w_ls, X_small, y_small)))

# %%
# Worked example by hand, part 3: a Jacobian.
section("worked example: Jacobian of (x1 x2, x1 + x2^2)")


def f_vec(x: FloatArray) -> FloatArray:
    """f(x) = (x1 x2, x1 + x2^2). x (2,) -> (2,)."""
    return np.array([x[0] * x[1], x[0] + x[1] ** 2])


xj = np.array([2.0, 3.0])
J_hand = np.array([[xj[1], xj[0]], [1.0, 2 * xj[1]]])  # (m, n) = (2, 2): row i = gradient of output i
print("J by hand at (2, 3) =", show(J_hand))
dx = np.array([0.01, 0.01])
print(f"f(x + dx) - f(x) = {show(f_vec(xj + dx) - f_vec(xj))}   J @ dx = {show(J_hand @ dx)}")

# %%
# From scratch: a numerical gradient for any shape, a numerical Jacobian, and a relative error.
section("from scratch: gradient checker")


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = H) -> FloatArray:
    """Central-difference gradient of a scalar function of an array of any shape. x (...) -> (...)."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):  # every entry, whatever the shape: (j,) for a vector, (i, j) for a matrix
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old  # restore: the caller's array must come back unchanged
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def numerical_jacobian(f: Callable[[FloatArray], FloatArray], x: FloatArray, h: float = H) -> FloatArray:
    """Central-difference Jacobian of f: R^n -> R^m. x (n,) -> (m, n); column j = d f / d x_j."""
    cols = []
    for j in range(x.size):
        e_j = np.zeros_like(x)
        e_j[j] = h
        cols.append((f(x + e_j) - f(x - e_j)) / (2 * h))  # (m,)
    return np.stack(cols, axis=1)  # (m, n)


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Relative error |a - b| / (|a| + |b|), safe when both are zero."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


TOL = 1e-6  # central differences with h = 1e-6 give ~1e-9 relative error on these smooth functions
w = rng.normal(size=N_FEATURES)  # (d,) a random point to check at


def report(name: str, analytic: FloatArray, numeric: FloatArray) -> None:
    """Print one gradient-check line. Tiny errors are bucketed: their last digits vary by machine."""
    err = rel_error(analytic, numeric)
    shown = "< 1e-8 " if err < 1e-8 else f"{err:.1e}"
    print(f"{name:<36} shape {str(analytic.shape):<8} rel error {shown}  {'PASS' if err < TOL else 'FAIL'}")


report("MSE: (2/n) X^T (Xw - y)", mse_grad(w, X, y), numerical_gradient(lambda v: mse(v, X, y), w))
report("MSE without the 2 (a typical bug)", mse_grad(w, X, y) / 2, numerical_gradient(lambda v: mse(v, X, y), w))

A = rng.normal(size=(4, N_FEATURES))  # (m, d)
b = rng.normal(size=4)  # (m,)
x_ab = rng.normal(size=N_FEATURES)  # (d,)


def sq_norm(v: FloatArray) -> float:
    """|A v - b|^2. v (d,) -> scalar."""
    r = A @ v - b
    return float(r @ r)


report("|Ax - b|^2: 2 A^T (Ax - b)", 2 * A.T @ (A @ x_ab - b), numerical_gradient(sq_norm, x_ab))

M = rng.normal(size=(N_FEATURES, N_FEATURES))  # (d, d), NOT symmetric
num_q = numerical_gradient(lambda v: float(v @ M @ v), x_ab)
report("w^T M w: (M + M^T) w", (M + M.T) @ x_ab, num_q)
report("w^T M w: 2 M w (only if M symmetric)", 2 * M @ x_ab, num_q)

D_IN, D_OUT = N_FEATURES, 2
W = rng.normal(size=(D_IN, D_OUT))  # (d_in, d_out), a layer's weight matrix
G = rng.normal(size=(N_ROWS, D_OUT))  # (n, d_out), an arbitrary "upstream gradient" dL/dZ
# L(W) = sum(G * (X W)) is the simplest loss whose gradient with respect to Z = X W is exactly G.
num_W = numerical_gradient(lambda V: float(np.sum(G * (X @ V))), W)
report("Z = XW: dL/dW = X^T G", X.T @ G, num_W)
assert (X.T @ G).shape == W.shape  # a gradient always has the shape of what it is a gradient of

J_num = numerical_jacobian(f_vec, xj)
print("numerical Jacobian at (2, 3) =", show(J_num), "  matches by hand:", np.allclose(J_num, J_hand, atol=1e-8))
z = rng.normal(size=4)
J_tanh = numerical_jacobian(np.tanh, z)
print("Jacobian of elementwise tanh is diagonal:", np.allclose(J_tanh, np.diag(1 - np.tanh(z) ** 2), atol=1e-8))
J_lin = numerical_jacobian(lambda v: A @ v, x_ab)
print("Jacobian of v -> A v is A itself:", np.allclose(J_lin, A, atol=1e-8), "  shape", J_lin.shape)

# %%
# With SciPy: approx_fprime / check_grad and differentiate.jacobian agree with ours.
section("with scipy")
g_scipy = optimize.approx_fprime(w, mse, np.sqrt(np.finfo(float).eps), X, y)  # forward differences
np.testing.assert_allclose(g_scipy, mse_grad(w, X, y), rtol=1e-5, atol=1e-7)
# rtol 1e-5: approx_fprime uses forward differences, accurate to about sqrt(eps) ~ 1e-8 in absolute terms
print("scipy.optimize.approx_fprime matches the analytic MSE gradient (rtol 1e-5): OK")
cg = optimize.check_grad(mse, mse_grad, w, X, y)
print(f"scipy.optimize.check_grad = |analytic - forward-difference| < 1e-6: {cg < 1e-6}")
cg_bad = optimize.check_grad(mse, lambda v, X_, y_: mse_grad(v, X_, y_) / 2, w, X, y)
print(f"check_grad with the bug = {cg_bad:.2f}  (large: the bug is caught)")
res = differentiate.jacobian(lambda v: np.stack([v[0] * v[1], v[0] + v[1] ** 2]), xj)
np.testing.assert_allclose(res.df, J_hand, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(J_num, res.df, rtol=1e-6, atol=1e-8)
print("scipy.differentiate.jacobian =", show(res.df), " matches ours and the hand result: OK")

# %%
# Key plot: contour lines of the MSE over (w0, w1), with the negative gradient as arrows.
section("key plot")
g0, g1 = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-0.6, 1.6, 200))
Wgrid = np.stack([g0.ravel(), g1.ravel()], axis=1)  # (40000, 2), one candidate w per row
R = Wgrid @ X_small.T - y_small  # (40000, 3) residuals of every candidate
L_grid = (R**2).mean(axis=1).reshape(g0.shape)  # (200, 200)
fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=120)
cs = ax.contour(g0, g1, L_grid, levels=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2], colors="0.55", linewidths=1)
ax.clabel(cs, fontsize=7, fmt="%.1f")
a0, a1 = np.meshgrid(np.linspace(-1.0, 2.0, 7), np.linspace(-0.4, 1.4, 6))
P = np.stack([a0.ravel(), a1.ravel()], axis=1)  # (42, 2) arrow tails
neg = -np.array([mse_grad(p, X_small, y_small) for p in P])  # (42, 2) downhill directions
neg_unit = neg / np.linalg.norm(neg, axis=1, keepdims=True)
ax.quiver(P[:, 0], P[:, 1], neg_unit[:, 0], neg_unit[:, 1], color="tab:red", angles="xy",
          scale_units="xy", scale=4.5, width=0.004, label="−gradient (direction only)")
ax.plot(*w_ls, "k*", ms=12, label=f"minimum (normal equations) {show(w_ls)}")
ax.plot(*w_s, "o", color="tab:blue", ms=7, label="w = (0, 1) of the worked example")
ax.set_aspect("equal")
ax.set_xlabel("w₀ (intercept)")
ax.set_ylabel("w₁ (slope)")
ax.set_title("The negative gradient points downhill, across the contour lines", fontsize=10)
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
path = IMG / "a4_2_gradient_contours.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set H = 1e-10 in numerical_gradient and rerun the checks. Which ones start to FAIL, and why (A4.1)?
# 2. Make M symmetric (M = M + M.T). Does the "2 M w" check pass now?
# 3. Check the gradient of the mean absolute error mean(|Xw - y|) with sign(r). Where can it fail?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
