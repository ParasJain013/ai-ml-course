# %% [markdown]
# # A4.2 Gradients, Jacobians and the shapes of matrix derivatives — lab
# Lesson: lessons/A/a4_2_gradients_jacobians.md
# Run: uv run python labs/A/a4_2_gradients_jacobians.py   (CPU, defaults: ~10 s, mostly importing PyTorch)
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
# Nothing here needs a GPU; the knobs size the random problems the gradient checker is tested on.
GPU_SCALE = False
N_ROWS = 1_000 if GPU_SCALE else 20  # rows n of the random regression problem
N_FEATURES = 50 if GPU_SCALE else 3  # features d
N_OUTPUTS = 10 if GPU_SCALE else 2  # outputs k of the matrix-weight problem
H_STEP = 1e-5  # central-difference step (A4.1: best near 1e-5 in float64)

# %%
# Data: the small matrices of the worked example, and a random regression problem.
section("data")
A3 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # (3, 2)
b3 = np.array([1.0, 2.0, 2.0])  # (3,)
X4 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])  # (4, 2): ones column + x, from A2.4
y4 = np.array([1.0, 3.0, 4.0, 4.0])  # (4,)
X = rng.normal(size=(N_ROWS, N_FEATURES))  # (n, d)
y = X @ rng.normal(size=N_FEATURES) + rng.normal(0.0, 0.1, size=N_ROWS)  # (n,)
print("A:", A3.shape, " b:", b3.shape, "  X4:", X4.shape, " y4:", y4.shape, "  random X:", X.shape, " y:", y.shape)

# %%
# Worked example by hand, part 1: a Jacobian of F: R^2 -> R^3.
section("worked example: a Jacobian")


def F(x: FloatArray) -> FloatArray:
    """F(x) = (x1 x2, x1 + x2^2, 3 x1). x (2,) -> (3,)."""
    return np.array([x[0] * x[1], x[0] + x[1] ** 2, 3.0 * x[0]])


x_pt = np.array([1.0, 2.0])
J_hand = np.array([[x_pt[1], x_pt[0]], [1.0, 2 * x_pt[1]], [3.0, 0.0]])  # (3, 2): row i = gradient of F_i
print("F(1, 2) =", show(F(x_pt)))
print("J by hand, shape", J_hand.shape, "=", show(J_hand))
delta = np.array([0.01, -0.02])
print("F(x + delta) =", show(F(x_pt + delta)), "  F(x) + J delta =", show(F(x_pt) + J_hand @ delta), " (linear approximation)")

# %%
# Worked example by hand, part 2: gradients of ||A x - b||^2 and of the MSE.
section("worked example: two gradients")


def sq_residual(x: FloatArray) -> float:
    """f(x) = ||A x - b||^2. x (2,) -> scalar."""
    r = A3 @ x - b3  # (3,)
    return float(r @ r)


def grad_sq_residual(x: FloatArray) -> FloatArray:
    """Gradient 2 A^T (A x - b). x (2,) -> (2,)."""
    return 2.0 * A3.T @ (A3 @ x - b3)


print("||Ax - b||^2 at x = (0, 0):  A^T b =", show(A3.T @ b3), "  gradient = -2 A^T b =", show(grad_sq_residual(np.zeros(2))))
x_star = np.linalg.solve(A3.T @ A3, A3.T @ b3)
print("setting the gradient to 0 gives the normal equations (A2.4): x* =", show(x_star), "  gradient at x* =", show(grad_sq_residual(x_star)))


def mse(w: FloatArray, X_: FloatArray, y_: FloatArray) -> float:
    """Mean squared error (1/n) ||X w - y||^2. X (n, d), w (d,), y (n,) -> scalar."""
    r = X_ @ w - y_  # (n,)
    return float(r @ r) / len(y_)


def grad_mse(w: FloatArray, X_: FloatArray, y_: FloatArray) -> FloatArray:
    """Gradient (2/n) X^T (X w - y). X (n, d), w (d,), y (n,) -> (d,)."""
    return 2.0 / len(y_) * X_.T @ (X_ @ w - y_)  # (d, n) @ (n,) -> (d,)


w0 = np.zeros(2)
print("MSE on the 4 points at w = (0, 0):", mse(w0, X4, y4), "  X^T y =", show(X4.T @ y4), "  gradient =", show(grad_mse(w0, X4, y4)))

# %%
# Worked example by hand, part 3: the gradient points uphill, perpendicular to the contour.
section("worked example: direction of steepest ascent")


def bowl(x: FloatArray) -> float:
    """f(x) = x1^2 + 3 x2^2. x (2,) -> scalar."""
    return float(x[0] ** 2 + 3.0 * x[1] ** 2)


g = np.array([2 * 1.0, 6 * 1.0])  # gradient (2 x1, 6 x2) at (1, 1)
print("gradient at (1, 1) =", show(g), "  length =", round(float(np.linalg.norm(g)), 4))
for name, u in (("along x1 (1, 0)", np.array([1.0, 0.0])), ("along the gradient", g / np.linalg.norm(g)),
                ("against the gradient", -g / np.linalg.norm(g)), ("along the contour (3, -1)/sqrt(10)", np.array([3.0, -1.0]) / np.sqrt(10))):
    print(f"  slope {name:<36} g . u = {g @ u:+.4f}")

# %%
# From scratch: a numerical gradient for any shape, a numerical Jacobian, and a relative-error check.
section("from scratch: gradient checker")


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = H_STEP) -> FloatArray:
    """Central-difference gradient of a scalar f at x of any shape. Returns the same shape as x."""
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):  # every coordinate: (i,) for a vector, (i, j) for a matrix
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old  # restore: the caller's array must come back unchanged
        grad[idx] = (f_plus - f_minus) / (2.0 * h)
    return grad


def numerical_jacobian(F_: Callable[[FloatArray], FloatArray], x: FloatArray, h: float = H_STEP) -> FloatArray:
    """Central-difference Jacobian of F: R^n -> R^m. x (n,) -> (m, n); column j = dF/dx_j."""
    cols = []
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = h
        cols.append((F_(x + step) - F_(x - step)) / (2.0 * h))  # (m,)
    return np.stack(cols, axis=1)  # (m, n)


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Largest relative error max |a - b| / max(|a|, |b|, 1e-12), elementwise."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)))


def check(name: str, analytic: FloatArray, numeric: FloatArray, tol: float = 1e-7) -> None:
    """Print one gradient-check line: shapes, relative error, PASS or FAIL."""
    assert analytic.shape == numeric.shape, (analytic.shape, numeric.shape)
    err = rel_error(analytic, numeric)
    print(f"{name:<44} shape {str(analytic.shape):<8} rel error {err:.1e}  {'PASS' if err < tol else 'FAIL'}")


print("Jacobian of F at (1, 2), numerical =", show(numerical_jacobian(F, x_pt)))
check("Jacobian of F (by hand)", J_hand, numerical_jacobian(F, x_pt))
x_r = rng.normal(size=2)
check("||Ax - b||^2: 2 A^T (Ax - b)", grad_sq_residual(x_r), numerical_gradient(sq_residual, x_r.copy()))
w_r = rng.normal(size=N_FEATURES)
check("MSE: (2/n) X^T (Xw - y)", grad_mse(w_r, X, y), numerical_gradient(lambda w: mse(w, X, y), w_r.copy()))
check("MSE with the 2 forgotten: (1/n) X^T (Xw - y)", grad_mse(w_r, X, y) / 2, numerical_gradient(lambda w: mse(w, X, y), w_r.copy()))

M = np.array([[1.0, 2.0], [0.0, 3.0]])  # (2, 2), NOT symmetric
quad = lambda x: float(x @ M @ x)  # noqa: E731  f(x) = x^T M x
check("x^T M x: (M + M^T) x", (M + M.T) @ x_r, numerical_gradient(quad, x_r.copy()))
check("x^T M x: 2 M x (true only if M symmetric)", 2 * M @ x_r, numerical_gradient(quad, x_r.copy()))

# %%
# From scratch: a matrix of weights. The gradient has the same shape as W.
section("from scratch: matrix shapes")
W = rng.normal(size=(N_FEATURES, N_OUTPUTS))  # (d, k)
Y = rng.normal(size=(N_ROWS, N_OUTPUTS))  # (n, k)


def mse_matrix(W_: FloatArray) -> float:
    """(1/n) ||X W - Y||_F^2: squared error summed over outputs, averaged over rows. W (d, k) -> scalar."""
    R = X @ W_ - Y  # (n, k)
    return float(np.sum(R * R)) / N_ROWS


R = X @ W - Y  # (n, k)
dW = 2.0 / N_ROWS * X.T @ R  # (d, n) @ (n, k) -> (d, k): the only product of X and R with W's shape
print("X", X.shape, " W", W.shape, " R = XW - Y", R.shape, " dW = (2/n) X^T R", dW.shape)
check("(1/n)||XW - Y||_F^2: (2/n) X^T (XW - Y)", dW, numerical_gradient(mse_matrix, W.copy()))

A_lin = rng.normal(size=(3, N_FEATURES))  # (m, n)
check("Jacobian of x -> A x is A itself", A_lin, numerical_jacobian(lambda x: A_lin @ x, w_r))
J_tanh = numerical_jacobian(np.tanh, w_r)
print("Jacobian of elementwise tanh at w =", show(J_tanh))
check("elementwise tanh: diag(1 - tanh^2)", np.diag(1 - np.tanh(w_r) ** 2), J_tanh)

# %%
# From scratch: the Hessian is the Jacobian of the gradient.
section("from scratch: the Hessian")
H_bowl = numerical_jacobian(lambda x: np.array([2 * x[0], 6 * x[1]]), np.array([1.0, 1.0]))
print("Hessian of x1^2 + 3 x2^2 =", show(H_bowl))
H_mse = numerical_jacobian(lambda w: grad_mse(w, X4, y4), np.zeros(2))
print("Hessian of the 4-point MSE =", show(H_mse), "  (2/n) X^T X =", show(2 / 4 * X4.T @ X4))
check("Hessian of MSE: (2/n) X^T X", 2 / 4 * X4.T @ X4, H_mse)

# %%
# With SciPy and PyTorch: library gradient checking and automatic differentiation agree.
section("with scipy and pytorch")
import torch  # noqa: E402
from scipy.optimize import approx_fprime, check_grad  # noqa: E402

g_scipy = approx_fprime(w_r, lambda w: mse(w, X, y), 1e-6)  # forward differences
print(f"scipy approx_fprime vs mine: rel error {rel_error(g_scipy, grad_mse(w_r, X, y)):.1e}  (forward differences: expect ~1e-5)")
print(f"scipy check_grad (norm of the difference): {check_grad(lambda w: mse(w, X, y), lambda w: grad_mse(w, X, y), w_r):.1e}")

Xt, yt = torch.tensor(X), torch.tensor(y)  # float64, because X and y are float64
wt = torch.tensor(w_r, requires_grad=True)
loss = ((Xt @ wt - yt) ** 2).mean()
(g_torch,) = torch.autograd.grad(loss, wt)
np.testing.assert_allclose(g_torch.numpy(), grad_mse(w_r, X, y), rtol=1e-6, atol=1e-8)
print("torch.autograd.grad of the MSE matches (2/n) X^T (Xw - y): OK")
J_torch = torch.autograd.functional.jacobian(
    lambda t: torch.stack([t[0] * t[1], t[0] + t[1] ** 2, 3 * t[0]]), torch.tensor(x_pt)
)
np.testing.assert_allclose(J_torch.numpy(), J_hand, rtol=1e-6, atol=1e-8)
print("torch.autograd.functional.jacobian of F matches J by hand: OK, shape", tuple(J_torch.shape))
Wt = torch.tensor(W, requires_grad=True)
(dW_torch,) = torch.autograd.grad(((torch.tensor(X) @ Wt - torch.tensor(Y)) ** 2).sum() / N_ROWS, Wt)
np.testing.assert_allclose(dW_torch.numpy(), dW, rtol=1e-6, atol=1e-8)
print("torch gradient for the matrix W matches (2/n) X^T (XW - Y): OK, shape", tuple(dW_torch.shape))

# %%
# Key plot: contours of x1^2 + 3 x2^2 with gradient arrows.
section("key plot")
g1, g2 = np.meshgrid(np.linspace(-2.2, 2.2, 200), np.linspace(-1.5, 1.5, 200))
Z = g1**2 + 3 * g2**2
fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
cs = ax.contour(g1, g2, Z, levels=[0.5, 1, 2, 4, 6], colors="0.5", linewidths=1)
ax.clabel(cs, fontsize=7, fmt="f = %g")
pts = np.array([[1.0, 1.0], [-1.5, 0.5], [0.5, -0.9], [-0.7, -0.6], [1.7, 0.0], [0.0, 1.05]])
for k, p in enumerate(pts):
    grad = np.array([2 * p[0], 6 * p[1]])
    u = 0.3 * grad / np.linalg.norm(grad)  # fixed length 0.3: the plot shows direction, not size
    ax.annotate("", xy=p + u, xytext=p, arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
    ax.annotate("", xy=p - u, xytext=p, arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8, ls="--"))
    ax.plot(*p, "ko", ms=4)
ax.plot([], [], "-", color="tab:red", lw=1.8, label="direction of ∇f: steepest uphill")
ax.plot([], [], "--", color="tab:blue", lw=1.8, label="direction of −∇f: steepest downhill")
ax.plot(0, 0, "k*", ms=12, label="minimum (gradient = 0)")
ax.set_aspect("equal")
ax.set_xlabel("x₁ (unitless)")
ax.set_ylabel("x₂ (unitless)")
ax.set_title("The gradient is perpendicular to the contours and points uphill")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
fig.tight_layout()
path = IMG / "a4_2_gradient_contours.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Change H_STEP to 1e-10 and rerun the checks. Which ones start to fail, and why (A4.1)?
# 2. Make M symmetric ((M + M.T) / 2). Does the "2 M x" check pass now?
# 3. Add an L2 penalty lam * ||w||^2 to mse() and check your gradient for it with numerical_gradient.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
