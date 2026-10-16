# %% [markdown]
# # A2.4 Projections and least squares — lab
# Lesson: lessons/A/a2_4_projections_least_squares.md
# Run: uv run python labs/A/a2_4_projections_least_squares.py   (CPU, defaults: ~20 s, mostly importing scikit-learn)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
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
OUT = ROOT / "outputs" / "a2_4"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knob sizes the noisy dataset.
GPU_SCALE = False
N_POINTS = 100_000 if GPU_SCALE else 50  # rows of the noisy line-fitting dataset
TRUE_W = np.array([2.0, 0.5])  # intercept and slope used to generate the data
NOISE_SD = 1.0

# %%
# Data: noisy points around the line y = 2 + 0.5 x.
section("data")
x_data = rng.uniform(0.0, 10.0, size=N_POINTS)  # (n,)
y_data = TRUE_W[0] + TRUE_W[1] * x_data + rng.normal(0.0, NOISE_SD, size=N_POINTS)  # (n,)
print("x:", x_data.shape, " y:", y_data.shape)
print("first three (x, y):", show(np.column_stack([x_data[:3], y_data[:3]])))

# %%
# Worked example by hand, part 1: project b onto the line through a.
section("worked example: projection onto a line")


def project_onto_line(b: FloatArray, a: FloatArray) -> FloatArray:
    """Closest point to b on the line through the origin along a. a, b (d,) -> (d,)."""
    return (a @ b) / (a @ a) * a


a = np.array([2.0, 1.0])  # (2,)
b = np.array([1.0, 3.0])  # (2,)
p = project_onto_line(b, a)
r = b - p
print("a . b =", a @ b, "  a . a =", a @ a, "  coefficient =", (a @ b) / (a @ a))
print("p =", show(p), "  residual r = b - p =", show(r), "  r . a =", r @ a)

# %%
# Worked example by hand, part 2: project b onto the plane spanned by two columns.
section("worked example: projection onto a plane")


def project_onto_columns(b: FloatArray, A: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Project b (n,) onto the column space of A (n, k). Returns (coefficients (k,), projection (n,))."""
    c = np.linalg.solve(A.T @ A, A.T @ b)  # the normal equations, (k, k) c = (k,)
    return c, A @ c


A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2): columns (1, 0, 1) and (0, 1, 1)
b3 = np.array([1.0, 2.0, 4.0])  # (3,): not in the plane
print("A^T A =", show(A.T @ A), "  A^T b =", show(A.T @ b3))
c, p3 = project_onto_columns(b3, A)
r3 = b3 - p3
print("c =", show(c), "  p = A c =", show(p3), "  r = b - p =", show(r3))
print("A^T r =", show(A.T @ r3), "-> the residual is orthogonal to both columns")
P = A @ np.linalg.inv(A.T @ A) @ A.T  # (3, 3) projection matrix; inv is fine for a 2x2 here
print("P =", show(P))
print("P @ b == p:", np.allclose(P @ b3, p3), "  P @ P == P:", np.allclose(P @ P, P), "  P.T == P:", np.allclose(P.T, P))
print("projecting twice changes nothing: P @ p == p:", np.allclose(P @ p3, p3))

# %%
# Worked example by hand, part 3: fit a line to four points with the normal equations.
section("worked example: line through four points")
x4 = np.array([0.0, 1.0, 2.0, 3.0])
y4 = np.array([1.0, 3.0, 4.0, 4.0])


def design_matrix(x: FloatArray) -> FloatArray:
    """Stack a column of ones (for the intercept) next to x. x (n,) -> (n, 2)."""
    return np.column_stack([np.ones_like(x), x])


X4 = design_matrix(x4)  # (4, 2)
print("X =", show(X4))
print("X^T X =", show(X4.T @ X4), "  X^T y =", show(X4.T @ y4))
w4 = np.linalg.solve(X4.T @ X4, X4.T @ y4)
res4 = y4 - X4 @ w4
print("w = (intercept, slope) =", show(w4))
print("predictions =", show(X4 @ w4), "  residuals =", show(res4))
print("sum of residuals =", round(res4.sum(), 10) + 0.0, "  sum of x * residual =", round(x4 @ res4, 10) + 0.0)
print("sum of squared residuals =", round(res4 @ res4, 4))

# %%
# From scratch on the noisy data: normal equations, then the checks.
section("from scratch: normal equations")


def fit_least_squares(X: FloatArray, y: FloatArray) -> FloatArray:
    """Solve the normal equations X^T X w = X^T y. X (n, d), y (n,) -> w (d,)."""
    XtX = X.T @ X  # (d, d)
    Xty = X.T @ y  # (d,)
    return np.linalg.solve(XtX, Xty)  # never inv(XtX) @ Xty; see A2.2, A3.4


X = design_matrix(x_data)  # (n, 2)
w = fit_least_squares(X, y_data)
resid = y_data - X @ w  # (n,)
print("X", X.shape, " X^T X", (X.T @ X).shape, " w", w.shape)
print("fitted w =", show(w), "  true w =", show(TRUE_W))
print("X^T residual =", show(X.T @ resid), "(zero up to rounding: orthogonal)")
sse = resid @ resid
print(f"sum of squared residuals at w: {sse:.4f}")
for dw in (np.array([0.1, 0.0]), np.array([-0.1, 0.0]), np.array([0.0, 0.02]), np.array([0.0, -0.02])):
    r_ = y_data - X @ (w + dw)
    print(f"  at w + {show(dw):<12} {r_ @ r_:.4f}  (larger)")
    assert r_ @ r_ > sse

# %%
# With NumPy and scikit-learn: lstsq and LinearRegression give the same line.
section("with numpy and scikit-learn")
from sklearn.linear_model import LinearRegression  # noqa: E402

w_lstsq, *_ = np.linalg.lstsq(X, y_data, rcond=None)
model = LinearRegression().fit(x_data.reshape(-1, 1), y_data)  # sklearn adds the intercept itself
w_sklearn = np.array([model.intercept_, model.coef_[0]])
np.testing.assert_allclose(w, w_lstsq, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(w, w_sklearn, rtol=1e-6, atol=1e-8)
print("normal equations   w =", show(w))
print("np.linalg.lstsq    w =", show(w_lstsq))
print("LinearRegression   w =", show(w_sklearn))
print("all three agree: OK")
print(f"condition numbers: X {np.linalg.cond(X):.1f}   X^T X {np.linalg.cond(X.T @ X):.1f}   (= cond(X)^2 {np.linalg.cond(X) ** 2:.1f})")

# %%
# Key plot: the fitted line and its residuals.
section("key plot")
order = np.argsort(x_data)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
ax1.scatter(x_data, y_data, s=14, marker="o", label="data")
ax1.plot(x_data[order], (X @ w)[order], "-", color="tab:orange", lw=2, label=f"least squares: {w[0]:.2f} + {w[1]:.2f} x")
ax1.plot(x_data[order], (X @ TRUE_W)[order], "--", color="0.3", lw=1.5, label="true line: 2 + 0.5 x")
for xi, yi, fi in zip(x_data[:8], y_data[:8], (X @ w)[:8], strict=True):
    ax1.plot([xi, xi], [yi, fi], color="tab:red", lw=0.8)
ax1.plot([], [], color="tab:red", lw=0.8, label="residuals (first 8)")
ax1.set_xlabel("x (unitless)")
ax1.set_ylabel("y (unitless)")
ax1.set_title("The fitted line is close to the true one")
ax1.legend(loc="upper left", fontsize=8)
ax2.scatter(x_data, resid, s=14, marker="x", color="tab:red")
ax2.axhline(0.0, color="0.3", lw=1)
ax2.set_xlabel("x (unitless)")
ax2.set_ylabel("residual y - ŷ (unitless)")
ax2.set_title("Residuals: centred on 0, no trend in x")
fig.tight_layout()
path = IMG / "a2_4_line_fit_residuals.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set NOISE_SD to 3. How far is w from TRUE_W now? Then set N_POINTS to 5000.
# 2. Replace y_data[0] with 100 (one corrupted row). How much does the slope move?
# 3. Drop the column of ones from X. What line do you fit, and what happens to sum(resid)?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
