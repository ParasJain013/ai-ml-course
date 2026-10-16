# %% [markdown]
# # A2.4 Projections and least squares — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a2_4_projections_least_squares_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
OUT = Path(__file__).resolve().parents[3] / "outputs" / "a2_4_answers"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
a, b = np.array([1.0, 1.0, 0.0]), np.array([3.0, 1.0, 2.0])
p = (a @ b) / (a @ a) * a
r = b - p
print("p =", show(p), " r =", show(r), " r.a =", r @ a, f" |b|^2 = {b @ b:g} = |p|^2 + |r|^2 = {p @ p:g} + {r @ r:g}")
x3, y3 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0])
X3 = np.column_stack([np.ones(3), x3])
w3 = np.linalg.solve(X3.T @ X3, X3.T @ y3)
r3 = y3 - X3 @ w3
print("X^T X =", show(X3.T @ X3), " X^T y =", show(X3.T @ y3), " w =", show(w3))
print("residuals =", show(r3), " sum =", round(r3.sum(), 10) + 0.0, " sum x*r =", round(x3 @ r3, 10) + 0.0)
print("line at mean x:", round(w3[0] + w3[1] * x3.mean(), 4), " mean y:", round(y3.mean(), 4))

# %%
section("exercise 2: quadratic least squares")
N_POINTS = 40
TRUE_W = np.array([1.0, -2.0, 0.5])
x = rng.uniform(-3.0, 4.0, size=N_POINTS)  # (40,)
y = TRUE_W[0] + TRUE_W[1] * x + TRUE_W[2] * x**2 + rng.normal(0.0, 1.0, size=N_POINTS)  # (40,)


def poly_design(x: FloatArray, degree: int) -> FloatArray:
    """Columns 1, x, x^2, ..., x^degree. x (n,) -> (n, degree + 1)."""
    return np.column_stack([x**k for k in range(degree + 1)])


def fit_least_squares(X: FloatArray, y: FloatArray) -> FloatArray:
    """Solve the normal equations X^T X w = X^T y. X (n, d), y (n,) -> w (d,)."""
    return np.linalg.solve(X.T @ X, X.T @ y)


X = poly_design(x, 2)  # (40, 3)
w = fit_least_squares(X, y)
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
w_polyfit = np.polyfit(x, y, deg=2)[::-1]  # polyfit returns the highest power first
np.testing.assert_allclose(w, w_lstsq, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(w, w_polyfit, rtol=1e-6, atol=1e-8)
print("X", X.shape, " w =", show(w), " true =", show(TRUE_W))
print("matches np.linalg.lstsq and np.polyfit (reversed): OK")
resid = y - X @ w
np.testing.assert_allclose(X.T @ resid, np.zeros(3), atol=1e-8)
print("X^T residual =", show(X.T @ resid), " orthogonal to all three columns: OK")
P = X @ np.linalg.solve(X.T @ X, X.T)  # (40, 40), without forming an explicit inverse
np.testing.assert_allclose(P @ P, P, atol=1e-10)
np.testing.assert_allclose(P.T, P, atol=1e-10)
print("P", P.shape, " idempotent and symmetric: OK   trace(P) =", round(np.trace(P), 4), "(= number of columns)")

# %%
section("exercise 2: straight line vs quadratic residuals")
X1 = poly_design(x, 1)
w1 = fit_least_squares(X1, y)
resid1 = y - X1 @ w1
print("line w =", show(w1), f" SSE line = {resid1 @ resid1:.2f}   SSE quadratic = {resid @ resid:.2f}")
print("line residuals still sum to", round(resid1.sum(), 10) + 0.0, "and are orthogonal to x:", round(x @ resid1, 8) + 0.0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)
for ax, rr, title in [(axes[0], resid1, "Straight line: residuals curve (U shape)"), (axes[1], resid, "Quadratic: residuals patternless")]:
    ax.scatter(x, rr, s=14, marker="x")
    ax.axhline(0.0, color="0.3", lw=1)
    ax.set_xlabel("x (unitless)")
    ax.set_title(title)
axes[0].set_ylabel("residual (unitless)")
fig.tight_layout()
fig.savefig(OUT / "line_vs_quadratic_residuals.png", dpi=120, bbox_inches="tight")
plt.close(fig)
order = np.argsort(x)
print("line residuals at the smallest, middle and largest x:", show(resid1[order][[0, N_POINTS // 2, -1]]))
print("saved outputs/a2_4_answers/line_vs_quadratic_residuals.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
