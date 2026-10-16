# %% [markdown]
# # A3.3 Covariance matrices — lab
# Lesson: lessons/A/a3_3_covariance_matrices.md
# Run: uv run python labs/A/a3_3_covariance_matrices.py   (CPU, defaults: ~3 s)
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
import pandas as pd

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
OUT = ROOT / "outputs" / "a3_3"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knob sizes the simulated 2-D dataset.
GPU_SCALE = False
N_POINTS = 1_000_000 if GPU_SCALE else 500  # rows of the correlated 2-D dataset
TRUE_MEAN = np.array([10.0, 5.0])
TRUE_COV = np.array([[4.0, 2.4], [2.4, 2.25]])  # sd 2 and 1.5, correlation 0.8

# %%
# Data: the 4-example worked example, and 2-D correlated data with a known covariance.
section("data")
X4 = np.array([[1.0, 4.0], [3.0, 5.0], [4.0, 4.0], [4.0, 7.0]])  # (4, 2): 4 examples, 2 features
T = np.linalg.cholesky(TRUE_COV)  # (2, 2) lower-triangular with T @ T.T == TRUE_COV (A3.4 explains it)
Z = rng.normal(size=(N_POINTS, 2))  # (n, 2) independent standard normals: covariance I
X = Z @ T.T + TRUE_MEAN  # (n, 2): each row is T z + mean, covariance T I T^T = TRUE_COV
print("worked example X4", X4.shape, "=", show(X4))
print("simulated X", X.shape, " first three rows:", show(X[:3]))

# %%
# Worked example by hand: center, multiply, divide by n - 1.
section("worked example: covariance by hand")
mean4 = X4.mean(axis=0)  # (2,)
Xc4 = X4 - mean4  # (4, 2) centered: broadcasting subtracts the mean row from every row
print("means =", show(mean4))
print("centered rows =", show(Xc4))
print("Xc^T Xc =", show(Xc4.T @ Xc4))
S4 = Xc4.T @ Xc4 / (len(X4) - 1)  # (2, 2), divide by n - 1 = 3
print("covariance (divide by n - 1 = 3) =", show(S4))
print("divide by n = 4 instead          =", show(Xc4.T @ Xc4 / len(X4)))
sd4 = np.sqrt(np.diag(S4))
print("standard deviations =", show(sd4), "  correlation =", round(S4[0, 1] / (sd4[0] * sd4[1]), 4))
vals4, vecs4 = np.linalg.eigh(S4)
print("eigenvalues (ascending) =", show(vals4), "  eigenvectors (columns) =", show(vecs4))
for w in (np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2)):
    proj = Xc4 @ w  # (4,): coordinate of each centered example along w
    print(f"along w = {show(w)}: projections = {show(proj)}  variance = {proj @ proj / 3:.4f}  w^T S w = {w @ S4 @ w:.4f}")

# %%
# From scratch: the covariance and correlation matrices of any (n, d) data matrix.
section("from scratch: covariance and correlation")


def covariance(X: FloatArray, ddof: int = 1) -> FloatArray:
    """Covariance matrix of the columns. X (n, d) -> (d, d). ddof=1 divides by n - 1."""
    Xc = X - X.mean(axis=0)  # (n, d)
    return Xc.T @ Xc / (X.shape[0] - ddof)  # (d, n) @ (n, d) -> (d, d)


def correlation(X: FloatArray) -> FloatArray:
    """Correlation matrix of the columns. X (n, d) -> (d, d), ones on the diagonal."""
    S = covariance(X)
    sd = np.sqrt(np.diag(S))  # (d,)
    return S / np.outer(sd, sd)  # entry (j, k) divided by sd_j * sd_k


def covariance_loops(X: FloatArray) -> FloatArray:
    """Same as covariance(X), one entry at a time, straight from the definition. X (n, d) -> (d, d)."""
    n, d = X.shape
    mean = X.mean(axis=0)
    S = np.zeros((d, d))
    for j in range(d):
        for k in range(d):
            S[j, k] = sum((X[i, j] - mean[j]) * (X[i, k] - mean[k]) for i in range(n)) / (n - 1)
    return S


S = covariance(X)
R = correlation(X)
np.testing.assert_allclose(S, covariance_loops(X), rtol=1e-10, atol=1e-12)
print("matrix formula == entry-by-entry loops: OK")
print("sample covariance =", show(S), "  true =", show(TRUE_COV))
print("sample correlation =", show(R), "  true correlation = 0.8")

# %%
# Why the covariance matrix is symmetric and positive semi-definite.
section("from scratch: symmetric and positive semi-definite")
print("symmetric: S == S.T:", np.allclose(S, S.T))
vals, vecs = np.linalg.eigh(S)
print("eigenvalues =", show(vals), " all >= 0:", bool(np.all(vals >= 0)))
W = rng.normal(size=(1000, 2))  # 1000 random directions as rows
quad = np.einsum("ij,jk,ik->i", W, S, W)  # w^T S w for every row w, (1000,)
var_proj = np.var((X - X.mean(axis=0)) @ W.T, axis=0, ddof=1)  # variance of the data along each w, (1000,)
np.testing.assert_allclose(quad, var_proj, rtol=1e-10, atol=1e-12)
print(f"w^T S w == variance of the data along w, for 1000 random w: OK   smallest value {quad.min():.4f} >= 0")
print("largest eigenvalue = max variance over unit directions:", round(vals[-1], 4), " direction:", show(vecs[:, -1]))

X3 = np.column_stack([X, X[:, 0] + X[:, 1]])  # (n, 3): third column is the sum of the first two
vals3 = np.linalg.eigvalsh(covariance(X3))
print("add a column = col1 + col2: eigenvalues =", show(vals3), "-> one is 0: semi-definite, singular")

M = np.array([[1.0, 0.5], [-2.0, 1.0]])  # (2, 2) any linear map applied to the rows: Y = X @ M
np.testing.assert_allclose(covariance(X @ M), M.T @ S @ M, rtol=1e-10, atol=1e-12)
print("Cov(X @ M) == M^T S M: OK")

# %%
# With NumPy and pandas: np.cov, np.corrcoef, DataFrame.cov, and their defaults.
section("with numpy and pandas")
print("np.cov(X).shape =", np.cov(X).shape, " <- rows treated as variables: a 500 x 500 matrix, wrong")
S_np = np.cov(X, rowvar=False)  # (2, 2): columns are the variables
np.testing.assert_allclose(S, S_np, rtol=1e-6, atol=1e-8)
print("np.cov(X, rowvar=False) matches: OK   (ddof=1 by default)")
np.testing.assert_allclose(covariance(X, ddof=0), np.cov(X, rowvar=False, ddof=0), rtol=1e-6, atol=1e-8)
print("ddof=0 versions match: OK")
np.testing.assert_allclose(R, np.corrcoef(X, rowvar=False), rtol=1e-6, atol=1e-8)
print("np.corrcoef(X, rowvar=False) matches: OK")
df = pd.DataFrame(X, columns=["x1", "x2"])
np.testing.assert_allclose(S, df.cov().to_numpy(), rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(R, df.corr().to_numpy(), rtol=1e-6, atol=1e-8)
print("DataFrame.cov() and .corr() match: OK")
print(f"variance of x1: np.var = {np.var(X[:, 0]):.4f} (ddof=0)   pandas .var() = {df['x1'].var():.4f} (ddof=1)")
print(f"worked example, n = 4: np.var = {np.var(X4[:, 0]):.4f}   pandas = {pd.Series(X4[:, 0]).var():.4f}")

# %%
# Key plot: the data, its covariance eigenvectors scaled by the square roots of the eigenvalues.
section("key plot")
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=120)
ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4, marker="o", label=f"{N_POINTS} samples")
center = X.mean(axis=0)
t = np.linspace(0, 2 * np.pi, 200)
ellipse = center + 2 * (np.stack([np.cos(t), np.sin(t)], axis=1) * np.sqrt(vals)) @ vecs.T  # 2 sd contour
ax.plot(ellipse[:, 0], ellipse[:, 1], "--", color="0.3", lw=1.2, label="2 standard deviations along each axis")
for j, color, style in ((1, "tab:red", "-"), (0, "tab:green", "-")):
    arrow = 2 * np.sqrt(vals[j]) * vecs[:, j]
    ax.annotate("", xy=center + arrow, xytext=center, arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
    ax.plot([], [], style, color=color, lw=2.5, label=f"eigenvector × 2√λ, λ = {vals[j]:.2f}")
ax.set_aspect("equal")
ax.set_xlabel("x₁ (units of feature 1)")
ax.set_ylabel("x₂ (units of feature 2)")
ax.set_title("The covariance eigenvectors are the axes of the data cloud")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
path = IMG / "a3_3_covariance_eigenvectors.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set TRUE_COV to [[4, 0], [0, 2.25]]. Which way do the eigenvectors point now?
# 2. Multiply column 0 of X by 1000 (change its units). How do S and R change? Which one changes less?
# 3. Set N_POINTS to 10. How far is the sample covariance from TRUE_COV? Rerun with SEED = 1.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
