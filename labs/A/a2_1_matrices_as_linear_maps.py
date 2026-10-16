# %% [markdown]
# # A2.1 Matrices as linear maps — lab
# Lesson: lessons/A/a2_1_matrices_as_linear_maps.md
# Run: uv run python labs/A/a2_1_matrices_as_linear_maps.py   (CPU, defaults: ~2 s)
# GPU: no

# %%
from __future__ import annotations

import math
import sys
import time
import warnings
from functools import reduce
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
OUT = ROOT / "outputs" / "a2_1"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knobs size the grid of points and the random agreement check.
GPU_SCALE = False
GRID_STEPS = 41 if GPU_SCALE else 11  # points per side of the grid we transform
N_RANDOM = 1_000_000 if GPU_SCALE else 10_000  # random points in the library check

# %%
# Data: a grid of 2-D points, one point per row, and the corners of the unit square.
section("data")
ticks = np.linspace(-1.0, 1.0, GRID_STEPS)
gx, gy = np.meshgrid(ticks, ticks)
GRID = np.column_stack([gx.ravel(), gy.ravel()])  # (GRID_STEPS**2, 2): rows are points
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])  # (5, 2), closed
print("grid of points:", GRID.shape, "  unit square corners:", SQUARE.shape)
print("first three grid points:", show(GRID[:3]))

# %%
# Worked example by hand: A = [[2, -1], [1, 1]], x = (1, 2).
section("worked example")


def matvec_rows(A: list[list[float]], x: list[float]) -> list[float]:
    """Row view: entry i of Ax is (row i of A) . x. A (m, d) as lists, x (d,) -> (m,)."""
    if any(len(row) != len(x) for row in A):
        raise ValueError(f"A has rows of length {len(A[0])}, x has length {len(x)}")
    return [sum(a_ij * x_j for a_ij, x_j in zip(row, x, strict=True)) for row in A]


def matvec_columns(A: list[list[float]], x: list[float]) -> list[float]:
    """Column view: Ax = x_1 * (column 1) + ... + x_d * (column d). A (m, d), x (d,) -> (m,)."""
    m, d = len(A), len(A[0])
    if d != len(x):
        raise ValueError(f"A has {d} columns, x has length {len(x)}")
    out = [0.0] * m
    for j in range(d):  # j is 0-based here; column j+1 in the math
        for i in range(m):
            out[i] += x[j] * A[i][j]
    return out


A_list = [[2.0, -1.0], [1.0, 1.0]]
x_list = [1.0, 2.0]
print("A e1 = column 1 =", matvec_rows(A_list, [1.0, 0.0]), "  A e2 = column 2 =", matvec_rows(A_list, [0.0, 1.0]))
print("row view    Ax =", matvec_rows(A_list, x_list))
print("column view Ax =", matvec_columns(A_list, x_list))
A = np.array(A_list)  # (2, 2)
x = np.array(x_list)  # (2,)
print("numpy       A @ x =", show(A @ x))
P3 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2): three points as rows
print("three points as rows, X @ A.T =", show(P3 @ A.T))
print("det(A) = 2*1 - (-1)*1 =", A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

# %%
# The four basic maps of the plane, built from where they send e1 and e2.
section("four maps applied to x = (1, 2)")


def rotation(theta_deg: float) -> FloatArray:
    """Rotate counter-clockwise by theta degrees. -> (2, 2)."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def scaling(sx: float, sy: float) -> FloatArray:
    """Stretch the x-axis by sx and the y-axis by sy. -> (2, 2)."""
    return np.array([[sx, 0.0], [0.0, sy]])


def shear(k: float) -> FloatArray:
    """Slide each point right by k times its height. -> (2, 2)."""
    return np.array([[1.0, k], [0.0, 1.0]])


def projection_onto(u: FloatArray) -> FloatArray:
    """Project onto the line through the origin along u (d,). -> (d, d)."""
    return np.outer(u, u) / (u @ u)  # u u^T / (u^T u); A2.4 derives this


def det2(M: FloatArray) -> float:
    """Determinant of a 2x2 matrix: the factor by which it scales areas (sign = flip)."""
    return float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])


MAPS = {
    "rotate 90": rotation(90.0),
    "rotate 30": rotation(30.0),
    "scale (1.5, 0.5)": scaling(1.5, 0.5),
    "shear k=1": shear(1.0),
    "project on (1, 1)": projection_onto(np.array([1.0, 1.0])),
}
for name, M in MAPS.items():
    print(f"{name:<18} M = {show(M):<34} M x = {show(M @ x):<16} det = {det2(M) + 0.0:.4f}")

# %%
# Linearity: M(a x + b y) = a Mx + b My for every matrix; adding a constant breaks it.
section("linearity")
u_vec, v_vec = rng.normal(size=2), rng.normal(size=2)  # (2,), (2,)
alpha, beta = 3.0, -2.0
M = MAPS["shear k=1"]
np.testing.assert_allclose(M @ (alpha * u_vec + beta * v_vec), alpha * (M @ u_vec) + beta * (M @ v_vec), rtol=1e-6, atol=1e-8)
print("shear(3u - 2v) == 3 shear(u) - 2 shear(v): OK")
shift = np.array([1.0, 0.0])
print("M @ 0 =", show(M @ np.zeros(2)), "  but (0 + shift) =", show(np.zeros(2) + shift), "-> a shift is not linear")

# %%
# Applying a matrix to many points at once. Points are rows, so the product is X @ M.T.
section("many points at once")
M = MAPS["rotate 30"]
loop = np.array([M @ p for p in GRID])  # (n, 2): one matrix-vector product per point
batch = GRID @ M.T  # (n, 2) @ (2, 2) -> (n, 2)
np.testing.assert_allclose(loop, batch, rtol=1e-6, atol=1e-8)
print(f"loop of M @ p over {len(GRID)} points == GRID @ M.T: OK")
wrong = GRID @ M  # forgetting the transpose: runs, gives a rotation the other way
print("GRID @ M (no transpose) equals rotating by -30 degrees:", np.allclose(wrong, GRID @ rotation(-30.0).T))
lengths_before = np.linalg.norm(GRID, axis=1)
lengths_after = np.linalg.norm(batch, axis=1)
print("rotation keeps every length:", np.allclose(lengths_before, lengths_after))

# %%
# The shape rule (n, d) @ (d, k) -> (n, k), as a checker that explains failures.
section("shape checker")
Shape = tuple[int, ...]


def check_shapes(*operands: Shape | FloatArray) -> Shape:
    """Predict the shape of operands[0] @ operands[1] @ ...; raise ValueError explaining a mismatch.

    Follows NumPy's matmul rules for 1-D and 2-D operands: a 1-D operand on the left acts as a
    row (1, d), on the right as a column (d, 1), and that temporary axis is dropped afterwards.
    """
    shapes = [tuple(op.shape) if hasattr(op, "shape") else tuple(op) for op in operands]
    if len(shapes) < 2:
        raise ValueError("need at least two operands")
    if any(len(s) not in (1, 2) for s in shapes):
        raise ValueError(f"only 1-D and 2-D operands are supported, got {shapes}")
    current = shapes[0]
    for step, nxt in enumerate(shapes[1:], start=1):
        left = (1, current[0]) if len(current) == 1 else current
        right = (nxt[0], 1) if len(nxt) == 1 else nxt
        if left[1] != right[0]:
            raise ValueError(
                f"product {step}: {current} @ {nxt}: inner sizes differ "
                f"({left[1]} columns on the left, {right[0]} rows on the right)"
            )
        out = (left[0], right[1])
        if len(current) == 1:
            out = out[1:]  # drop the temporary row axis
        if len(nxt) == 1:
            out = out[:-1]  # drop the temporary column axis
        current = out
    return current


CASES: list[list[Shape]] = [
    [(100, 3), (3, 2)],  # data (n, d) times weights (d, k)
    [(5, 3), (3, 4), (4, 2)],  # a chain: two layers without non-linearities
    [(3,), (3, 4)],  # one example as a 1-D vector
    [(5, 3), (3,)],  # a weight vector: one prediction per row
    [(3,), (3,)],  # two vectors: the dot product, a scalar
    [(5, 3), (5, 2)],  # forgot a transpose
    [(5, 3), (3, 4), (2, 4)],  # the second product fails
]
for case in CASES:
    try:
        predicted: Shape | str = check_shapes(*case)
    except ValueError as err:
        predicted = f"error: {err}"
    try:
        actual: Shape | str = reduce(np.matmul, [np.ones(s) for s in case]).shape
    except ValueError:
        actual = "error"
    agree = (isinstance(predicted, str) and actual == "error") or predicted == actual
    print(f"{' @ '.join(str(s) for s in case):<28} -> {predicted}   numpy agrees: {agree}")
    assert agree, (case, predicted, actual)

# %%
# With NumPy: @, np.matmul and np.einsum are the same product.
section("with numpy: @, matmul, einsum")
X_big = rng.normal(size=(N_RANDOM, 2))  # (N_RANDOM, 2)
M = MAPS["shear k=1"]
by_at = X_big @ M.T
by_matmul = np.matmul(X_big, M.T)
by_einsum = np.einsum("ij,nj->ni", M, X_big)  # out[n, i] = sum_j M[i, j] X[n, j]
mine = np.array([matvec_columns(M.tolist(), p.tolist()) for p in X_big[:200]])  # slow, so 200 rows
np.testing.assert_allclose(by_at, by_matmul, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(by_at, by_einsum, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(mine, by_at[:200], rtol=1e-6, atol=1e-8)
print(f"{N_RANDOM} points: X @ M.T == np.matmul == np.einsum: OK")
print("from-scratch column view (first 200 points) matches numpy: OK")

# %%
# Determinant = area scale factor: transform the unit square and measure its area (shoelace formula).
section("determinant as area")


def polygon_area(P: FloatArray) -> float:
    """Signed area of a closed polygon P (k, 2), first row repeated last (shoelace formula)."""
    x_, y_ = P[:, 0], P[:, 1]
    return float(0.5 * np.sum(x_[:-1] * y_[1:] - x_[1:] * y_[:-1]))


for name, M in [("A (worked example)", A), *MAPS.items()]:
    area = polygon_area(SQUARE @ M.T)
    print(f"{name:<20} area of image of unit square = {area + 0.0:.4f}   det = {det2(M) + 0.0:.4f}")
    np.testing.assert_allclose(area, det2(M), rtol=1e-6, atol=1e-8)
print("area == det for every map: OK")

# %%
# Composition: doing one map after another. (A2.2 turns this into matrix multiplication.)
section("composition order matters")
R, S = MAPS["rotate 90"], MAPS["shear k=1"]
print("shear then rotate:", show(R @ (S @ x)), "  rotate then shear:", show(S @ (R @ x)))

# %%
# Key plot: each map applied to a grid and to the unit square.
section("key plot")
panels = ["rotate 30", "scale (1.5, 0.5)", "shear k=1", "project on (1, 1)"]
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=120)
for ax, name in zip(axes.ravel(), panels, strict=True):
    M = MAPS[name]
    moved = GRID @ M.T  # (n, 2)
    ax.scatter(GRID[:, 0], GRID[:, 1], s=4, color="0.75", marker="o", label="grid before")
    ax.scatter(moved[:, 0], moved[:, 1], s=6, color="tab:blue", marker="x", label="grid after")
    ax.plot(SQUARE[:, 0], SQUARE[:, 1], "k--", lw=1.2, label="unit square before")
    sq = SQUARE @ M.T
    ax.fill(sq[:, 0], sq[:, 1], color="tab:orange", alpha=0.35, label="unit square after")
    for col, style, lab in [(M[:, 0], "-", "M e1 (column 1)"), (M[:, 1], ":", "M e2 (column 2)")]:
        ax.annotate("", xy=col, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2, "linestyle": style, "color": "tab:red"})
        ax.plot([], [], color="tab:red", linestyle=style, label=lab)
    ax.set_title(f"{name}: det = {det2(M) + 0.0:.2f}")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x1 (unitless)")
    ax.set_ylabel("x2 (unitless)")
    ax.grid(True, lw=0.3)
axes[0, 0].legend(loc="lower left", fontsize=7)
fig.suptitle("A matrix moves every point; its columns are where e1 and e2 land")
fig.tight_layout()
path = IMG / "a2_1_four_linear_maps.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Build the matrix that reflects across the x-axis from where it sends e1 and e2. What is its det?
# 2. Apply rotation(30) twelve times to x. Where do you end up, and why?
# 3. Make a 3x2 matrix and apply it to GRID. What shape comes out, and what does the picture lose?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
