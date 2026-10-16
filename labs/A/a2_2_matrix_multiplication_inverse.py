# %% [markdown]
# # A2.2 Matrix multiplication, transpose, identity and inverse — lab
# Lesson: lessons/A/a2_2_matrix_multiplication_inverse.md
# Run: uv run python labs/A/a2_2_matrix_multiplication_inverse.py   (CPU, defaults: ~3 s)
# GPU: no

# %%
from __future__ import annotations

import math
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
OUT = ROOT / "outputs" / "a2_2"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knobs size the random matrices in the checks and the timing.
GPU_SCALE = False
SHAPE = (60, 40, 30) if GPU_SCALE else (12, 8, 5)  # (n, d, k) for the agreement checks
TIME_SIZE = 200 if GPU_SCALE else 80  # square size for the loop-vs-@ timing

# %%
# Data: the small matrices of the worked example, plus random ones for the checks.
section("data")
A = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
B = np.array([[0.0, 1.0], [1.0, 0.0]])  # (2, 2): swaps the two coordinates
n, d, k = SHAPE
X_rand = rng.normal(size=(n, d))  # (n, d)
W_rand = rng.normal(size=(d, k))  # (d, k)
print("A =", show(A), "  B =", show(B))
print("random X", X_rand.shape, " random W", W_rand.shape)

# %%
# Worked example by hand: AB and BA, entry by entry.
section("worked example")
AB, BA = A @ B, B @ A
print("AB =", show(AB), "  (AB)[0, 0] = 1*0 + 2*1 =", A[0] @ B[:, 0])
print("BA =", show(BA))
print("AB == BA:", np.array_equal(AB, BA))
C = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])  # (2, 3)
D = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])  # (3, 2)
print("C (2, 3) @ D (3, 2) =", show(C @ D), " shape", (C @ D).shape)
print("D (3, 2) @ C (2, 3) =", show(D @ C), " shape", (D @ C).shape)

# %%
# From scratch: three ways to multiply, all checked against @.
section("from scratch: three ways to multiply")


def matmul_loops(A: FloatArray, B: FloatArray) -> FloatArray:
    """Triple loop: C[i, j] = sum_p A[i, p] B[p, j]. A (n, d), B (d, k) -> (n, k)."""
    n_, d_ = A.shape
    d2, k_ = B.shape
    if d_ != d2:
        raise ValueError(f"inner sizes differ: A is {A.shape}, B is {B.shape}")
    C = np.zeros((n_, k_))
    for i in range(n_):
        for j in range(k_):
            total = 0.0
            for p in range(d_):
                total += A[i, p] * B[p, j]
            C[i, j] = total
    return C


def matmul_dots(A: FloatArray, B: FloatArray) -> FloatArray:
    """Each entry is a dot product: (row i of A) . (column j of B). A (n, d), B (d, k) -> (n, k)."""
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"inner sizes differ: A is {A.shape}, B is {B.shape}")
    return np.array([[A[i] @ B[:, j] for j in range(B.shape[1])] for i in range(A.shape[0])])


def matmul_outer(A: FloatArray, B: FloatArray) -> FloatArray:
    """Sum of d outer products: (column p of A)(row p of B). A (n, d), B (d, k) -> (n, k)."""
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"inner sizes differ: A is {A.shape}, B is {B.shape}")
    C = np.zeros((A.shape[0], B.shape[1]))
    for p in range(A.shape[1]):
        C += np.outer(A[:, p], B[p])  # (n,) outer (k,) -> (n, k), a rank-1 matrix
    return C


reference = X_rand @ W_rand  # (n, k)
for fn in (matmul_loops, matmul_dots, matmul_outer):
    np.testing.assert_allclose(fn(X_rand, W_rand), reference, rtol=1e-6, atol=1e-8)
    print(f"{fn.__name__:<13} {X_rand.shape} @ {W_rand.shape} -> {fn(X_rand, W_rand).shape} matches @: OK")
print("worked example, loops:", show(matmul_loops(A, B)))
try:
    matmul_loops(X_rand, X_rand)
except ValueError as err:
    print("shape error caught:", err)

# %%
# Speed: the triple loop versus NumPy's compiled @ (timings vary, so the lines start with ~).
section("speed")
S1, S2 = rng.normal(size=(TIME_SIZE, TIME_SIZE)), rng.normal(size=(TIME_SIZE, TIME_SIZE))
t = time.perf_counter()
matmul_loops(S1, S2)
t_loop = time.perf_counter() - t
t = time.perf_counter()
for _ in range(100):
    S1 @ S2
t_at = (time.perf_counter() - t) / 100
print(f"multiplying two {TIME_SIZE} x {TIME_SIZE} matrices takes {TIME_SIZE**3:,} multiply-adds")
print(f"~ triple loop: {t_loop * 1000:.1f} ms   @: {t_at * 1000:.3f} ms   ratio ~{t_loop / t_at:,.0f}x")

# %%
# Transpose and identity.
section("transpose and identity")
print("A.T =", show(A.T))
np.testing.assert_allclose((X_rand @ W_rand).T, W_rand.T @ X_rand.T, rtol=1e-6, atol=1e-8)
print(f"(XW).T == W.T X.T: OK   shapes {(X_rand @ W_rand).T.shape} == {(W_rand.T @ X_rand.T).shape}")
print("X.T X is", (X_rand.T @ X_rand).shape, "and symmetric:", np.allclose(X_rand.T @ X_rand, (X_rand.T @ X_rand).T))
I2 = np.eye(2)
print("I =", show(I2), "  A I == A:", np.array_equal(A @ I2, A), "  I A == A:", np.array_equal(I2 @ A, A))

# %%
# The 2x2 inverse by formula: swap the diagonal, negate the off-diagonal, divide by det.
section("2x2 inverse")


def inverse_2x2(M: FloatArray) -> FloatArray:
    """Inverse of a 2x2 matrix [[a, b], [c, d]] = [[d, -b], [-c, a]] / (ad - bc)."""
    a, b, c, d_ = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d_ - b * c
    if math.isclose(det, 0.0, abs_tol=1e-12):
        raise ValueError(f"det = {det:g}: the matrix squashes the plane onto a line, no inverse")
    return np.array([[d_, -b], [-c, a]]) / det


A_inv = inverse_2x2(A)
print("det(A) = 1*4 - 2*3 =", A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
print("A^-1 =", show(A_inv))
print("A @ A^-1 =", show(A @ A_inv), "  A^-1 @ A =", show(A_inv @ A))
SING = np.array([[1.0, 2.0], [2.0, 4.0]])  # column 2 = 2 * column 1
try:
    inverse_2x2(SING)
except ValueError as err:
    print("singular [[1, 2], [2, 4]]:", err)

# %%
# Gauss-Jordan inversion of a 3x3 matrix, printing every step.
section("gauss-jordan on a 3x3")
A3 = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]])  # (3, 3); top-left is 0


def show_aug(aug: FloatArray) -> str:
    """Format an augmented matrix [L | R] (m, 2m) as m lines with a bar in the middle."""
    m = aug.shape[0]
    lines = []
    for row in np.round(aug, 4) + 0.0:
        left = " ".join(f"{v:5g}" for v in row[:m])
        right = " ".join(f"{v:5g}" for v in row[m:])
        lines.append(f"    [{left} |{right} ]")
    return "\n".join(lines)


def gauss_jordan_inverse(M: FloatArray, verbose: bool = False, tol: float = 1e-12) -> FloatArray:
    """Invert a square matrix by row-reducing [M | I] to [I | M^-1], with partial pivoting. (m, m) -> (m, m)."""
    m = M.shape[0]
    if M.shape != (m, m):
        raise ValueError(f"only square matrices have inverses, got {M.shape}")
    aug = np.hstack([M.astype(float), np.eye(m)])  # (m, 2m): the augmented matrix [M | I]
    for col in range(m):
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))  # largest entry at or below
        if abs(aug[pivot_row, col]) < tol:
            raise ValueError(f"column {col + 1} has no usable pivot: the matrix is singular")
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]  # swap two rows
            if verbose:
                print(f"swap R{col + 1} and R{pivot_row + 1}\n{show_aug(aug)}")
        pivot = aug[col, col]
        if pivot != 1.0:
            aug[col] = aug[col] / pivot  # make the pivot 1
            if verbose:
                print(f"R{col + 1} <- R{col + 1} / ({pivot + 0.0:g})\n{show_aug(aug)}")
        for r in range(m):
            if r != col and aug[r, col] != 0.0:
                factor = aug[r, col]
                aug[r] = aug[r] - factor * aug[col]  # clear the rest of the column
                if verbose:
                    print(f"R{r + 1} <- R{r + 1} - ({factor + 0.0:g}) * R{col + 1}\n{show_aug(aug)}")
    return aug[:, m:]  # the right half is now M^-1


print(f"start: [A3 | I]\n{show_aug(np.hstack([A3, np.eye(3)]))}")
A3_inv = gauss_jordan_inverse(A3, verbose=True)
print("A3^-1 =", show(A3_inv))
print("A3 @ A3^-1 =", show(A3 @ A3_inv))
np.testing.assert_allclose(A3 @ A3_inv, np.eye(3), rtol=1e-6, atol=1e-8)
print("A3 @ A3^-1 == I: OK")

# %%
# With NumPy: np.linalg.inv agrees; np.linalg.solve answers A x = b without forming the inverse.
section("with numpy")
np.testing.assert_allclose(gauss_jordan_inverse(A3), np.linalg.inv(A3), rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(inverse_2x2(A), np.linalg.inv(A), rtol=1e-6, atol=1e-8)
print("gauss_jordan_inverse and inverse_2x2 match np.linalg.inv: OK")
R_big = rng.normal(size=(50, 50))  # a random square matrix is invertible with probability 1
np.testing.assert_allclose(gauss_jordan_inverse(R_big), np.linalg.inv(R_big), rtol=1e-6, atol=1e-8)
print("random 50 x 50: matches np.linalg.inv: OK")
try:
    np.linalg.inv(SING)
except np.linalg.LinAlgError as err:
    print("np.linalg.inv on the singular matrix raises LinAlgError:", err)
b3 = np.array([3.0, 2.0, 4.0])
print("x = A3^-1 b =", show(A3_inv @ b3), "  np.linalg.solve(A3, b) =", show(np.linalg.solve(A3, b3)))
np.testing.assert_allclose(np.linalg.inv(A @ B), np.linalg.inv(B) @ np.linalg.inv(A), rtol=1e-6, atol=1e-8)
print("(AB)^-1 == B^-1 A^-1: OK")

# %%
# Composition: R @ S means "apply S first, then R". Order changes the result.
section("composition order")


def rotation(theta_deg: float) -> FloatArray:
    """Rotate counter-clockwise by theta degrees. -> (2, 2)."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


R = rotation(90.0)
S = np.array([[1.0, 1.0], [0.0, 1.0]])  # shear
print("R @ S (shear, then rotate) =", show(R @ S))
print("S @ R (rotate, then shear) =", show(S @ R))
x = np.array([1.0, 2.0])
print("(R @ S) @ x =", show((R @ S) @ x), " == R @ (S @ x) =", show(R @ (S @ x)))

# %%
# Key plot: the letter F under R @ S and under S @ R.
section("key plot")
F_SHAPE = np.array(
    [[0, 0], [0.25, 0], [0.25, 0.5], [0.6, 0.5], [0.6, 0.7], [0.25, 0.7], [0.25, 0.8], [0.8, 0.8], [0.8, 1.0], [0, 1.0], [0, 0]]
)  # (11, 2): a closed polygon, points as rows; an F shows flips and turns
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=120, sharex=True, sharey=True)
for ax, (title, M) in zip(axes, [("original", np.eye(2)), ("R @ S: shear, then rotate", R @ S), ("S @ R: rotate, then shear", S @ R)], strict=True):
    ax.fill(F_SHAPE[:, 0], F_SHAPE[:, 1], color="0.85", label="original F")
    moved = F_SHAPE @ M.T  # (11, 2)
    ax.fill(moved[:, 0], moved[:, 1], color="tab:blue", alpha=0.6, hatch="//", label="after the map")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.2, 2.2)
    ax.set_xlabel("x1 (unitless)")
    ax.grid(True, lw=0.3)
axes[0].set_ylabel("x2 (unitless)")
axes[1].legend(loc="lower right", fontsize=7)
fig.suptitle("Matrix products compose maps right to left, and order matters: RS ≠ SR")
fig.tight_layout()
path = IMG / "a2_2_composition_order.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Time matmul_dots and matmul_outer against matmul_loops. Which is fastest, and why?
# 2. Call gauss_jordan_inverse on [[1e-20, 1], [1, 1]] with the pivot search removed. What breaks?
# 3. Check that (A^T)^-1 == (A^-1)^T for A3. Can you explain it from (AB)^T = B^T A^T?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
