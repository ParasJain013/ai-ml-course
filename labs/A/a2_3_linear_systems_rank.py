# %% [markdown]
# # A2.3 Systems of linear equations and rank — lab
# Lesson: lessons/A/a2_3_linear_systems_rank.md
# Run: uv run python labs/A/a2_3_linear_systems_rank.py   (CPU, defaults: ~2 s)
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
OUT = ROOT / "outputs" / "a2_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


def show_aug(aug: FloatArray) -> str:
    """Format an augmented matrix [A | b] (m, d + 1) as m lines with a bar before the last column."""
    lines = []
    for row in np.round(aug, 4) + 0.0:
        left = " ".join(f"{v:7g}" for v in row[:-1])
        lines.append(f"    [{left} | {row[-1]:7g} ]")
    return "\n".join(lines)


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the knobs size the random systems in the agreement checks.
GPU_SCALE = False
N_RANDOM_SYSTEMS = 200 if GPU_SCALE else 50  # random square systems checked against np.linalg.solve
SIZE = 100 if GPU_SCALE else 20  # their size

# %%
# Data: three 3x3 systems with one, no and infinitely many solutions.
section("data")
A_u = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 5.0], [2.0, 5.0, -1.0]])  # (3, 3), independent rows
b_u = np.array([6.0, -4.0, 27.0])  # (3,)
A_s = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])  # (3, 3), row 3 = row 1 + row 2
b_inf = np.array([6.0, 14.0, 20.0])  # 20 = 6 + 14: consistent
b_none = np.array([6.0, 14.0, 21.0])  # 21 != 6 + 14: contradiction
print("A_u =", show(A_u), "  b_u =", show(b_u))
print("A_s =", show(A_s), "  b_inf =", show(b_inf), "  b_none =", show(b_none))

# %%
# Worked example by hand: Gaussian elimination with partial pivoting on [A_u | b_u].
section("worked example: elimination")


def row_echelon(aug: FloatArray, verbose: bool = False, tol: float = 1e-10) -> tuple[FloatArray, list[int]]:
    """Reduce a matrix to row echelon form with partial pivoting. (m, c) -> ((m, c), pivot columns).

    Works column by column: pick the row with the largest absolute entry as the pivot, swap it up,
    and subtract multiples of it from every row below so the column is zero under the pivot.
    """
    R = aug.astype(float).copy()
    m, c = R.shape
    pivots: list[int] = []
    row = 0
    for col in range(c):
        if row == m:
            break
        best = row + int(np.argmax(np.abs(R[row:, col])))
        if abs(R[best, col]) < tol:
            continue  # no pivot in this column: it is a free column
        if best != row:
            R[[row, best]] = R[[best, row]]
            if verbose:
                print(f"swap R{row + 1} and R{best + 1}\n{show_aug(R)}")
        for r in range(row + 1, m):
            factor = R[r, col] / R[row, col]
            if factor != 0.0:
                R[r] -= factor * R[row]
                R[r, col] = 0.0  # exactly zero, not 1e-17
                if verbose:
                    print(f"R{r + 1} <- R{r + 1} - ({factor:g}) * R{row + 1}\n{show_aug(R)}")
        pivots.append(col)
        row += 1
    return R, pivots


def back_substitute(U: FloatArray, c: FloatArray) -> FloatArray:
    """Solve U x = c for upper-triangular U (d, d) with non-zero diagonal, from the last row up. -> (d,)."""
    d = U.shape[0]
    x = np.zeros(d)
    for i in range(d - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


aug_u = np.column_stack([A_u, b_u])  # (3, 4): the augmented matrix [A | b]
print(f"start: [A_u | b_u]\n{show_aug(aug_u)}")
R_u, piv_u = row_echelon(aug_u, verbose=True)
x_u = back_substitute(R_u[:, :3], R_u[:, 3])
print("pivot columns:", piv_u, "  back substitution: x =", show(x_u))
print("check A_u @ x =", show(A_u @ x_u), "== b_u:", np.allclose(A_u @ x_u, b_u))

# %%
# From scratch: solve, rank, and classification of all three systems.
section("from scratch: rank and classification")


def rank(A: FloatArray, tol: float = 1e-10) -> int:
    """Rank = number of pivots after row reduction. A (m, d) -> int."""
    _, pivots = row_echelon(A, tol=tol)
    return len(pivots)


def solve(A: FloatArray, b: FloatArray) -> FloatArray:
    """Solve the square system A x = b by elimination and back substitution. A (d, d), b (d,) -> (d,)."""
    d = A.shape[0]
    if A.shape != (d, d):
        raise ValueError(f"solve needs a square matrix, got {A.shape}")
    R, pivots = row_echelon(np.column_stack([A, b]))
    if pivots[:d] != list(range(d)):
        raise ValueError(f"A is singular (rank {rank(A)} < {d}): no unique solution")
    return back_substitute(R[:, :d], R[:, d])


def classify(A: FloatArray, b: FloatArray) -> str:
    """Count the solutions of A x = b from rank(A), rank([A | b]) and d."""
    r_A, r_Ab, d = rank(A), rank(np.column_stack([A, b])), A.shape[1]
    if r_Ab > r_A:
        verdict = "no solution (b is not reachable)"
    elif r_A == d:
        verdict = "exactly one solution"
    else:
        verdict = f"infinitely many solutions ({d - r_A} free variable{'s' if d - r_A > 1 else ''})"
    return f"rank(A) = {r_A}, rank([A|b]) = {r_Ab}, d = {d}: {verdict}"


for name, A_, b_ in [("unique", A_u, b_u), ("infinite", A_s, b_inf), ("none", A_s, b_none)]:
    print(f"{name:<8} {classify(A_, b_)}")
print(f"elimination of [A_s | b_none]:\n{show_aug(row_echelon(np.column_stack([A_s, b_none]))[0])}")
print("last row says 0 x1 + 0 x2 + 0 x3 = -1: impossible")
print(f"elimination of [A_s | b_inf]:\n{show_aug(row_echelon(np.column_stack([A_s, b_inf]))[0])}")
print("last row says 0 = 0: one equation was redundant")
for x_try in (np.array([-2.0, 8.0, 0.0]), np.array([-1.0, 6.0, 1.0]), np.array([0.0, 4.0, 2.0])):
    print("  candidate", show(x_try), " A_s @ x =", show(A_s @ x_try), " solves it:", np.allclose(A_s @ x_try, b_inf))
try:
    solve(A_s, b_inf)
except ValueError as err:
    print("solve(A_s, b_inf):", err)

# %%
# Why partial pivoting: a tiny pivot destroys the answer in floating point.
section("why pivot")
A_tiny = np.array([[1e-20, 1.0], [1.0, 1.0]])  # (2, 2)
b_tiny = np.array([1.0, 2.0])  # true solution is x = (1.0000..., 0.9999...), both about 1


def solve_no_pivot(A: FloatArray, b: FloatArray) -> FloatArray:
    """Gaussian elimination WITHOUT row swaps, for comparison only. A (d, d), b (d,) -> (d,)."""
    U = np.column_stack([A, b]).astype(float)
    d = A.shape[0]
    for col in range(d):
        for r in range(col + 1, d):
            U[r] -= (U[r, col] / U[col, col]) * U[col]
    return back_substitute(U[:, :d], U[:, d])


print("without pivoting x =", show(solve_no_pivot(A_tiny, b_tiny)))
print("with pivoting    x =", show(solve(A_tiny, b_tiny)))
print("np.linalg.solve  x =", show(np.linalg.solve(A_tiny, b_tiny)))

# %%
# With NumPy: solve, matrix_rank and lstsq.
section("with numpy")
np.testing.assert_allclose(solve(A_u, b_u), np.linalg.solve(A_u, b_u), rtol=1e-6, atol=1e-8)
for _ in range(N_RANDOM_SYSTEMS):
    M = rng.normal(size=(SIZE, SIZE))
    v = rng.normal(size=SIZE)
    np.testing.assert_allclose(solve(M, v), np.linalg.solve(M, v), rtol=1e-6, atol=1e-8)
print(f"solve matches np.linalg.solve on the worked example and {N_RANDOM_SYSTEMS} random {SIZE}x{SIZE} systems: OK")
LOW = rng.normal(size=(50, 4)) @ rng.normal(size=(4, 10))  # (50, 10) built to have rank 4
TALL = rng.normal(size=(50, 10))  # (50, 10), full column rank
for name, M in [("A_u", A_u), ("A_s", A_s), ("[A_s | b_none]", np.column_stack([A_s, b_none])), ("low-rank (50, 10)", LOW), ("random (50, 10)", TALL)]:
    mine, theirs = rank(M), int(np.linalg.matrix_rank(M))
    assert mine == theirs, (name, mine, theirs)
    print(f"rank {name:<18} mine {mine}   np.linalg.matrix_rank {theirs}")
try:
    np.linalg.solve(A_s, b_inf)
except np.linalg.LinAlgError as err:
    print("np.linalg.solve(A_s, b_inf) raises LinAlgError:", err, "(even though solutions exist)")
x_ls, residual, r_ls, _ = np.linalg.lstsq(A_s, b_inf, rcond=None)
print("np.linalg.lstsq(A_s, b_inf): x =", show(x_ls), " rank", r_ls, " A_s @ x =", show(A_s @ x_ls))
x_ls2, *_ = np.linalg.lstsq(A_s, b_none, rcond=None)
print("np.linalg.lstsq(A_s, b_none): x =", show(x_ls2), " A_s @ x =", show(A_s @ x_ls2), "(closest reachable b, see A2.4)")

# %%
# Key plot: the row picture of three 2x2 systems.
section("key plot")
systems = [
    ("one solution", np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([3.0, 1.0])),
    ("no solution", np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([3.0, 1.0])),
    ("infinitely many", np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([3.0, 6.0])),
]
xs = np.linspace(-1.0, 4.0, 50)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=120, sharex=True, sharey=True)
for ax, (title, M, v) in zip(axes, systems, strict=True):
    for i, style in enumerate(["-", "--"]):
        a1, a2 = M[i]
        ax.plot(xs, (v[i] - a1 * xs) / a2, linestyle=style, lw=2 + 1.5 * (1 - i), label=f"{a1:g} x1 + {a2:g} x2 = {v[i]:g}")
    r_A, r_Ab = rank(M), rank(np.column_stack([M, v]))
    if r_A == r_Ab == 2:
        sol = solve(M, v)
        ax.plot(*sol, "ko", ms=7, label=f"solution {show(sol)}")
    ax.set_title(f"{title}\nrank(A) = {r_A}, rank([A|b]) = {r_Ab}")
    ax.set_xlabel("x1 (unitless)")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.grid(True, lw=0.3)
    ax.legend(loc="upper right", fontsize=7)
axes[0].set_ylabel("x2 (unitless)")
fig.suptitle("Each equation is a line; the solutions are where all the lines meet")
fig.tight_layout()
path = IMG / "a2_3_three_systems.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Change b_none[2] from 21 to 20.000001. What do rank() and np.linalg.matrix_rank say? Why?
# 2. Make LOW with rank 4 then add rng.normal(scale=1e-12, size=LOW.shape). What is its rank now?
# 3. Count the operations of row_echelon for a (d, d) matrix. How does the cost grow with d?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
