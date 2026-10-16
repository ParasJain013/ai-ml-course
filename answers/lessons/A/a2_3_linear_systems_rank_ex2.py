# %% [markdown]
# # A2.3 Linear systems and rank — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a2_3_linear_systems_rank_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy.linalg import null_space

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
A1 = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
b1 = np.array([8.0, -11.0, -3.0])
print("x =", show(np.linalg.solve(A1, b1)))
M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
print("rank(M) =", np.linalg.matrix_rank(M))
for b in (np.array([1.0, 2.0, 0.0]), np.array([1.0, 3.0, 0.0])):
    print(f"b = {show(b)}: rank([M|b]) = {np.linalg.matrix_rank(np.column_stack([M, b]))}")
x_p, n_vec = np.array([0.0, 0.5, 0.0]), np.array([-1.0, -1.0, 1.0])
for t in (0.0, 1.0, -2.5):
    print(f"  t = {t:4}: M (x_p + t n) = {show(M @ (x_p + t * n_vec))}")

# %%
section("exercise 2: solve_general")


def rref(aug: FloatArray, tol: float = 1e-10) -> tuple[FloatArray, list[int]]:
    """Reduced row echelon form with partial pivoting. (m, c) -> ((m, c), pivot columns)."""
    R = aug.astype(float).copy()
    m, c = R.shape
    pivots: list[int] = []
    row = 0
    for col in range(c):
        if row == m:
            break
        best = row + int(np.argmax(np.abs(R[row:, col])))
        if abs(R[best, col]) < tol:
            continue
        R[[row, best]] = R[[best, row]]
        R[row] /= R[row, col]  # pivot becomes 1
        for r in range(m):  # clear above AND below
            if r != row:
                R[r] -= R[r, col] * R[row]
        pivots.append(col)
        row += 1
    R[np.abs(R) < tol] = 0.0
    return R, pivots


def solve_general(A: FloatArray, b: FloatArray, tol: float = 1e-10) -> tuple[FloatArray, FloatArray] | None:
    """All solutions of A x = b as (x_p (d,), N (d, d - rank)), or None if there are none. A (m, d), b (m,)."""
    m, d = A.shape
    R, pivots = rref(np.column_stack([A, b]), tol=tol)
    if d in pivots:  # a pivot in the b column: a row reads 0 = non-zero
        return None
    free = [j for j in range(d) if j not in pivots]
    x_p = np.zeros(d)
    for i, pc in enumerate(pivots):  # free variables = 0: each pivot variable equals its row's b entry
        x_p[pc] = R[i, d]
    N = np.zeros((d, len(free)))
    for k, fc in enumerate(free):  # set one free variable to 1, the others to 0, b to 0
        N[fc, k] = 1.0
        for i, pc in enumerate(pivots):
            N[pc, k] = -R[i, fc]
    return x_p, N


A_s = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
CASES = [
    ("A_s, b_inf", A_s, np.array([6.0, 14.0, 20.0])),
    ("A_s, b_none", A_s, np.array([6.0, 14.0, 21.0])),
]
wide = rng.normal(size=(4, 6))  # (4, 6): rank 4, so 2 free variables
CASES.append(("random (4, 6)", wide, rng.normal(size=4)))
tall = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 4))  # (6, 4) with rank 2
CASES.append(("rank-2 (6, 4), b in column space", tall, tall @ rng.normal(size=4)))
CASES.append(("rank-2 (6, 4), random b", tall, rng.normal(size=6)))

for name, A, b in CASES:
    result = solve_general(A, b)
    if result is None:
        print(f"{name}: no solution   (np rank(A) = {np.linalg.matrix_rank(A)}, rank([A|b]) = {np.linalg.matrix_rank(np.column_stack([A, b]))})")
        continue
    x_p, N = result
    for _ in range(5):
        t = rng.normal(size=N.shape[1])
        np.testing.assert_allclose(A @ (x_p + N @ t), b, rtol=1e-6, atol=1e-8)
    assert N.shape[1] == null_space(A).shape[1], (N.shape, null_space(A).shape)
    print(f"{name}: x_p = {show(x_p)}  N {N.shape}  A(x_p + N t) == b for random t: OK  null-space dim matches scipy: OK")
    if name == "A_s, b_inf":
        print("  N =", show(N))

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
