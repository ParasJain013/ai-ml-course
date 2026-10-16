# %% [markdown]
# # A2.2 Matrix multiplication and inverse — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a2_2_matrix_multiplication_inverse_ex2.py   (CPU, ~2 s)

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
P = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])  # (2, 3)
Q = np.array([[2.0, 1.0], [0.0, 1.0], [1.0, 0.0]])  # (3, 2)
PQ, QP = P @ Q, Q @ P
print("PQ =", show(PQ), PQ.shape, "  QP =", show(QP), QP.shape)
print("Q^T P^T =", show(Q.T @ P.T), " == (PQ)^T:", np.array_equal(Q.T @ P.T, PQ.T))
det = PQ[0, 0] * PQ[1, 1] - PQ[0, 1] * PQ[1, 0]
PQ_inv = np.array([[PQ[1, 1], -PQ[0, 1]], [-PQ[1, 0], PQ[0, 0]]]) / det
print("det(PQ) =", det, "  (PQ)^-1 =", show(PQ_inv), "  PQ (PQ)^-1 =", show(PQ @ PQ_inv))
print("rank(QP) =", np.linalg.matrix_rank(QP), " det(QP) =", round(np.linalg.det(QP), 10) + 0.0)

# %%
section("exercise 2: matpow by repeated squaring")


# from labs/A/a2_2_matrix_multiplication_inverse.py
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


def matpow(M: FloatArray, k: int) -> tuple[FloatArray, int]:
    """M^k for square M (m, m) and k >= 0 by repeated squaring. Returns (M^k, number of products)."""
    m = M.shape[0]
    if M.shape != (m, m):
        raise ValueError(f"matrix powers need a square matrix, got {M.shape}")
    if k < 0:
        raise ValueError("k must be >= 0 (negative powers need the inverse)")
    result = np.eye(m)
    base = M.astype(float)
    products = 0
    first = True  # multiplying the identity by base is free: just copy
    while k > 0:
        if k & 1:  # this bit of k is set: include the current power of M
            if first:
                result, first = base.copy(), False
            else:
                result = matmul_loops(result, base)
                products += 1
        k >>= 1
        if k:  # square only if there are bits left
            base = matmul_loops(base, base)
            products += 1
    return result, products


M = rng.normal(size=(4, 4))
for k in [0, 1, 2, 5, 8, 13, 64]:
    mine, count = matpow(M, k)
    np.testing.assert_allclose(mine, np.linalg.matrix_power(M, k), rtol=1e-6, atol=1e-8)
    print(f"k = {k:>2}: matches np.linalg.matrix_power: OK   products: {count:>2} (naive: {max(k - 1, 0)})")

F = np.array([[1.0, 1.0], [1.0, 0.0]])
F30, _ = matpow(F, 30)
print("F^30 =", show(F30))
print("F_30 = top-right entry =", int(F30[0, 1]))

# %%
section("exercise 2: k = 100 in float64")
F100, _ = matpow(F, 100)
exact = [0, 1]
for _ in range(99):
    exact.append(exact[-1] + exact[-2])
f100_exact = exact[100]  # Python ints never overflow
print("F_100 exact (Python int) =", f100_exact)
print("F_100 from float64       =", int(F100[0, 1]))
print("difference               =", int(F100[0, 1]) - f100_exact)
print("2**53 =", 2**53, " (largest range where float64 holds every integer exactly)")
try:
    with np.errstate(over="raise"):
        Fi = np.array([[1, 1], [1, 0]], dtype=np.int64)
        print("int64 F^100 top-right =", np.linalg.matrix_power(Fi, 100)[0, 1], "(wrapped around: wrong)")
except FloatingPointError as err:
    print("int64:", err)

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
