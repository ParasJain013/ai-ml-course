# %% [markdown]
# # A3.3 Covariance matrices — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a3_3_covariance_matrices_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import time
from collections.abc import Iterator

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
X3 = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 6.0]])
Xc = X3 - X3.mean(axis=0)
S = Xc.T @ Xc / 2
print("means =", show(X3.mean(axis=0)), " centered =", show(Xc))
print("covariance (n - 1) =", show(S), "  (n) =", show(Xc.T @ Xc / 3))
print("correlation =", round(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]), 4))
w = np.array([1.0, 1.0]) / np.sqrt(2)
print("projections onto w =", show(Xc @ w), " variance =", round((Xc @ w) @ (Xc @ w) / 2, 4), " w^T S w =", round(w @ S @ w, 4))
print("eigenvalues of S =", show(np.linalg.eigvalsh(S)), " (4 -/+ sqrt 13 =", show(4 + np.array([-1, 1]) * np.sqrt(13)), ")")

# %%
section("exercise 2: covariance in one pass over chunks")
N_ROWS, CHUNK = 100_000, 10_000
COV_TRUE = np.array([[1.0, 0.6, 0.0], [0.6, 2.0, -0.5], [0.0, -0.5, 0.5]])
L = np.linalg.cholesky(COV_TRUE)
Z = rng.normal(size=(N_ROWS, 3)) @ L.T  # (n, 3), covariance COV_TRUE, mean 0


def chunks(X: FloatArray, size: int) -> Iterator[FloatArray]:
    """Yield consecutive row blocks of X, as if read from a file. X (n, d) -> blocks (<= size, d)."""
    for start in range(0, X.shape[0], size):
        yield X[start : start + size]


def cov_streaming_naive(blocks: Iterator[FloatArray]) -> FloatArray:
    """One pass: sum of x and of x x^T, then S = (sum xx^T - n mean mean^T) / (n - 1). -> (d, d)."""
    n, sx, sxx = 0, None, None
    for B in blocks:
        n += B.shape[0]
        sx = B.sum(axis=0) if sx is None else sx + B.sum(axis=0)  # (d,)
        sxx = B.T @ B if sxx is None else sxx + B.T @ B  # (d, d)
    mean = sx / n
    return (sxx - n * np.outer(mean, mean)) / (n - 1)


def cov_streaming_shifted(blocks: Iterator[FloatArray]) -> FloatArray:
    """Same, after subtracting a shift K (the first row seen); covariance does not change under a shift."""
    n, sx, sxx, K = 0, None, None, None
    for B in blocks:
        if K is None:
            K = B[0].copy()  # (d,), close to the mean in scale, so the sums stay small
        Bk = B - K
        n += Bk.shape[0]
        sx = Bk.sum(axis=0) if sx is None else sx + Bk.sum(axis=0)
        sxx = Bk.T @ Bk if sxx is None else sxx + Bk.T @ Bk
    mean_k = sx / n
    return (sxx - n * np.outer(mean_k, mean_k)) / (n - 1)


for offset in (0.0, 1e4, 1e8):
    X = Z + offset  # same spread, shifted far from 0 (think: timestamps, sensor counts, prices in paise)
    ref = np.cov(X, rowvar=False)
    naive = cov_streaming_naive(chunks(X, CHUNK))
    shifted = cov_streaming_shifted(chunks(X, CHUNK))
    err_naive = np.abs(naive - ref).max()
    err_shift = np.abs(shifted - ref).max()
    print(f"offset {offset:>8.0e}: max |error| naive = {err_naive:.1e}   shifted = {err_shift:.1e}")
    if offset == 1e8:
        print("   naive S =", show(naive))
        print("   true  S =", show(ref), "  naive eigenvalues =", show(np.linalg.eigvalsh(naive)))
np.testing.assert_allclose(cov_streaming_shifted(chunks(Z + 1e8, CHUNK)), np.cov(Z + 1e8, rowvar=False), rtol=1e-6, atol=1e-6)
print("shifted one-pass covariance matches np.cov at every offset: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
