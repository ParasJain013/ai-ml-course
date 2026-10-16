# %% [markdown]
# # A3.1 Eigenvalues and eigenvectors — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a3_1_eigenvalues_eigenvectors_ex2.py   (CPU, ~1 s)

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
A = np.array([[5.0, 2.0], [2.0, 2.0]])
print("trace =", np.trace(A), " det =", round(np.linalg.det(A), 10), " eigh values =", show(np.linalg.eigh(A)[0]))
for v, lam in ((np.array([2.0, 1.0]), 6.0), (np.array([1.0, -2.0]), 1.0)):
    print(f"A @ {show(v)} = {show(A @ v)} = {lam:g} * {show(v)}:", np.allclose(A @ v, lam * v))
x = np.array([1.0, 0.0])
for k in range(1, 4):
    x_next = A @ x
    rq = (x_next @ A @ x_next) / (x_next @ x_next)
    print(f"A^{k} x0 = {show(x_next):<16} Rayleigh quotient = {rq:.5f}")
    x = x_next
print("eigenvalues of A^2:", show(np.linalg.eigvalsh(A @ A)), " A + 3I:", show(np.linalg.eigvalsh(A + 3 * np.eye(2))),
      " A^-1:", show(np.linalg.eigvalsh(np.linalg.inv(A))))

# %%
section("exercise 2: inverse iteration")


def inverse_iteration(
    M: FloatArray, shift: float, tol: float = 1e-12, max_iters: int = 10_000, seed: int = SEED
) -> tuple[float, FloatArray, int]:
    """Eigenpair of symmetric M closest to shift. M (d, d) -> (eigenvalue, unit eigenvector (d,), iterations)."""
    d = M.shape[0]
    shifted = M - shift * np.eye(d)  # (d, d); eigenvalues lambda_j - shift, same eigenvectors
    x = np.random.default_rng(seed).normal(size=d)
    x /= np.linalg.norm(x)
    for k in range(1, max_iters + 1):
        y = np.linalg.solve(shifted, x)  # y = shifted^-1 x, without forming the inverse
        y /= np.linalg.norm(y)
        if 1.0 - abs(y @ x) < tol:
            x = y
            break
        x = y
    else:
        raise RuntimeError(f"inverse iteration did not converge in {max_iters} iterations")
    return float(x @ M @ x), x, k  # Rayleigh quotient on the ORIGINAL matrix gives lambda itself


G = rng.normal(size=(6, 6))
S = (G + G.T) / 2  # (6, 6), the same matrix as the lab
vals, vecs = np.linalg.eigh(S)
print("eigh eigenvalues (ascending):", show(vals))
for shift in (0.0, -1.0, 1.4, 1.0):
    lam, v, iters = inverse_iteration(S, shift)
    j = np.argmin(np.abs(vals - shift))
    np.testing.assert_allclose(lam, vals[j], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(v * np.sign(v @ vecs[:, j]), vecs[:, j], rtol=1e-5, atol=1e-6)
    dist = np.sort(np.abs(vals - shift))  # distances of every eigenvalue to the shift
    ratio = dist[0] / dist[1]  # |lambda2 / lambda1| of the inverted matrix
    print(f"shift {shift:>5}: lambda = {lam:.4f}  iterations = {iters:>4}  "
          f"ratio for the inverted matrix = {ratio:.3f}  matches eigh: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
