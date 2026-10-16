# %% [markdown]
# # A3.4 Numerical stability — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a3_4_numerical_stability_ex2.py   (CPU, ~3 s, mostly importing scikit-learn)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# from labs/A/a3_4_numerical_stability.py
def cholesky(C: FloatArray) -> FloatArray:
    """Lower-triangular L with L L^T = C. C (d, d) symmetric positive definite -> L (d, d)."""
    d = C.shape[0]
    L = np.zeros((d, d))
    for j in range(d):
        diag = C[j, j] - L[j, :j] @ L[j, :j]
        if diag <= 0:
            raise np.linalg.LinAlgError(f"not positive definite (pivot {j} is {diag:.3g})")
        L[j, j] = np.sqrt(diag)
        L[j + 1 :, j] = (C[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def forward_substitution(L: FloatArray, b: FloatArray) -> FloatArray:
    """Solve L y = b for lower-triangular L. L (d, d), b (d,) -> (d,)."""
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_substitution(U: FloatArray, y: FloatArray) -> FloatArray:
    """Solve U x = y for upper-triangular U. U (d, d), y (d,) -> (d,)."""
    x = np.zeros_like(y)
    for i in range(len(y) - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


# %%
section("exercise 1 check")
A = np.array([[1.0, 4.0], [2.0, 2.0], [2.0, 5.0]])
Q, R = np.linalg.qr(A)
signs = np.sign(np.diag(R))
Q, R = Q * signs, R * signs[:, None]
print("Q =", show(Q), " R =", show(R), "  (Q * 3 =", show(Q * 3), ")")
y = np.ones(3)
w = back_substitution(R, Q.T @ y)
print("Q^T y =", show(Q.T @ y), " w =", show(w), " normal equations give", show(np.linalg.solve(A.T @ A, A.T @ y)))
print("cholesky([[9, 3], [3, 5]]) =", show(cholesky(np.array([[9.0, 3.0], [3.0, 5.0]]))))
try:
    cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
except np.linalg.LinAlgError as err:
    print("cholesky([[1, 2], [2, 1]]):", err)
print("cond(diag(100, 0.01)) =", round(np.linalg.cond(np.diag([100.0, 0.01])), 4))

# %%
section("exercise 2: ridge regression by Cholesky")
N, D = 100, 5
X = rng.normal(size=(N, D))  # (100, 5)
X[:, 1] = X[:, 0] + 1e-4 * rng.normal(size=N)  # column 1 is column 0 plus a tiny bit of noise
w_true = np.array([1.0, 1.0, -2.0, 0.5, 3.0])
y = X @ w_true + 0.1 * rng.normal(size=N)  # (100,)
print(f"cond(X) = {np.linalg.cond(X):.2e}   cond(X^T X) = {np.linalg.cond(X.T @ X):.2e}")


def ridge_cholesky(X: FloatArray, y: FloatArray, lam: float) -> FloatArray:
    """Minimize |y - X w|^2 + lam |w|^2 by Cholesky on X^T X + lam I. X (n, d), y (n,) -> w (d,)."""
    A = X.T @ X + lam * np.eye(X.shape[1])  # (d, d), symmetric positive definite for lam > 0
    L = cholesky(A)
    return back_substitution(L.T, forward_substitution(L, X.T @ y))


print(f"{'lambda':>8} {'cond(X^T X + lam I)':>20}  {'w0':>9} {'w1':>9} {'w0 + w1':>8}  matches sklearn")
for lam in (1e-8, 1e-4, 1e-2, 1.0, 10.0):
    w = ridge_cholesky(X, y, lam)
    w_sk = Ridge(alpha=lam, fit_intercept=False, solver="cholesky").fit(X, y).coef_
    ok = np.allclose(w, w_sk, rtol=1e-6, atol=1e-8)
    cond = np.linalg.cond(X.T @ X + lam * np.eye(D))
    print(f"{lam:8.0e} {cond:20.2e}  {w[0]:9.4f} {w[1]:9.4f} {w[0] + w[1]:8.4f}  {ok}")
w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
print("plain least squares (lstsq): w0, w1 =", show(w_ls[:2]), " w0 + w1 =", round(w_ls[0] + w_ls[1], 4))
print("other weights at lambda = 1e-2:", show(ridge_cholesky(X, y, 1e-2)[2:]), " true:", show(w_true[2:]))

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
