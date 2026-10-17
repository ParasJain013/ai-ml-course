# %% [markdown]
# # B1.5 Linear regression — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b1_5_linear_regression_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
MAX_ITERS, GRAD_TOL = 20_000, 1e-8
NEAR_COPY_SD = 0.01


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


def add_intercept(X: FloatArray) -> FloatArray:
    """Prepend a column of ones. (n, d) -> (n, d + 1)."""
    # from labs/B/b1_5_linear_regression.py
    return np.column_stack([np.ones(len(X)), X])


def gradient_descent(A: FloatArray, y: FloatArray, lr: float) -> tuple[FloatArray, int]:
    """Batch GD on the MSE from theta = 0. A (n, k), y (n,) -> (theta (k,), steps taken)."""
    # from labs/B/b1_5_linear_regression.py, without the loss history
    theta = np.zeros(A.shape[1])
    for k in range(1, MAX_ITERS + 1):
        g = 2.0 / len(y) * (A.T @ (A @ theta - y))
        if np.linalg.norm(g) < GRAD_TOL:
            return theta, k - 1
        theta = theta - lr * g
    return theta, MAX_ITERS


# %%
# Exercise 1 check.
section("exercise 1 check")
x5, y5 = np.array([0.0, 1, 2, 3, 4]), np.array([1.0, 3, 2, 5, 4])
dx, dy = x5 - x5.mean(), y5 - y5.mean()
slope = dx @ dy / (dx @ dx)
icpt = y5.mean() - slope * x5.mean()
res = y5 - (icpt + slope * x5)
print(f"(a) sum dx*dy = {dx @ dy}  sum dx^2 = {dx @ dx}  slope {slope:.4f}  intercept {icpt:.4f}")
print(f"    residuals {show(res)}  SSE {res @ res:.4f}  SST {dy @ dy:.4f}  R^2 {1 - res @ res / (dy @ dy):.4f}")
A5 = add_intercept(x5)
print(f"(b) A^T A = {show(A5.T @ A5)}  A^T y = {show(A5.T @ y5)}  solve -> {show(np.linalg.solve(A5.T @ A5, A5.T @ y5))}")
H5 = 2.0 / len(x5) * A5.T @ A5
lam5 = np.linalg.eigvalsh(H5)
print(f"(c) Hessian {show(H5)}  eigenvalues {show(lam5)} (7 -/+ sqrt(41))  condition number {lam5[1] / lam5[0]:.2f}  "
      f"max stable lr {2 / lam5[1]:.4f}")
sd5 = x5.std()
AZ5 = add_intercept((x5 - x5.mean()) / sd5)
g0 = 2.0 / len(x5) * AZ5.T @ (AZ5 @ np.zeros(2) - y5)
th1 = -0.5 * g0
g1 = 2.0 / len(x5) * AZ5.T @ (AZ5 @ th1 - y5)
print(f"(d) sd {sd5:.4f}  z {show(AZ5[:, 1])}  gradient at 0 {show(g0)}  after one step {show(th1)}  gradient there {show(g1)}")
print(f"    back to raw: slope {th1[1] / sd5:.4f}  intercept {th1[0] - th1[1] / sd5 * x5.mean():.4f}")

# %%
# Exercise 2: the lab's diabetes split, plus an exact or a near copy of bmi.
section("exercise 2: data")
diab = load_diabetes(as_frame=True, scaled=False)
X_all = diab.data.to_numpy(dtype=np.float64)  # (442, 10)
y_all = diab.target.to_numpy(dtype=np.float64)  # (442,)
BMI = list(diab.feature_names).index("bmi")
# the lab's split (labs/B/b1_5_linear_regression.py): same seed, same order of draws
split = np.random.default_rng(SEED)
rows = split.permutation(len(y_all))
n_test = n_val = round(0.2 * len(y_all))
test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]
i_tr, i_va = split.permutation(train), split.permutation(val)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
noise = np.random.default_rng(SEED)
near_tr = X_tr[:, BMI] + noise.normal(0, NEAR_COPY_SD, len(i_tr))
near_va = X_va[:, BMI] + noise.normal(0, NEAR_COPY_SD, len(i_va))
cases = {
    "(i) exact copy": (np.column_stack([X_tr, X_tr[:, BMI]]), np.column_stack([X_va, X_va[:, BMI]])),
    "(ii) near copy": (np.column_stack([X_tr, near_tr]), np.column_stack([X_va, near_va])),
}
base = LinearRegression().fit(X_tr, y_tr)
print(f"train {X_tr.shape}  val {X_va.shape}  without a copy: bmi weight {base.coef_[BMI]:.4f}  "
      f"val R^2 {r2_score(y_va, base.predict(X_va)):.4f}")
print(f"near copy: corr(bmi, bmi + noise) on train = {np.corrcoef(X_tr[:, BMI], near_tr)[0, 1]:.8f}")

# %%
section("exercise 2: four fits")
for label, (Xt, Xv) in cases.items():
    A, Av = add_intercept(Xt), add_intercept(Xv)
    AtA = A.T @ A
    print(f"{label}: condition number of A^T A {np.linalg.cond(AtA):.3e}   rank {np.linalg.matrix_rank(A)} of {A.shape[1]}")
    fits: dict[str, FloatArray] = {}
    try:
        fits["solve"] = np.linalg.solve(AtA, A.T @ y_tr)
    except np.linalg.LinAlgError as e:
        print(f"  solve              LinAlgError: {e}")
    fits["lstsq"] = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    lr_sk = LinearRegression().fit(Xt, y_tr)
    fits["LinearRegression"] = np.concatenate([[lr_sk.intercept_], lr_sk.coef_])
    mu, sd = Xt.mean(0), Xt.std(0)
    AZ = add_intercept((Xt - mu) / sd)
    lam = np.linalg.eigvalsh(2.0 / len(y_tr) * AZ.T @ AZ)
    # lr 1/lambda_max, not 2/(lambda_min + lambda_max): with lambda_min ~ 0 the "best" rate sits at the edge
    # of stability and the steepest direction then shrinks by only 1 - 2/kappa per step
    theta_z, steps = gradient_descent(AZ, y_tr, 1.0 / lam[-1])
    w = theta_z[1:] / sd
    fits["GD from zero"] = np.concatenate([[theta_z[0] - w @ mu], w])
    for name, th in fits.items():
        extra = f"   ({steps} steps, smallest Hessian eigenvalue {lam[0]:.2e})" if name == "GD from zero" else ""
        print(f"  {name:18s} bmi weights {th[1 + BMI]:12.4f} {th[-1]:12.4f}   sum {th[1 + BMI] + th[-1]:8.4f}   "
              f"val R^2 {r2_score(y_va, Av @ th):.4f}{extra}")
    if "solve" not in fits or label.startswith("(i)"):
        mn = np.linalg.pinv(A) @ y_tr
        print(f"  pinv (min-norm)    bmi weights {mn[1 + BMI]:12.4f} {mn[-1]:12.4f}")

# %%
section("done")
print(f"~ elapsed {time.perf_counter() - T0:.1f} s")
