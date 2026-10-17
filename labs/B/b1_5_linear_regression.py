# %% [markdown]
# # B1.5 Linear regression from scratch versus scikit-learn — lab
# Lesson: lessons/B/b1_5_linear_regression.md
# Run: uv run python labs/B/b1_5_linear_regression.py   (CPU, defaults: ~5 s)
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
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
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
OUT = ROOT / "outputs" / "b1_5"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only runs gradient descent on the raw features for longer.
GPU_SCALE = False
MAX_ITERS = 200_000 if GPU_SCALE else 20_000  # GD steps allowed (raw features never get there)
GRAD_TOL = 1e-8  # stop when |gradient| falls below this
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: the diabetes set in raw units (age in years, sex coded 1/2, bmi in kg/m^2, blood pressure in
# mm Hg, six blood measurements s1..s6), split exactly as in B1.1 so the baselines carry over.
section("data")
diab = load_diabetes(as_frame=True, scaled=False)
names = list(diab.feature_names)
X_all = diab.data.to_numpy(dtype=np.float64)  # (442, 10)
y_all = diab.target.to_numpy(dtype=np.float64)  # (442,)


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py (without the stratify option, which regression does not need)
    rows = gen.permutation(n)
    n_test, n_val = round(test_frac * n), round(val_frac * n)
    test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]
    return gen.permutation(train), gen.permutation(val), gen.permutation(test)


i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
n, d = X_tr.shape
print(f"train X {X_tr.shape}  val X {X_va.shape}  (test {len(i_te)} rows kept aside)")
print("feature means (train):", dict(zip(names, np.round(X_tr.mean(0), 2).tolist())))
print("feature sds   (train):", dict(zip(names, np.round(X_tr.std(0), 2).tolist())))

# %%
# Worked example by hand: four points, closed form, R^2, and one GD step on the standardized feature.
section("worked example by hand")
x4 = np.array([1.0, 2.0, 3.0, 4.0])
y4 = np.array([2.0, 3.0, 5.0, 6.0])
dx, dy = x4 - x4.mean(), y4 - y4.mean()
slope = (dx @ dy) / (dx @ dx)
intercept = y4.mean() - slope * x4.mean()
y4_hat = intercept + slope * x4
res4 = y4 - y4_hat
print(f"x-bar = {x4.mean()}   y-bar = {y4.mean()}   dx = {dx.tolist()}   dy = {dy.tolist()}")
print(f"sum dx*dy = {dx @ dy}   sum dx^2 = {dx @ dx}   slope = {slope:.4f}   intercept = {intercept:.4f}")
print(f"predictions = {show(y4_hat)}   residuals = {show(res4)}")
print(f"SSE = {res4 @ res4:.4f}   SST = {dy @ dy:.4f}   R^2 = 1 - SSE/SST = {1 - (res4 @ res4) / (dy @ dy):.4f}")
X4 = np.column_stack([np.ones(4), x4])  # (4, 2)
print(f"normal equations: X^T X = {show(X4.T @ X4)}   X^T y = {show(X4.T @ y4)}   "
      f"solution = {show(np.linalg.solve(X4.T @ X4, X4.T @ y4))}")
sd4 = x4.std()
z4 = dx / sd4
Z4 = np.column_stack([np.ones(4), z4])  # (4, 2) intercept + standardized x
g0 = 2 / 4 * Z4.T @ (Z4 @ np.zeros(2) - y4)
print(f"standardized x = {show(z4)}   (sd = sqrt(1.25) = {sd4:.4f})")
print(f"Z^T Z = {show(Z4.T @ Z4)}   gradient at w = 0: (2/n) Z^T (Z w - y) = {show(g0)}")
w1 = np.zeros(2) - 0.5 * g0
print(f"one step with lr = 0.5: w = {show(w1)}   gradient there = {show(2 / 4 * Z4.T @ (Z4 @ w1 - y4))}")
print(f"back to raw units: slope = {w1[1]:.4f} / {sd4:.4f} = {w1[1] / sd4:.4f}   "
      f"intercept = {w1[0]:.4f} - slope * {x4.mean()} = {w1[0] - w1[1] / sd4 * x4.mean():.4f}")

# %%
# From scratch: the closed form. Add a column of ones, solve X^T X w = X^T y (A2.4).
section("from scratch: normal equations")


def add_intercept(X: FloatArray) -> FloatArray:
    """Prepend a column of ones. X (n, d) -> (n, d + 1)."""
    return np.column_stack([np.ones(len(X)), X])


def fit_normal_equations(X: FloatArray, y: FloatArray) -> tuple[float, FloatArray]:
    """Least squares with an intercept. X (n, d), y (n,) -> (b, w (d,))."""
    A = add_intercept(X)  # (n, d + 1)
    theta = np.linalg.solve(A.T @ A, A.T @ y)  # (d + 1,); never inv(A^T A) @ ..., see A3.4
    return float(theta[0]), theta[1:]


def predict(X: FloatArray, b: float, w: FloatArray) -> FloatArray:
    """Linear model. X (n, d), w (d,) -> (n,)."""
    return X @ w + b


b_ne, w_ne = fit_normal_equations(X_tr, y_tr)
r_tr = y_tr - predict(X_tr, b_ne, w_ne)  # (n,) training residuals
print(f"intercept b = {b_ne:.4f}")
print("weights w (raw units):", dict(zip(names, np.round(w_ne, 4).tolist())))
print(f"checks: mean training residual {r_tr.mean():.2e}   max |X^T r| {np.abs(X_tr.T @ r_tr).max():.2e}")
A_tr = add_intercept(X_tr)
print(f"condition number of A^T A (raw features + ones): {np.linalg.cond(A_tr.T @ A_tr):.3e}")

# %%
# From scratch: batch gradient descent on standardized features, then convert back to raw units.
section("from scratch: gradient descent")


def mse(theta: FloatArray, A: FloatArray, y: FloatArray) -> float:
    """Mean squared error. A (n, k), theta (k,), y (n,) -> scalar."""
    r = A @ theta - y
    return float(r @ r / len(y))


def mse_grad(theta: FloatArray, A: FloatArray, y: FloatArray) -> FloatArray:
    """Gradient (2/n) A^T (A theta - y). -> (k,)."""
    return 2.0 / len(y) * (A.T @ (A @ theta - y))


def gradient_descent(
    A: FloatArray, y: FloatArray, lr: float, max_iters: int = MAX_ITERS, tol: float = GRAD_TOL
) -> tuple[FloatArray, list[float], int]:
    """Batch GD from theta = 0. A (n, k), y (n,) -> (theta (k,), loss per step, steps taken)."""
    # from labs/A/a4_4_gradient_descent.py, returning when it stops instead of raising
    theta = np.zeros(A.shape[1])
    losses = [mse(theta, A, y)]
    for k in range(1, max_iters + 1):
        g = mse_grad(theta, A, y)
        if np.linalg.norm(g) < tol:
            return theta, losses, k - 1
        theta = theta - lr * g
        losses.append(mse(theta, A, y))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"diverged at step {k}: lr = {lr} is too large")
    return theta, losses, max_iters


mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)  # (d,) learned on train only (B1.4)
Z_tr = (X_tr - mu) / sd  # (n, d)
AZ = add_intercept(Z_tr)  # (n, d + 1)
H_z = 2.0 / n * AZ.T @ AZ  # Hessian of the MSE (A4.4)
lam_z = np.linalg.eigvalsh(H_z)
lr_z = 2.0 / (lam_z[0] + lam_z[-1])  # best fixed learning rate for a quadratic (A4.4)
theta_z, loss_z, steps_z = gradient_descent(AZ, y_tr, lr_z)
print(f"standardized: Hessian eigenvalues {lam_z[0]:.4f} .. {lam_z[-1]:.4f}   condition number "
      f"{lam_z[-1] / lam_z[0]:.1f}   lr {lr_z:.4f}   converged in {steps_z} steps")
b_z, w_z = theta_z[0], theta_z[1:]
print("weights per 1 sd of each feature:", dict(zip(names, np.round(w_z, 4).tolist())))
w_back = w_z / sd  # (d,) per raw unit
b_back = b_z - w_back @ mu
print(f"converted back: b = {b_back:.4f}   max |w - w_normal_equations| = {np.abs(w_back - w_ne).max():.2e}")
np.testing.assert_allclose(w_back, w_ne, rtol=0, atol=1e-4)
np.testing.assert_allclose(b_back, b_ne, rtol=0, atol=1e-4)
print("GD on standardized features matches the normal equations to 4 decimals: OK")

# %%
# From scratch: the same GD on the raw features. The condition number decides everything.
section("from scratch: gradient descent without scaling")
H_raw = 2.0 / n * A_tr.T @ A_tr
lam_raw = np.linalg.eigvalsh(H_raw)
lr_raw = 2.0 / (lam_raw[0] + lam_raw[-1])
print(f"raw: Hessian eigenvalues {lam_raw[0]:.3e} .. {lam_raw[-1]:.3e}   condition number "
      f"{lam_raw[-1] / lam_raw[0]:.3e}   best lr {lr_raw:.3e}")
try:
    with np.errstate(over="ignore", invalid="ignore"):  # the overflow is the point; report it once
        gradient_descent(A_tr, y_tr, lr_z, max_iters=50)
except FloatingPointError as err:
    print(f"with the standardized run's lr {lr_z:.4f}: {err}")
theta_raw, loss_raw, steps_raw = gradient_descent(A_tr, y_tr, lr_raw)
best = mse(np.concatenate([[b_ne], w_ne]), A_tr, y_tr)
print(f"with lr {lr_raw:.3e}: {steps_raw} steps   train MSE {loss_raw[-1]:.2f}   (minimum {best:.2f})")
print(f"weights after {steps_raw} steps: bmi {theta_raw[1 + names.index('bmi')]:.4f} (target {w_ne[names.index('bmi')]:.4f})   "
      f"s5 {theta_raw[1 + names.index('s5')]:.4f} (target {w_ne[names.index('s5')]:.4f})")
kappa = lam_raw[-1] / lam_raw[0]
rate = (kappa - 1) / (kappa + 1)
print(f"slowest direction shrinks by (kappa - 1)/(kappa + 1) = {rate:.10f} per step -> "
      f"about {np.log(1e-3) / np.log(rate):.3e} steps to cut its error 1,000-fold")

# %%
# With scikit-learn: LinearRegression and np.linalg.lstsq give the same numbers.
section("with scikit-learn")
lr_sk = LinearRegression().fit(X_tr, y_tr)
np.testing.assert_allclose(lr_sk.coef_, w_ne, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(lr_sk.intercept_, b_ne, rtol=1e-6)
print(f"LinearRegression intercept_ {lr_sk.intercept_:.4f}   matches normal equations: OK")
print(f"LinearRegression coef_ {show(lr_sk.coef_)}")
theta_ls, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
np.testing.assert_allclose(theta_ls, np.concatenate([[b_ne], w_ne]), rtol=1e-6, atol=1e-8)
print("np.linalg.lstsq matches: OK")
pred_va = lr_sk.predict(X_va)
np.testing.assert_allclose(pred_va, predict(X_va, b_ne, w_ne), rtol=1e-9)
mse_va = mean_squared_error(y_va, pred_va)
print(f"validation: MSE {mse_va:.2f}  RMSE {np.sqrt(mse_va):.2f}  R^2 {r2_score(y_va, pred_va):.4f}")
print(f"train:      MSE {mean_squared_error(y_tr, lr_sk.predict(X_tr)):.2f}  R^2 {lr_sk.score(X_tr, y_tr):.4f}")
base = np.full(len(y_va), y_tr.mean())
print(f"B1.1 baselines on the same validation rows: mean R^2 {r2_score(y_va, base):.4f} "
      f"(RMSE {np.sqrt(mean_squared_error(y_va, base)):.2f}), bmi rule R^2 0.3461 (RMSE 61.92)")
one_feature = LinearRegression().fit(X_tr[:, [names.index("bmi")]], y_tr)
print(f"bmi alone, linear: val R^2 {one_feature.score(X_va[:, [names.index('bmi')]], y_va):.4f}   "
      f"slope {one_feature.coef_[0]:.4f} per kg/m^2")
print(f"correlation of s1 and s2 on train: {np.corrcoef(X_tr[:, names.index('s1')], X_tr[:, names.index('s2')])[0, 1]:.4f}   "
      f"their weights: s1 {w_ne[names.index('s1')]:.4f}  s2 {w_ne[names.index('s2')]:.4f}")

# %%
# Key plot: the same loss, the same algorithm, two parametrizations.
section("key plot")
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=120)
gap_z = np.maximum(np.array(loss_z) - best, 1e-6)  # below ~1e-6 it is floating-point noise
ax.plot(np.arange(1, len(loss_z) + 1), gap_z, "-", lw=2, label=f"standardized features (lr {lr_z:.2f})")
ax.plot(np.arange(1, len(loss_raw) + 1), np.array(loss_raw) - best, "--", lw=2,
        label=f"raw features (best lr {lr_raw:.1e})")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim(1e-6, 1e5)
ax.set(xlabel="gradient-descent step (log scale)", ylabel="train MSE minus its minimum (log scale)",
       title=f"Same GD: standardized converges in {steps_z:,} steps; raw is stuck")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
path_png = IMG / "b1_5_gd_scaled_vs_raw.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Standardize only bmi and bp, leave the rest raw. What is the condition number now?
# 2. Duplicate the bmi column (X_tr with bmi twice). What does np.linalg.solve do? What does lstsq do?
# 3. Fit on 30 training rows instead of 266. Compare train and val R^2 (a preview of B2.2).

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
