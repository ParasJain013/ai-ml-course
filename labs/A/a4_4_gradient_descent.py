# %% [markdown]
# # A4.4 Gradient descent — lab
# Lesson: lessons/A/a4_4_gradient_descent.md
# Run: uv run python labs/A/a4_4_gradient_descent.py   (CPU, defaults: ~3 s)
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
from scipy import optimize
from sklearn.linear_model import LinearRegression, SGDRegressor

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
OUT = ROOT / "outputs" / "a4_4"  # scratch figures, gitignored
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
# Nothing here needs a GPU. The data is the A2.4 line-fitting set, regenerated with the same seed.
GPU_SCALE = False
N_POINTS = 100_000 if GPU_SCALE else 50  # rows (same as A2.4)
TRUE_W = np.array([2.0, 0.5])  # intercept and slope used to generate the data (A2.4)
NOISE_SD = 1.0
MAX_ITERS = 100_000  # cap for batch gradient descent
GRAD_TOL = 1e-8  # stop when the gradient's length falls below this
BATCH_SIZE = 10  # mini-batch size for SGD
EPOCHS = 200  # passes over the data for SGD

# %%
# Data: exactly the A2.4 points, so the normal-equation answer is known: w = [1.7575, 0.5523].
section("data")
x_data = rng.uniform(0.0, 10.0, size=N_POINTS)  # (n,)
y_data = TRUE_W[0] + TRUE_W[1] * x_data + rng.normal(0.0, NOISE_SD, size=N_POINTS)  # (n,)
X = np.column_stack([np.ones(N_POINTS), x_data])  # (n, 2) design matrix: ones column, then x
n = len(y_data)
w_ne = np.linalg.solve(X.T @ X, X.T @ y_data)  # (2,) normal equations (A2.4)
print("X", X.shape, " y", y_data.shape, "  normal-equation w =", show(w_ne))


def mse(w: FloatArray, X: FloatArray, y: FloatArray) -> float:
    """Mean squared error (1/n) |X w - y|^2. X (n, d), w (d,), y (n,) -> scalar."""
    r = X @ w - y
    return float(r @ r / len(y))


def mse_grad(w: FloatArray, X: FloatArray, y: FloatArray) -> FloatArray:
    """Gradient (2/n) X^T (X w - y) (A4.2). -> (d,)."""
    return 2.0 / len(y) * (X.T @ (X @ w - y))


loss_star = mse(w_ne, X, y_data)
print(f"minimum MSE (at the normal-equation w) = {loss_star:.4f}")

# %%
# Worked example by hand, part 1: L(w) = (w - 3)^2 with four learning rates.
section("worked example: one parameter")
for lr in (0.1, 0.5, 0.9, 1.1):
    w, path = 0.0, []
    for _ in range(4):
        w = w - lr * 2 * (w - 3)  # gradient of (w - 3)^2 is 2 (w - 3)
        path.append(round(w, 4))
    print(f"lr = {lr:<4} factor (1 - 2 lr) = {1 - 2 * lr:<5.1f} w after 1..4 steps = {path}")

# %%
# Worked example by hand, part 2: one step on the four-point line of A2.4, and its Hessian.
section("worked example: four points")
X4 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])  # (4, 2)
y4 = np.array([1.0, 3.0, 4.0, 4.0])  # (4,): A2.4's example, least squares w = (1.5, 1.0)
w = np.zeros(2)
g = mse_grad(w, X4, y4)
w1 = w - 0.1 * g
print(f"loss at w = 0: {mse(w, X4, y4):.5f}   gradient = {show(g)}")
print(f"after one step with lr = 0.1: w = {show(w1)}   loss = {mse(w1, X4, y4):.5f}")
H4 = 2.0 / 4 * X4.T @ X4  # (2, 2) Hessian of the MSE
lam4 = np.linalg.eigvalsh(H4)  # ascending
print(f"Hessian (2/n) X^T X = {show(H4)}   eigenvalues = {show(lam4)}")
print(f"largest stable lr 2/lambda_max = {2 / lam4[-1]:.4f}   condition number = {lam4[-1] / lam4[0]:.2f}")
w = np.zeros(2)
for k in range(1, 1001):
    w = w - 0.1 * mse_grad(w, X4, y4)
    if k in (10, 100, 300):
        print(f"after {k:>3} steps: w = {show(w)}   distance to (1.5, 1.0) = {np.linalg.norm(w - [1.5, 1.0]):.6f}")

# %%
# From scratch: batch gradient descent on the A2.4 data.
section("from scratch: batch gradient descent")


def gradient_descent(
    X: FloatArray, y: FloatArray, lr: float, max_iters: int = MAX_ITERS, tol: float = GRAD_TOL
) -> tuple[FloatArray, list[float], int]:
    """Batch GD on the MSE from w = 0. X (n, d), y (n,) -> (w (d,), loss per step, iterations)."""
    w = np.zeros(X.shape[1])  # (d,)
    losses = [mse(w, X, y)]
    for k in range(1, max_iters + 1):
        g = mse_grad(w, X, y)  # (d,)
        if np.linalg.norm(g) < tol:
            return w, losses, k - 1
        w = w - lr * g
        losses.append(mse(w, X, y))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"diverged at step {k}: lr = {lr} is too large")
    raise RuntimeError(f"no convergence in {max_iters} steps (|grad| = {np.linalg.norm(g):.2e})")


H = 2.0 / n * X.T @ X  # (2, 2) Hessian, constant for the MSE
lam = np.linalg.eigvalsh(H)
lr_max = 2.0 / lam[-1]
lr_best = 2.0 / (lam[0] + lam[-1])  # the best fixed learning rate for a quadratic
print(f"Hessian eigenvalues = {show(lam)}   condition number = {lam[-1] / lam[0]:.1f}")
print(f"largest stable lr = {lr_max:.4f}   best fixed lr = {lr_best:.4f}")
w_gd, losses_gd, iters = gradient_descent(X, y_data, lr_best)
print(f"GD with lr = {lr_best:.4f}: {iters} steps   w = {show(w_gd)}   normal equations w = {show(w_ne)}")
np.testing.assert_allclose(w_gd, w_ne, rtol=0, atol=1e-6)
print("GD matches the normal equations to 4 decimals: OK")
kappa = lam[-1] / lam[0]
g0 = np.linalg.norm(mse_grad(np.zeros(2), X, y_data))
rate = (kappa - 1) / (kappa + 1)  # the slowest error component shrinks by this factor per step
print(f"|grad| at w = 0: {g0:.1f}   rate (kappa - 1)/(kappa + 1) = {rate:.4f}"
      f"   predicted steps to reach |grad| < 1e-8: about {np.log(GRAD_TOL / g0) / np.log(rate):.0f}")

# %%
# From scratch: the same fit with a standardized feature. The condition number collapses.
section("from scratch: standardizing the feature")
mu, sd = x_data.mean(), x_data.std()
Xs = np.column_stack([np.ones(n), (x_data - mu) / sd])  # (n, 2): ones, standardized x
Hs = 2.0 / n * Xs.T @ Xs
lam_s = np.linalg.eigvalsh(Hs)
lr_s = 2.0 / (lam_s[0] + lam_s[-1])
ws, _, iters_s = gradient_descent(Xs, y_data, lr_s)
w_back = np.array([ws[0] - ws[1] * mu / sd, ws[1] / sd])  # undo the scaling: y = a + b (x - mu)/sd
print(f"Hessian eigenvalues = {show(lam_s)}   condition number = {lam_s[-1] / lam_s[0]:.1f}")
print(f"GD with lr = {lr_s:.4f}: {iters_s} steps   w (standardized) = {show(ws)}")
print(f"converted back to the raw x: w = {show(w_back)}   matches normal equations: {np.allclose(w_back, w_ne, atol=1e-6)}")

# %%
# From scratch: mini-batch SGD. Shuffle every epoch; constant versus decaying learning rate.
section("from scratch: mini-batch SGD")


def sgd(
    X: FloatArray, y: FloatArray, lr0: float, epochs: int, batch_size: int, decay: bool, seed: int = SEED
) -> FloatArray:
    """Mini-batch SGD on the MSE from w = 0. lr_t = lr0 / (1 + epoch / 10) if decay. -> w (d,)."""
    gen = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        lr = lr0 / (1 + epoch / 10) if decay else lr0
        order = gen.permutation(len(y))  # a fresh shuffle every epoch
        for start in range(0, len(y), batch_size):
            idx = order[start : start + batch_size]  # (batch_size,) row indices
            w = w - lr * mse_grad(w, X[idx], y[idx])  # gradient on the batch only
    return w


for decay in (False, True):
    w_sgd = sgd(Xs, y_data, lr_s, EPOCHS, BATCH_SIZE, decay)
    w_raw = np.array([w_sgd[0] - w_sgd[1] * mu / sd, w_sgd[1] / sd])
    print(f"SGD, batch {BATCH_SIZE}, {EPOCHS} epochs, {'decaying' if decay else 'constant'} lr: w = {show(w_raw)}"
          f"   distance to normal equations = {np.linalg.norm(w_raw - w_ne):.1e}"
          f"   excess loss = {mse(w_raw, X, y_data) - loss_star:.1e}")

# %%
# With SciPy and scikit-learn: a library optimizer, a library SGD, and the closed form.
section("with scipy and scikit-learn")
res = optimize.minimize(mse, np.zeros(2), args=(X, y_data), jac=mse_grad, method="BFGS", options={"gtol": 1e-10})
np.testing.assert_allclose(res.x, w_ne, rtol=1e-6, atol=1e-8)
print(f"scipy.optimize.minimize(BFGS): w = {show(res.x)}   {res.nit} iterations   matches normal equations: OK")
lin = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
print(f"LinearRegression: w = {show(np.array([lin.intercept_, lin.coef_[0]]))}")
# Default schedule ("invscaling", a decaying lr from eta0 = 0.01); penalty=None turns off its default
# L2 penalty; tol=None runs all max_iter epochs instead of stopping early.
sgd_sk = SGDRegressor(penalty=None, max_iter=1000, tol=None, random_state=SEED)
sgd_sk.fit(Xs[:, 1:], y_data)  # standardized x; SGDRegressor fits its own intercept
w_sk = np.array([sgd_sk.intercept_[0] - sgd_sk.coef_[0] * mu / sd, sgd_sk.coef_[0] / sd])
print(f"SGDRegressor (standardized x): w = {show(w_sk)}   distance to normal equations = "
      f"{np.linalg.norm(w_sk - w_ne):.1e}")
np.testing.assert_allclose(w_sk, w_ne, atol=1e-3)  # SGD is noisy: agreement to ~3 decimals, not 6
print("SGDRegressor agrees with the normal equations to about 3 decimals: OK")

# %%
# Key plot: excess loss for three learning rates, and the path of GD on the raw-feature contours.
section("key plot")
rates = {"too small (0.02 x max)": 0.02 * lr_max, "best fixed lr": lr_best, "too large (1.05 x max)": 1.05 * lr_max}
STEPS = 1000
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=120)
for (label, lr), style in zip(rates.items(), ("o-", "s-", "^-")):
    w = np.zeros(2)
    excess = []
    for _ in range(STEPS):
        excess.append(mse(w, X, y_data) - loss_star)
        w = w - lr * mse_grad(w, X, y_data)
    ax1.semilogy(excess, style, ms=2.5, markevery=50, lw=1.2, label=f"{label}: lr = {lr:.4f}")
    print(f"{label:<24} lr = {lr:.4f}   excess loss after {STEPS} steps = {excess[-1]:.2e}")
ax1.set_ylim(1e-12, 1e4)  # the diverging run leaves the chart upwards
ax1.set_xlabel("step")
ax1.set_ylabel("MSE − minimum MSE (log scale)")
ax1.set_title("Too small crawls, too large explodes", fontsize=10)
ax1.legend(loc="lower left", fontsize=7)
LR_PATH = 0.8 * lr_max  # each step multiplies the steep error component by 1 - 0.8 * 2 = -0.6
w, path = np.zeros(2), [np.zeros(2)]
for _ in range(40):
    w = w - LR_PATH * mse_grad(w, X, y_data)
    path.append(w)
path_arr = np.array(path)  # (41, 2)
lo, hi = path_arr.min(axis=0), path_arr.max(axis=0)
lo, hi = np.minimum(lo, w_ne) - 0.15, np.maximum(hi, w_ne) + 0.15
g0, g1 = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
Wg = np.stack([g0.ravel(), g1.ravel()], axis=1)
Lg = ((Wg @ X.T - y_data) ** 2).mean(axis=1).reshape(g0.shape)
ax2.contour(g0, g1, Lg, levels=loss_star + np.array([0.05, 0.2, 0.8, 3.2, 12.8]), colors="0.6", linewidths=0.8)
ax2.plot(path_arr[:, 0], path_arr[:, 1], "o-", ms=3, lw=1, color="tab:orange",
         label=f"first 40 GD steps, lr = 0.8 x max = {LR_PATH:.4f}")
ax2.plot(*w_ne, "k*", ms=12, label="normal equations")
ax2.set_xlabel("w₀ (intercept)")
ax2.set_ylabel("w₁ (slope)")
ax2.set_title("Raw x: a long, thin valley; GD zig-zags across it, then crawls along it", fontsize=10)
ax2.legend(loc="upper right", fontsize=7)
fig.suptitle("Gradient descent on the A2.4 line fit: the learning rate and the condition number decide the speed")
fig.tight_layout()
path_png = IMG / "a4_4_learning_rates.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Run gradient_descent on X with lr = 0.99 * lr_max. Faster or slower than the best lr? Why?
# 2. Change BATCH_SIZE to 1 and to 50 in sgd(). How does the final distance change? What is batch 50?
# 3. Stop GD when the loss changes by less than 1e-10 instead of using the gradient. How far is w then?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
