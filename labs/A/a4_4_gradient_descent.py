# %% [markdown]
# # A4.4 Gradient descent — lab
# Lesson: lessons/A/a4_4_gradient_descent.md
# Run: uv run python labs/A/a4_4_gradient_descent.py   (CPU, defaults: ~15 s, mostly importing PyTorch and scikit-learn)
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
# Nothing here needs a GPU. N_POINTS must stay 50 to reproduce the A2.4 weights exactly.
GPU_SCALE = False
N_POINTS = 100_000 if GPU_SCALE else 50  # rows of the A2.4 noisy line (50 reproduces A2.4)
TRUE_W = np.array([2.0, 0.5])  # intercept and slope used to generate the data (as in A2.4)
NOISE_SD = 1.0
GRAD_TOL = 1e-10  # batch GD stops when the gradient norm falls below this
MAX_ITERS = 100_000
BATCH_SIZE = 10  # mini-batch SGD
EPOCHS = 200

# %%
# Data: exactly the A2.4 dataset (same seed, same draws in the same order).
section("data")
x_data = rng.uniform(0.0, 10.0, size=N_POINTS)  # (n,)
y_data = TRUE_W[0] + TRUE_W[1] * x_data + rng.normal(0.0, NOISE_SD, size=N_POINTS)  # (n,)


def design_matrix(x: FloatArray) -> FloatArray:
    """Stack a column of ones (for the intercept) next to x. x (n,) -> (n, 2)."""
    return np.column_stack([np.ones_like(x), x])


X = design_matrix(x_data)  # (n, 2)
w_normal = np.linalg.solve(X.T @ X, X.T @ y_data)  # the A2.4 answer, the target for GD
print("X", X.shape, " y", y_data.shape, "  first three (x, y):", show(np.column_stack([x_data[:3], y_data[:3]])))
print("normal-equation weights (A2.4) =", show(w_normal))


def mse(w: FloatArray, X_: FloatArray, y_: FloatArray) -> float:
    """Mean squared error. X (n, d), w (d,), y (n,) -> scalar."""
    r = X_ @ w - y_  # (n,)
    return float(r @ r) / len(y_)


def grad_mse(w: FloatArray, X_: FloatArray, y_: FloatArray) -> FloatArray:
    """Gradient (2/n) X^T (X w - y) (A4.2). X (n, d), w (d,), y (n,) -> (d,)."""
    return 2.0 / len(y_) * X_.T @ (X_ @ w - y_)


# %%
# Worked example by hand: two steps of gradient descent on the four A2.4 points.
section("worked example: two steps by hand")
X4 = design_matrix(np.array([0.0, 1.0, 2.0, 3.0]))  # (4, 2)
y4 = np.array([1.0, 3.0, 4.0, 4.0])
w = np.zeros(2)
lr = 0.1
print(f"step 0: w = {show(w)}  loss = {mse(w, X4, y4):.5f}")
for step in (1, 2):
    g = grad_mse(w, X4, y4)
    w = w - lr * g
    print(f"step {step}: gradient = {show(g)}  w = {show(w)}  residuals = {show(X4 @ w - y4)}  loss = {mse(w, X4, y4):.5f}")

H4 = 2 / 4 * X4.T @ X4  # (2, 2) Hessian of the MSE (A4.2)
eig4 = np.linalg.eigvalsh(H4)  # ascending
print("Hessian (2/n) X^T X =", show(H4), "  eigenvalues =", show(eig4))
print(f"largest stable learning rate 2 / lambda_max = {2 / eig4[-1]:.4f}   condition number = {eig4[-1] / eig4[0]:.2f}")
print(f"with lr = 0.1, the slow direction shrinks by 1 - lr * lambda_min = {1 - 0.1 * eig4[0]:.4f} per step")

# %%
# Worked example: three learning rates on the four points.
section("worked example: three learning rates")


def gradient_descent(
    X_: FloatArray, y_: FloatArray, lr_: float, n_iters: int, w0: FloatArray | None = None
) -> tuple[FloatArray, list[float]]:
    """Batch gradient descent on the MSE for a fixed number of steps. Returns (w (d,), loss per step)."""
    w_ = np.zeros(X_.shape[1]) if w0 is None else w0.copy()
    losses = [mse(w_, X_, y_)]
    for _ in range(n_iters):
        w_ -= lr_ * grad_mse(w_, X_, y_)
        losses.append(mse(w_, X_, y_))
    return w_, losses


w_ls = np.linalg.solve(X4.T @ X4, X4.T @ y4)
print("least-squares answer for the 4 points: w =", show(w_ls), f"  loss = {mse(w_ls, X4, y4):.4f}")
for lr_ in (0.01, 0.2, 0.25):
    w_, losses = gradient_descent(X4, y4, lr_, 100)
    print(f"lr = {lr_:<5} loss after 10 steps = {losses[10]:>10.4f}   after 100 = {losses[100]:>12.4f}   w = {show(w_)}")

# %%
# From scratch: batch gradient descent on the 50 A2.4 points, raw x, until the gradient vanishes.
section("from scratch: batch gradient descent")


def gd_until_converged(
    X_: FloatArray, y_: FloatArray, lr_: float, tol: float = GRAD_TOL, max_iters: int = MAX_ITERS
) -> tuple[FloatArray, int]:
    """Batch GD on the MSE from w = 0 until ||gradient|| < tol. Returns (w (d,), iterations)."""
    w_ = np.zeros(X_.shape[1])
    for k in range(1, max_iters + 1):
        g = grad_mse(w_, X_, y_)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) > 1e12:
            raise FloatingPointError(f"gradient norm {np.linalg.norm(g):.1e} at iteration {k}: learning rate {lr_} is too large")
        if np.linalg.norm(g) < tol:
            return w_, k
        w_ -= lr_ * g
    raise RuntimeError(f"no convergence in {max_iters} iterations (gradient norm {np.linalg.norm(g):.2e})")


H = 2 / N_POINTS * X.T @ X  # (2, 2)
lam_min, lam_max = np.linalg.eigvalsh(H)
print(f"Hessian eigenvalues = [{lam_min:.4f}, {lam_max:.4f}]   condition number = {lam_max / lam_min:.1f}"
      f"   largest stable lr = 2 / lambda_max = {2 / lam_max:.5f}")
lr_raw = 0.025
w_gd, iters = gd_until_converged(X, y_data, lr_raw)
print(f"lr = {lr_raw}: converged in {iters} iterations   w = {show(w_gd)}")
np.testing.assert_allclose(w_gd, w_normal, rtol=1e-6, atol=1e-8)
print("matches the normal equations (A2.4) to 4 decimals: OK   (max difference "
      f"{np.max(np.abs(w_gd - w_normal)):.1e})")
try:
    gd_until_converged(X, y_data, 0.03)
except FloatingPointError as e:
    print("lr = 0.03 (above 2 / lambda_max):", str(e).split(":")[0], "-> diverged")

# %%
# From scratch: standardize x first. The Hessian becomes nearly round and GD needs far fewer steps.
section("from scratch: standardized feature")
x_mean, x_std = x_data.mean(), x_data.std()
Xs = design_matrix((x_data - x_mean) / x_std)  # (n, 2), feature has mean 0 and standard deviation 1
Hs = 2 / N_POINTS * Xs.T @ Xs
lam_s = np.linalg.eigvalsh(Hs)
print(f"Hessian eigenvalues = {show(lam_s)}   condition number = {lam_s[-1] / lam_s[0]:.2f}")
lr_std = 2 / (lam_s[0] + lam_s[-1])  # the best fixed step for a quadratic
w_s, iters_s = gd_until_converged(Xs, y_data, lr_std)
slope = w_s[1] / x_std  # undo the scaling: y = a + b (x - mean) / std
intercept = w_s[0] - slope * x_mean
w_back = np.array([intercept, slope])
print(f"lr = {lr_std:.4f}: converged in {iters_s} iterations   w (standardized) = {show(w_s)}   back on the x scale = {show(w_back)}")
np.testing.assert_allclose(w_back, w_normal, rtol=1e-6, atol=1e-8)
print("matches the normal equations: OK")
_, iters_typical = gd_until_converged(Xs, y_data, 0.1)
print(f"with a typical lr = 0.1 instead: {iters_typical} iterations (raw x needed {iters})")

# %%
# From scratch: mini-batch SGD on the standardized feature.
section("from scratch: mini-batch SGD")


def minibatch_sgd(
    X_: FloatArray, y_: FloatArray, lr0: float, batch_size: int, epochs: int, decay: bool, seed: int = SEED
) -> FloatArray:
    """Mini-batch SGD on the MSE from w = 0. If decay, lr = lr0 / (1 + epoch / 10). Returns w (d,)."""
    gen = np.random.default_rng(seed)
    n = len(y_)
    w_ = np.zeros(X_.shape[1])
    for epoch in range(epochs):
        lr_ = lr0 / (1 + epoch / 10) if decay else lr0
        order = gen.permutation(n)  # reshuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]  # (batch_size,) row indices
            w_ -= lr_ * grad_mse(w_, X_[idx], y_[idx])  # gradient of the batch's mean loss
    return w_


for decay in (False, True):
    w_sgd = minibatch_sgd(Xs, y_data, 0.1, BATCH_SIZE, EPOCHS, decay)
    print(f"SGD, batch {BATCH_SIZE}, {EPOCHS} epochs, lr {'0.1 / (1 + epoch / 10)' if decay else '0.1 constant':<22}"
          f" w = {show(w_sgd)}   distance to the GD answer = {np.linalg.norm(w_sgd - w_s):.1e}")
print(f"{EPOCHS} epochs x {N_POINTS // BATCH_SIZE} batches = {EPOCHS * N_POINTS // BATCH_SIZE} updates, each using {BATCH_SIZE} of {N_POINTS} rows")

# %%
# With PyTorch and scikit-learn: torch.optim.SGD takes exactly our steps; SGDRegressor gets close.
section("with pytorch and scikit-learn")
import torch  # noqa: E402
from sklearn.linear_model import SGDRegressor  # noqa: E402

wt = torch.zeros(2, dtype=torch.float64, requires_grad=True)
opt = torch.optim.SGD([wt], lr=lr_raw)
Xt, yt = torch.tensor(X), torch.tensor(y_data)
for _ in range(iters - 1):  # the same number of updates our loop made
    opt.zero_grad()
    loss = torch.mean((Xt @ wt - yt) ** 2)
    loss.backward()
    opt.step()
w_torch = wt.detach().numpy()
np.testing.assert_allclose(w_torch, w_gd, rtol=1e-10, atol=1e-12)
print(f"torch.optim.SGD, {iters - 1} full-batch steps at lr {lr_raw}: w = {show(w_torch)}   identical to our loop: OK")

sk = SGDRegressor(loss="squared_error", penalty=None, learning_rate="invscaling", eta0=0.01,
                  max_iter=1000, tol=1e-6, random_state=SEED)
sk.fit(Xs[:, 1:], y_data)  # sklearn adds its own intercept; pass the standardized feature only
sk_slope = sk.coef_[0] / x_std
w_sk = np.array([sk.intercept_[0] - sk_slope * x_mean, sk_slope])
print(f"SGDRegressor on the standardized feature: {sk.n_iter_} epochs   w back on the x scale = {show(w_sk)}"
      f"   distance to the normal equations = {np.linalg.norm(w_sk - w_normal):.1e}")

# %%
# Key plot: loss curves for three learning rates on the raw A2.4 data, and the paths they take.
section("key plot")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=120)
styles = {0.002: ("tab:blue", "-", "lr = 0.002: too small, crawls"),
          0.025: ("tab:green", "--", "lr = 0.025: converges"),
          0.0305: ("tab:red", ":", "lr = 0.0305 > 2/λ_max: diverges")}
loss_min = mse(w_normal, X, y_data)
for lr_, (color, ls, label) in styles.items():
    _, losses = gradient_descent(X, y_data, lr_, 300)
    ax1.semilogy(np.array(losses) - loss_min + 1e-16, ls, color=color, lw=2, label=label)
ax1.set_ylim(1e-6, 1e4)
ax1.set_xlabel("iteration")
ax1.set_ylabel("loss − minimum loss (log scale)")
ax1.set_title("Too small crawls, too large blows up")
ax1.legend(loc="upper left", fontsize=8)

b0, b1 = np.meshgrid(np.linspace(-0.5, 3.5, 200), np.linspace(-0.2, 1.4, 200))
resid = b0[..., None] + b1[..., None] * x_data - y_data  # (200, 200, n): residuals at every grid point
Z = np.mean(resid**2, axis=-1)  # (200, 200) loss surface
ax2.contour(b0, b1, Z, levels=np.geomspace(loss_min * 1.01, 60, 12), colors="0.6", linewidths=0.8)
for lr_, color, ls, n_show in ((0.012, "tab:green", "--", 300), (0.026, "tab:purple", "-", 60)):
    path = [np.zeros(2)]
    for _ in range(n_show):
        path.append(path[-1] - lr_ * grad_mse(path[-1], X, y_data))
    path = np.array(path)
    ax2.plot(path[:, 0], path[:, 1], ls, color=color, marker=".", ms=4, lw=1,
             label=f"lr = {lr_}: {n_show} steps")
ax2.plot(*w_normal, "k*", ms=12, label="normal-equation answer")
ax2.set_xlim(-0.5, 3.5)
ax2.set_ylim(-0.2, 1.4)
ax2.set_xlabel("intercept w₀")
ax2.set_ylabel("slope w₁")
ax2.set_title("Raw x: a long narrow valley, so GD zig-zags across it")
ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout()
path_img = IMG / "a4_4_learning_rates.png"
fig.savefig(path_img, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_img.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Multiply x_data by 100 (centimetres instead of metres). What happens to 2 / lambda_max and to the
#    number of iterations? Then standardize and compare.
# 2. Run minibatch_sgd with BATCH_SIZE = 1 and = 50. Which one is plain gradient descent?
# 3. Start gd_until_converged from w = (10, -10). Does the answer change? Why not (A4.5)?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
