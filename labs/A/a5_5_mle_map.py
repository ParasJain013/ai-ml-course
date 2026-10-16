# %% [markdown]
# # A5.5 MLE and MAP — lab
# Lesson: lessons/A/a5_5_mle_map.md
# Run: uv run python labs/A/a5_5_mle_map.py   (CPU, defaults: ~3 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import log_loss

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
OUT = ROOT / "outputs" / "a5_5"  # scratch figures, gitignored
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
# Nothing here needs a GPU. The regression data is the A2.4 / A4.4 line, regenerated with the same seed.
GPU_SCALE = False
N_POINTS = 50  # regression rows (same as A2.4 and A4.4)
TRUE_W = np.array([2.0, 0.5])  # intercept and slope used to generate the line (A2.4)
NOISE_SD = 1.0  # sigma of the Gaussian noise on y
N_GAUSS = 100_000 if GPU_SCALE else 40  # draws for the Gaussian MLE
TRUE_MU, TRUE_SD = 5.0, 2.0
N_FLIPS = 100_000 if GPU_SCALE else 100  # coin flips for the Bernoulli MLE
TRUE_P = 0.3
PRIOR_SD = 0.1  # tau: prior standard deviation of the slope for MAP
LR = 0.1  # gradient-descent learning rate for the Gaussian NLL
MAX_ITERS = 10_000
GRAD_TOL = 1e-10

# %%
# Data: the A2.4 line (first, so it matches A2.4 and A4.4 exactly), then a Gaussian sample and coin flips.
section("data")
x_line = rng.uniform(0.0, 10.0, size=N_POINTS)  # (n,)
y_line = TRUE_W[0] + TRUE_W[1] * x_line + rng.normal(0.0, NOISE_SD, size=N_POINTS)  # (n,)
X = np.column_stack([np.ones(N_POINTS), x_line])  # (n, 2) design matrix
x_gauss = rng.normal(TRUE_MU, TRUE_SD, size=N_GAUSS)  # (N_GAUSS,)
flips = (rng.random(N_FLIPS) < TRUE_P).astype(np.float64)  # (N_FLIPS,) 1 = heads
print(f"line: X {X.shape}  y {y_line.shape}   Gaussian sample: {x_gauss.shape}, first 5 = {show(x_gauss[:5])}")
print(f"coin: {N_FLIPS} flips, {int(flips.sum())} heads")

# %%
# Worked example by hand.
section("worked example: Gaussian MLE for x = (2, 4, 6)")
xs = np.array([2.0, 4.0, 6.0])


def gauss_nll(x: FloatArray, mu: float, sigma: float) -> float:
    """Total negative log-likelihood of x under N(mu, sigma^2). x (n,) -> scalar."""
    n = len(x)
    return float(0.5 * n * np.log(2 * np.pi * sigma**2) + ((x - mu) ** 2).sum() / (2 * sigma**2))


mu_hat, var_hat = xs.mean(), ((xs - xs.mean()) ** 2).mean()
print(f"mu_hat = mean = {mu_hat:.4f}   sigma2_hat = (4 + 0 + 4)/3 = {var_hat:.4f}   (ddof=1 would give {xs.var(ddof=1):.4f})")
for mu in (3.0, 4.0, 5.0):
    print(f"NLL with sigma = 2 at mu = {mu}: {gauss_nll(xs, mu, 2.0):.4f}")

section("worked example: Bernoulli MLE for 3 heads in 4 flips")
k, n4 = 3, 4
for p in (0.5, 0.75, 0.9):
    print(f"p = {p:<4}: likelihood p^3 (1 - p) = {p**3 * (1 - p):.4f}   mean NLL (log-loss) = "
          f"{-(k * np.log(p) + (n4 - k) * np.log(1 - p)) / n4:.4f}")

section("worked example: MAP for a one-weight line")
x2, y2 = np.array([1.0, 2.0]), np.array([1.0, 3.0])
print(f"least squares (MLE) w = sum(xy)/sum(x^2) = {x2 @ y2:.0f}/{x2 @ x2:.0f} = {x2 @ y2 / (x2 @ x2):.4f}")
for lam in (1.0, 5.0):
    print(f"MAP with lambda = sigma^2/tau^2 = {lam}: w = {x2 @ y2:.0f}/({x2 @ x2:.0f} + {lam:.0f}) = "
          f"{x2 @ y2 / (x2 @ x2 + lam):.4f}")

# %%
# From scratch: Gaussian MLE in closed form and by gradient descent on the mean NLL.
section("from scratch: Gaussian MLE, closed form and gradient descent")
mu_cf = x_gauss.mean()
sigma_cf = np.sqrt(((x_gauss - mu_cf) ** 2).mean())  # MLE divides by n, not n - 1
print(f"closed form: mu_hat = {mu_cf:.4f}   sigma_hat = {sigma_cf:.4f}   (true 5.0 and 2.0, n = {N_GAUSS})")


def gauss_mean_nll(theta: FloatArray, x: FloatArray) -> float:
    """Mean NLL of x under N(mu, exp(s)^2), theta = (mu, s) with s = log sigma. -> scalar."""
    mu, s = theta
    return float(0.5 * np.log(2 * np.pi) + s + ((x - mu) ** 2).mean() / (2 * np.exp(2 * s)))


def gauss_mean_nll_grad(theta: FloatArray, x: FloatArray) -> FloatArray:
    """Gradient of gauss_mean_nll with respect to (mu, s). -> (2,)."""
    mu, s = theta
    var = np.exp(2 * s)
    d_mu = -(x - mu).mean() / var  # d/dmu of (x - mu)^2 / (2 var), averaged
    d_s = 1.0 - ((x - mu) ** 2).mean() / var  # d/ds of s + (x - mu)^2 exp(-2s) / 2
    return np.array([d_mu, d_s])


def numerical_grad(f: Callable[[FloatArray], float], theta: FloatArray, h: float = 1e-6) -> FloatArray:
    """Central-difference gradient (from labs/A/a4_1_derivatives.py). theta (p,) -> (p,)."""
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h
        g[j] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g


theta_test = np.array([3.0, 0.2])
g_an = gauss_mean_nll_grad(theta_test, x_gauss)
g_num = numerical_grad(lambda t: gauss_mean_nll(t, x_gauss), theta_test)
np.testing.assert_allclose(g_an, g_num, rtol=1e-6, atol=1e-8)
print(f"gradient check at (mu, s) = {theta_test.tolist()}: analytic {show(g_an)}  numerical {show(g_num)}: OK")


def gradient_descent(
    grad: Callable[[FloatArray], FloatArray], theta0: FloatArray, lr: float,
    max_iters: int = MAX_ITERS, tol: float = GRAD_TOL,
) -> tuple[FloatArray, list[FloatArray]]:
    """Plain gradient descent (adapted from labs/A/a4_4_gradient_descent.py). -> (theta, path)."""
    theta, path = theta0.astype(np.float64), [theta0.astype(np.float64)]
    for _ in range(max_iters):
        g = grad(theta)
        if np.linalg.norm(g) < tol:
            return theta, path
        theta = theta - lr * g
        path.append(theta)
    raise RuntimeError(f"no convergence in {max_iters} steps (|grad| = {np.linalg.norm(g):.2e})")


theta_gd, path_gd = gradient_descent(lambda t: gauss_mean_nll_grad(t, x_gauss), np.array([0.0, 0.0]), LR)
print(f"gradient descent from (mu, s) = (0, 0), lr = {LR}: {len(path_gd) - 1} steps   mu = {theta_gd[0]:.4f}   "
      f"sigma = exp(s) = {np.exp(theta_gd[1]):.4f}")
np.testing.assert_allclose([theta_gd[0], np.exp(theta_gd[1])], [mu_cf, sigma_cf], rtol=1e-6)
print("gradient descent matches the closed form: OK")

# %%
# From scratch: Bernoulli MLE, and log-loss is its mean NLL.
section("from scratch: Bernoulli MLE")
p_hat = flips.mean()


def bern_mean_nll(p: float | FloatArray, y: FloatArray) -> float | FloatArray:
    """Mean NLL (log-loss) of 0/1 outcomes y under Bernoulli(p). y (n,) -> scalar (or grid of p)."""
    k = y.sum()
    return -(k * np.log(p) + (len(y) - k) * np.log(1 - p)) / len(y)


grid_p = np.linspace(0.01, 0.99, 9801)
p_grid_best = grid_p[np.argmin(bern_mean_nll(grid_p, flips))]
print(f"p_hat = heads / flips = {p_hat:.4f}   grid search minimum of the NLL at p = {p_grid_best:.4f}")
print(f"mean NLL at p_hat = {bern_mean_nll(p_hat, flips):.4f}   at p = 0.5: {bern_mean_nll(0.5, flips):.4f}")

# %%
# From scratch: regression with Gaussian noise. The NLL minimizer is the MSE minimizer.
section("from scratch: squared error is Gaussian MLE")
w_ls = np.linalg.lstsq(X, y_line, rcond=None)[0]  # (2,)


def reg_nll(theta: FloatArray) -> float:
    """Total NLL of y_line under y ~ N(X w, exp(s)^2); theta = (w0, w1, s). -> scalar."""
    w, s = theta[:2], theta[2]
    r = y_line - X @ w  # (n,)
    return float(0.5 * N_POINTS * np.log(2 * np.pi) + N_POINTS * s + (r @ r) / (2 * np.exp(2 * s)))


def mse(w: FloatArray) -> float:
    """Mean squared error of the line w. w (2,) -> scalar."""
    r = y_line - X @ w
    return float(r @ r / N_POINTS)


res_nll = optimize.minimize(reg_nll, np.zeros(3), method="BFGS", options={"gtol": 1e-8})
res_mse = optimize.minimize(mse, np.zeros(2), method="BFGS", options={"gtol": 1e-10})
w_nll, sigma_nll = res_nll.x[:2], np.exp(res_nll.x[2])
print(f"least squares w = {show(w_ls)}")
print(f"argmin MSE      w = {show(res_mse.x)}")
print(f"argmin Gaussian NLL over (w, sigma): w = {show(w_nll)}   sigma = {sigma_nll:.4f}   "
      f"sqrt(MSE at w) = {np.sqrt(mse(w_ls)):.4f}")
np.testing.assert_allclose(w_nll, w_ls, atol=1e-5)
np.testing.assert_allclose(res_mse.x, w_ls, atol=1e-5)
print("MSE minimizer = Gaussian NLL minimizer = least squares: OK")

# %%
# From scratch: MAP with a Gaussian prior on the slope is ridge regression.
section("from scratch: MAP with a Gaussian prior = ridge")
lam = NOISE_SD**2 / PRIOR_SD**2  # lambda = sigma^2 / tau^2
D = np.diag([0.0, 1.0])  # (2, 2) the prior is on the slope only; the intercept is left free


def neg_log_posterior(w: FloatArray) -> float:
    """-log p(w | data) up to a constant: sum r^2 / (2 sigma^2) + w1^2 / (2 tau^2). w (2,) -> scalar."""
    r = y_line - X @ w
    return float(r @ r / (2 * NOISE_SD**2) + w[1] ** 2 / (2 * PRIOR_SD**2))


w_map_closed = np.linalg.solve(X.T @ X + lam * D, X.T @ y_line)  # (X^T X + lambda D) w = X^T y
w_map_opt = optimize.minimize(neg_log_posterior, np.zeros(2), method="BFGS", options={"gtol": 1e-10}).x
print(f"sigma = {NOISE_SD}, prior sd of slope tau = {PRIOR_SD}  ->  lambda = sigma^2 / tau^2 = {lam:.0f}")
print(f"MAP by minimizing the negative log posterior: w = {show(w_map_opt)}")
print(f"ridge closed form (X^T X + lambda D)^-1 X^T y: w = {show(w_map_closed)}")
print(f"MLE (no prior): w = {show(w_ls)}   the slope shrank from {w_ls[1]:.4f} to {w_map_closed[1]:.4f}")
np.testing.assert_allclose(w_map_opt, w_map_closed, atol=1e-5)
print("MAP = ridge: OK")

# %%
# With SciPy and scikit-learn.
section("with scipy and scikit-learn")
loc, scale = stats.norm.fit(x_gauss)
print(f"stats.norm.fit: loc = {loc:.4f}   scale = {scale:.4f}   (our closed form {mu_cf:.4f}, {sigma_cf:.4f})")
np.testing.assert_allclose([loc, scale], [mu_cf, sigma_cf], rtol=1e-6)
print("stats.norm.fit is the MLE (divides by n): OK")
ll = log_loss(flips, np.full(N_FLIPS, p_hat))
print(f"sklearn log_loss at p_hat = {ll:.4f}   our Bernoulli mean NLL = {bern_mean_nll(p_hat, flips):.4f}")
np.testing.assert_allclose(ll, bern_mean_nll(p_hat, flips), rtol=1e-10)
lin = LinearRegression().fit(x_line.reshape(-1, 1), y_line)
print(f"LinearRegression: w = {show(np.array([lin.intercept_, lin.coef_[0]]))}   (Gaussian MLE)")
ridge = Ridge(alpha=lam).fit(x_line.reshape(-1, 1), y_line)  # penalizes coef_, not intercept_
w_ridge = np.array([ridge.intercept_, ridge.coef_[0]])
print(f"Ridge(alpha = {lam:.0f}): w = {show(w_ridge)}   (MAP with tau = {PRIOR_SD})")
np.testing.assert_allclose(w_ridge, w_map_closed, rtol=1e-6, atol=1e-8)
print("matches scikit-learn: OK")

# %%
# Key plot: Gaussian NLL in mu, Bernoulli NLL in p, and the MAP slope as the prior widens.
section("key plot")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), dpi=120)
ax = axes[0]
mus = np.linspace(0, 7, 300)
ax.plot(mus, [gauss_mean_nll(np.array([m, np.log(sigma_cf)]), x_gauss) for m in mus], "-", label=f"mean NLL, σ fixed at {sigma_cf:.2f}")
path_arr = np.array(path_gd)
ax.plot(path_arr[::4, 0], [gauss_mean_nll(np.array([m, np.log(sigma_cf)]), x_gauss) for m in path_arr[::4, 0]],
        "o", ms=3, color="tab:orange", label="μ along gradient descent (every 4th step)")
ax.axvline(mu_cf, color="k", ls="--", lw=1, label=f"sample mean {mu_cf:.2f}")
ax.set(xlabel="μ", ylabel="mean negative log-likelihood", title="Gaussian: the NLL is lowest at the sample mean")
ax.legend(loc="upper center", fontsize=7)
ax = axes[1]
ax.plot(grid_p, bern_mean_nll(grid_p, flips), "-", color="tab:green", label=f"log-loss of {N_FLIPS} flips")
ax.axvline(p_hat, color="k", ls="--", lw=1, label=f"heads / flips = {p_hat:.2f}")
ax.set_ylim(0.5, 2.0)
ax.set(xlabel="p", ylabel="mean negative log-likelihood", title="Bernoulli: log-loss is lowest at k / n")
ax.legend(loc="upper center", fontsize=7)
ax = axes[2]
taus = np.logspace(-2.5, 1, 60)
slopes = [np.linalg.solve(X.T @ X + (NOISE_SD**2 / t**2) * D, X.T @ y_line)[1] for t in taus]
ax.semilogx(taus, slopes, "-", color="tab:purple", label="MAP slope (closed form)")
for t in (0.01, 0.03, 0.1, 1.0):
    r = Ridge(alpha=NOISE_SD**2 / t**2).fit(x_line.reshape(-1, 1), y_line)
    ax.semilogx(t, r.coef_[0], "kD", ms=4)
ax.plot([], [], "kD", ms=4, label="sklearn Ridge(α = σ²/τ²)")
ax.axhline(w_ls[1], color="tab:blue", ls="--", lw=1, label=f"MLE slope {w_ls[1]:.3f}")
ax.axhline(0, color="0.5", lw=0.8)
ax.set(xlabel="prior standard deviation τ of the slope (log scale)", ylabel="slope",
       title="MAP: a tight prior pulls the slope to 0")
ax.legend(loc="center right", fontsize=7)
fig.tight_layout()
path_png = IMG / "a5_5_nll_and_map.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set N_GAUSS = 5 and compare sigma_hat (divides by n) with x_gauss.std(ddof=1). Which is closer to 2?
# 2. Put the prior on the intercept too (D = np.eye(2)) and compare with Ridge(fit_intercept=False) on X.
# 3. Replace the Gaussian noise on y_line with rng.standard_t(2): which loss would MLE suggest now?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
