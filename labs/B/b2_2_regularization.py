# %% [markdown]
# # B2.2 Regularization — lab
# Lesson: lessons/B/b2_2_regularization.md
# Run: uv run python labs/B/b2_2_regularization.py   (CPU, defaults: ~10 s)
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
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import r2_score

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
OUT = ROOT / "outputs" / "b2_2"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only makes the coefficient paths finer.
GPU_SCALE = False
N_ALPHAS = 400 if GPU_SCALE else 60  # points on each coefficient path
CD_TOL = 1e-10  # coordinate descent stops when no weight moves more than this in a sweep
CD_MAX_SWEEPS = 100_000
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: the diabetes set in raw units, split exactly as in B1.1 and B1.5, then standardized with
# training statistics. The target is centred on its training mean, so the intercept is that mean
# and is never penalized (A5.5).
section("data")
diab = load_diabetes(as_frame=True, scaled=False)
names = list(diab.feature_names)
X_all = diab.data.to_numpy(dtype=np.float64)  # (442, 10)
y_all = diab.target.to_numpy(dtype=np.float64)  # (442,)


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_5_linear_regression.py
    rows = gen.permutation(n)
    n_test, n_val = round(test_frac * n), round(val_frac * n)
    test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]
    return gen.permutation(train), gen.permutation(val), gen.permutation(test)


i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
mu, sd = X_all[i_tr].mean(axis=0), X_all[i_tr].std(axis=0)  # (d,) training statistics only
Z_tr, Z_va = (X_all[i_tr] - mu) / sd, (X_all[i_va] - mu) / sd  # (266, 10), (88, 10)
y_tr, y_va = y_all[i_tr], y_all[i_va]
y_bar = y_tr.mean()
yc_tr = y_tr - y_bar  # (n,) centred target
n, d = Z_tr.shape
print(f"train Z {Z_tr.shape}  val Z {Z_va.shape}  (test {len(i_te)} rows kept aside)")
print(f"training target mean (the intercept) {y_bar:.4f}")
print(f"correlation of s1 and s2 on train: {np.corrcoef(Z_tr[:, 4], Z_tr[:, 5])[0, 1]:.4f}")

# %%
# Worked example by hand: one centred feature, ridge and lasso in closed form.
section("worked example by hand")


def soft_threshold(a: float, t: float) -> float:
    """S(a, t) = sign(a) max(|a| - t, 0): shrink a towards 0 by t, and stop at 0."""
    return float(np.sign(a) * max(abs(a) - t, 0.0))


x4 = np.array([-2.0, -1.0, 1.0, 2.0])
y4 = np.array([-3.0, -1.0, 0.0, 4.0])
sxy, sxx = float(x4 @ y4), float(x4 @ x4)
print(f"x = {x4.tolist()}   y = {y4.tolist()}   (both have mean 0, so b = 0)")
print(f"sum xy = {sxy}   sum x^2 = {sxx}   least squares w = {sxy / sxx:.4f}")
for lam in [5.0, 10.0]:
    print(f"ridge, lambda = {lam:>4}: w = sum xy / (sum x^2 + lambda) = {sxy}/{sxx + lam} = {sxy / (sxx + lam):.4f}")
print(f"lasso: a = sum xy / n = {sxy / 4}   c = sum x^2 / n = {sxx / 4}")
for lam in [1.0, 3.75, 4.0]:
    w_l = soft_threshold(sxy / 4, lam) / (sxx / 4)
    print(f"lasso, lambda = {lam:>4}: w = S(a, lambda) / c = S({sxy / 4}, {lam}) / {sxx / 4} = {w_l:.4f}")
r4 = Ridge(alpha=5.0).fit(x4[:, None], y4).coef_[0]
l4 = Lasso(alpha=1.0).fit(x4[:, None], y4).coef_[0]
print(f"sklearn Ridge(alpha=5) w = {r4:.4f}   Lasso(alpha=1) w = {l4:.4f}")

# %%
# From scratch: ridge in closed form (A5.5). Penalty lambda * |w|^2 on the weights, not the intercept.
section("from scratch: ridge")


def fit_ridge(Z: FloatArray, y: FloatArray, lam: float) -> tuple[float, FloatArray]:
    """Minimize |y - Z w - b|^2 + lam |w|^2 for standardized Z. Z (n, d), y (n,) -> (b, w (d,))."""
    b = float(y.mean())  # with centred columns the intercept decouples: it is the target mean
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - b))  # (d,)
    return b, w


for lam in [0.0, 10.0, 100.0]:
    b_r, w_r = fit_ridge(Z_tr, y_tr, lam)
    sk_r = Ridge(alpha=lam).fit(Z_tr, y_tr)
    np.testing.assert_allclose(w_r, sk_r.coef_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b_r, sk_r.intercept_, rtol=1e-6, atol=1e-8)
    print(f"lambda {lam:>6}: s1 {w_r[4]:8.4f}  s2 {w_r[5]:8.4f}  bmi {w_r[2]:7.4f}  |w| {np.linalg.norm(w_r):7.4f}  "
          f"val R^2 {r2_score(y_va, Z_va @ w_r + b_r):.4f}")
print("matches sklearn Ridge(alpha=lambda) at every lambda: OK")
eig = np.linalg.eigvalsh(Z_tr.T @ Z_tr)
print(f"eigenvalues of Z^T Z: smallest {eig[0]:.4f}  largest {eig[-1]:.4f}  "
      f"condition number {eig[-1] / eig[0]:.1f}")
print(f"with lambda = 10: condition number of Z^T Z + 10 I = {(eig[-1] + 10) / (eig[0] + 10):.1f}")

# %%
# From scratch: lasso by coordinate descent with soft-thresholding.
section("from scratch: lasso by coordinate descent")


def fit_elastic_net_cd(
    Z: FloatArray, y: FloatArray, alpha: float, l1_ratio: float = 1.0, tol: float = CD_TOL
) -> tuple[float, FloatArray, int]:
    """Minimize (1/2n)|y - Zw - b|^2 + alpha*l1_ratio*|w|_1 + (alpha*(1 - l1_ratio)/2)|w|^2.

    Z (n, d) standardized, y (n,) -> (b, w (d,), sweeps). l1_ratio = 1 is the lasso.
    """
    n_rows, n_feat = Z.shape
    b = float(y.mean())
    r = y - b  # (n,) residual of the current model, kept up to date
    w = np.zeros(n_feat)
    col_sq = (Z**2).sum(axis=0) / n_rows  # (d,) each is 1 for standardized columns
    for sweep in range(1, CD_MAX_SWEEPS + 1):
        max_change = 0.0
        for j in range(n_feat):
            r += Z[:, j] * w[j]  # partial residual: the model without feature j
            a_j = Z[:, j] @ r / n_rows  # how much feature j alone would explain
            w_new = soft_threshold(a_j, alpha * l1_ratio) / (col_sq[j] + alpha * (1 - l1_ratio))
            r -= Z[:, j] * w_new  # put feature j back with its new weight
            max_change = max(max_change, abs(w_new - w[j]))
            w[j] = w_new
        if max_change < tol:
            return b, w, sweep
    return b, w, CD_MAX_SWEEPS


for alpha in [0.1, 1.0, 5.0]:
    b_l, w_l, sweeps = fit_elastic_net_cd(Z_tr, y_tr, alpha)
    sk_l = Lasso(alpha=alpha, tol=1e-12, max_iter=1_000_000).fit(Z_tr, y_tr)
    # both solvers stop at a tolerance; 1e-6 absolute on weights of size ~1-40 is far below what matters
    np.testing.assert_allclose(w_l, sk_l.coef_, atol=1e-6)
    kept = [names[j] for j in np.flatnonzero(w_l)]
    print(f"alpha {alpha:>4}: {sweeps:>4} sweeps  non-zero {len(kept):>2}/{d}  val R^2 "
          f"{r2_score(y_va, Z_va @ w_l + b_l):.4f}  kept {kept}")
print("matches sklearn Lasso(alpha) at every alpha: OK")
b_l1, w_l1, _ = fit_elastic_net_cd(Z_tr, y_tr, 1.0)
print("lasso alpha 1 weights:", dict(zip(names, (np.round(w_l1, 4) + 0.0).tolist())))
alpha_max = float(np.max(np.abs(Z_tr.T @ yc_tr)) / n)
print(f"smallest alpha that zeroes every weight: max_j |z_j . (y - y_bar)| / n = {alpha_max:.4f}")

# %%
# Elastic net, and what each penalty does with two correlated features (s1 and s2).
section("from scratch: elastic net and correlated features")
b_en, w_en, _ = fit_elastic_net_cd(Z_tr, y_tr, alpha=1.0, l1_ratio=0.5)
sk_en = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=1e-12, max_iter=1_000_000).fit(Z_tr, y_tr)
np.testing.assert_allclose(w_en, sk_en.coef_, atol=1e-6)
print(f"elastic net alpha 1, l1_ratio 0.5: non-zero {np.count_nonzero(w_en)}/{d}   matches sklearn ElasticNet: OK")
pair = [4, 5]  # s1 (total cholesterol) and s2 (LDL), correlation 0.88 on train
for label, w_pair in [
    ("least squares", fit_ridge(Z_tr, y_tr, 0.0)[1][pair]),
    ("ridge, lambda 10", fit_ridge(Z_tr, y_tr, 10.0)[1][pair]),
    ("lasso, alpha 1", w_l1[pair]),
    ("elastic net, alpha 1, l1_ratio 0.5", w_en[pair]),
]:
    print(f"{label:<36} s1 {w_pair[0] + 0.0:9.4f}   s2 {w_pair[1] + 0.0:9.4f}")

# %%
# Adding L2 to the B2.1 logistic regression. Our objective is mean log-loss + (lam/2)|w|^2;
# scikit-learn's is C * sum(log-loss) + (1/2)|w|^2. Divide that by C n: lam = 1 / (C n).
section("L2 logistic regression from scratch")


def sigmoid(z: FloatArray) -> FloatArray:
    """Stable sigmoid. z (...) -> (...)."""
    # from labs/B/b2_1_logistic_regression.py, written with |z|
    ez = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


def fit_logistic_l2(
    Z: FloatArray, y: FloatArray, lam: float, max_iters: int = 200_000, tol: float = 1e-8
) -> tuple[float, FloatArray, int]:
    """GD on mean log-loss + (lam/2)|w|^2, bias unpenalized. Z (n, d), y (n,) -> (b, w (d,), steps)."""
    A = np.column_stack([np.ones(len(Z)), Z])  # (n, d + 1)
    theta = np.zeros(A.shape[1])
    pen = np.r_[0.0, np.ones(Z.shape[1])]  # (d + 1,) 0 for the bias: never penalized
    lr = 1.0 / (np.linalg.eigvalsh(A.T @ A / len(y))[-1] / 4 + lam)  # curvature bound + lam
    for step in range(1, max_iters + 1):
        grad = A.T @ (sigmoid(A @ theta) - y) / len(y) + lam * pen * theta  # (d + 1,)
        if np.linalg.norm(grad) < tol:
            return float(theta[0]), theta[1:], step
        theta -= lr * grad
    return float(theta[0]), theta[1:], max_iters


bc = load_breast_cancer()
Xb, yb = bc.data.astype(np.float64), (bc.target == 0).astype(np.float64)  # all 30 features, 1 = malignant
ib = np.random.default_rng(SEED).permutation(len(yb))
ib_tr, ib_va = ib[:400], ib[400:]
mub, sdb = Xb[ib_tr].mean(0), Xb[ib_tr].std(0)
Zb_tr, Zb_va = (Xb[ib_tr] - mub) / sdb, (Xb[ib_va] - mub) / sdb
for C in [0.1, 1.0]:
    lam = 1.0 / (C * len(ib_tr))
    b_lg, w_lg, steps_lg = fit_logistic_l2(Zb_tr, yb[ib_tr], lam)
    sk_lg = LogisticRegression(C=C, max_iter=10_000, tol=1e-10).fit(Zb_tr, yb[ib_tr])
    p_mine = sigmoid(Zb_va @ w_lg + b_lg)
    p_sk = sk_lg.predict_proba(Zb_va)[:, 1]
    np.testing.assert_allclose(p_mine, p_sk, atol=1e-5)
    print(f"C {C:>4} -> lambda = 1/(C n) = {lam:.6f}: {steps_lg:>5} steps  |w| {np.linalg.norm(w_lg):.4f}  "
          f"max |p_mine - p_sklearn| {np.max(np.abs(p_mine - p_sk)):.1e}")
print("L2 logistic regression matches LogisticRegression(C) on all 30 features: OK")
sk_l1 = LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=100_000, tol=1e-8).fit(Zb_tr, yb[ib_tr])
print(f"L1 logistic regression (C 0.1, l1_ratio 1, saga): non-zero weights {np.count_nonzero(sk_l1.coef_)}/30")

# %%
# Choosing the strength: validation R^2 along the ridge and lasso paths.
section("choosing lambda on the validation set")
ridge_lams = np.logspace(-2, 4, N_ALPHAS)
lasso_alphas = np.logspace(-2, np.log10(alpha_max), N_ALPHAS)
ridge_path = np.array([fit_ridge(Z_tr, y_tr, lam)[1] for lam in ridge_lams])  # (N_ALPHAS, d)
lasso_path = np.array([fit_elastic_net_cd(Z_tr, y_tr, a)[1] for a in lasso_alphas])  # (N_ALPHAS, d)
r2_ridge = np.array([r2_score(y_va, Z_va @ w + y_bar) for w in ridge_path])
r2_lasso = np.array([r2_score(y_va, Z_va @ w + y_bar) for w in lasso_path])
print(f"ridge: best val R^2 {r2_ridge.max():.4f} at lambda {ridge_lams[r2_ridge.argmax()]:.2f}   "
      f"(lambda -> 0: {r2_ridge[0]:.4f})")
print(f"lasso: best val R^2 {r2_lasso.max():.4f} at alpha {lasso_alphas[r2_lasso.argmax()]:.3f} "
      f"with {np.count_nonzero(lasso_path[r2_lasso.argmax()])} non-zero weights")
order = np.argsort([lasso_alphas[np.flatnonzero(lasso_path[:, j] != 0).max()] if np.any(lasso_path[:, j]) else 0
                    for j in range(d)])[::-1]
print("order in which features enter the lasso as alpha falls:", [names[j] for j in order])
print("ridge weights at lambda = 1e4 (none exactly 0):", show(ridge_path[-1]))

# %%
# Key plot: left, why L1 gives zeros (geometry); right, the two coefficient paths.
section("key plot")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
# Left: contours of a 2-feature squared-error loss whose minimum is at (2, 0.6), with an L1 ball
# and an L2 ball of the same radius t = 1. The constrained solutions are where the smallest contour
# touches each ball. Computed on a fine grid of the ball's boundary.
w_ls = np.array([2.0, 0.6])
Q = np.array([[1.0, 0.4], [0.4, 0.6]])  # (2, 2) curvature of the loss: elongated, tilted ellipses
g = np.linspace(-1.6, 3.2, 400)
G0, G1 = np.meshgrid(g, g)
D0, D1 = G0 - w_ls[0], G1 - w_ls[1]
loss_grid = Q[0, 0] * D0**2 + 2 * Q[0, 1] * D0 * D1 + Q[1, 1] * D1**2
ang = np.linspace(0, 2 * np.pi, 4001)
l2_ball = np.column_stack([np.cos(ang), np.sin(ang)])  # (4001, 2) |w|_2 = 1
l1_ball = l2_ball / np.abs(l2_ball).sum(axis=1, keepdims=True)  # (4001, 2) |w|_1 = 1


def quad_loss(W: FloatArray) -> FloatArray:
    """Loss (w - w_ls)^T Q (w - w_ls) for rows of W. W (m, 2) -> (m,)."""
    D = W - w_ls
    return np.einsum("mi,ij,mj->m", D, Q, D)


w_l2_star = l2_ball[quad_loss(l2_ball).argmin()]
w_l1_star = l1_ball[quad_loss(l1_ball).argmin()]
print(f"geometry: unconstrained minimum {show(w_ls)}   L2 ball solution {show(w_l2_star)}   "
      f"L1 ball solution {show(w_l1_star)}")
levels = sorted({float(quad_loss(w_l1_star[None])[0]), float(quad_loss(w_l2_star[None])[0]), 0.3, 3.0, 5.0})
ax0.contour(G0, G1, loss_grid, levels=levels, colors="0.6", linewidths=0.8)
ax0.plot(l1_ball[:, 0], l1_ball[:, 1], "-", color="tab:blue", label="L1 ball |w|_1 = 1")
ax0.plot(l2_ball[:, 0], l2_ball[:, 1], "--", color="tab:orange", label="L2 ball |w|_2 = 1")
ax0.plot(*w_ls, "k+", ms=10, label="least squares")
ax0.plot(*w_l1_star, "o", color="tab:blue", ms=7, label="L1 solution (w2 = 0)")
ax0.plot(*w_l2_star, "s", color="tab:orange", ms=7, label="L2 solution")
ax0.axhline(0, color="0.85", lw=0.8, zorder=0)
ax0.axvline(0, color="0.85", lw=0.8, zorder=0)
ax0.set_aspect("equal")
ax0.set_xlim(-1.6, 3.2)
ax0.set_ylim(-1.6, 2.2)
ax0.set_xlabel("weight w1")
ax0.set_ylabel("weight w2")
ax0.set_title("Loss contours first touch the L1 diamond at a corner")
ax0.legend(loc="lower right", fontsize=7)
for j, style in zip([2, 8, 4, 5], ["-", "--", "-.", ":"]):  # bmi, s5, s1, s2
    ax1.plot(lasso_alphas, lasso_path[:, j], style, label=f"{names[j]} (lasso)")
ax1.set_xscale("log")
ax1.axhline(0, color="0.7", lw=0.8)
ax1.set_xlabel("lasso alpha (log scale)")
ax1.set_ylabel("weight per 1 sd of the feature")
ax1.set_title("Lasso weights hit exactly zero one by one")
ax1.legend(loc="upper right", fontsize=7)
fig.tight_layout()
fig.savefig(IMG / "b2_2_l1_vs_l2.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b2_2_l1_vs_l2.png")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)  # scratch: the ridge path, for comparison
for j in range(d):
    ax.plot(ridge_lams, ridge_path[:, j], label=names[j])
ax.set_xscale("log")
ax.set_xlabel("ridge lambda (log scale)")
ax.set_ylabel("weight per 1 sd")
ax.set_title("Ridge weights shrink smoothly and never reach zero")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "ridge_path.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# %%
# Try this (not graded):
# 1. Do not standardize: fit the lasso on raw diabetes units. Which features does it drop first, and why?
# 2. Penalize the intercept too (add a column of ones to Z and run fit_ridge). What happens to predictions?
# 3. Run fit_elastic_net_cd with l1_ratio 0.1, 0.5, 0.9 and watch the s1/s2 weights change.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
