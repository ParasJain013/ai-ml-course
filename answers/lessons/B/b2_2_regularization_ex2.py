# %% [markdown]
# # B2.2 Regularization — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b2_2_regularization_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Lasso, Ridge

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_BOOT = 200
ALPHA, L1_RATIO = 1.0, 0.5
CD_TOL, CD_MAX_SWEEPS = 1e-8, 100_000


def section(name: str) -> None:
    print(f"\n== {name} ==")


def soft_threshold(a: float, t: float) -> float:
    """S(a, t) = sign(a) max(|a| - t, 0)."""
    return float(np.sign(a) * max(abs(a) - t, 0.0))


def fit_elastic_net_cd(Z: FloatArray, y: FloatArray, alpha: float, l1_ratio: float = 1.0) -> FloatArray:
    """Coordinate descent on scikit-learn's elastic-net objective. Z (n, d) standardized, y (n,) -> w (d,)."""
    # from labs/B/b2_2_regularization.py, returning only the weights
    n_rows, n_feat = Z.shape
    r = y - y.mean()
    w = np.zeros(n_feat)
    col_sq = (Z**2).sum(axis=0) / n_rows
    for _ in range(CD_MAX_SWEEPS):
        max_change = 0.0
        for j in range(n_feat):
            r += Z[:, j] * w[j]
            w_new = soft_threshold(Z[:, j] @ r / n_rows, alpha * l1_ratio) / (col_sq[j] + alpha * (1 - l1_ratio))
            r -= Z[:, j] * w_new
            max_change = max(max_change, abs(w_new - w[j]))
            w[j] = w_new
        if max_change < CD_TOL:
            break
    return w


# %%
section("exercise 1 check")
n = 10
a = np.array([3.0, -0.5])  # (1/n) Z^T (y - y_bar); (1/n) Z^T Z = I
print(f"(a) least squares w = {a.tolist()}")
for lam in [1.0, 4.0]:
    print(f"(b) lasso lambda {lam}: w = {[soft_threshold(v, lam) + 0.0 for v in a]}")
print(f"(c) ridge (sum form) lambda 10: w = {(n * a / (n + 10.0)).tolist()}")
print(f"(d) elastic net lambda 1, rho 0.5: w = {np.round([soft_threshold(v, 0.5) / 1.5 + 0.0 for v in a], 4).tolist()}")
print(f"(e) lambda_max = max |a_j| = {np.max(np.abs(a))}")
# A concrete Z with exactly these statistics, to check against scikit-learn.
Q, _ = np.linalg.qr(np.random.default_rng(SEED).normal(size=(n, 3)))  # orthonormal columns (n, 3)
Q -= Q.mean(axis=0)  # centre; re-orthonormalize so (1/n) Z^T Z = I exactly
Q, _ = np.linalg.qr(np.column_stack([np.ones(n), Q]))
Z1 = np.sqrt(n) * Q[:, 1:3]  # (10, 2): columns mean 0, (1/n) Z^T Z = I
y1 = Z1 @ a + Q[:, 3]  # plus a bit of noise orthogonal to everything
print(f"    sklearn Lasso(1) {np.round(Lasso(alpha=1.0).fit(Z1, y1).coef_, 4) + 0.0}  "
      f"Ridge(10) {np.round(Ridge(alpha=10.0).fit(Z1, y1).coef_, 4)}  "
      f"ElasticNet(1, 0.5) {np.round(ElasticNet(alpha=1.0, l1_ratio=0.5).fit(Z1, y1).coef_, 4) + 0.0}")

# %%
section("bootstrap stability on diabetes")
diab = load_diabetes(as_frame=True, scaled=False)
names = list(diab.feature_names)
X_all = diab.data.to_numpy(dtype=np.float64)
y_all = diab.target.to_numpy(dtype=np.float64)
gen = np.random.default_rng(SEED)  # the lab's split (B1.5)
rows = gen.permutation(len(y_all))
n_te, n_va = round(0.2 * len(y_all)), round(0.2 * len(y_all))
i_tr = gen.permutation(rows[n_te + n_va:])
X_tr, y_tr = X_all[i_tr], y_all[i_tr]
rng = np.random.default_rng(SEED)
W_lasso = np.zeros((N_BOOT, X_tr.shape[1]))
W_enet = np.zeros((N_BOOT, X_tr.shape[1]))
for k in range(N_BOOT):
    idx = rng.integers(0, len(y_tr), len(y_tr))  # sample rows with replacement
    Xb, yb = X_tr[idx], y_tr[idx]
    Zb = (Xb - Xb.mean(0)) / Xb.std(0)  # each sample standardized with its own statistics
    W_lasso[k] = fit_elastic_net_cd(Zb, yb, ALPHA)
    W_enet[k] = fit_elastic_net_cd(Zb, yb, ALPHA, L1_RATIO)
    if k == 0:
        sk = Lasso(alpha=ALPHA, tol=1e-12, max_iter=1_000_000).fit(Zb, yb)
        np.testing.assert_allclose(W_lasso[0], sk.coef_, atol=1e-5)
        print("bootstrap sample 0 matches sklearn Lasso: OK")
print(f"{'feature':<8} {'lasso: % non-zero':>18} {'mean':>8} {'sd':>7}   {'enet: % non-zero':>17} {'mean':>8} {'sd':>7}")
for j, name in enumerate(names):
    print(f"{name:<8} {100 * np.mean(W_lasso[:, j] != 0):>17.0f}% {W_lasso[:, j].mean():>8.2f} {W_lasso[:, j].std():>7.2f}   "
          f"{100 * np.mean(W_enet[:, j] != 0):>16.0f}% {W_enet[:, j].mean():>8.2f} {W_enet[:, j].std():>7.2f}")
both = np.mean((W_lasso[:, 4] != 0) & (W_lasso[:, 5] != 0))
only_s1 = np.mean((W_lasso[:, 4] != 0) & (W_lasso[:, 5] == 0))
only_s2 = np.mean((W_lasso[:, 4] == 0) & (W_lasso[:, 5] != 0))
print(f"lasso s1/s2: both non-zero {both:.2f}   only s1 {only_s1:.2f}   only s2 {only_s2:.2f}   "
      f"neither {1 - both - only_s1 - only_s2:.2f}")
print(f"lasso: number of non-zero weights per fit ranges from {int((W_lasso != 0).sum(1).min())} "
      f"to {int((W_lasso != 0).sum(1).max())}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
