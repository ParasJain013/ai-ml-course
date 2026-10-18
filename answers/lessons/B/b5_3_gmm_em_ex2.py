# %% [markdown]
# # B5.3 Gaussian mixture models and EM — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b5_3_gmm_em_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time
import warnings

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_PER_COMPONENT = 300
N_ITER = 100
REG_COVAR = 1e-6


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
x = np.array([0.0, 2.0, 6.0])


def resp_1d(x: FloatArray, w: FloatArray, mu: FloatArray, var: FloatArray) -> FloatArray:
    """Responsibilities for a 1-D mixture. x (n,), w, mu, var (K,) -> (n, K)."""
    log_joint = np.log(w) - 0.5 * np.log(2 * np.pi * var) - (x[:, None] - mu) ** 2 / (2 * var)
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


g = resp_1d(x, np.array([0.5, 0.5]), np.array([0.0, 6.0]), np.array([4.0, 4.0]))
print("(a) gamma_i1 =", np.round(g[:, 0], 5).tolist())
n1 = g[:, 0].sum()
mu1 = g[:, 0] @ x / n1
var1 = g[:, 0] @ (x - mu1) ** 2 / n1
print(f"(b) N_1 = {n1:.5f}   pi_1 = {n1 / 3:.5f}   mu_1 = {mu1:.5f}   sigma_1^2 = {var1:.5f}")
g_c = resp_1d(x, np.array([0.2, 0.8]), np.array([0.0, 6.0]), np.array([4.0, 4.0]))
print(f"(c) with pi = (0.2, 0.8): gamma_21 = {g_c[1, 0]:.5f}")
K_, d_ = 4, 3
print(f"(d) K=4, d=3: full {(K_ - 1) + K_ * d_ + K_ * d_ * (d_ + 1) // 2} parameters, diagonal {(K_ - 1) + 2 * K_ * d_}")

# %%
# The lab's 2-D data, generated with the same seed and the same calls.
section("data (as in the lab)")
rng = np.random.default_rng(SEED)
counts_1d = (np.array([0.5, 0.3, 0.2]) * 3 * N_PER_COMPONENT).astype(int)
_ = np.concatenate([rng.normal(m, s, size=c) for m, s, c in zip([-2.0, 1.0, 4.0], [0.8, 0.5, 1.0], counts_1d)])
_ = rng.permutation(_)
true_means = np.array([[0.0, 0.0], [4.0, 3.0], [-1.0, 5.0]])
true_covs = np.array([[[2.0, 1.5], [1.5, 1.5]], [[0.5, 0.0], [0.0, 0.5]], [[1.5, -1.2], [-1.2, 1.2]]])
sizes = [N_PER_COMPONENT * 2, N_PER_COMPONENT, N_PER_COMPONENT]
parts = [rng.multivariate_normal(m, c, size=s) for m, c, s in zip(true_means, true_covs, sizes)]
order = rng.permutation(sum(sizes))
X = np.concatenate(parts)[order]
n, d = X.shape
print(f"X {X.shape}")


# %%
def e_step_diag(X: FloatArray, weights: FloatArray, means: FloatArray, variances: FloatArray) -> tuple[FloatArray, float]:
    """Responsibilities and mean log-likelihood for diagonal covariances. X (n, d), weights (K,), means and variances (K, d)
    -> (n, K), float."""
    # log N(x | mu, diag(v)) = sum over features of the 1-D log-densities: shape (n, K, d) summed over d
    D2 = (X[:, None, :] - means[None]) ** 2 / variances[None]  # (n, K, d)
    log_dens = -0.5 * np.sum(np.log(2 * np.pi * variances)[None] + D2, axis=2)  # (n, K)
    log_joint = np.log(weights)[None] + log_dens
    log_px = logsumexp(log_joint, axis=1, keepdims=True)  # (n, 1)
    return np.exp(log_joint - log_px), float(np.mean(log_px))


def m_step_diag(X: FloatArray, resp: FloatArray, reg: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Weighted MLE with diagonal covariances. X (n, d), resp (n, K) -> weights (K,), means (K, d), variances (K, d)."""
    nk = resp.sum(axis=0)  # (K,)
    means = resp.T @ X / nk[:, None]  # (K, d)
    # sum_i g_ik (x_ij - mu_kj)^2 / N_k = E_g[x^2] - mu^2, computed directly to avoid cancellation
    variances = np.stack([resp[:, k] @ (X - means[k]) ** 2 for k in range(len(nk))]) / nk[:, None] + reg  # (K, d)
    return nk / len(X), means, variances


section("diagonal EM from scratch versus GaussianMixture(covariance_type='diag')")
K = 3
init_rng = np.random.default_rng(SEED)
w0, m0 = np.full(K, 1 / K), X[init_rng.choice(n, size=K, replace=False)].copy()
v0 = np.repeat(X.var(axis=0)[None], K, axis=0)  # (K, d)
w, m, v = w0, m0, v0
hist = []
for _ in range(N_ITER):
    resp, ll = e_step_diag(X, w, m, v)
    hist.append(ll)
    w, m, v = m_step_diag(X, resp, REG_COVAR)
assert np.all(np.diff(hist) >= -1e-10), "log-likelihood went down"
gm = GaussianMixture(K, covariance_type="diag", max_iter=N_ITER, tol=0.0, reg_covar=REG_COVAR,
                     weights_init=w0, means_init=m0, precisions_init=1 / v0, random_state=SEED)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    gm.fit(X)
np.testing.assert_allclose(w, gm.weights_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(m, gm.means_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(v, gm.covariances_, rtol=1e-6, atol=1e-8)
print(f"final mean log-likelihood {hist[-1]:.4f} (never decreased); weights, means, variances match sklearn: OK")

# %%
section("BIC: full versus diagonal, K = 1..6")


def n_params(K_: int, d_: int, cov_type: str) -> int:
    """Free parameters: K-1 weights, K d means, and K d(d+1)/2 (full) or K d (diag) covariance entries."""
    cov = K_ * d_ * (d_ + 1) // 2 if cov_type == "full" else K_ * d_
    return (K_ - 1) + K_ * d_ + cov


print(f"{'K':>2}{'full params':>13}{'full BIC':>10}{'diag params':>13}{'diag BIC':>10}")
best = (np.inf, "", 0)
for k_ in range(1, 7):
    row = [f"{k_:>2}"]
    for ct in ["full", "diag"]:
        g_ = GaussianMixture(k_, covariance_type=ct, n_init=3, random_state=SEED).fit(X)
        bic = -2 * g_.score(X) * n + n_params(k_, d, ct) * np.log(n)
        np.testing.assert_allclose(bic, g_.bic(X), rtol=1e-10)
        best = min(best, (bic, ct, k_))
        row.append(f"{n_params(k_, d, ct):>13}{bic:10.1f}")
    print("".join(row))
print(f"lowest BIC: covariance_type={best[1]!r}, K = {best[2]} (BIC {best[0]:.1f})")
diag_best_k = min(range(1, 7), key=lambda k_: GaussianMixture(k_, covariance_type="diag", n_init=3, random_state=SEED).fit(X).bic(X))
print(f"best K among diagonal models: {diag_best_k}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
