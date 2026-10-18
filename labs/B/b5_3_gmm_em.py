# %% [markdown]
# # B5.3 Gaussian mixture models and the EM algorithm — lab
# Lesson: lessons/B/b5_3_gmm_em.md
# Run: uv run python labs/B/b5_3_gmm_em.py   (CPU, defaults: ~10 s)
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
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

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
OUT = ROOT / "outputs" / "b5_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the synthetic datasets larger.
GPU_SCALE = False
N_PER_COMPONENT = 3_000 if GPU_SCALE else 300
N_ITER = 100  # EM iterations, the same for the from-scratch code and scikit-learn
REG_COVAR = 1e-6  # added to every covariance diagonal, scikit-learn's default

# %%
# Data: a 1-D mixture of three Gaussians and a 2-D mixture of three tilted, unequal clusters.
section("data")
true_1d = {"weights": [0.5, 0.3, 0.2], "means": [-2.0, 1.0, 4.0], "stds": [0.8, 0.5, 1.0]}
counts_1d = (np.array(true_1d["weights"]) * 3 * N_PER_COMPONENT).astype(int)
x_1d = np.concatenate([rng.normal(m, s, size=c) for m, s, c in zip(true_1d["means"], true_1d["stds"], counts_1d)])
x_1d = rng.permutation(x_1d)
print(f"1-D: {x_1d.shape} points from 3 components, sizes {counts_1d.tolist()}")

true_means_2d = np.array([[0.0, 0.0], [4.0, 3.0], [-1.0, 5.0]])
true_covs_2d = np.array([[[2.0, 1.5], [1.5, 1.5]], [[0.5, 0.0], [0.0, 0.5]], [[1.5, -1.2], [-1.2, 1.2]]])
sizes_2d = [N_PER_COMPONENT * 2, N_PER_COMPONENT, N_PER_COMPONENT]
parts = [rng.multivariate_normal(m, c, size=s) for m, c, s in zip(true_means_2d, true_covs_2d, sizes_2d)]
labels_2d = np.concatenate([np.full(s, k) for k, s in enumerate(sizes_2d)])
order = rng.permutation(len(labels_2d))
X = np.concatenate(parts)[order]
labels_2d = labels_2d[order]
n, d = X.shape
assert X.shape == (sum(sizes_2d), 2), X.shape
print(f"2-D: X {X.shape}, true component sizes {sizes_2d}")


# %%
# The model, from scratch. Every function works for any d; the 1-D case passes X of shape (n, 1).
def log_gaussian(X: FloatArray, mean: FloatArray, cov: FloatArray) -> FloatArray:
    """Log density of N(mean, cov) at each row. X (n, d), mean (d,), cov (d, d) -> (n,)."""
    d_ = X.shape[1]
    L = np.linalg.cholesky(cov)  # cov = L L^T; fails loudly if cov is not positive definite
    z = np.linalg.solve(L, (X - mean).T)  # (d, n): whitened deviations, so sum of squares = Mahalanobis^2
    maha2 = np.sum(z**2, axis=0)  # (n,)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d_ * np.log(2 * np.pi) + log_det + maha2)


def e_step(X: FloatArray, weights: FloatArray, means: FloatArray, covs: FloatArray) -> tuple[FloatArray, float]:
    """Responsibilities and mean log-likelihood. X (n, d), weights (K,), means (K, d), covs (K, d, d) -> (n, K), float."""
    log_joint = np.column_stack([np.log(w) + log_gaussian(X, m, c) for w, m, c in zip(weights, means, covs)])  # (n, K)
    log_px = logsumexp(log_joint, axis=1, keepdims=True)  # (n, 1): log p(x_i), computed stably
    log_resp = log_joint - log_px  # Bayes' rule in logs: log P(z=k | x) = log p(x, z=k) - log p(x)
    return np.exp(log_resp), float(np.mean(log_px))


def m_step(X: FloatArray, resp: FloatArray, reg: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Weighted MLE updates. X (n, d), resp (n, K) -> weights (K,), means (K, d), covs (K, d, d)."""
    nk = resp.sum(axis=0)  # (K,): effective number of points per component
    weights = nk / len(X)
    means = (resp.T @ X) / nk[:, None]  # (K, d): responsibility-weighted means
    covs = np.empty((resp.shape[1], X.shape[1], X.shape[1]))
    for k in range(resp.shape[1]):
        D = X - means[k]  # (n, d)
        covs[k] = (resp[:, k, None] * D).T @ D / nk[k] + reg * np.eye(X.shape[1])  # (d, d)
    return weights, means, covs


def fit_em(
    X: FloatArray, weights: FloatArray, means: FloatArray, covs: FloatArray, n_iter: int, reg: float
) -> tuple[FloatArray, FloatArray, FloatArray, list[float]]:
    """Run n_iter rounds of E-step then M-step. Returns the final parameters and the log-likelihood before each M-step."""
    history = []
    for _ in range(n_iter):
        resp, ll = e_step(X, weights, means, covs)
        history.append(ll)
        weights, means, covs = m_step(X, resp, reg)
    return weights, means, covs, history


# %%
# Worked example by hand: one E-step and one M-step on four points in 1-D.
section("worked example by hand")
x4 = np.array([[1.0], [2.0], [4.0], [5.0]])
w0, m0, c0 = np.array([0.5, 0.5]), np.array([[1.0], [5.0]]), np.array([[[1.0]], [[1.0]]])
resp4, ll4 = e_step(x4, w0, m0, c0)
print("responsibilities of component 1:", np.round(resp4[:, 0], 4).tolist())
w1, m1, c1 = m_step(x4, resp4, reg=0.0)
print(f"N_1 = {resp4[:, 0].sum():.4f}   pi_1 = {w1[0]:.4f}   mu_1 = {m1[0, 0]:.4f}   sigma_1^2 = {c1[0, 0, 0]:.4f}")
print(f"N_2 = {resp4[:, 1].sum():.4f}   pi_2 = {w1[1]:.4f}   mu_2 = {m1[1, 0]:.4f}   sigma_2^2 = {c1[1, 0, 0]:.4f}")
_, ll4_after = e_step(x4, w1, m1, c1)
print(f"total log-likelihood: before {4 * ll4:.4f}   after one EM step {4 * ll4_after:.4f}")

# %%
# From scratch, 1-D: three components. Run 1 starts from three random data points; run 2 from the 1/6, 1/2 and
# 5/6 quantiles. Both start with equal weights and the overall variance.
section("from scratch: 1-D EM")
X1 = x_1d[:, None]  # (n, 1)
c_init1 = np.repeat(np.var(X1) * np.eye(1)[None], 3, axis=0)
starts_1d = {"random points": X1[rng.choice(len(X1), size=3, replace=False)].copy(),
             "quantiles": np.quantile(X1, [1 / 6, 1 / 2, 5 / 6])[:, None]}
fits_1d = {}
for name, m_init1 in starts_1d.items():
    w_1, m_1, c_1, hist_1 = fit_em(X1, np.full(3, 1 / 3), m_init1, c_init1, N_ITER, REG_COVAR)
    assert np.all(np.diff(hist_1) >= -1e-10), "log-likelihood went down: a bug in the E- or M-step"
    fits_1d[name] = (w_1, m_1, c_1)
    print(f"start from {name:<13}: initial means {np.round(np.sort(m_init1[:, 0]), 3).tolist()}; mean log-likelihood per point: "
          f"iteration 1 {hist_1[0]:.4f}, 10 {hist_1[9]:.4f}, {N_ITER} {hist_1[-1]:.4f}  (never decreased: OK)")
print(f"{'':16}{'weight':>8}{'mean':>8}{'std':>8}")
for name, (w_1, m_1, c_1) in fits_1d.items():
    for k in np.argsort(m_1[:, 0]):
        print(f"{name:16}{w_1[k]:8.3f}{m_1[k, 0]:8.3f}{np.sqrt(c_1[k, 0, 0]):8.3f}")
for wt, mu, sd in zip(true_1d["weights"], true_1d["means"], true_1d["stds"]):
    print(f"{'true':16}{wt:8.3f}{mu:8.3f}{sd:8.3f}")

# %%
# From scratch, 2-D: full covariances. Start from three random points, the data covariance and equal weights.
section("from scratch: 2-D EM")
K = 3
w_init, m_init = np.full(K, 1 / K), X[rng.choice(n, size=K, replace=False)].copy()
c_init = np.repeat(np.cov(X, rowvar=False, bias=True)[None], K, axis=0)
w_2, m_2, c_2, hist_2 = fit_em(X, w_init, m_init, c_init, N_ITER, REG_COVAR)
assert np.all(np.diff(hist_2) >= -1e-10), "log-likelihood went down"
assert m_2.shape == (K, d) and c_2.shape == (K, d, d)
print(f"mean log-likelihood per point: iteration 1 {hist_2[0]:.4f}, 10 {hist_2[9]:.4f}, {N_ITER} {hist_2[-1]:.4f}  (never decreased: OK)")
resp_2, _ = e_step(X, w_2, m_2, c_2)
hard = resp_2.argmax(axis=1)
print(f"weights {np.round(np.sort(w_2), 3).tolist()}   (true {np.round(np.sort(np.array(sizes_2d) / n), 3).tolist()})")
print(f"adjusted Rand index of the hard assignments against the true components: {adjusted_rand_score(labels_2d, hard):.4f}")
unsure = np.sum(resp_2.max(axis=1) < 0.9)
print(f"points whose largest responsibility is below 0.9: {unsure} of {n} ({unsure / n:.1%})")

# %%
# With scikit-learn: the same starting parameters, the same number of iterations, so both run the same EM.
section("with scikit-learn: GaussianMixture")
gm = GaussianMixture(n_components=K, covariance_type="full", max_iter=N_ITER, tol=0.0, reg_covar=REG_COVAR,
                     weights_init=w_init, means_init=m_init, precisions_init=np.linalg.inv(c_init), random_state=SEED)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)  # tol=0 forces all N_ITER iterations, so sklearn warns
    gm.fit(X)
# sklearn computes the same updates through Cholesky factors of the precisions; float rounding differs in the
# last digits and EM can amplify it slightly over 100 iterations, so 1e-6 relative (not 1e-12) is the honest check.
np.testing.assert_allclose(w_2, gm.weights_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(m_2, gm.means_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(c_2, gm.covariances_, rtol=1e-6, atol=1e-8)
print("weights, means and covariances match sklearn GaussianMixture: OK")
np.testing.assert_allclose(e_step(X, w_2, m_2, c_2)[1], gm.score(X), rtol=1e-6)
np.testing.assert_allclose(resp_2, gm.predict_proba(X), rtol=1e-6, atol=1e-8)
print("mean log-likelihood and responsibilities match gm.score and gm.predict_proba: OK")
print(f"sklearn's own convergence check (tol=1e-3, default init): converged in {GaussianMixture(K, random_state=SEED).fit(X).n_iter_} iterations")

# %%
# Choosing K with BIC = -2 log L + p log n, with p the number of free parameters.
section("choosing K with BIC")


def n_params_full(K_: int, d_: int) -> int:
    """Free parameters of a full-covariance GMM: K-1 weights, K d means, K d(d+1)/2 covariance entries."""
    return (K_ - 1) + K_ * d_ + K_ * d_ * (d_ + 1) // 2


print(f"{'K':>2}{'params':>8}{'log L':>12}{'BIC (mine)':>12}{'BIC (sklearn)':>15}")
bics = []
for k_ in range(1, 7):
    g = GaussianMixture(k_, covariance_type="full", n_init=3, random_state=SEED).fit(X)
    log_l = g.score(X) * n
    bic = -2 * log_l + n_params_full(k_, d) * np.log(n)
    np.testing.assert_allclose(bic, g.bic(X), rtol=1e-10)
    bics.append(bic)
    print(f"{k_:>2}{n_params_full(k_, d):>8}{log_l:12.1f}{bic:12.1f}{g.bic(X):15.1f}")
print(f"lowest BIC at K = {int(np.argmin(bics)) + 1}; from-scratch BIC matches g.bic: OK")

# %%
# k-means is the hard-assignment limit: equal weights, shared spherical variance s^2 -> 0.
section("k-means as the limit of EM")
xq = np.array([[2.6]])  # 1.6 from the mean at 1 and 2.4 from the mean at 5
for s2 in [4.0, 1.0, 0.25, 0.01]:
    r, _ = e_step(xq, np.array([0.5, 0.5]), np.array([[1.0], [5.0]]), np.array([[[s2]], [[s2]]]))
    print(f"shared variance {s2:5.2f}: responsibility of the nearer mean (1.0) = {r[0, 0]:.4f}")
km = KMeans(K, n_init=10, random_state=SEED).fit(X)
print(f"adjusted Rand index on the tilted 2-D clusters: k-means {adjusted_rand_score(labels_2d, km.labels_):.4f}   "
      f"GMM (full covariance) {adjusted_rand_score(labels_2d, hard):.4f}")

# %%
# Why reg_covar exists: a component sitting on a single point has a likelihood that grows without bound.
section("degenerate component")
for s in [1.0, 0.1, 0.01, 0.001]:
    print(f"std {s:6.3f}: log density at its own mean = {multivariate_normal(0.0, s**2).logpdf(0.0):8.3f}")

# %%
# Key plot: the fitted ellipses (1 and 2 standard deviations) and the log-likelihood per iteration.
section("key plot: fitted ellipses")


def ellipse_axes(cov: FloatArray, n_std: float) -> tuple[float, float, float]:
    """Width, height and angle (degrees) of the n_std contour of N(., cov). cov (2, 2) -> 3 floats."""
    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues; columns are the axes' directions (A3.3)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))  # direction of the largest-variance axis
    return 2 * n_std * np.sqrt(vals[1]), 2 * n_std * np.sqrt(vals[0]), float(angle)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=120, gridspec_kw={"width_ratios": [1.3, 1]})
markers = ["o", "s", "^"]
for k in range(K):
    pts = X[hard == k]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, marker=markers[k], label=f"component {k + 1} ({w_2[k]:.2f})")
    for n_std, ls in [(1.0, "-"), (2.0, "--")]:
        wdt, hgt, ang = ellipse_axes(c_2[k], n_std)
        ax.add_patch(Ellipse(m_2[k], wdt, hgt, angle=ang, fill=False, color="black", linestyle=ls, linewidth=1.2))
    ax.plot(*m_2[k], "kx", markersize=8)
ax.set_xlabel("feature 1 (synthetic units)")
ax.set_ylabel("feature 2 (synthetic units)")
ax.set_title("EM finds tilted clusters of unequal size")
ax.legend(loc="lower right", fontsize=8, markerscale=2)
ax.set_aspect("equal", adjustable="datalim")
ax2.plot(np.arange(1, N_ITER + 1), hist_2, marker=".", markersize=3)
ax2.set_xscale("log")
ax2.set_xlabel("EM iteration (log scale)")
ax2.set_ylabel("mean log-likelihood per point (nats)")
ax2.set_title("The log-likelihood never goes down")
fig.tight_layout()
fig.savefig(IMG / "b5_3_em_ellipses.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b5_3_em_ellipses.png (solid: 1 std, dashed: 2 std)")

# %%
# Try this (not graded):
# 1. Start the 2-D EM from three points in the same true cluster (e.g. X[labels_2d == 0][:3]). Where does it end up,
#    and does n_init in GaussianMixture rescue it?
# 2. Fit GaussianMixture with covariance_type="spherical" and "tied". Which assumption does each one break here?
# 3. Set REG_COVAR = 0 and add 50 copies of one point to X. Run EM with K = 4 and watch one variance collapse.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
