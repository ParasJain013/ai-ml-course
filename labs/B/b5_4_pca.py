# %% [markdown]
# # B5.4 PCA from scratch via SVD, explained variance — lab
# Lesson: lessons/B/b5_4_pca.md
# Run: uv run python labs/B/b5_4_pca.py   (CPU, defaults: ~20 s)
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
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
OUT = ROOT / "outputs" / "b5_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def fmt(a: FloatArray, digits: int = 4) -> list:
    """Round an array for printing as a plain nested list."""
    return np.round(np.asarray(a, dtype=float), digits).tolist()


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU: PCA on 1,797 x 64 digits takes milliseconds. GPU_SCALE only adds more
# random directions to the "no direction beats PC1" check and more CV folds.
GPU_SCALE = False
N_RANDOM_DIRECTIONS = 100_000 if GPU_SCALE else 10_000
CV_FOLDS = 10 if GPU_SCALE else 5
K_SHOW = [2, 10, 30]  # the reconstructions the syllabus asks for
TEST_FRAC = 0.25

# %%
# Data: scikit-learn's bundled digits (8 x 8 grey images, pixel values 0-16), no download needed.
section("data")
digits = load_digits()
X_all = digits.data.astype(np.float64)  # (1797, 64)
y_all = digits.target.astype(np.int64)  # (1797,)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=TEST_FRAC, random_state=SEED, stratify=y_all)
n, d = X_tr.shape
print(f"digits: X {X_all.shape}, pixel values {X_all.min():.0f}..{X_all.max():.0f}, classes {np.unique(y_all).tolist()}")
print(f"train {X_tr.shape}   test {X_te.shape}")
print(f"pixels that are 0 in every training image: {int(np.sum(X_tr.std(axis=0) == 0))} of {d}")

# %%
# Worked example by hand: the four points of A3.3, now as a PCA.
section("worked example by hand")
X4 = np.array([[1.0, 4.0], [3.0, 5.0], [4.0, 4.0], [4.0, 7.0]])  # (4, 2)
mu4 = X4.mean(axis=0)  # (2,)
Xc4 = X4 - mu4  # (4, 2)
U4, s4, Vt4 = np.linalg.svd(Xc4, full_matrices=False)  # U (4, 2), s (2,), Vt (2, 2)
Vt4 = Vt4 * np.sign(Vt4[np.arange(2), np.argmax(np.abs(Vt4), axis=1)])[:, None]  # sign convention: see pca_fit
print(f"mean = {fmt(mu4)}   centred rows = {fmt(Xc4)}")
print(f"singular values of Xc = {fmt(s4)}   squared = {fmt(s4**2)}")
print(f"variances along the components = s^2/(n-1) = {fmt(s4**2 / 3)}   (eigenvalues of S in A3.3: 3 and 1)")
print(f"explained-variance ratio = {fmt(s4**2 / np.sum(s4**2))}")
print(f"components (rows of Vt) = {fmt(Vt4)}")
z4 = Xc4 @ Vt4[0]  # (4,) scores on PC1
print(f"PC1 scores Xc v1 = {fmt(z4)}   (= [-3, 0, 0, 3]/sqrt(2))")
X4_hat = mu4 + np.outer(z4, Vt4[0])  # (4, 2) rank-1 reconstruction
print(f"rank-1 reconstruction = {fmt(X4_hat)}")
print(f"squared reconstruction error = {np.sum((X4 - X4_hat) ** 2):.4f}   = sigma_2^2 = {s4[1] ** 2:.4f}")
x_new = np.array([5.0, 5.0])
z_new = (x_new - mu4) @ Vt4[0]
print(f"new point {fmt(x_new)}: score {z_new:.4f}, reconstruction {fmt(mu4 + z_new * Vt4[0])}")

# %%
# Why PC1 is the direction of largest variance: no random unit direction does better.
section("no direction beats PC1")
S_tr = np.cov(X_tr, rowvar=False)  # (64, 64), divides by n - 1
lam = np.linalg.eigvalsh(S_tr)[::-1]  # (64,), descending
W = rng.normal(size=(N_RANDOM_DIRECTIONS, d))  # (m, 64)
W /= np.linalg.norm(W, axis=1, keepdims=True)  # unit directions
var_along = np.einsum("md,de,me->m", W, S_tr, W)  # w^T S w for every direction, (m,)
print(f"largest eigenvalue of S (variance along PC1) = {lam[0]:.4f}")
print(f"best of {N_RANDOM_DIRECTIONS} random unit directions = {var_along.max():.4f}   mean = {var_along.mean():.4f}")
print(f"mean over random directions = trace(S)/d = {np.trace(S_tr) / d:.4f}")


# %%
# From scratch: PCA as the SVD of the centred training data.
def pca_fit(X: FloatArray, k: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit PCA on X (n, d). -> mean (d,), components (k, d), explained variance (k,), ratio (k,)."""
    mean = X.mean(axis=0)  # (d,)
    Xc = X - mean  # (n, d), centred
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # s (r,), Vt (r, d), r = min(n, d)
    # Each component's sign is arbitrary. Same convention as scikit-learn: largest-|entry| positive.
    signs = np.sign(Vt[np.arange(len(s)), np.argmax(np.abs(Vt), axis=1)])  # (r,)
    Vt = Vt * signs[:, None]
    var = s**2 / (len(X) - 1)  # (r,) variance along each component
    return mean, Vt[:k], var[:k], var[:k] / var.sum()


def pca_transform(X: FloatArray, mean: FloatArray, components: FloatArray) -> FloatArray:
    """Project onto the components. X (m, d), components (k, d) -> scores Z (m, k)."""
    return (X - mean) @ components.T


def pca_inverse(Z: FloatArray, mean: FloatArray, components: FloatArray) -> FloatArray:
    """Map scores back to feature space. Z (m, k), components (k, d) -> X_hat (m, d)."""
    return Z @ components + mean


def recon_error(X: FloatArray, X_hat: FloatArray) -> float:
    """Mean squared reconstruction error per example: average over rows of the squared distance."""
    return float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))


section("from scratch: PCA on digits")
mean_f, comps_f, var_f, ratio_f = pca_fit(X_tr, d)  # all 64 components
assert comps_f.shape == (d, d), comps_f.shape
print(f"components {comps_f.shape}, orthonormal: {np.allclose(comps_f @ comps_f.T, np.eye(d))}")
print(f"explained variance (first 5) = {fmt(var_f[:5], 3)}")
print(f"explained-variance ratio (first 5) = {fmt(ratio_f[:5])}")
cum = np.cumsum(ratio_f)
for target in [0.80, 0.90, 0.95, 0.99]:
    print(f"components needed for {target:.0%} of the variance: {int(np.searchsorted(cum, target) + 1)}")
print(f"sum of variances = total variance trace(S): {np.isclose(var_f.sum(), np.trace(S_tr))}")
print(f"variance along PC1 = largest eigenvalue of S: {np.isclose(var_f[0], lam[0])}")
for k in K_SHOW + [d]:
    Z_tr = pca_transform(X_tr, mean_f, comps_f[:k])  # (n, k)
    err_tr = recon_error(X_tr, pca_inverse(Z_tr, mean_f, comps_f[:k]))
    err_te = recon_error(X_te, pca_inverse(pca_transform(X_te, mean_f, comps_f[:k]), mean_f, comps_f[:k]))
    pred = (n - 1) * var_f[k:].sum() / n  # the dropped variance, as a per-row mean square
    print(f"k = {k:2d}: scores {Z_tr.shape}  train error {err_tr:8.4f} (dropped variance x (n-1)/n = {pred:8.4f})  "
          f"test error {err_te:8.4f}")

# %%
# With scikit-learn: same components (up to sign), variances and reconstructions.
section("with scikit-learn")
k = 10
pca = PCA(n_components=k, svd_solver="full").fit(X_tr)
mean_k, comps_k, var_k, ratio_k = pca_fit(X_tr, k)
signs = np.sign(np.sum(pca.components_ * comps_k, axis=1))  # +1 or -1 per component
print(f"sign flips needed to align with sklearn: {int(np.sum(signs < 0))} of {k}")
np.testing.assert_allclose(signs[:, None] * comps_k, pca.components_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(var_k, pca.explained_variance_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(ratio_k, pca.explained_variance_ratio_, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(pca_transform(X_te, mean_k, comps_k) * signs, pca.transform(X_te), rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(pca_inverse(pca_transform(X_te, mean_k, comps_k), mean_k, comps_k),
                           pca.inverse_transform(pca.transform(X_te)), rtol=1e-6, atol=1e-8)
print("components, explained_variance_, explained_variance_ratio_, transform, inverse_transform match sklearn PCA: OK")
pca95 = PCA(n_components=0.95, svd_solver="full").fit(X_tr)
print(f"PCA(n_components=0.95) keeps {pca95.n_components_} components (ratio sum {pca95.explained_variance_ratio_.sum():.4f})")

# %%
# Standardize or not: on wine, one feature (proline, in the hundreds) dwarfs the others.
section("standardize or not (wine)")
wine = load_wine()
Xw = wine.data
names = wine.feature_names
stds = Xw.std(axis=0, ddof=1)
print(f"wine: X {Xw.shape}; largest std {names[int(np.argmax(stds))]} = {stds.max():.1f}, smallest "
      f"{names[int(np.argmin(stds))]} = {stds.min():.3f}")
for label, Xin in [("raw", Xw), ("standardized", StandardScaler().fit_transform(Xw))]:
    _, comps_w, _, ratio_w = pca_fit(Xin, 2)
    top = int(np.argmax(np.abs(comps_w[0])))
    print(f"{label:<13} PC1 ratio {ratio_w[0]:.4f}, PC2 ratio {ratio_w[1]:.4f};  PC1's largest weight: "
          f"{names[top]} ({comps_w[0, top]:.4f})")

# %%
# Fit on training rows only: a PCA fitted on train + test makes the test rows look easier to reconstruct.
section("fit on train only")
for k in [10, 30]:
    m_tr, c_tr, _, _ = pca_fit(X_tr, k)
    m_all, c_all, _, _ = pca_fit(np.r_[X_tr, X_te], k)
    e_clean = recon_error(X_te, pca_inverse(pca_transform(X_te, m_tr, c_tr), m_tr, c_tr))
    e_leak = recon_error(X_te, pca_inverse(pca_transform(X_te, m_all, c_all), m_all, c_all))
    print(f"k = {k}: test reconstruction error, PCA fitted on train {e_clean:.4f}  on train+test {e_leak:.4f}  "
          f"({100 * (1 - e_leak / e_clean):.1f}% optimistic)")

# %%
# PCA inside a Pipeline: scaler, PCA and classifier refitted on every training fold.
section("PCA inside a Pipeline")
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
print(f"{'k':>3} {'cum. ratio':>10} {'CV accuracy':>12} {'std':>6}")
for k in [2, 5, 10, 20, 30, 64]:
    pipe = make_pipeline(PCA(n_components=k, svd_solver="full"), LogisticRegression(max_iter=2000))
    acc = cross_val_score(pipe, X_tr, y_tr, cv=cv)
    print(f"{k:>3} {cum[k - 1]:>10.4f} {acc.mean():>12.4f} {acc.std():>6.4f}")

# %%
# Key plot: cumulative explained variance, and test digits reconstructed from 2, 10 and 30 components.
section("key plot")
fig = plt.figure(figsize=(9, 4.2), dpi=120)
gs = fig.add_gridspec(4, 6, width_ratios=[2.2, 2.2, 1, 1, 1, 1], wspace=0.15, hspace=0.15)
ax = fig.add_subplot(gs[:, :2])
ks = np.arange(1, d + 1)
ax.plot(ks, cum, marker=".", label="cumulative explained-variance ratio")
for k_mark, marker in zip(K_SHOW, ["o", "s", "^"]):
    ax.plot(k_mark, cum[k_mark - 1], marker=marker, linestyle="none", markersize=9,
            label=f"k = {k_mark}: {cum[k_mark - 1]:.2f}")
ax.axhline(0.95, linestyle=":", color="gray", label="95% of the variance")
ax.set_xlabel("number of components k")
ax.set_ylabel("share of total variance kept")
ax.set_ylim(0, 1.02)
ax.set_xlim(0, d + 1)
ax.set_title("10 of 64 components keep 74% of the variance")
ax.legend(loc="lower right", fontsize=7)
show_rows = [int(np.flatnonzero(y_te == c)[0]) for c in [0, 3, 5, 8]]
col_titles = [f"k = {k_}" for k_ in K_SHOW] + ["original"]
for r, i in enumerate(show_rows):
    for c, k_ in enumerate(K_SHOW + [None]):
        axi = fig.add_subplot(gs[r, 2 + c])
        if k_ is None:
            img = X_te[i]
        else:
            img = pca_inverse(pca_transform(X_te[i:i + 1], mean_f, comps_f[:k_]), mean_f, comps_f[:k_])[0]
        axi.imshow(img.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        axi.set_xticks([])
        axi.set_yticks([])
        if r == 0:
            axi.set_title(col_titles[c], fontsize=8)
fig.savefig(IMG / "b5_4_explained_variance_reconstructions.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b5_4_explained_variance_reconstructions.png")
print(f"test digits shown (rows): {[int(y_te[i]) for i in show_rows]}")

# %%
# Try this (not graded):
# 1. Plot comps_f[0].reshape(8, 8) and comps_f[1].reshape(8, 8) as images: which strokes does each one weigh?
# 2. Scatter the first two PCA scores of X_tr coloured by y_tr. Which digits overlap? Compare with B5.5's t-SNE.
# 3. Add StandardScaler() before PCA in the pipeline cell. Does it help or hurt on pixels, which all share one unit
#    (0-16 ink), unlike wine's features? Look at what it does to the nearly-always-blank border pixels.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
