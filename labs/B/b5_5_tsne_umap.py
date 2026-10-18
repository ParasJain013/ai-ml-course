# %% [markdown]
# # B5.5 t-SNE and UMAP — lab
# Lesson: lessons/B/b5_5_tsne_umap.md
# Run: uv run python labs/B/b5_5_tsne_umap.py   (CPU, defaults: ~60 s; the very first run adds ~60 s
#      while numba compiles UMAP, cached afterwards)
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
# umap-learn 0.5.12 warns that random_state forces n_jobs=1; we want that (reproducible output)
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import umap
from scipy.spatial.distance import squareform
from scipy.special import entr
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.manifold._t_sne import _joint_probabilities  # private API, VERIFY (as of 2026-10)
from sklearn.metrics import pairwise_distances, silhouette_score

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
OUT = ROOT / "outputs" / "b5_5"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU (scikit-learn's t-SNE and umap-learn run on the CPU); GPU_SCALE only makes
# the synthetic demos larger and runs t-SNE longer, which a Colab/Kaggle CPU runtime handles in a few minutes.
GPU_SCALE = False
PERPLEXITIES = [5, 30, 100]
N_NEIGHBORS = [5, 15, 50]
TSNE_ITER = 2_000 if GPU_SCALE else 1_000
N_PER_CLUSTER = 1_000 if GPU_SCALE else 300  # rows per cluster in the "it lies" demos
N_NOISE = 2_000 if GPU_SCALE else 500
K_EVAL = 10  # neighbourhood size for trustworthiness and kNN preservation

# %%
# Data: scikit-learn's bundled 8x8 handwritten digits (no download), 64 pixel features with values 0-16.
section("data")
digits = load_digits()
X_dig, y_dig = digits.data.astype(np.float64), digits.target.astype(np.int64)
n, d = X_dig.shape
assert X_dig.shape == (1797, 64), X_dig.shape
print(f"digits X {X_dig.shape}, labels 0-9, counts per digit {np.bincount(y_dig).tolist()}")
print(f"pixel values {X_dig.min():.0f}-{X_dig.max():.0f}; median distance between two digits {np.median(pairwise_distances(X_dig)):.2f}")


# %%
# Worked example by hand: four 1-D points x = 0, 1, 2, 4.
section("worked example by hand")


def conditional_p(sq_dist_row: FloatArray, sigma: float) -> FloatArray:
    """Gaussian neighbour probabilities p_{j|i} of one point. sq_dist_row (m,) squared distances to the others -> (m,)."""
    logits = -sq_dist_row / (2 * sigma**2)
    w = np.exp(logits - logits.max())  # subtract the max: same result, no underflow (A3.4)
    return w / w.sum()


def perplexity_of(p: FloatArray) -> float:
    """exp(entropy in nats) of a probability vector: the effective number of neighbours."""
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


x_toy = np.array([0.0, 1.0, 2.0, 4.0])
sq_from_0 = (x_toy[1:] - x_toy[0]) ** 2  # squared distances from x=0 to x=1, 2, 4
p_0 = conditional_p(sq_from_0, sigma=1.0)
h_0 = -np.sum(p_0 * np.log(p_0))
print(f"point x=0, sigma=1: squared distances {sq_from_0.tolist()}, weights "
      f"{np.round(np.exp(-sq_from_0 / 2), 4).tolist()}")
print(f"p_(j|0) for j = x=1, x=2, x=4: {np.round(p_0, 4).tolist()}")
print(f"entropy {h_0:.4f} nats -> perplexity exp(H) = {np.exp(h_0):.4f}")
for s in [2.0, 0.5]:
    print(f"sigma={s}: p = {np.round(conditional_p(sq_from_0, s), 4).tolist()}, perplexity {perplexity_of(conditional_p(sq_from_0, s)):.4f}")


def sigma_for_perplexity(sq_dist_row: FloatArray, target: float) -> float:
    """Binary search (on log sigma) for the sigma whose p_{j|i} has the target perplexity."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if perplexity_of(conditional_p(sq_dist_row, float(np.exp(mid)))) > target:
            hi = mid
        else:
            lo = mid
    return float(np.exp((lo + hi) / 2))


s2 = sigma_for_perplexity(sq_from_0, 2.0)
print(f"binary search: sigma for perplexity 2 = {s2:.4f}, p = {np.round(conditional_p(sq_from_0, s2), 4).tolist()}")

# Student-t similarities in a 1-D embedding y = 0, 1, 2, 4 (the same positions), over all ordered pairs i != j.
y_toy = x_toy.copy()
num = 1.0 / (1.0 + (y_toy[:, None] - y_toy[None, :]) ** 2)
np.fill_diagonal(num, 0.0)
q_toy = num / num.sum()
print(f"Student-t kernel 1/(1+d^2) for d = 1, 2, 3, 4: {[round(1 / (1 + t**2), 4) for t in [1, 2, 3, 4]]}")
print(f"Gaussian kernel exp(-d^2/2)  for d = 1, 2, 3, 4: {[round(float(np.exp(-t**2 / 2)), 4) for t in [1, 2, 3, 4]]}")
print(f"sum over ordered pairs of 1/(1+d^2) = {num.sum():.4f};  q_(0,1) = {q_toy[0, 1]:.4f}, q_(0,4) = {q_toy[0, 3]:.4f}")
# KL asymmetry: one pair's contribution p log(p/q)
for p_, q_ in [(0.2, 0.01), (0.01, 0.2)]:
    print(f"KL term p log(p/q) with p={p_}, q={q_}: {p_ * np.log(p_ / q_):+.4f}")


# %%
# From scratch: t-SNE input affinities P for all digits, with a vectorized binary search on beta = 1/(2 sigma^2).
section("from scratch: t-SNE affinities P")


def tsne_affinities(X: FloatArray, perplexity: float, n_steps: int = 100) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Symmetric t-SNE input affinities. X (n, d) -> P (n, n) summing to 1, sigma (n,), achieved perplexity (n,)."""
    D = pairwise_distances(X, squared=True)  # (n, n) squared Euclidean distances
    np.fill_diagonal(D, np.inf)  # a point is not its own neighbour
    D = D - D.min(axis=1, keepdims=True)  # shift each row: p_{j|i} unchanged, exp() no longer underflows
    target = np.log(perplexity)  # target entropy in nats
    beta = np.ones(len(X))  # (n,) precision 1/(2 sigma^2) per point
    lo, hi = np.zeros(len(X)), np.full(len(X), np.inf)
    for _ in range(n_steps):
        W = np.exp(-D * beta[:, None])  # (n, n), zero on the diagonal
        P_cond = W / W.sum(axis=1, keepdims=True)  # row i is p_{.|i}
        H = entr(P_cond).sum(axis=1)  # (n,) entropy in nats; entr(p) = -p log p, with entr(0) = 0
        too_flat = H > target  # entropy too high: sigma too large, so raise beta
        lo = np.where(too_flat, beta, lo)
        hi = np.where(too_flat, hi, beta)
        beta = np.where(np.isinf(hi), beta * 2, (lo + hi) / 2)
    P = (P_cond + P_cond.T) / (2 * len(X))  # symmetrize: p_ij = (p_{j|i} + p_{i|j}) / 2n
    return P, np.sqrt(1 / (2 * beta)), np.exp(H)


P_mine, sigmas, eff = tsne_affinities(X_dig, perplexity=30)
assert P_mine.shape == (n, n), P_mine.shape
print(f"P shape {P_mine.shape}, sum {P_mine.sum():.6f}, symmetric: {np.allclose(P_mine, P_mine.T)}")
print(f"perplexity of every p_(.|i) after the search: min {eff.min():.4f}, max {eff.max():.4f} (target 30)")
print(f"sigma per point: min {sigmas.min():.2f}, median {np.median(sigmas):.2f}, max {sigmas.max():.2f} "
      "(dense regions get small sigmas)")

P_sk = squareform(_joint_probabilities(pairwise_distances(X_dig, squared=True).astype(np.float32), 30.0, 0))
# scikit-learn works in float32 and stops its search at an entropy tolerance of 1e-5, and floors P at
# machine epsilon (1e-16 in float64 terms, ~2e-16): compare to 1e-3 relative on the entries that matter.
np.fill_diagonal(P_sk, 0.0)
big = P_sk > 1e-8
np.testing.assert_allclose(P_mine[big], P_sk[big], rtol=1e-3)
print(f"max |P_mine - P_sklearn| = {np.abs(P_mine - P_sk).max():.2e} (largest entry {P_sk.max():.2e})")
print("matches sklearn.manifold._t_sne._joint_probabilities: OK")


# %%
# From scratch: two measures of how faithful an embedding is to the input neighbourhoods.
section("from scratch: trustworthiness and kNN preservation")


def knn_indices(X: FloatArray, k: int) -> IntArray:
    """Indices of each row's k nearest other rows. X (n, d) -> (n, k)."""
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def knn_preservation(X: FloatArray, Y: FloatArray, k: int) -> float:
    """Average share of each point's k input-space neighbours that are also among its k embedding neighbours."""
    a, b = knn_indices(X, k), knn_indices(Y, k)
    return float(np.mean([len(np.intersect1d(a[i], b[i])) / k for i in range(len(X))]))


def trustworthiness_mine(X: FloatArray, Y: FloatArray, k: int) -> float:
    """1 - penalty for embedding neighbours that were far in the input (rank r > k). X (n, d), Y (n, 2) -> scalar."""
    m = len(X)
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    rank = np.empty((m, m), dtype=np.int64)  # rank[i, j] = 1 for i's nearest input neighbour, 2 for the next...
    rank[np.arange(m)[:, None], np.argsort(D, axis=1)] = np.arange(1, m + 1)
    emb_nn = knn_indices(Y, k)  # (m, k)
    excess = rank[np.arange(m)[:, None], emb_nn] - k  # > 0 only for intruders
    penalty = np.sum(excess[excess > 0])
    return float(1 - 2 / (m * k * (2 * m - 3 * k - 1)) * penalty)


# Toy check (the lesson's hand calculation): 4 points on a line at 0, 1, 3, 7, embedded unchanged, and with the
# first and last points swapped (no ties in either space).
x4 = np.array([[0.0], [1.0], [3.0], [7.0]])
print(f"toy: same order -> trustworthiness(k=1) {trustworthiness_mine(x4, x4, 1):.4f};  "
      f"0 and 7 swapped -> {trustworthiness_mine(x4, x4[[3, 1, 2, 0]], 1):.4f}")

pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_dig)
t_mine = trustworthiness_mine(X_dig, pca2, K_EVAL)
t_sk = trustworthiness(X_dig, pca2, n_neighbors=K_EVAL)
np.testing.assert_allclose(t_mine, t_sk, rtol=1e-10)
print(f"PCA 2-D of digits: trustworthiness(k={K_EVAL}) mine {t_mine:.4f}, sklearn {t_sk:.4f}")
print("matches sklearn.manifold.trustworthiness: OK")


# %%
# With scikit-learn and umap-learn: t-SNE at three perplexities and UMAP at three n_neighbors, against PCA.
section("with scikit-learn and umap-learn: digits")


def centroid_distance_corr(X: FloatArray, Y: FloatArray, labels: IntArray) -> float:
    """Spearman correlation between the 45 distances of the digit centroids in X and in Y (global structure)."""
    cx = np.array([X[labels == c].mean(axis=0) for c in np.unique(labels)])
    cy = np.array([Y[labels == c].mean(axis=0) for c in np.unique(labels)])
    return float(spearmanr(squareform(pairwise_distances(cx), checks=False),
                           squareform(pairwise_distances(cy), checks=False)).statistic)


def run_tsne(X: FloatArray, perplexity: float) -> FloatArray:
    return TSNE(perplexity=perplexity, max_iter=TSNE_ITER, init="pca", random_state=SEED).fit_transform(X)


def run_umap(X: FloatArray, n_neighbors: int) -> FloatArray:
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=SEED).fit_transform(X)


embeddings: dict[str, FloatArray] = {"PCA": pca2}
timings: dict[str, float] = {}
for p in PERPLEXITIES:
    t = time.perf_counter()
    embeddings[f"t-SNE perplexity={p}"] = run_tsne(X_dig, p)
    timings[f"t-SNE perplexity={p}"] = time.perf_counter() - t
for k in N_NEIGHBORS:
    t = time.perf_counter()
    embeddings[f"UMAP n_neighbors={k}"] = run_umap(X_dig, k)
    timings[f"UMAP n_neighbors={k}"] = time.perf_counter() - t

print(f"{'embedding':<22} {'trust@10':>8} {'kNN@10':>7} {'centroid rho':>12} {'silhouette':>10}")
for name, Y in embeddings.items():
    print(f"{name:<22} {trustworthiness(X_dig, Y, n_neighbors=K_EVAL):>8.4f} {knn_preservation(X_dig, Y, K_EVAL):>7.4f} "
          f"{centroid_distance_corr(X_dig, Y, y_dig):>12.4f} {silhouette_score(Y, y_dig):>10.4f}")
print(f"{'64-D input (reference)':<22} {'1.0000':>8} {'1.0000':>7} {'1.0000':>12} {silhouette_score(X_dig, y_dig):>10.4f}")
for name, s in timings.items():
    print(f"~ {name}: {s:.1f} s")

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=100)
for ax, name in zip(axes.ravel(), [k for k in embeddings if k != "PCA"]):
    Y = embeddings[name]
    for c in range(10):
        ax.scatter(*Y[y_dig == c].T, s=3, label=str(c))
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].legend(markerscale=4, fontsize=7, ncol=2, loc="best")
fig.suptitle("Digits: the picture changes with perplexity and n_neighbors (axes have no units)")
fig.tight_layout()
fig.savefig(OUT / "digits_grid.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved outputs/b5_5/digits_grid.png")


# %%
# "It lies" demo 1 and 2: cluster sizes and distances between clusters. Three Gaussian clusters in 10-D.
section("it lies 1: cluster sizes")


def blob(center: FloatArray, std: float, size: int, gen: np.random.Generator) -> FloatArray:
    """size points from N(center, std^2 I). center (d,) -> (size, d)."""
    return center + std * gen.standard_normal((size, len(center)))


def spread(Y: FloatArray) -> float:
    """Root-mean-square distance of the points to their centroid."""
    return float(np.sqrt(np.mean(np.sum((Y - Y.mean(axis=0)) ** 2, axis=1))))


gen = np.random.default_rng(SEED)
D_DEMO = 10
e1 = np.eye(D_DEMO)[0]
X_size = np.vstack([blob(0 * e1, 1.0, N_PER_CLUSTER, gen), blob(40 * e1, 10.0, N_PER_CLUSTER, gen)])
lab_size = np.repeat([0, 1], N_PER_CLUSTER)
size_emb = {"input 10-D": X_size, "PCA": PCA(2, random_state=SEED).fit_transform(X_size),
            "t-SNE perplexity=30": run_tsne(X_size, 30), "UMAP n_neighbors=15": run_umap(X_size, 15)}
print("cluster A: std 1, cluster B: std 10 (both 300 points). Spread = RMS distance to the cluster's centroid.")
for name, Y in size_emb.items():
    sa, sb = spread(Y[lab_size == 0]), spread(Y[lab_size == 1])
    print(f"{name:<20} spread A {sa:7.3f}   spread B {sb:7.3f}   ratio B/A {sb / sa:6.2f}")

section("it lies 2: distances between clusters")
X_gap = np.vstack([blob(0 * e1, 1.0, N_PER_CLUSTER, gen), blob(10 * e1, 1.0, N_PER_CLUSTER, gen),
                   blob(100 * e1, 1.0, N_PER_CLUSTER, gen)])
lab_gap = np.repeat([0, 1, 2], N_PER_CLUSTER)
gap_emb = {"input 10-D": X_gap, "PCA": PCA(2, random_state=SEED).fit_transform(X_gap),
           "t-SNE perplexity=30": run_tsne(X_gap, 30), "UMAP n_neighbors=15": run_umap(X_gap, 15)}
print("centres on one axis at 0 (A), 10 (B) and 100 (C): C is 10x farther from A than B is.")
for name, Y in gap_emb.items():
    c = np.array([Y[lab_gap == g].mean(axis=0) for g in range(3)])
    ab, ac = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[0])
    print(f"{name:<20} d(A,B) {ab:8.2f}   d(A,C) {ac:8.2f}   ratio d(A,C)/d(A,B) {ac / ab:6.2f}")

# %%
section("it lies 3: clusters in pure noise")
X_noise = np.random.default_rng(SEED + 1).standard_normal((N_NOISE, 100))  # no structure at all
noise_emb = {"input 100-D": X_noise, "PCA": PCA(2, random_state=SEED).fit_transform(X_noise),
             "t-SNE perplexity=2": run_tsne(X_noise, 2), "t-SNE perplexity=30": run_tsne(X_noise, 30),
             "UMAP n_neighbors=5": run_umap(X_noise, 5)}


def clumpiness(Y: FloatArray) -> float:
    """Median ratio of the distance to the 1st and to the 16th nearest neighbour. Even 2-D scatter: about 1/4 = sqrt(1/16)."""
    D = pairwise_distances(Y)
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    return float(np.median(D[:, 0] / D[:, 15]))


print(f"{N_NOISE} points from N(0, I) in 100-D. Clumpiness = median d(1st neighbour) / d(16th neighbour), "
      "lower = tighter clumps;")
print("silhouette of k-means with k=6 run on that space.")
for name, Y in noise_emb.items():
    km = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(Y)
    print(f"{name:<20} clumpiness {clumpiness(Y):.4f}   k-means silhouette {silhouette_score(Y, km.labels_):.4f}")


# %%
# Key plot: the three lies. Top row t-SNE (perplexity 30, or 2 for the noise), bottom row PCA.
section("key plot: three ways t-SNE misleads")
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), dpi=90)
panels = [
    (size_emb, lab_size, ["A: std 1", "B: std 10"], "t-SNE perplexity=30", "Sizes: B is 10x wider in 10-D"),
    (gap_emb, lab_gap, ["A at 0", "B at 10", "C at 100"], "t-SNE perplexity=30", "Gaps: C is 10x farther than B"),
    (noise_emb, np.zeros(N_NOISE, dtype=np.int64), ["pure noise"], "t-SNE perplexity=2", "Noise: no clusters exist"),
]
markers = ["o", "s", "^"]
for col, (emb, lab, names, tsne_key, title) in enumerate(panels):
    for row, key in enumerate([tsne_key, "PCA"]):
        ax = axes[row, col]
        Y = emb[key]
        for g, gname in enumerate(names):
            ax.scatter(*Y[lab == g].T, s=3, marker=markers[g], label=gname)
        ax.set_title(f"{title}\n{key}" if row == 0 else key, fontsize=9)
        ax.set_xlabel("component 1 (no units)", fontsize=8)
        ax.set_ylabel("component 2 (no units)", fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0 and len(names) > 1:
            ax.legend(markerscale=3, fontsize=7, loc="best")
fig.suptitle("t-SNE equalizes cluster sizes, compresses big gaps and finds clumps in noise; PCA does not")
fig.tight_layout()
fig.savefig(IMG / "b5_5_three_lies.png", dpi=90, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved lessons/B/img/b5_5_three_lies.png ({(IMG / 'b5_5_three_lies.png').stat().st_size // 1024} KB)")

# %%
# Try this (not graded):
# 1. Rerun t-SNE on digits with random_state=1, 2, 3. Which numbers in the comparison table move most?
# 2. In "it lies 2", set UMAP min_dist=0.8 and n_neighbors=200. Does the gap ratio come closer to 10?
# 3. Standardize the digits pixels (StandardScaler) before t-SNE. Rare edge pixels now count as much as the centre ones;
#    does the picture improve or get worse?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
