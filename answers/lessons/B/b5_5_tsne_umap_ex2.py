# %% [markdown]
# # B5.5 t-SNE and UMAP — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b5_5_tsne_umap_ex2.py   (CPU, ~60 s)

# %%
from __future__ import annotations

import time
import warnings

import numpy as np
import numpy.typing as npt
import umap
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")  # random_state forces n_jobs=1 in umap-learn

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
x = np.array([0.0, 1.0, 3.0])
sq = (x[1:] - x[0]) ** 2  # squared distances from x=0
for sigma in [1.0, 2.0]:
    w = np.exp(-sq / (2 * sigma**2))
    p = w / w.sum()
    h = -np.sum(p * np.log(p))
    print(f"(a/b) sigma={sigma}: weights {np.round(w, 4).tolist()}, p {np.round(p, 4).tolist()}, "
          f"H {h:.4f} nats, perplexity {np.exp(h):.4f}")
y = np.array([0.0, 1.0, 2.0])  # the embedding
num = 1 / (1 + (y[:, None] - y[None, :]) ** 2)
np.fill_diagonal(num, 0)
q = num / num.sum()
print(f"(c) sum of 1/(1+d^2) over ordered pairs {num.sum():.4f}; q_01 {q[0, 1]:.4f}, q_02 {q[0, 2]:.4f}, q_12 {q[1, 2]:.4f}")
for p_, q_ in [(0.3, 0.03), (0.03, 0.3)]:
    print(f"(d) p={p_}, q={q_}: p log(p/q) = {p_ * np.log(p_ / q_):+.4f}")


# %%
def kmeans_labels(Z: FloatArray, k: int = 10) -> IntArray:
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)


def spaces(X: FloatArray, tsne_seed: int = SEED) -> dict[str, FloatArray]:
    """The five spaces the exercise clusters in. X (n, d) -> {name: (n, d')}."""
    return {
        "raw 64-D": X,
        "PCA 2": PCA(2, random_state=SEED).fit_transform(X),
        "PCA 10": PCA(10, random_state=SEED).fit_transform(X),
        "t-SNE 2 (perp 30)": TSNE(perplexity=30, init="pca", random_state=tsne_seed).fit_transform(X),
        "UMAP 2 (nn 15, md 0)": umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=SEED).fit_transform(X),
    }


X, y = load_digits(return_X_y=True)
X = X.astype(np.float64)

section("(a) k-means (k=10) on digits in each space")
digit_spaces = spaces(X)
labels: dict[str, IntArray] = {}
print(f"{'space':<22} {'ARI vs digit':>12} {'silhouette (own space)':>23} {'silhouette (64-D)':>18}")
for name, Z in digit_spaces.items():
    labels[name] = kmeans_labels(Z)
    print(f"{name:<22} {adjusted_rand_score(y, labels[name]):>12.4f} {silhouette_score(Z, labels[name]):>23.4f} "
          f"{silhouette_score(X, labels[name]):>18.4f}")

section("(b) t-SNE seed stability")
runs = [labels["t-SNE 2 (perp 30)"]] + [
    kmeans_labels(TSNE(perplexity=30, init="pca", random_state=s).fit_transform(X)) for s in [1, 2]
]
runs_rand = [kmeans_labels(TSNE(perplexity=30, init="random", random_state=s).fit_transform(X)) for s in [0, 1, 2]]
for tag, rr in [("init='pca'", runs), ("init='random'", runs_rand)]:
    pairs = [adjusted_rand_score(rr[i], rr[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    aris = [adjusted_rand_score(y, r) for r in rr]
    print(f"{tag:<14} ARI between seeds {np.round(pairs, 4).tolist()}   ARI vs digit per seed {np.round(aris, 4).tolist()}")

section("(c) the same pipeline on pure Gaussian noise (1797, 64)")
X_noise = np.random.default_rng(SEED).standard_normal(X.shape)
print(f"{'space':<22} {'silhouette (own space)':>23} {'silhouette (64-D)':>18}")
for name, Z in spaces(X_noise).items():
    lab = kmeans_labels(Z)
    print(f"{name:<22} {silhouette_score(Z, lab):>23.4f} {silhouette_score(X_noise, lab):>18.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
