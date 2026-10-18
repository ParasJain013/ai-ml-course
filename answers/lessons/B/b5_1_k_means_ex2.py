# %% [markdown]
# # B5.1 k-means — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b5_1_k_means_ex2.py   (CPU, ~10 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# from labs/B/b5_1_k_means.py
def sq_dists(X: FloatArray, C: FloatArray) -> FloatArray:
    """Squared Euclidean distances. X (n, d), C (k, d) -> (n, k)."""
    D2 = (X**2).sum(axis=1)[:, None] - 2 * X @ C.T + (C**2).sum(axis=1)[None, :]
    return np.maximum(D2, 0.0)


def kmeans_plus_plus(X: FloatArray, k: int, gen: np.random.Generator) -> FloatArray:
    """k-means++ seeding. -> (k, d)."""
    C = [X[gen.integers(len(X))]]
    for _ in range(1, k):
        D2 = sq_dists(X, np.array(C)).min(axis=1)
        C.append(X[gen.choice(len(X), p=D2 / D2.sum())])
    return np.array(C)


def lloyd(X: FloatArray, C0: FloatArray, max_iter: int = 300, tol: float = 1e-4) -> tuple[FloatArray, IntArray, float]:
    """Lloyd's algorithm. -> centroids (k, d), labels (n,), inertia."""
    tol_abs = tol * float(np.mean(X.var(axis=0)))
    C = C0.copy()
    for _ in range(max_iter):
        labels = np.argmin(sq_dists(X, C), axis=1)
        C_new = C.copy()
        for j in range(len(C)):
            if np.any(labels == j):
                C_new[j] = X[labels == j].mean(axis=0)
        shift = float(((C_new - C) ** 2).sum())
        C = C_new
        if shift <= tol_abs:
            break
    labels = np.argmin(sq_dists(X, C), axis=1)
    return C, labels, float(((X - C[labels]) ** 2).sum())


def kmeans(X: FloatArray, k: int, n_init: int, gen: np.random.Generator) -> tuple[FloatArray, IntArray, float]:
    """Best of n_init k-means++ runs. -> centroids, labels, inertia."""
    runs = [lloyd(X, kmeans_plus_plus(X, k, gen)) for _ in range(n_init)]
    return min(runs, key=lambda r: r[2])


# %%
section("exercise 1 check")
x = np.array([0.0, 2.0, 3.0, 9.0, 11.0])[:, None]
C = np.array([[0.0], [2.0]])
for it in range(1, 5):
    lab = np.argmin(sq_dists(x, C), axis=1)
    C_new = np.array([[x[lab == j].mean()] for j in range(2)])
    J = float(((x - C_new[lab]) ** 2).sum())
    print(f"iteration {it}: labels {lab.tolist()} -> centroids {C_new.ravel().tolist()}, inertia {J:g}")
    if np.allclose(C_new, C):
        break
    C = C_new
for cut in range(1, 5):  # in 1-D the optimal 2-clustering splits the sorted points at one cut
    left, right = x[:cut], x[cut:]
    J = float(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())
    print(f"(b) split after {x[cut - 1, 0]:g}: inertia {J:.4f}")
D2 = sq_dists(x, x[[0]]).ravel()
print(f"(c) k-means++ after picking 0: D^2 = {D2.tolist()}, P(pick 11) = {D2[4] / D2.sum():.4f}")
lab = np.array([0, 0, 0, 1, 1])
dist = np.abs(x - x.T)
a = dist[2, [0, 1]].mean()
b = dist[2, [3, 4]].mean()
print(f"(d) silhouette of 3: a = {a:g}, b = {b:g}, s = {(b - a) / max(a, b):.4f}")
print(f"    ARI([0,0,0,1,1], [0,0,1,1,1]) = {adjusted_rand_score([0, 0, 0, 1, 1], [0, 0, 1, 1, 1]):.4f}")

# %%
section("wine: raw versus standardized")
X_raw, cultivar = load_wine(return_X_y=True)
names = load_wine().feature_names
print(f"X {X_raw.shape}, cultivars {np.bincount(cultivar).tolist()}")
stds = X_raw.std(axis=0)
print(f"largest feature std: {names[int(np.argmax(stds))]} {stds.max():.1f}; smallest: {names[int(np.argmin(stds))]} {stds.min():.3f}")
X_std = (X_raw - X_raw.mean(axis=0)) / stds  # no train/test split in this exploration, so all rows are used
for name, Xv in [("raw", X_raw), ("standardized", X_std)]:
    C, labels, J = kmeans(Xv, 3, n_init=10, gen=np.random.default_rng(SEED))
    print(f"{name:<13} k=3: inertia {J:12.1f}  silhouette {silhouette_score(Xv, labels):.4f}  "
          f"ARI vs cultivar {adjusted_rand_score(cultivar, labels):.4f}  sizes {sorted(np.bincount(labels).tolist())}")
C_raw, lab_raw, _ = kmeans(X_raw, 3, n_init=10, gen=np.random.default_rng(SEED))
print("raw centroids, proline column:", np.round(np.sort(C_raw[:, names.index("proline")]), 1).tolist())

# %%
section("choosing k on standardized wine")
print(f"{'k':>2} {'inertia':>8} {'silhouette':>10} {'boot ARI mean':>13} {'boot ARI min':>12}")
for k in range(2, 9):
    _, labels, J = kmeans(X_std, k, n_init=10, gen=np.random.default_rng(SEED))
    g = np.random.default_rng(SEED)
    boots = []
    for _b in range(20):
        rows = g.integers(0, len(X_std), size=len(X_std))
        C_b, _l, _j = kmeans(X_std[rows], k, n_init=10, gen=g)
        boots.append(adjusted_rand_score(labels, np.argmin(sq_dists(X_std, C_b), axis=1)))
    print(f"{k:>2} {J:8.1f} {silhouette_score(X_std, labels):10.4f} {np.mean(boots):13.4f} {np.min(boots):12.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
