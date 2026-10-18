# %% [markdown]
# # B5.2 Hierarchical clustering and DBSCAN — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b5_2_hierarchical_dbscan_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

FloatArray = npt.NDArray[np.float64]
SEED = 0
MIN_SAMPLES = 5
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
x = np.array([[1.0], [2.0], [4.5], [9.0], [10.2], [16.0]])
names = "ABCDEF"
for method in ["single", "complete"]:
    Z = linkage(x, method=method)
    print(f"{method:<8} merge heights: {np.round(Z[:, 2], 4).tolist()}")
    for k in [3, 2]:
        lab = fcluster(Z, t=k, criterion="maxclust")
        groups = ["".join(names[i] for i in np.flatnonzero(lab == c)) for c in np.unique(lab)]
        print(f"         cut into {k}: {' | '.join(sorted(groups))}")
Zw = linkage(x, method="ward")
print(f"ward     first three merge heights: {np.round(Zw[:3, 2], 4).tolist()}   by hand sqrt(2 * 2/3 * 3^2) = {np.sqrt(12):.4f}")
for eps, ms in [(1.5, 2), (2.6, 3)]:
    m = DBSCAN(eps=eps, min_samples=ms).fit(x)
    core = np.zeros(len(x), dtype=bool)
    core[m.core_sample_indices_] = True
    kinds = ["core" if c else ("border" if l != -1 else "noise") for c, l in zip(core, m.labels_)]
    print(f"DBSCAN eps={eps} min_samples={ms}: labels {m.labels_.tolist()}  {kinds}")

# %%
# The lab's three-density blobs.
X, y = make_blobs(n_samples=[200, 200, 200], centers=[[0, 0], [6, 0], [3, 5]],
                  cluster_std=[0.3, 1.0, 2.0], random_state=SEED)

section("(a) one eps for three densities")
best = (-1.0, 0.0)
print(f"{'eps':>4} {'clusters':>8} {'noise':>6} {'ARI':>7}")
for eps in np.round(np.arange(0.1, 2.01, 0.1), 1):
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
    ari = adjusted_rand_score(y, lab)
    best = max(best, (ari, eps))
    if eps in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0):
        print(f"{eps:>4} {lab.max() + 1:>8} {np.mean(lab == -1):>6.3f} {ari:>7.4f}")
print(f"best ARI over the grid {best[0]:.4f} at eps = {best[1]}")
lab = DBSCAN(eps=best[1], min_samples=MIN_SAMPLES).fit_predict(X)
for g in range(3):
    print(f"  true blob {g} (std {[0.3, 1.0, 2.0][g]}): noise share {np.mean(lab[y == g] == -1):.3f}")

section("(b) HDBSCAN")
for mcs in [10, 20, 50]:
    lab = HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(X)
    print(f"min_cluster_size={mcs}: {lab.max() + 1} clusters, noise share {np.mean(lab == -1):.3f}, ARI {adjusted_rand_score(y, lab):.4f}")

section("(c) units: stretched moons")
Xm, ym = make_moons(n_samples=500, noise=0.06, random_state=SEED)
Xs = Xm * np.array([1.0, 10.0])
for name, data in [("raw moons", Xm), ("feature 2 x10", Xs), ("x10 then standardized", StandardScaler().fit_transform(Xs))]:
    D = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=-1))
    kd = np.sort(D, axis=1)[:, MIN_SAMPLES - 1]
    eps = float(np.quantile(kd, 0.99))  # rule of thumb: just above almost every point's 5-distance
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(data)
    print(f"{name:<22} eps (99th pct of 5-distance) {eps:.4f}: {lab.max() + 1} clusters, "
          f"noise {np.mean(lab == -1):.3f}, ARI {round(adjusted_rand_score(ym, lab), 4) + 0.0:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
