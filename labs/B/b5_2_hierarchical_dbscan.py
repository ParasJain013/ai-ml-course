# %% [markdown]
# # B5.2 Hierarchical clustering and DBSCAN — lab
# Lesson: lessons/B/b5_2_hierarchical_dbscan.md
# Run: uv run python labs/B/b5_2_hierarchical_dbscan.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from collections import deque
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

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
OUT = ROOT / "outputs" / "b5_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the shape datasets larger (the from-scratch
# DBSCAN builds an (n, n) distance matrix, so 5,000 points need about 200 MB).
GPU_SCALE = False
N_SHAPES = 5_000 if GPU_SCALE else 500  # points in each of two-moons and circles
N_LINKAGE_CHECK = 40  # points for the from-scratch vs scipy linkage check (the naive version is O(n^3))
MIN_SAMPLES = 5
EPS_MOONS, EPS_CIRCLES = 0.15, 0.15

# %%
# Data: the five 1-D points of the worked example, and the two shape datasets where k-means fails.
section("data")
x_hand = np.array([[0.0], [2.0], [5.0], [9.0], [10.5]])  # A, B, C, D, E
names = "ABCDE"
print(f"worked example: {x_hand.shape[0]} points on a line: {dict(zip(names, x_hand[:, 0].tolist()))}")
X_moons, y_moons = make_moons(n_samples=N_SHAPES, noise=0.06, random_state=SEED)
X_circ, y_circ = make_circles(n_samples=N_SHAPES, factor=0.5, noise=0.05, random_state=SEED)
print(f"two moons {X_moons.shape}, concentric circles {X_circ.shape}; true labels kept only for scoring")


# %%
# Hierarchical clustering from scratch: start with every point alone, merge the closest pair of clusters,
# repeat. The output has scipy's format: one row per merge, [cluster a, cluster b, height, new size];
# original points are clusters 0..n-1 and the cluster made by merge i gets id n + i.
def cluster_distance(A: FloatArray, B: FloatArray, method: str) -> float:
    """Linkage distance between two clusters. A (n_a, d), B (n_b, d) -> scalar."""
    if method == "ward":
        n_a, n_b = len(A), len(B)
        gap = A.mean(axis=0) - B.mean(axis=0)  # (d,)
        delta_sse = n_a * n_b / (n_a + n_b) * float(gap @ gap)  # rise in within-cluster sum of squares
        return float(np.sqrt(2 * delta_sse))  # scipy's scale: two single points -> their distance
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1))  # (n_a, n_b)
    return float(D.min() if method == "single" else D.max())  # "complete" -> max


def linkage_scratch(X: FloatArray, method: str) -> FloatArray:
    """Naive agglomerative clustering. X (n, d) -> merge table (n - 1, 4) in scipy's format."""
    n = len(X)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}  # active cluster id -> its rows
    Z = np.zeros((n - 1, 4))
    for step in range(n - 1):
        ids = sorted(members)
        best = (np.inf, -1, -1)
        for i_pos, a in enumerate(ids):
            for b in ids[i_pos + 1:]:
                dist = cluster_distance(X[members[a]], X[members[b]], method)
                if dist < best[0]:
                    best = (dist, a, b)
        dist, a, b = best
        members[n + step] = members.pop(a) + members.pop(b)
        Z[step] = [a, b, dist, len(members[n + step])]
    return Z


# %%
section("worked example by hand: linkage")


def merge_story(Z: FloatArray, n: int) -> str:
    """Describe a merge table with letter names: 'DE@1.5, ...'."""
    label = {i: names[i] for i in range(n)}
    parts = []
    for step, (a, b, h, _) in enumerate(Z):
        label[n + step] = "".join(sorted(label[int(a)] + label[int(b)]))
        parts.append(f"{label[n + step]}@{h:.4f}")
    return ", ".join(parts)


for method in ["single", "complete", "ward"]:
    Z_hand = linkage_scratch(x_hand, method)
    print(f"{method:<8} merges: {merge_story(Z_hand, len(x_hand))}")
print(f"ward AB+C by hand: dSSE = (2*1/3) * (5 - 1)^2 = {2 / 3 * 16:.4f}, height sqrt(2 dSSE) = {np.sqrt(2 * 2 / 3 * 16):.4f}")
for method in ["single", "complete", "ward"]:
    labels2 = fcluster(linkage(x_hand, method=method), t=2, criterion="maxclust")
    print(f"{method:<8} cut into 2 clusters: {''.join(names[i] for i in np.flatnonzero(labels2 == labels2[0]))} | "
          f"{''.join(names[i] for i in np.flatnonzero(labels2 != labels2[0]))}")

# %%
# The same merge tables from scipy, on random 2-D data (no ties, so the merge order is unique).
section("from scratch: linkage matches scipy")
X_check = rng.normal(size=(N_LINKAGE_CHECK, 2))
for method in ["single", "complete", "ward"]:
    Z_mine, Z_scipy = linkage_scratch(X_check, method), linkage(X_check, method=method)
    np.testing.assert_allclose(Z_mine, Z_scipy, rtol=1e-6, atol=1e-8)
    print(f"{method:<8} {N_LINKAGE_CHECK} points: merge table matches scipy.cluster.hierarchy.linkage: OK "
          f"(last merge height {Z_mine[-1, 2]:.4f})")


# %%
# DBSCAN from scratch. Neighbourhood of a point: every point within eps, itself included.
# Core point: at least min_samples points in its neighbourhood. Clusters grow outward from cores;
# a non-core point inside a core's neighbourhood joins as a border point; everything else is noise (-1).
def dbscan_scratch(X: FloatArray, eps: float, min_samples: int) -> tuple[IntArray, IntArray]:
    """DBSCAN by breadth-first region growing. X (n, d) -> labels (n,) with -1 = noise, core mask (n,)."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))  # (n, n) pairwise distances
    neighbours = [np.flatnonzero(row <= eps) for row in D]  # each includes the point itself
    is_core = np.array([len(nb) >= min_samples for nb in neighbours])  # (n,)
    labels = np.full(len(X), -1, dtype=np.int64)
    cluster = 0
    for start in range(len(X)):  # seed clusters in index order, as scikit-learn does
        if not is_core[start] or labels[start] != -1:
            continue
        labels[start] = cluster
        queue = deque([start])
        while queue:  # only core points pass the cluster on to their neighbours
            p = queue.popleft()
            for q in neighbours[p]:
                if labels[q] == -1:
                    labels[q] = cluster
                    if is_core[q]:
                        queue.append(q)
        cluster += 1
    return labels, is_core.astype(np.int64)


section("worked example by hand: DBSCAN")
x_db = np.array([[0.0], [1.0], [2.0], [3.4], [6.0], [7.0], [8.0], [12.0]])
lab_db, core_db = dbscan_scratch(x_db, eps=1.5, min_samples=3)
counts = [int(np.sum(np.abs(x_db[:, 0] - v) <= 1.5)) for v in x_db[:, 0]]
kind = ["core" if c else ("border" if l != -1 else "noise") for c, l in zip(core_db, lab_db)]
print("x:               ", x_db[:, 0].tolist())
print("points within 1.5:", counts)
print("kind:            ", kind)
print("labels:          ", lab_db.tolist())

# %%
section("from scratch: DBSCAN matches scikit-learn")
for name, X, eps in [("two moons", X_moons, EPS_MOONS), ("circles", X_circ, EPS_CIRCLES)]:
    mine, core = dbscan_scratch(X, eps, MIN_SAMPLES)
    sk = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X)
    np.testing.assert_array_equal(np.flatnonzero(core), sk.core_sample_indices_)
    np.testing.assert_array_equal(mine, sk.labels_)
    print(f"{name:<9} eps={eps} min_samples={MIN_SAMPLES}: {mine.max() + 1} clusters, {int(core.sum())} core, "
          f"{int(np.sum((core == 0) & (mine != -1)))} border, {int(np.sum(mine == -1))} noise; "
          f"labels match sklearn.cluster.DBSCAN: OK")


# %%
# Choosing eps: the k-distance curve. For each point, the distance to its (min_samples)-th nearest
# neighbour (counting itself as the first). Cores are exactly the points whose value is <= eps.
def k_distance(X: FloatArray, k: int) -> FloatArray:
    """Distance from each point to its k-th nearest neighbour, itself included. X (n, d) -> (n,)."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    return np.sort(D, axis=1)[:, k - 1]


section("choosing eps: the k-distance curve")
for name, X in [("two moons", X_moons), ("circles", X_circ)]:
    kd = k_distance(X, MIN_SAMPLES)
    q = np.quantile(kd, [0.5, 0.9, 0.95, 0.99])
    print(f"{name:<9} {MIN_SAMPLES}-distance quantiles 50/90/95/99%: {np.round(q, 4).tolist()}  max {kd.max():.4f}")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.plot(np.sort(k_distance(X_moons, MIN_SAMPLES)), label="two moons")
ax.plot(np.sort(k_distance(X_circ, MIN_SAMPLES)), linestyle="--", label="circles")
ax.axhline(EPS_MOONS, color="gray", linestyle=":", label=f"eps = {EPS_MOONS}")
ax.set_xlabel("points, sorted by k-distance")
ax.set_ylabel(f"distance to {MIN_SAMPLES}-th nearest neighbour")
ax.set_title("Pick eps near the bend, where the curve shoots up")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "k_distance.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved outputs/b5_2/k_distance.png")

# %%
# With scikit-learn and scipy: the four methods on the two shapes, scored against the true labels with the
# adjusted Rand index (1 = identical partition, about 0 = no better than random labels).
section("with scikit-learn: who finds the shapes")
methods = {
    "k-means": lambda X: KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(X),
    "Ward": lambda X: AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X),
    "complete": lambda X: AgglomerativeClustering(n_clusters=2, linkage="complete").fit_predict(X),
    "single": lambda X: AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X),
}
results: dict[tuple[str, str], IntArray] = {}
print(f"{'method':<9} {'moons ARI':>9} {'circles ARI':>11}")
for mname, fit in methods.items():
    for dname, X in [("moons", X_moons), ("circles", X_circ)]:
        results[(mname, dname)] = fit(X)
    print(f"{mname:<9} {adjusted_rand_score(y_moons, results[(mname, 'moons')]):>9.4f} "
          f"{adjusted_rand_score(y_circ, results[(mname, 'circles')]):>11.4f}")
results[("DBSCAN", "moons")] = DBSCAN(eps=EPS_MOONS, min_samples=MIN_SAMPLES).fit_predict(X_moons)
results[("DBSCAN", "circles")] = DBSCAN(eps=EPS_CIRCLES, min_samples=MIN_SAMPLES).fit_predict(X_circ)
print(f"{'DBSCAN':<9} {adjusted_rand_score(y_moons, results[('DBSCAN', 'moons')]):>9.4f} "
      f"{adjusted_rand_score(y_circ, results[('DBSCAN', 'circles')]):>11.4f}   (noise points count as their own group)")
# sklearn's AgglomerativeClustering and scipy's linkage + fcluster give the same partition
lab_sp = fcluster(linkage(X_moons, method="single"), t=2, criterion="maxclust")
assert adjusted_rand_score(lab_sp, results[("single", "moons")]) == 1.0
print("single linkage: sklearn AgglomerativeClustering matches scipy linkage + fcluster: OK")

# %%
# Two failure modes. (1) Single linkage chains through a thin bridge of points. (2) DBSCAN has one eps for
# the whole dataset, so clusters of very different density cannot all be found at once.
section("failure modes: chaining and varying density")
X_two, y_two = make_blobs(n_samples=[150, 150], centers=[[-3, 0], [3, 0]], cluster_std=0.6, random_state=SEED)
bridge = np.c_[np.linspace(-2.2, 2.2, 12), np.zeros(12)]  # 12 points in a line between the blobs
X_br = np.r_[X_two, bridge]
y_br = np.r_[y_two, np.full(12, -1)]  # bridge points are scored with neither blob
mask = y_br != -1
for link in ["single", "complete", "ward"]:
    lab = AgglomerativeClustering(n_clusters=2, linkage=link).fit_predict(X_br)
    sizes = np.bincount(lab).tolist()
    print(f"blobs + bridge, {link:<8}: cluster sizes {sizes}, ARI on the blob points {adjusted_rand_score(y_br[mask], lab[mask]):.4f}")
X_dens, y_dens = make_blobs(n_samples=[200, 200, 200], centers=[[0, 0], [6, 0], [3, 5]],
                            cluster_std=[0.3, 1.0, 2.0], random_state=SEED)
for eps in [0.2, 0.5, 1.0]:
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X_dens)
    print(f"three densities (std 0.3, 1, 2), DBSCAN eps={eps}: {lab.max() + 1} clusters, noise share "
          f"{np.mean(lab == -1):.3f}, ARI {adjusted_rand_score(y_dens, lab):.4f}")
lab = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_dens)
print(f"three densities, Ward with 3 clusters: ARI {adjusted_rand_score(y_dens, lab):.4f}")
X_stretch = X_moons * np.array([1.0, 10.0])  # the same moons, second feature in different units
lab = DBSCAN(eps=EPS_MOONS, min_samples=MIN_SAMPLES).fit_predict(X_stretch)
print(f"two moons with feature 2 x10, same eps: {lab.max() + 1} clusters, noise share {np.mean(lab == -1):.3f}, "
      f"ARI {adjusted_rand_score(y_moons, lab):.4f}")

# %%
# Dendrogram of the worked example and of 30 blob points (scratch figure).
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=120)
for ax, method in zip(axes, ["single", "complete", "ward"]):
    dendrogram(linkage(x_hand, method=method), labels=list(names), ax=ax, color_threshold=0)
    ax.set_title(f"{method} linkage")
    ax.set_xlabel("point")
    ax.set_ylabel("merge height (distance)")
fig.suptitle("Same points, same first merges, different heights for the late merges")
fig.tight_layout()
fig.savefig(OUT / "dendrograms.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved outputs/b5_2/dendrograms.png")

# %%
# Key plot: four methods, two shapes. Clusters are shown by colour AND marker; DBSCAN noise is a grey x.
section("key plot: four methods on two shapes")
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), dpi=100)
markers = ["o", "s", "^", "D"]
for row, (dname, X, y_true) in enumerate([("moons", X_moons, y_moons), ("circles", X_circ, y_circ)]):
    for col, mname in enumerate(["k-means", "Ward", "single", "DBSCAN"]):
        ax = axes[row, col]
        lab = results[(mname, dname)]
        for c in np.unique(lab):
            pts = X[lab == c]
            if c == -1:
                ax.scatter(pts[:, 0], pts[:, 1], s=14, marker="x", color="gray", label="noise")
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=8, marker=markers[c % 4], label=f"cluster {c}")
        ax.set_title(f"{mname} on {dname}: ARI {round(adjusted_rand_score(y_true, lab), 2) + 0.0:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("feature 2")
        if row == 1:
            ax.set_xlabel("feature 1")
fig.suptitle("k-means and Ward cut the shapes in half; single linkage and DBSCAN follow them")
fig.tight_layout()
fig.savefig(IMG / "b5_2_shapes_four_methods.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved lessons/B/img/b5_2_shapes_four_methods.png ({(IMG / 'b5_2_shapes_four_methods.png').stat().st_size // 1000} KB)")

# %%
# Try this (not graded):
# 1. Raise the moons noise to 0.12. Which of single linkage and DBSCAN breaks first, and why?
# 2. Set MIN_SAMPLES = 2 (DBSCAN then has no border points: every point with a neighbour is core). Compare with single linkage cut at eps.
# 3. Run sklearn.cluster.HDBSCAN(min_cluster_size=20) on the three-density blobs.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
