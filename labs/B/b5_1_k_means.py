# %% [markdown]
# # B5.1 k-means from scratch, and choosing k — lab
# Lesson: lessons/B/b5_1_k_means.md
# Run: uv run python labs/B/b5_1_k_means.py   (CPU, defaults: ~20 s)
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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

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
OUT = ROOT / "outputs" / "b5_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the dataset and the restart experiment larger.
GPU_SCALE = False
N_SAMPLES = 60_000 if GPU_SCALE else 600
N_RESTART_TRIALS = 1_000 if GPU_SCALE else 100  # single-init runs used to count bad local minima
K_RANGE = range(1, 11)
MAX_ITER = 300
TOL = 1e-4  # relative to the mean feature variance, as in scikit-learn

# %%
# Data: six round blobs on a 3 x 2 grid in 2-D (think customers described by two standardized features).
section("data")
CENTERS = np.array([[0, 0], [4, 0], [8, 0], [0, 5], [4, 5], [8, 5]], dtype=float)
X, y_true = make_blobs(n_samples=N_SAMPLES, centers=CENTERS, cluster_std=1.0, random_state=SEED)
y_true = y_true.astype(np.int64)
n, d = X.shape
print(f"X {X.shape}, true groups {np.bincount(y_true).tolist()} (labels are never shown to k-means)")
print(f"feature means {np.round(X.mean(axis=0), 3).tolist()}, stds {np.round(X.std(axis=0), 3).tolist()}")


# %%
# The building blocks: squared distances, the assignment step, the update step, the objective (inertia).
def sq_dists(X: FloatArray, C: FloatArray) -> FloatArray:
    """Squared Euclidean distances. X (n, d), C (k, d) -> (n, k)."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 (B2.3); clip tiny negatives from rounding
    D2 = (X**2).sum(axis=1)[:, None] - 2 * X @ C.T + (C**2).sum(axis=1)[None, :]
    return np.maximum(D2, 0.0)


def assign(X: FloatArray, C: FloatArray) -> IntArray:
    """Assignment step: index of the nearest centroid. X (n, d), C (k, d) -> (n,)."""
    return np.argmin(sq_dists(X, C), axis=1)


def update(X: FloatArray, labels: IntArray, C_old: FloatArray) -> FloatArray:
    """Update step: each centroid becomes the mean of its points (an empty cluster keeps its old centroid). -> (k, d)."""
    C = C_old.copy()
    for j in range(len(C)):
        members = X[labels == j]
        if len(members):
            C[j] = members.mean(axis=0)
    return C


def inertia(X: FloatArray, C: FloatArray, labels: IntArray) -> float:
    """Within-cluster sum of squared distances J. X (n, d), C (k, d), labels (n,) -> scalar."""
    return float(((X - C[labels]) ** 2).sum())


# %%
# Worked example by hand (lesson): four points, k = 2, a bad start at A and B.
section("worked example by hand")
P = np.array([[1.0, 1.0], [1.0, 3.0], [5.0, 1.0], [6.0, 3.0]])  # A, B, C, D
names = ["A", "B", "C", "D"]
C = P[[0, 1]].copy()  # start the centroids on A and B
for it in range(1, 4):
    lab = assign(P, C)
    D2 = sq_dists(P, C)
    print(f"iteration {it}: squared distances to (c1, c2): "
          + "  ".join(f"{nm} ({D2[i, 0]:g}, {D2[i, 1]:g})" for i, nm in enumerate(names)))
    C_new = update(P, lab, C)
    groups = ["{" + ",".join(nm for nm, l in zip(names, lab) if l == j) + "}" for j in range(2)]
    print(f"  clusters {groups[0]} {groups[1]} -> centroids {C_new[0].tolist()} {C_new[1].tolist()}, "
          f"inertia {inertia(P, C_new, lab):g}")
    if np.allclose(C_new, C):
        print("  centroids did not move: converged")
        break
    C = C_new
good = np.array([0, 0, 1, 1])
C_good = update(P, good, np.zeros((2, 2)))
print(f"the other split {{A,B}} {{C,D}}: centroids {C_good[0].tolist()} {C_good[1].tolist()}, inertia {inertia(P, C_good, good):g}")
D2A = sq_dists(P[[0]], P)[0]
print(f"k-means++ after picking A first: D^2 to B, C, D = {D2A[1]:g}, {D2A[2]:g}, {D2A[3]:g} -> "
      f"probabilities {D2A[1] / D2A.sum():.3f}, {D2A[2] / D2A.sum():.3f}, {D2A[3] / D2A.sum():.3f}")
dist_A = np.sqrt(D2A)
a_A, b_A = dist_A[1], (dist_A[2] + dist_A[3]) / 2
print(f"silhouette of A in {{A,B}} {{C,D}}: a = {a_A:.3f}, b = ({dist_A[2]:.3f} + {dist_A[3]:.3f}) / 2 = {b_A:.3f}, "
      f"s = {(b_A - a_A) / max(a_A, b_A):.3f}")


# %%
# From scratch: k-means++ initialization and Lloyd's algorithm.
def kmeans_plus_plus(X: FloatArray, k: int, gen: np.random.Generator) -> FloatArray:
    """k-means++ seeding: first centroid uniform, each next one with probability proportional to D^2. -> (k, d)."""
    C = [X[gen.integers(len(X))]]
    for _ in range(1, k):
        D2 = sq_dists(X, np.array(C)).min(axis=1)  # (n,) squared distance to the nearest chosen centroid
        C.append(X[gen.choice(len(X), p=D2 / D2.sum())])
    return np.array(C)


def random_init(X: FloatArray, k: int, gen: np.random.Generator) -> FloatArray:
    """Forgy initialization: k distinct data points chosen uniformly. -> (k, d)."""
    return X[gen.choice(len(X), size=k, replace=False)].copy()


def lloyd(X: FloatArray, C0: FloatArray, max_iter: int = MAX_ITER, tol: float = TOL) -> tuple[FloatArray, IntArray, float, list[float]]:
    """Lloyd's algorithm from initial centroids C0 (k, d). -> centroids (k, d), labels (n,), inertia, inertia per iteration."""
    tol_abs = tol * float(np.mean(X.var(axis=0)))  # stop when centroids move less than this (squared)
    C, history = C0.copy(), []
    for _ in range(max_iter):
        labels = assign(X, C)
        C_new = update(X, labels, C)
        history.append(inertia(X, C_new, labels))
        shift = float(((C_new - C) ** 2).sum())
        C = C_new
        if shift <= tol_abs:
            break
    labels = assign(X, C)
    return C, labels, inertia(X, C, labels), history


def kmeans(X: FloatArray, k: int, n_init: int, gen: np.random.Generator, init: str = "k-means++") -> tuple[FloatArray, IntArray, float]:
    """Run Lloyd n_init times from fresh initializations and keep the lowest inertia. -> centroids, labels, inertia."""
    best: tuple[FloatArray, IntArray, float] | None = None
    for _ in range(n_init):
        C0 = kmeans_plus_plus(X, k, gen) if init == "k-means++" else random_init(X, k, gen)
        C, labels, J, _ = lloyd(X, C0)
        if best is None or J < best[2]:
            best = (C, labels, J)
    assert best is not None
    return best


section("from scratch: Lloyd's algorithm with k-means++")
K = 6
C0 = kmeans_plus_plus(X, K, np.random.default_rng(SEED))
C_mine, lab_mine, J_mine, hist = lloyd(X, C0)
assert C_mine.shape == (K, d) and lab_mine.shape == (n,), (C_mine.shape, lab_mine.shape)
print(f"converged in {len(hist)} iterations; inertia per iteration: {[round(h, 2) for h in hist]}")
assert all(a >= b - 1e-9 for a, b in zip(hist, hist[1:])), "inertia must never increase"
print("inertia never increased: OK")
print(f"cluster sizes {sorted(np.bincount(lab_mine, minlength=K).tolist())}, final inertia {J_mine:.4f}")


# %%
# Adjusted Rand index from scratch: pair-counting agreement between two labelings, corrected for chance.
def comb2(m: npt.NDArray[np.int64] | int) -> FloatArray:
    """Number of pairs m (m - 1) / 2, elementwise."""
    m = np.asarray(m, dtype=np.float64)
    return m * (m - 1) / 2


def adjusted_rand(a: IntArray, b: IntArray) -> float:
    """ARI between two labelings a (n,), b (n,): 1 identical up to relabelling, about 0 for random labelings."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)  # contingency table n_uv
    np.add.at(table, (ai, bi), 1)
    together_both = comb2(table).sum()  # pairs in the same group in a AND in b
    together_a, together_b = comb2(table.sum(axis=1)).sum(), comb2(table.sum(axis=0)).sum()
    expected = together_a * together_b / comb2(len(a))  # expected overlap if b were a random relabelling
    max_index = (together_a + together_b) / 2
    return float((together_both - expected) / (max_index - expected))


section("adjusted Rand index")
toy_a, toy_b = np.array([0, 0, 1, 1, 2, 2]), np.array([5, 5, 3, 3, 9, 9])
print(f"same groups, different names: ARI {adjusted_rand(toy_a, toy_b):.4f}")
print(f"one point moved: ARI {adjusted_rand(toy_a, np.array([0, 0, 1, 1, 1, 2])):.4f}")
shuffled = np.random.default_rng(SEED).permutation(lab_mine)
print(f"k-means labels vs the same labels shuffled across points: ARI {adjusted_rand(lab_mine, shuffled):.4f}")
print(f"k-means labels vs the true blobs: ARI {adjusted_rand(lab_mine, y_true):.4f}")
for x_, y_ in [(toy_a, toy_b), (lab_mine, shuffled), (lab_mine, y_true)]:
    np.testing.assert_allclose(adjusted_rand(x_, y_), adjusted_rand_score(x_, y_), rtol=1e-10, atol=1e-12)
print("matches sklearn.metrics.adjusted_rand_score: OK")

# %%
# With scikit-learn: KMeans from the same initial centroids reaches the same clustering, up to relabelling.
section("with scikit-learn: KMeans")
sk = KMeans(n_clusters=K, init=C0, n_init=1, algorithm="lloyd", max_iter=MAX_ITER, tol=TOL, random_state=SEED).fit(X)
print(f"sklearn from the same start: inertia {sk.inertia_:.4f}, iterations {sk.n_iter_}")
print(f"ARI(mine, sklearn) = {adjusted_rand_score(lab_mine, sk.labels_):.4f}")
# relabel: map each of my clusters to the sklearn cluster most of its points fall in
mapping = np.array([np.bincount(sk.labels_[lab_mine == j], minlength=K).argmax() for j in range(K)])
assert len(set(mapping.tolist())) == K, "the mapping must be a permutation"
np.testing.assert_array_equal(mapping[lab_mine], sk.labels_)
np.testing.assert_allclose(C_mine, sk.cluster_centers_[mapping], rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(J_mine, sk.inertia_, rtol=1e-6)
print("matches sklearn.cluster.KMeans (labels up to relabelling, centroids, inertia): OK")
sk_default = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
_, lab_best, J_best = kmeans(X, K, n_init=10, gen=np.random.default_rng(SEED))
print(f"independent runs, n_init=10 each: inertia mine {J_best:.4f}, sklearn {sk_default.inertia_:.4f}, "
      f"ARI between them {adjusted_rand_score(lab_best, sk_default.labels_):.4f}")

# %%
# Bad local minima: one initialization per run, many runs. How often does Lloyd get stuck?
section("bad local minima and n_init")
gen = np.random.default_rng(SEED)
J_best_known = min(J_best, sk_default.inertia_)
results: dict[str, FloatArray] = {}
for init in ["random", "k-means++"]:
    results[init] = np.array([kmeans(X, K, n_init=1, gen=gen, init=init)[2] for _ in range(N_RESTART_TRIALS)])
    stuck = results[init] > 1.01 * J_best_known
    print(f"init={init:<10} n_init=1: {int(stuck.sum())} of {N_RESTART_TRIALS} runs stuck (inertia > 1% above best "
          f"{J_best_known:.2f}); worst {results[init].max():.2f}")
for init in ["random", "k-means++"]:
    J10 = np.array([kmeans(X, K, n_init=10, gen=gen, init=init)[2] for _ in range(N_RESTART_TRIALS // 5)])
    print(f"init={init:<10} n_init=10: {int((J10 > 1.01 * J_best_known).sum())} of {N_RESTART_TRIALS // 5} runs stuck")
worst_seed_run = None
g2 = np.random.default_rng(SEED)
for _ in range(N_RESTART_TRIALS):  # find one stuck random-init run to draw
    C0_bad = random_init(X, K, g2)
    C_bad, lab_bad, J_bad, _ = lloyd(X, C0_bad)
    if J_bad > 1.2 * J_best_known:
        worst_seed_run = (C_bad, lab_bad, J_bad)
        break
assert worst_seed_run is not None
C_bad, lab_bad, J_bad = worst_seed_run
print(f"an example stuck run: inertia {J_bad:.2f}, cluster sizes {sorted(np.bincount(lab_bad, minlength=K).tolist())}, "
      f"ARI with the true blobs {adjusted_rand_score(y_true, lab_bad):.4f} (best run: {adjusted_rand_score(y_true, lab_best):.4f})")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=120)
for ax, (labs, Cs, J, title) in zip(axes, [(lab_bad, C_bad, J_bad, "stuck"), (lab_best, None, J_best, "best of n_init=10")]):
    ax.scatter(X[:, 0], X[:, 1], c=labs, cmap="tab10", s=8)
    if Cs is not None:
        ax.scatter(Cs[:, 0], Cs[:, 1], marker="x", s=120, c="black")
    ax.set_title(f"{title}: inertia {J:.0f}")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
fig.tight_layout()
fig.savefig(OUT / "local_minimum.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b5_1/local_minimum.png")


# %%
# Choosing k: the elbow (inertia against k) and the silhouette score, both from scratch.
def silhouette(X: FloatArray, labels: IntArray) -> FloatArray:
    """Silhouette value of each point. X (n, d), labels (n,) -> (n,); points alone in their cluster get 0."""
    D = np.sqrt(sq_dists(X, X))  # (n, n) Euclidean distances
    ks = np.unique(labels)
    sums = np.stack([D[:, labels == j].sum(axis=1) for j in ks], axis=1)  # (n, k) total distance to each cluster
    sizes = np.array([np.sum(labels == j) for j in ks])  # (k,)
    own = np.searchsorted(ks, labels)  # column of each point's own cluster
    own_size = sizes[own]
    a = sums[np.arange(len(X)), own] / np.maximum(own_size - 1, 1)  # mean distance to the OTHER members
    mean_other = sums / sizes[None, :]
    mean_other[np.arange(len(X)), own] = np.inf
    b = mean_other.min(axis=1)  # mean distance to the nearest other cluster
    s = (b - a) / np.maximum(a, b)
    return np.where(own_size > 1, s, 0.0)


section("choosing k: elbow and silhouette")
inertias, sils = {}, {}
for k in K_RANGE:
    _, labs, J = kmeans(X, k, n_init=10, gen=np.random.default_rng(SEED))
    inertias[k] = J
    if k >= 2:
        sils[k] = float(silhouette(X, labs).mean())
        np.testing.assert_allclose(sils[k], silhouette_score(X, labs), rtol=1e-6, atol=1e-8)
print(f"{'k':>2} {'inertia':>9} {'drop':>8} {'silhouette':>10}")
for k in K_RANGE:
    drop = "" if k == 1 else f"{inertias[k - 1] - inertias[k]:8.1f}"
    sil = "" if k == 1 else f"{sils[k]:10.4f}"
    print(f"{k:>2} {inertias[k]:9.1f} {drop:>8} {sil:>10}")
print("from-scratch silhouette matches sklearn.metrics.silhouette_score: OK")
print(f"best silhouette at k = {max(sils, key=sils.get)}")

# %%
# Stability: does the clustering come back if we change the seed or resample the rows?
section("stability: seeds and bootstrap")
base = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
seed_aris = [adjusted_rand_score(base.labels_, KMeans(n_clusters=K, n_init=10, random_state=s).fit(X).labels_) for s in range(1, 6)]
print(f"k={K}, 5 other seeds: ARI with seed 0 = {[round(a, 4) for a in seed_aris]}")
for k_try in [3, 6, 9]:
    ref = KMeans(n_clusters=k_try, n_init=10, random_state=SEED).fit(X)
    boot = []
    g = np.random.default_rng(SEED)
    for _ in range(10):
        rows = g.integers(0, n, size=n)  # bootstrap resample
        m = KMeans(n_clusters=k_try, n_init=10, random_state=SEED).fit(X[rows])
        boot.append(adjusted_rand_score(ref.labels_, m.predict(X)))  # compare on ALL original points
    print(f"k={k_try}: ARI of 10 bootstrap refits with the full-data clustering: mean {np.mean(boot):.4f}, min {np.min(boot):.4f}")

# %%
# Key plot: the elbow and the silhouette side by side.
section("key plot: choosing k")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=120)
ks = list(K_RANGE)
ax1.plot(ks, [inertias[k] for k in ks], "o-")
ax1.axvline(K, color="gray", linestyle=":", label=f"true k = {K}")
ax1.set_xlabel("number of clusters k")
ax1.set_ylabel("inertia (sum of squared distances)")
ax1.set_title(f"Elbow: inertia falls slowly after k = {K}")
ax1.legend(loc="best")
ax2.plot(list(sils), list(sils.values()), "s-", color="tab:orange")
ax2.axvline(K, color="gray", linestyle=":", label=f"true k = {K}")
ax2.set_xlabel("number of clusters k")
ax2.set_ylabel("mean silhouette score")
ax2.set_title(f"Silhouette peaks at k = {max(sils, key=sils.get)}")
ax2.legend(loc="best")
fig.tight_layout()
fig.savefig(IMG / "b5_1_elbow_silhouette.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b5_1_elbow_silhouette.png")

# %%
# Try this (not graded):
# 1. Set cluster_std=1.6 in make_blobs. Does the elbow or the silhouette still point at 6?
# 2. Stretch feature 2 by 10 (X[:, 1] *= 10) without re-standardizing. Which clusters does k-means find now?
# 3. Replace one blob by 20 points far away (an outlier group). How do the centroids and the elbow move?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
