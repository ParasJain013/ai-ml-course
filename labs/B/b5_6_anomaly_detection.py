# %% [markdown]
# # B5.6 Anomaly detection — lab
# Lesson: lessons/B/b5_6_anomaly_detection.md
# Run: uv run python labs/B/b5_6_anomaly_detection.py   (CPU, defaults: ~10 s)
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
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

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
OUT = ROOT / "outputs" / "b5_6"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only grows the forest and the bootstrap.
GPU_SCALE = False
N_TREES = 500 if GPU_SCALE else 100  # isolation trees
SUBSAMPLE = 256  # psi in Liu et al. (2008); also scikit-learn's default max_samples
N_ANOMALIES = 15  # malignant rows kept as the "rare, labelled" anomalies
K_NN = 5  # neighbours for the k-NN distance score
N_BOOT = 2_000 if GPU_SCALE else 1_000
EULER_GAMMA = 0.5772156649

# %%
# Worked example by hand: z-scores mask two outliers, Tukey's fences do not.
section("worked example by hand: rules")
x = np.array([10.0, 11, 12, 12, 13, 14, 50, 52])
z = (x - x.mean()) / x.std()  # ddof=0, as in B1.2
q1, q3 = np.quantile(x, [0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"x = {x.tolist()}   mean {x.mean():.4f}   std (ddof=0) {x.std():.4f}")
print(f"z = {np.round(z, 4).tolist()}   largest possible |z| for n = 8: sqrt(7) = {np.sqrt(7):.4f}")
print(f"Q1 {q1:.4f}  Q3 {q3:.4f}  IQR {iqr:.4f}  fences [{lo:.4f}, {hi:.4f}]  flagged {x[(x < lo) | (x > hi)].tolist()}")
x6 = x[:6]
z6 = (x6 - x6.mean()) / x6.std()
print(f"without the two outliers: mean {x6.mean():.4f}  std {x6.std():.4f}  -> z of 50 would be {(50 - x6.mean()) / x6.std():.2f}")


# %%
section("worked example by hand: isolation and k-NN")


def harmonic(i: int) -> float:
    """Exact harmonic number H(i) = 1 + 1/2 + ... + 1/i."""
    return float(np.sum(1.0 / np.arange(1, i + 1)))


def c_factor(n: int) -> float:
    """Average path length of an unsuccessful search in a BST of n points (Liu et al. 2008)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    h = np.log(n - 1) + EULER_GAMMA  # H(n-1) ~ ln(n-1) + gamma
    return float(2 * h - 2 * (n - 1) / n)


pts = np.array([1.0, 2, 3, 4, 10])
print(f"1-D points {pts.tolist()}: P(first uniform cut isolates 10) = (10 - 4) / (10 - 1) = {(10 - 4) / (10 - 1):.4f}")
print(f"H(255) exact {harmonic(255):.4f}   ln(255) + gamma {np.log(255) + EULER_GAMMA:.4f}")
c256 = c_factor(256)
print(f"c(256) = 2 H(255) - 2 * 255/256 = {c256:.4f}   (exact-H version {2 * harmonic(255) - 2 * 255 / 256:.4f})")
print(f"c(256) = 2 H(256) - 2 (same number) = {2 * harmonic(256) - 2:.4f}")
for h in [3.0, c256, 12.0]:
    print(f"mean path length {h:.4f} -> score s = 2^(-h / c) = {2 ** (-h / c256):.4f}")
P = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1], [4, 4]])
Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(Dm, np.inf)
Ds = np.sort(Dm, axis=1)
print(f"k-NN scores, k = 1: {np.round(Ds[:, 0], 4).tolist()}   k = 2: {np.round(Ds[:, 1], 4).tolist()}")


# %%
# Check the c(n) derivation: brute-force random BSTs, and the identity sum_{g=1}^{N} H(g) = (N+1) H(N) - N.
section("check: c(n) is the mean depth of a new point in a random BST")


def mean_depth_random_bst(m: int, trials: int, gen: np.random.Generator) -> float:
    """Insert m random keys in random order, then drop a random query in; average its depth (number of keys on its path)."""
    total = 0
    for _ in range(trials):
        order = gen.permutation(m)  # insertion order of keys 0..m-1
        g = gen.integers(0, m + 1)  # query falls in gap g (between key g-1 and key g)
        # key j is on the query's path iff it was inserted before every key strictly between j and the gap
        rank = np.empty(m, dtype=np.int64)
        rank[order] = np.arange(m)
        depth = 0
        for j in range(m):
            between = range(j + 1, g) if j < g else range(g, j)
            if all(rank[j] < rank[i] for i in between):
                depth += 1
        total += depth
    return total / trials


gen = np.random.default_rng(SEED)
for n_pts in [2, 5, 10]:
    sim = mean_depth_random_bst(n_pts - 1, 20_000, gen)
    print(f"n = {n_pts:>2} points: simulated {sim:.3f}   2 H(n) - 2 = {2 * harmonic(n_pts) - 2:.3f}")
print(f"identity at N = 7: sum H(g) = {sum(harmonic(g) for g in range(1, 8)):.6f}   (N+1) H(N) - N = {8 * harmonic(7) - 7:.6f}")

# %%
# Data: breast cancer (Wisconsin diagnostic), 30 numeric features. All 357 benign tumours plus a random 15 of the
# 212 malignant ones play "normal traffic plus a few rare anomalies". The labels are used only to evaluate.
section("data")
bc = load_breast_cancer()
benign = np.flatnonzero(bc.target == 1)  # in this dataset 1 = benign
malignant = np.flatnonzero(bc.target == 0)
keep = np.r_[benign, np.random.default_rng(SEED).choice(malignant, size=N_ANOMALIES, replace=False)]
X_raw = bc.data[keep]
y = (bc.target[keep] == 0).astype(np.int64)  # 1 = anomaly (malignant)
# Robust scaling with the median and IQR of the whole batch: no labels are used, and the few anomalies barely move
# a median, whereas they inflate a mean and a standard deviation (the masking effect above).
med = np.median(X_raw, axis=0)
q1_all, q3_all = np.quantile(X_raw, [0.25, 0.75], axis=0)
X = (X_raw - med) / (q3_all - q1_all)
n, d = X.shape
assert X.shape == (372, 30), X.shape
print(f"X {X.shape}: {int(y.sum())} anomalies, prevalence {y.mean():.4f} (the AP of a random score)")


# %%
# From scratch 1: statistical rules turned into scores (one number per row, higher = more anomalous).
section("from scratch: z-score and IQR scores")


def zscore_score(X: FloatArray) -> FloatArray:
    """Largest absolute z-score across features. X (n, d) -> (n,)."""
    Z = (X - X.mean(axis=0)) / X.std(axis=0)  # (n, d)
    return np.abs(Z).max(axis=1)


def iqr_score(X: FloatArray) -> FloatArray:
    """Largest distance outside Tukey's fences, in IQR units (0 inside the fences). X (n, d) -> (n,)."""
    q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)  # (d,) each
    iqr = q3 - q1
    below = (q1 - 1.5 * iqr - X) / iqr
    above = (X - q3 - 1.5 * iqr) / iqr
    return np.maximum(np.maximum(below, above), 0).max(axis=1)


s_z, s_iqr = zscore_score(X), iqr_score(X)
print(f"rows with some |z| > 3: {int((s_z > 3).sum())} (anomalies among them {int(y[s_z > 3].sum())})")
print(f"rows outside a fence in some feature: {int((s_iqr > 0).sum())} (anomalies among them {int(y[s_iqr > 0].sum())})")
print("-> a per-feature rule over 30 features flags many normal rows: each feature gets its own chance to fire")


# %%
# From scratch 2: an isolation forest. A tree is a nested tuple: ("leaf", size) or ("split", feature, value, left, right).
section("from scratch: isolation forest")
Tree = tuple


def grow_tree(X: FloatArray, depth: int, max_depth: int, gen: np.random.Generator) -> Tree:
    """Split on a random feature at a uniform random value until a point is alone or max_depth is hit. X (m, d)."""
    m = len(X)
    if m <= 1 or depth >= max_depth:
        return ("leaf", m)
    lo, hi = X.min(axis=0), X.max(axis=0)
    usable = np.flatnonzero(hi > lo)  # a constant feature cannot split
    if len(usable) == 0:
        return ("leaf", m)
    j = int(gen.choice(usable))
    v = float(gen.uniform(lo[j], hi[j]))
    left = X[:, j] < v
    return ("split", j, v, grow_tree(X[left], depth + 1, max_depth, gen), grow_tree(X[~left], depth + 1, max_depth, gen))


def path_length(x: FloatArray, tree: Tree, depth: int = 0) -> float:
    """Depth at which x lands, plus c(size) for an unexpanded leaf. x (d,) -> scalar."""
    if tree[0] == "leaf":
        return depth + c_factor(tree[1])
    _, j, v, left, right = tree
    return path_length(x, left if x[j] < v else right, depth + 1)


def isolation_forest(X: FloatArray, n_trees: int, psi: int, gen: np.random.Generator) -> list[Tree]:
    """Grow n_trees isolation trees, each on psi rows drawn without replacement. X (n, d)."""
    psi = min(psi, len(X))
    max_depth = int(np.ceil(np.log2(psi)))  # 8 for psi = 256: anomalies are isolated well before this
    return [grow_tree(X[gen.choice(len(X), size=psi, replace=False)], 0, max_depth, gen) for _ in range(n_trees)]


def iforest_score(X: FloatArray, forest: list[Tree], psi: int) -> FloatArray:
    """Anomaly score s = 2^(-mean path length / c(psi)), in (0, 1]; above 0.5 is suspicious. X (n, d) -> (n,)."""
    h = np.array([[path_length(x, t) for t in forest] for x in X]).mean(axis=1)  # (n,)
    return 2.0 ** (-h / c_factor(min(psi, len(X))))


forest = isolation_forest(X, N_TREES, SUBSAMPLE, np.random.default_rng(SEED))
s_if = iforest_score(X, forest, SUBSAMPLE)
print(f"{N_TREES} trees of {SUBSAMPLE} rows, max depth {int(np.ceil(np.log2(SUBSAMPLE)))}")
print(f"mean score: normal rows {s_if[y == 0].mean():.4f}   anomalies {s_if[y == 1].mean():.4f}   rows above 0.5: {int((s_if > 0.5).sum())}")


# %%
# From scratch 3: distance to the k-th nearest neighbour (a density proxy: far neighbours = sparse region).
section("from scratch: k-NN distance score")


def knn_score(X: FloatArray, k: int) -> FloatArray:
    """Distance from each row to its k-th nearest other row. X (n, d) -> (n,)."""
    sq = (X**2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)  # (n, n) squared distances (B2.3)
    np.fill_diagonal(D2, np.inf)  # a row is not its own neighbour
    return np.sqrt(np.sort(D2, axis=1)[:, k - 1])


s_knn = knn_score(X, K_NN)
print(f"k = {K_NN}: median score normal rows {np.median(s_knn[y == 0]):.4f}   anomalies {np.median(s_knn[y == 1]):.4f}")


# %%
# With scikit-learn: IsolationForest, NearestNeighbors and LocalOutlierFactor.
section("with scikit-learn")
iso = IsolationForest(n_estimators=N_TREES, max_samples=SUBSAMPLE, random_state=SEED).fit(X)
s_if_sk = -iso.score_samples(X)  # score_samples returns minus the paper's score
rho = spearmanr(s_if, s_if_sk).statistic
print(f"IsolationForest: score range [{s_if_sk.min():.4f}, {s_if_sk.max():.4f}]   Spearman rank correlation with ours {rho:.4f}")
print(f"  (different random trees, so not equal; same algorithm, so the rankings agree closely)")
nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(X)
dist, _ = nn.kneighbors(X)  # column 0 is the row itself at distance 0
np.testing.assert_allclose(s_knn, dist[:, K_NN], rtol=1e-6, atol=1e-8)
print("k-NN distance matches sklearn.neighbors.NearestNeighbors: OK")
lof = LocalOutlierFactor(n_neighbors=20).fit(X)
s_lof = -lof.negative_outlier_factor_
print(f"LocalOutlierFactor (n_neighbors=20): {int((s_lof > 1.5).sum())} rows with LOF > 1.5")
for c in [0.04, 0.10]:
    flagged = IsolationForest(n_estimators=N_TREES, max_samples=SUBSAMPLE, contamination=c, random_state=SEED).fit(X).predict(X) == -1
    print(f"contamination={c}: flags {int(flagged.sum())} rows ({int(y[flagged].sum())} anomalies) - it only sets the threshold")


# %%
# Evaluation with few labels: average precision (PR-AUC), precision at an alert budget, bootstrap CIs.
section("evaluation: AP, precision@15, bootstrap CI")
scores = {"z-score (max |z|)": s_z, "IQR (max excess)": s_iqr, "isolation forest": s_if,
          "sklearn IsolationForest": s_if_sk, f"k-NN distance (k={K_NN})": s_knn, "LOF (sklearn)": s_lof}


def precision_at_k(y: IntArray, s: FloatArray, k: int) -> float:
    """Share of anomalies among the k highest scores (stable sort: ties go to the earlier row)."""
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())


def bootstrap_ap(y: IntArray, s: FloatArray, n_boot: int, gen: np.random.Generator) -> tuple[float, float]:
    """95% percentile bootstrap CI for average precision, resampling rows (A6.2)."""
    aps = []
    for _ in range(n_boot):
        idx = gen.integers(0, len(y), size=len(y))
        if y[idx].sum() == 0:
            continue
        aps.append(average_precision_score(y[idx], s[idx]))
    return float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


print(f"{'score':<24} {'AP':>6} {'95% CI':>16} {'P@15':>6}")
for name, s in scores.items():
    lo_ci, hi_ci = bootstrap_ap(y, s, N_BOOT, np.random.default_rng(SEED))
    print(f"{name:<24} {average_precision_score(y, s):>6.4f}   [{lo_ci:.4f}, {hi_ci:.4f}] {precision_at_k(y, s, N_ANOMALIES):>6.4f}")
print(f"random score: AP = prevalence = {y.mean():.4f}")
aps_seed = [average_precision_score(y, iforest_score(X, isolation_forest(X, N_TREES, SUBSAMPLE, np.random.default_rng(s)), SUBSAMPLE))
            for s in range(1, 6)]
print(f"our isolation forest with seeds 1-5: AP {np.round(aps_seed, 4).tolist()}  (the forest's own randomness)")

# %%
# Key plot: precision-recall curves of the from-scratch scores against the prevalence baseline.
section("key plot: precision-recall curves")
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
for (name, s), style in zip([(k, scores[k]) for k in ["z-score (max |z|)", "IQR (max excess)", "isolation forest",
                                                    f"k-NN distance (k={K_NN})"]], ["-", "--", "-.", ":"]):
    prec, rec, _ = precision_recall_curve(y, s)
    ax.step(rec, prec, where="post", linestyle=style, label=f"{name}, AP {average_precision_score(y, s):.2f}")
ax.axhline(y.mean(), color="gray", linestyle=(0, (1, 3)), label=f"random score, AP {y.mean():.2f}")
ax.set_xlabel("recall (share of the 15 anomalies found)")
ax.set_ylabel("precision (share of flagged rows that are anomalies)")
ax.set_ylim(0, 1.02)
ax.set_title("All four scores beat random; isolation forest ranks best")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b5_6_pr_curves.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b5_6_pr_curves.png")

# %%
# Try this (not graded):
# 1. Skip the scaling, or scale with the mean and standard deviation. Only the k-NN and LOF scores change: why?
# 2. Set N_ANOMALIES = 60 (14% prevalence). What happens to the isolation forest's advantage (masking, swamping)?
# 3. Vary K_NN over 1, 5, 20 and SUBSAMPLE over 32, 256. Plot AP against each.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
