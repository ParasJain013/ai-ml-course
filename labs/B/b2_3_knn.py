# %% [markdown]
# # B2.3 K-nearest neighbours — lab
# Lesson: lessons/B/b2_3_knn.md
# Run: uv run python labs/B/b2_3_knn.py   (CPU, defaults: ~10 s)
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
from sklearn.datasets import load_diabetes, load_wine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
OUT = ROOT / "outputs" / "b2_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the curse-of-dimensionality simulation bigger.
GPU_SCALE = False
SIM_N = 10_000 if GPU_SCALE else 1_000  # points in the unit cube
SIM_QUERIES = 1_000 if GPU_SCALE else 100  # query points per dimension
SIM_DIMS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
NOISE_DIMS = [0, 10, 30, 100, 300, 1000]  # pure-noise features added to wine
K_VALUES = [1, 3, 5, 7, 9, 15, 25]
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: the wine dataset bundled with scikit-learn. 178 wines from three cultivars, 13 chemical
# measurements on wildly different scales (proline in the hundreds, hue around 1).
section("data")
wine = load_wine()
wine_names = list(wine.feature_names)
X_all = wine.data.astype(np.float64)  # (178, 13)
y_all = wine.target.astype(np.int64)  # (178,) classes 0, 1, 2


def train_val_test_split(
    n: int,
    val_frac: float,
    test_frac: float,
    gen: np.random.Generator,
    stratify: IntArray | None = None,
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_all)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
print(f"train X {X_tr.shape}  val X {X_va.shape}  (test {len(i_te)} rows kept aside)")
print(f"class counts: train {np.bincount(y_tr).tolist()}  val {np.bincount(y_va).tolist()}")
sds = X_tr.std(axis=0)
print(f"feature sd (train): smallest {sds.min():.4f} ({wine_names[sds.argmin()]})   "
      f"largest {sds.max():.2f} ({wine_names[sds.argmax()]})   ratio {sds.max() / sds.min():.0f}")

# %%
# Worked example by hand: five 2-D training points, one query.
section("worked example by hand")
P5 = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 2.0], [5.0, 5.0], [6.0, 3.0]])  # (5, 2)
c5 = np.array([0, 0, 1, 1, 1])  # class labels
t5 = np.array([1.0, 2.0, 3.0, 6.0, 5.0])  # regression targets
q = np.array([3.0, 3.0])
d_euc = np.sqrt(((P5 - q) ** 2).sum(axis=1))  # (5,)
d_man = np.abs(P5 - q).sum(axis=1)  # (5,)
order = np.argsort(d_euc)
print(f"query {q.tolist()}")
print(f"Euclidean distances {show(d_euc)}   Manhattan distances {show(d_man)}")
print(f"neighbours by Euclidean distance (nearest first): {order.tolist()}   their classes {c5[order].tolist()}")
for k in [1, 3, 5]:
    votes = np.bincount(c5[order[:k]], minlength=2)
    print(f"k = {k}: votes {votes.tolist()} -> class {votes.argmax()}   "
          f"regression: mean of targets {t5[order[:k]].tolist()} = {t5[order[:k]].mean():.4f}")
# The same idea with features in different units: the raw distance is all income.
P4 = np.array([[25.0, 40_000.0], [60.0, 41_000.0], [30.0, 46_000.0], [55.0, 52_000.0]])  # age, income
c4 = np.array([0, 1, 0, 1])
q4 = np.array([58.0, 45_000.0])
d_raw = np.sqrt(((P4 - q4) ** 2).sum(axis=1))
mu4, sd4 = P4.mean(axis=0), P4.std(axis=0)
d_std = np.sqrt((((P4 - q4) / sd4) ** 2).sum(axis=1))
print(f"age/income example: raw distances {np.round(d_raw, 1).tolist()} -> nearest row {d_raw.argmin()} "
      f"(class {c4[d_raw.argmin()]})")
print(f"  sd = {np.round(sd4, 2).tolist()}   standardized distances {show(d_std)} -> nearest row "
      f"{d_std.argmin()} (class {c4[d_std.argmin()]})")

# %%
# From scratch: the vectorized distance matrix (A1.2) and the vote.
section("from scratch: kNN classifier")


def pairwise_sq_dists(A: FloatArray, B: FloatArray) -> FloatArray:
    """Squared Euclidean distances via |a|^2 + |b|^2 - 2 a.b. A (m, d), B (n, d) -> (m, n)."""
    # from labs/A/a1_2_numpy_arrays.py (pairwise_trick), kept squared: the ranking is the same
    a_sq = (A**2).sum(axis=1)[:, None]  # (m, 1)
    b_sq = (B**2).sum(axis=1)[None, :]  # (1, n)
    return np.maximum(a_sq + b_sq - 2.0 * A @ B.T, 0.0)  # (m, n), clip tiny negatives from rounding


def manhattan_dists(A: FloatArray, B: FloatArray) -> FloatArray:
    """L1 distances by broadcasting. A (m, d), B (n, d) -> (m, n)."""
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=-1)  # (m, n, d) -> (m, n)


def knn_indices(D: FloatArray, k: int) -> IntArray:
    """Indices of the k smallest entries in each row, nearest first. D (m, n) -> (m, k)."""
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]  # (m, k) the k nearest, in no particular order
    rows = np.arange(len(D))[:, None]
    return idx[rows, np.argsort(D[rows, idx], axis=1)]  # sort just those k


def knn_classify(X_train: FloatArray, y_train: IntArray, X_query: FloatArray, k: int,
                 metric: str = "euclidean") -> IntArray:
    """Majority vote of the k nearest training rows; ties go to the smallest label. -> (m,)."""
    D = pairwise_sq_dists(X_query, X_train) if metric == "euclidean" else manhattan_dists(X_query, X_train)
    nn = knn_indices(D, k)  # (m, k)
    n_classes = int(y_train.max()) + 1
    votes = np.zeros((len(X_query), n_classes), dtype=np.int64)  # (m, K)
    np.add.at(votes, (np.arange(len(X_query))[:, None], y_train[nn]), 1)
    return votes.argmax(axis=1)  # argmax returns the first maximum: the smallest label wins ties


mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
Z_tr, Z_va = (X_tr - mu) / sd, (X_va - mu) / sd
print(f"{'k':>3}  {'raw val acc':>11}  {'scaled val acc':>14}  {'scaled, manhattan':>17}")
for k in K_VALUES:
    acc_raw = np.mean(knn_classify(X_tr, y_tr, X_va, k) == y_va)
    acc_std = np.mean(knn_classify(Z_tr, y_tr, Z_va, k) == y_va)
    acc_man = np.mean(knn_classify(Z_tr, y_tr, Z_va, k, "manhattan") == y_va)
    print(f"{k:>3}  {acc_raw:>11.4f}  {acc_std:>14.4f}  {acc_man:>17.4f}")
raw_nn = knn_indices(pairwise_sq_dists(X_va, X_tr), 1)[:, 0]
diff_raw = np.abs(X_va - X_tr[raw_nn])  # (m, d) per-feature gap to the raw nearest neighbour
shares = (diff_raw**2 / (diff_raw**2).sum(axis=1, keepdims=True)).mean(axis=0)  # (d,) mean share per feature
top = np.argsort(shares)[::-1][:3]
print("share of the squared distance to the raw nearest neighbour, top 3 features:",
      {wine_names[j]: round(float(shares[j]), 4) for j in top}, f"(together {shares[top[:2]].sum():.4f} for the top 2)")

# %%
# With scikit-learn: KNeighborsClassifier in a pipeline after the scaler.
section("with scikit-learn")
for k in [1, 5, 15]:
    mine = knn_classify(Z_tr, y_tr, Z_va, k)
    sk = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(Z_tr, y_tr).predict(Z_va)
    assert np.array_equal(mine, sk), k
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_tr, y_tr)
    assert np.array_equal(pipe.predict(X_va), mine), k
print("from-scratch predictions identical to KNeighborsClassifier (k = 1, 5, 15), also inside a Pipeline: OK")
mine_man = knn_classify(Z_tr, y_tr, Z_va, 5, "manhattan")
sk_man = KNeighborsClassifier(n_neighbors=5, metric="manhattan", algorithm="brute").fit(Z_tr, y_tr).predict(Z_va)
assert np.array_equal(mine_man, sk_man)
print("Manhattan metric identical too: OK")
proba = KNeighborsClassifier(n_neighbors=5).fit(Z_tr, y_tr).predict_proba(Z_va[:3])
print(f"predict_proba for 3 val wines (k = 5): {show(proba)}   (vote shares, multiples of 1/5)")

# %%
# kNN regression on the diabetes data (B1.5 split): predict the mean target of the k nearest rows.
section("from scratch: kNN regression on diabetes")
diab = load_diabetes(as_frame=True, scaled=False)
Xd, yd = diab.data.to_numpy(dtype=np.float64), diab.target.to_numpy(dtype=np.float64)
# with stratify=None this makes the same random draws as B1.5's split, so the rows are identical
id_tr, id_va, _ = train_val_test_split(len(yd), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
mud, sdd = Xd[id_tr].mean(0), Xd[id_tr].std(0)
Zd_tr, Zd_va = (Xd[id_tr] - mud) / sdd, (Xd[id_va] - mud) / sdd


def knn_regress(X_train: FloatArray, y_train: FloatArray, X_query: FloatArray, k: int) -> FloatArray:
    """Mean target of the k nearest training rows. -> (m,)."""
    nn = knn_indices(pairwise_sq_dists(X_query, X_train), k)  # (m, k)
    return y_train[nn].mean(axis=1)


for k in [1, 5, 10, 20, 40]:
    pred = knn_regress(Zd_tr, yd[id_tr], Zd_va, k)
    sk_pred = KNeighborsRegressor(n_neighbors=k, algorithm="brute").fit(Zd_tr, yd[id_tr]).predict(Zd_va)
    np.testing.assert_allclose(pred, sk_pred, rtol=1e-12)
    print(f"k = {k:>2}: val R^2 {r2_score(yd[id_va], pred):.4f}")
print("matches KNeighborsRegressor at every k: OK")
lin = LinearRegression().fit(Zd_tr, yd[id_tr])
print(f"linear regression on the same split (B1.5): val R^2 {r2_score(yd[id_va], lin.predict(Zd_va)):.4f}")

# %%
# The curse of dimensionality, simulated: uniform points in the unit cube [0, 1]^d.
section("curse of dimensionality: distance contrast")
ratios, contrasts = [], []
for dim in SIM_DIMS:
    pts = rng.random((SIM_N, dim))  # (N, d)
    qs = rng.random((SIM_QUERIES, dim))  # (Q, d)
    D = np.sqrt(pairwise_sq_dists(qs, pts))  # (Q, N)
    near, far = D.min(axis=1), D.max(axis=1)
    ratios.append(float(np.mean(near / far)))
    contrasts.append(float(np.mean((far - near) / near)))
    edge = 0.01 ** (1 / dim)  # side of a sub-cube that holds 1% of the volume
    print(f"d = {dim:>4}: mean nearest/farthest {ratios[-1]:.4f}   relative contrast (far - near)/near "
          f"{contrasts[-1]:8.4f}   cube holding 1% of points has side {edge:.4f}")

# %%
section("curse of dimensionality: noise features on wine")
noise_acc = []
gen_noise = np.random.default_rng(SEED + 1)
for m in NOISE_DIMS:
    noise = gen_noise.normal(size=(len(y_all), m))  # (178, m) standardized noise, no signal at all
    Xn_tr = np.column_stack([Z_tr, noise[i_tr]])
    Xn_va = np.column_stack([Z_va, noise[i_va]])
    noise_acc.append(float(np.mean(knn_classify(Xn_tr, y_tr, Xn_va, 5) == y_va)))
    print(f"13 real + {m:>4} noise features: kNN (k = 5) val accuracy {noise_acc[-1]:.4f}")

# %%
# Key plot: the distance contrast collapses with dimension, and noise features drown kNN.
section("key plot")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
ax0.plot(SIM_DIMS, ratios, "o-")
ax0.set_xscale("log")
ax0.set_ylim(0, 1)
ax0.set_xlabel("dimension d (log scale)")
ax0.set_ylabel("nearest / farthest distance")
ax0.set_title("In high dimensions the nearest point is almost as far as the farthest")
ax1.plot([max(m, 1) for m in NOISE_DIMS], noise_acc, "s--", color="tab:orange")
ax1.set_xscale("log")
ax1.set_ylim(0, 1.05)
ax1.axhline(np.bincount(y_tr).max() / len(y_tr), color="0.6", ls=":", label="majority-class rate")
ax1.set_xlabel("noise features added to 13 real ones (log scale; 0 plotted at 1)")
ax1.set_ylabel("validation accuracy (k = 5)")
ax1.set_title("Irrelevant features drown the real ones")
ax1.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b2_3_curse_of_dimensionality.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b2_3_curse_of_dimensionality.png")

# %%
# Try this (not graded):
# 1. Use KNeighborsClassifier(weights="distance") and compare validation accuracy for k = 15 and 25.
# 2. Scale with MinMaxScaler instead of StandardScaler. Does the best k or accuracy change?
# 3. In the simulation, draw points from a 2-D plane embedded in d = 1000 dimensions. Does the contrast collapse?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
