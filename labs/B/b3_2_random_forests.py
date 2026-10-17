# %% [markdown]
# # B3.2 Random forests — lab
# Lesson: lessons/B/b3_2_random_forests.md
# Run: uv run python labs/B/b3_2_random_forests.py   (CPU, defaults: ~25 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Node = dict[str, Any]  # a tree node: a leaf has "counts"; an internal node also has a split
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
OUT = ROOT / "outputs" / "b3_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here uses a GPU; GPU_SCALE only grows the forests (on any machine, it takes longer).
GPU_SCALE = False
N_TREES_SCRATCH = 200 if GPU_SCALE else 100  # trees in the from-scratch random forest
N_TREES_BAG_SCRATCH = 100 if GPU_SCALE else 25  # from-scratch bagging is slower (all 64 features per split)
N_TREES_CURVE = 1000 if GPU_SCALE else 300  # trees in the scikit-learn forests behind the key plot
N_SEEDS = 5  # scikit-learn forests with different random_state, to show the run-to-run spread
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: scikit-learn's handwritten digits. 1,797 images of 8 x 8 pixels with intensities 0..16,
# flattened to 64 features, 10 classes. A single tree does poorly here, which makes the gains of
# averaging easy to see.
section("data")
digits = load_digits()
X_all = digits.data.astype(np.float64)  # (1797, 64)
y_all = digits.target.astype(np.int64)  # (1797,)
K = 10


def train_val_test_split(
    n: int,
    val_frac: float,
    test_frac: float,
    gen: np.random.Generator,
    stratify: IntArray | None = None,
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    if stratify is None:
        groups = [np.arange(n)]
    else:
        groups = [np.flatnonzero(stratify == c) for c in np.unique(stratify)]  # one group per class
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


i_tr, i_va, i_te = train_val_test_split(
    len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), stratify=y_all
)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
n, d = X_tr.shape
print(f"train X {X_tr.shape}  val X {X_va.shape}  (test {len(i_te)} rows kept aside)")
print(f"classes {K}, train counts per class {np.bincount(y_tr).tolist()}")
print(f"pixels that are 0 in every training image: {int(np.sum(X_tr.std(0) == 0))} of {d}")

# %%
# Worked example by hand: three small calculations behind bagging.
section("worked example by hand")
for m in [5, 100, 1000]:
    print(f"P(a given row is out of bag), n = {m:>4}: (1 - 1/n)^n = {(1 - 1 / m) ** m:.4f}")
print(f"limit 1/e = {np.exp(-1):.4f}")
p_ok = 0.7
maj3 = 3 * p_ok**2 * (1 - p_ok) + p_ok**3
print(f"3 independent classifiers, each right with p = {p_ok}: P(majority right) = 3p^2(1-p) + p^3 = {maj3:.4f}")
sim_rng = np.random.default_rng(SEED)
right = sim_rng.random((100_000, 3)) < p_ok  # (trials, classifiers)
print(f"simulated, 100,000 trials: {np.mean(right.sum(1) >= 2):.4f}")
sigma2, rho = 1.0, 0.5
for b in [1, 10, 100]:
    print(f"average of B = {b:>3} trees, variance {sigma2}, correlation {rho}: "
          f"rho*s2 + (1-rho)*s2/B = {rho * sigma2 + (1 - rho) * sigma2 / b:.4f}")
cov = sigma2 * (rho * np.ones((10, 10)) + (1 - rho) * np.eye(10))  # (10, 10) correlated "tree outputs"
draws = sim_rng.multivariate_normal(np.zeros(10), cov, size=100_000)
print(f"simulated variance of the mean of 10 correlated outputs: {draws.mean(1).var():.4f}")
toy_samples = [[0, 0, 2, 3], [1, 2, 2, 3], [0, 1, 1, 3]]  # three bootstrap samples of rows 0..3
toy_votes = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]])  # (3 trees, 4 rows) each tree's class
toy_y = np.array([1, 0, 0, 1])
oob_right = []
for i in range(4):
    oob_trees = [b for b, s in enumerate(toy_samples) if i not in s]
    oob_votes = [int(toy_votes[b, i]) for b in oob_trees]
    if oob_votes:
        oob_right.append(np.bincount(oob_votes, minlength=2).argmax() == toy_y[i])
    print(f"row {i} (y = {toy_y[i]}): out of bag for trees {oob_trees}  their votes {oob_votes}")
all_votes = (toy_votes.mean(axis=0) > 0.5).astype(int)
print(f"OOB accuracy over rows with a vote: {np.mean(oob_right):.4f}   "
      f"accuracy of all trees' majority on the training rows: {np.mean(all_votes == toy_y):.4f}")


# %%
# The B3.1 tree, with one change: at each node it searches a random subset of the features.
def impurity(counts: FloatArray, criterion: str) -> FloatArray:
    """Gini or entropy (bits) of each row of class counts. counts (m, K) -> (m,)."""
    # from labs/B/b3_1_decision_trees.py
    totals = counts.sum(axis=1, keepdims=True)
    p = counts / np.maximum(totals, 1)
    if criterion == "gini":
        return 1.0 - (p**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=1)


def split_scores(x: FloatArray, Y: FloatArray, criterion: str, min_leaf: int) -> tuple[FloatArray, FloatArray]:
    """Weighted child impurity for every threshold on one feature. x (m,), Y one-hot (m, K) -> (m - 1,) twice."""
    # from labs/B/b3_1_decision_trees.py
    order = np.argsort(x, kind="stable")
    xs, Ys = x[order], Y[order]
    m = len(xs)
    left = np.cumsum(Ys, axis=0)[:-1]
    right = Ys.sum(axis=0) - left
    n_left = np.arange(1, m)
    score = (n_left * impurity(left, criterion) + (m - n_left) * impurity(right, criterion)) / m
    valid = (xs[:-1] < xs[1:]) & (n_left >= min_leaf) & (m - n_left >= min_leaf)
    return np.where(valid, score, np.inf), (xs[:-1] + xs[1:]) / 2


def best_split_subset(
    X: FloatArray, Y: FloatArray, criterion: str, min_leaf: int, max_features: int, gen: np.random.Generator
) -> tuple[int, float, float]:
    """Best split among max_features randomly chosen non-constant features. X (m, d), Y (m, K)."""
    best = (-1, 0.0, np.inf)
    tried = 0
    for j in gen.permutation(X.shape[1]):  # random feature order
        if tried >= max_features:
            break
        x = X[:, j]
        if x.min() == x.max():  # constant in this node: skip it, as scikit-learn does
            continue
        tried += 1
        scores, thresholds = split_scores(x, Y, criterion, min_leaf)
        i = int(np.argmin(scores))
        if scores[i] < best[2]:
            best = (int(j), float(thresholds[i]), float(scores[i]))
    return best


def grow(X: FloatArray, Y: FloatArray, depth: int, max_depth: int | None, min_leaf: int,
         max_features: int, gen: np.random.Generator, criterion: str = "gini") -> Node:
    """Grow a CART tree with a random feature subset at every node. X (m, d), Y one-hot (m, K)."""
    counts = Y.sum(axis=0)
    node: Node = {"counts": counts, "n": len(Y)}
    if np.count_nonzero(counts) == 1 or (max_depth is not None and depth >= max_depth) or len(Y) < 2 * min_leaf:
        return node
    j, t, _ = best_split_subset(X, Y, criterion, min_leaf, max_features, gen)
    if j < 0:
        return node
    go_left = X[:, j] <= t
    node.update(
        feature=j,
        threshold=t,
        left=grow(X[go_left], Y[go_left], depth + 1, max_depth, min_leaf, max_features, gen, criterion),
        right=grow(X[~go_left], Y[~go_left], depth + 1, max_depth, min_leaf, max_features, gen, criterion),
    )
    return node


def predict_proba_tree(node: Node, X: FloatArray) -> FloatArray:
    """Class proportions of the leaf each row lands in. X (m, d) -> (m, K)."""
    # from labs/B/b3_1_decision_trees.py
    if "feature" not in node:
        return np.tile(node["counts"] / node["n"], (len(X), 1))
    go_left = X[:, node["feature"]] <= node["threshold"]
    out = np.empty((len(X), len(node["counts"])))
    out[go_left] = predict_proba_tree(node["left"], X[go_left])
    out[~go_left] = predict_proba_tree(node["right"], X[~go_left])
    return out


# %%
# From scratch: the forest. Each tree gets a bootstrap sample (n rows drawn with replacement) and a
# random feature subset at every split. We keep the in-bag mask to compute the out-of-bag estimate.
section("from scratch: a random forest")


def fit_forest(X: FloatArray, y: IntArray, n_trees: int, max_features: int, gen: np.random.Generator,
               min_leaf: int = 1) -> tuple[list[Node], npt.NDArray[np.bool_]]:
    """Fit n_trees on bootstrap samples. X (n, d), y (n,) -> trees and in-bag mask (n_trees, n)."""
    m = len(y)
    Y = np.eye(K)[y]  # (n, K) one-hot
    trees, in_bag = [], np.zeros((n_trees, m), dtype=bool)
    for b in range(n_trees):
        rows = gen.integers(0, m, m)  # bootstrap: m draws with replacement
        in_bag[b, rows] = True
        trees.append(grow(X[rows], Y[rows], 0, None, min_leaf, max_features, gen))
    return trees, in_bag


def tree_probas(trees: list[Node], X: FloatArray) -> FloatArray:
    """Every tree's class probabilities. X (m, d) -> (n_trees, m, K)."""
    return np.stack([predict_proba_tree(t, X) for t in trees])


def oob_accuracy(P_train: FloatArray, in_bag: npt.NDArray[np.bool_], y: IntArray) -> tuple[float, int]:
    """Accuracy of each row's prediction from only the trees that did not see it. P_train (B, n, K)."""
    oob = ~in_bag  # (B, n)
    summed = (P_train * oob[:, :, None]).sum(axis=0)  # (n, K) sum over trees where the row was out of bag
    has_vote = oob.sum(axis=0) > 0
    pred = np.argmax(summed[has_vote], axis=1)
    return float(np.mean(pred == y[has_vote])), int(np.sum(~has_vote))


max_feat = int(np.sqrt(d))  # 8 of 64 features per split, scikit-learn's default for classification
t_fit = time.perf_counter()
trees_rf, in_bag_rf = fit_forest(X_tr, y_tr, N_TREES_SCRATCH, max_feat, np.random.default_rng(SEED))
print(f"~ fitted {N_TREES_SCRATCH} trees in {time.perf_counter() - t_fit:.1f} s")
P_va_rf = tree_probas(trees_rf, X_va)  # (B, n_val, K)
P_tr_rf = tree_probas(trees_rf, X_tr)  # (B, n_train, K)
soft = np.argmax(P_va_rf.mean(axis=0), axis=1)  # average the probabilities, then take the largest
votes = np.argmax(P_va_rf, axis=2)  # (B, n_val) each tree's class
hard = np.array([np.bincount(votes[:, i], minlength=K).argmax() for i in range(votes.shape[1])])
oob_acc, no_vote = oob_accuracy(P_tr_rf, in_bag_rf, y_tr)
print(f"{N_TREES_SCRATCH} trees, max_features {max_feat}: each tree saw {in_bag_rf.mean():.4f} of the distinct training rows")
print(f"single tree (tree 0) val accuracy {np.mean(np.argmax(P_va_rf[0], 1) == y_va):.4f}")
print(f"forest val accuracy: average of probabilities {np.mean(soft == y_va):.4f}   majority vote {np.mean(hard == y_va):.4f}")
print(f"out-of-bag accuracy {oob_acc:.4f}   (training rows never out of bag: {no_vote})")

# %%
# Bagging = the same forest with every feature available at every split. Slower, so fewer trees.
section("from scratch: bagging versus random forest")
trees_bag, in_bag_bag = fit_forest(X_tr, y_tr, N_TREES_BAG_SCRATCH, d, np.random.default_rng(SEED))
P_va_bag = tree_probas(trees_bag, X_va)
P_va_rf_small = P_va_rf[:N_TREES_BAG_SCRATCH]


def mean_tree_accuracy(P: FloatArray) -> float:
    """Average accuracy of the individual trees. P (B, m, K)."""
    return float(np.mean(np.argmax(P, axis=2) == y_va[None, :]))


def mean_pairwise_agreement(P: FloatArray) -> float:
    """How often two different trees predict the same class, averaged over pairs. P (B, m, K)."""
    pred = np.argmax(P, axis=2)  # (B, m)
    same = (pred[:, None, :] == pred[None, :, :]).mean(axis=2)  # (B, B)
    b = len(pred)
    return float((same.sum() - b) / (b * (b - 1)))  # drop the diagonal (a tree always agrees with itself)


def mean_error_correlation(P: FloatArray) -> float:
    """Average correlation between two trees' 0/1 error indicators on the validation rows. P (B, m, K)."""
    wrong = (np.argmax(P, axis=2) != y_va[None, :]).astype(np.float64)  # (B, m)
    C = np.corrcoef(wrong)  # (B, B)
    b = len(C)
    return float((C.sum() - b) / (b * (b - 1)))


for label, P in [(f"bagging ({N_TREES_BAG_SCRATCH} trees, all {d} features)", P_va_bag),
                 (f"random forest (first {N_TREES_BAG_SCRATCH} trees, {max_feat} features)", P_va_rf_small)]:
    print(f"{label:<45}: mean single-tree accuracy {mean_tree_accuracy(P):.4f}   "
          f"tree-pair agreement {mean_pairwise_agreement(P):.4f}   error correlation {mean_error_correlation(P):.4f}   "
          f"ensemble accuracy {np.mean(np.argmax(P.mean(0), 1) == y_va):.4f}")

# %%
# With scikit-learn: RandomForestClassifier does the same thing (bootstrap + max_features="sqrt" +
# averaged probabilities + oob_score). Its random numbers differ from ours, so we compare the
# numbers with the spread across several random_state values instead of asking for equality.
section("with scikit-learn")
sk_val, sk_oob = [], []
for seed in range(N_SEEDS):
    rf = RandomForestClassifier(n_estimators=N_TREES_SCRATCH, oob_score=True, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    sk_val.append(rf.score(X_va, y_va))
    sk_oob.append(rf.oob_score_)
print(f"RandomForestClassifier({N_TREES_SCRATCH} trees), random_state 0..{N_SEEDS - 1}:")
print(f"  val accuracy {[round(v, 4) for v in sk_val]}   mean {np.mean(sk_val):.4f}")
print(f"  OOB accuracy {[round(v, 4) for v in sk_oob]}   mean {np.mean(sk_oob):.4f}")
print(f"ours: val {np.mean(soft == y_va):.4f}   OOB {oob_acc:.4f}")
lo, hi = min(sk_val) - 0.01, max(sk_val) + 0.01
assert lo <= np.mean(soft == y_va) <= hi, "from-scratch forest is outside scikit-learn's seed spread"
print("from-scratch val accuracy within scikit-learn's seed-to-seed spread (+-0.01): OK")
rf0 = RandomForestClassifier(n_estimators=N_TREES_SCRATCH, random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
P_sk = np.stack([t.predict_proba(X_va) for t in rf0.estimators_])  # (B, m, K)
np.testing.assert_allclose(P_sk.mean(axis=0), rf0.predict_proba(X_va))
print("sklearn's predict_proba = mean of its trees' predict_proba (soft voting): OK")

# %%
# Key plot: validation error against the number of trees, for one tree, bagging and a forest.
section("key plot: error versus number of trees")
curves = {}
for label, mf in [("bagging (max_features=None)", None), ("random forest (max_features='sqrt')", "sqrt")]:
    model = RandomForestClassifier(n_estimators=N_TREES_CURVE, max_features=mf, oob_score=True,
                                   random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
    P = np.stack([t.predict_proba(X_va) for t in model.estimators_])  # (B, m, K)
    running = np.cumsum(P, axis=0)  # (B, m, K) summed probabilities of the first b trees
    curves[label] = 1 - np.mean(np.argmax(running, axis=2) == y_va[None, :], axis=1)  # (B,)
    print(f"{label:<36}: val error at 1, 10, 50, 100, {N_TREES_CURVE} trees = "
          + ", ".join(f"{curves[label][b - 1]:.4f}" for b in [1, 10, 50, 100, N_TREES_CURVE])
          + f"   OOB error at {N_TREES_CURVE}: {1 - model.oob_score_:.4f}")
single = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
single_err = 1 - single.score(X_va, y_va)
print(f"single tree on all training rows: val error {single_err:.4f}")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
b_axis = np.arange(1, N_TREES_CURVE + 1)
ax.axhline(single_err, color="gray", linestyle=":", label="one fully grown tree")
for (label, err), style in zip(curves.items(), ["--", "-"]):
    ax.plot(b_axis, err, style, label=label)
ax.set_xscale("log")
ax.set_xlabel("number of trees (log scale)")
ax.set_ylabel("validation error rate")
ax.set_ylim(0, None)
ax.set_title("Averaging cuts the error; random feature subsets cut it further")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(IMG / "b3_2_error_vs_trees.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b3_2_error_vs_trees.png")

# %%
# Try this (not graded):
# 1. Set max_features to 1, 4, 16 and 32 in RandomForestClassifier. Where is the validation error lowest?
# 2. Give the forest min_samples_leaf=5. Does averaging deep trees or shallower trees work better here?
# 3. Pass max_samples=0.3 (each tree sees 30% of the rows). What happens to the OOB estimate and to accuracy?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
