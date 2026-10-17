# %% [markdown]
# # B3.1 Decision trees — lab
# Lesson: lessons/B/b3_1_decision_trees.md
# Run: uv run python labs/B/b3_1_decision_trees.py   (CPU, defaults: ~10 s)
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
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text

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
OUT = ROOT / "outputs" / "b3_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU (trees run on the CPU); GPU_SCALE only widens the depth sweep.
GPU_SCALE = False
MAX_DEPTH_SWEEP = 20 if GPU_SCALE else 10  # depths 1..MAX_DEPTH_SWEEP in the overfitting curve
N_BOOT = 5  # bootstrap resamples in the instability demo
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: the Wisconsin breast cancer set (B2.1). All 30 features, unscaled: a tree only compares a
# feature with a threshold, so the units do not matter. Malignant is the positive class (y = 1).
section("data")
bc = load_breast_cancer()
NAMES = [str(f) for f in bc.feature_names]
X_all = bc.data.astype(np.float64)  # (569, 30)
y_all = (bc.target == 0).astype(np.int64)  # (569,) 1 = malignant


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
print(f"malignant share: train {y_tr.mean():.4f}  val {y_va.mean():.4f}")
print(f"feature ranges differ wildly (no scaling needed): 'mean area' {X_tr[:, 3].min():.1f}..{X_tr[:, 3].max():.1f}"
      f"   'mean smoothness' {X_tr[:, 4].min():.4f}..{X_tr[:, 4].max():.4f}")

# %%
# Impurity measures and the split search. Everything works on class counts.
def impurity(counts: FloatArray, criterion: str) -> FloatArray:
    """Gini or entropy (bits) of each row of class counts. counts (m, K) -> (m,)."""
    totals = counts.sum(axis=1, keepdims=True)  # (m, 1)
    p = counts / np.maximum(totals, 1)  # (m, K) class proportions; empty rows give all zeros
    if criterion == "gini":
        return 1.0 - (p**2).sum(axis=1)
    if criterion == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, -p * np.log2(p), 0.0)  # 0 log 0 = 0
        return terms.sum(axis=1)
    raise ValueError(f"unknown criterion {criterion!r}")


def split_scores(x: FloatArray, Y: FloatArray, criterion: str, min_leaf: int) -> tuple[FloatArray, FloatArray]:
    """Weighted child impurity for every threshold on one feature. x (m,), Y one-hot (m, K) -> (m - 1,) twice.

    Entry i describes the cut between the i-th and (i+1)-th smallest values; invalid cuts are +inf.
    """
    order = np.argsort(x, kind="stable")
    xs, Ys = x[order], Y[order]
    m = len(xs)
    left = np.cumsum(Ys, axis=0)[:-1]  # (m - 1, K) class counts left of each cut
    right = Ys.sum(axis=0) - left  # (m - 1, K)
    n_left = np.arange(1, m)  # (m - 1,)
    score = (n_left * impurity(left, criterion) + (m - n_left) * impurity(right, criterion)) / m
    valid = (xs[:-1] < xs[1:]) & (n_left >= min_leaf) & (m - n_left >= min_leaf)  # no cut between equal values
    thresholds = (xs[:-1] + xs[1:]) / 2  # midpoints, as CART and scikit-learn use
    return np.where(valid, score, np.inf), thresholds


def best_split(X: FloatArray, Y: FloatArray, criterion: str, min_leaf: int) -> tuple[int, float, float]:
    """Exhaustive search over features and thresholds. X (m, d), Y (m, K) -> (feature, threshold, child impurity)."""
    best = (-1, 0.0, np.inf)
    for j in range(X.shape[1]):
        scores, thresholds = split_scores(X[:, j], Y, criterion, min_leaf)
        i = int(np.argmin(scores))
        if scores[i] < best[2]:  # strict: on a tie the lower feature index wins
            best = (j, float(thresholds[i]), float(scores[i]))
    return best


# %%
# Worked example by hand: six points, two features, four negatives and two positives.
section("worked example by hand")
X_toy = np.array([[1, 3], [2, 4], [3, 5], [4, 2], [5, 1], [6, 6]], dtype=np.float64)  # (6, 2)
y_toy = np.array([0, 0, 0, 1, 0, 1])
Y_toy = np.eye(2)[y_toy]  # (6, 2) one-hot
root_counts = Y_toy.sum(0, keepdims=True)
print(f"root counts (neg, pos) {Y_toy.sum(0).astype(int).tolist()}   Gini {impurity(root_counts, 'gini')[0]:.4f}"
      f"   entropy {impurity(root_counts, 'entropy')[0]:.4f} bits")
for crit in ["gini", "entropy"]:
    parent = impurity(root_counts, crit)[0]
    for j in range(2):
        scores, thr = split_scores(X_toy[:, j], Y_toy, crit, 1)
        cells = "  ".join(f"t={t:.1f}: gain {parent - s:.4f}" for s, t in zip(scores, thr) if np.isfinite(s))
        print(f"{crit:<7} x{j + 1}: {cells}")
j0, t0, _ = best_split(X_toy, Y_toy, "gini", 1)
right = X_toy[:, j0] > t0
j1, t1, _ = best_split(X_toy[right], Y_toy[right], "gini", 1)
print(f"root split: x{j0 + 1} <= {t0}   right child {y_toy[right].tolist()} splits on x{j1 + 1} <= {t1}")
toy_tree = DecisionTreeClassifier(random_state=SEED).fit(X_toy, y_toy)
print("scikit-learn's tree on the same six points:")
print(export_text(toy_tree, feature_names=["x1", "x2"]).rstrip())

# %%
# From scratch: grow the tree recursively, then predict by walking it.
section("from scratch: grow and predict")


def grow(X: FloatArray, Y: FloatArray, depth: int, max_depth: int | None, min_leaf: int, criterion: str) -> Node:
    """Grow a CART classification tree. X (m, d), Y one-hot (m, K) -> nested dict."""
    counts = Y.sum(axis=0)  # (K,)
    node: Node = {"counts": counts, "n": len(Y), "depth": depth}
    pure = np.count_nonzero(counts) == 1
    if pure or (max_depth is not None and depth >= max_depth) or len(Y) < 2 * min_leaf:
        return node
    j, t, child_impurity = best_split(X, Y, criterion, min_leaf)
    if j < 0:  # every feature is constant here: nothing to split on
        return node
    go_left = X[:, j] <= t
    node.update(
        feature=j,
        threshold=t,
        gain=float(impurity(counts[None, :], criterion)[0] - child_impurity),
        left=grow(X[go_left], Y[go_left], depth + 1, max_depth, min_leaf, criterion),
        right=grow(X[~go_left], Y[~go_left], depth + 1, max_depth, min_leaf, criterion),
    )
    return node


def fit_tree(X: FloatArray, y: IntArray, n_classes: int, max_depth: int | None = None,
             min_leaf: int = 1, criterion: str = "gini") -> Node:
    """Fit a tree on labels 0..n_classes-1. X (n, d), y (n,) -> root node."""
    return grow(X, np.eye(n_classes)[y], 0, max_depth, min_leaf, criterion)


def predict_proba_tree(node: Node, X: FloatArray) -> FloatArray:
    """Class proportions of the leaf each row lands in. X (m, d) -> (m, K)."""
    if "feature" not in node:
        return np.tile(node["counts"] / node["n"], (len(X), 1))
    go_left = X[:, node["feature"]] <= node["threshold"]
    out = np.empty((len(X), len(node["counts"])))
    out[go_left] = predict_proba_tree(node["left"], X[go_left])
    out[~go_left] = predict_proba_tree(node["right"], X[~go_left])
    return out


def predict_tree(node: Node, X: FloatArray) -> IntArray:
    """Majority class of each row's leaf. X (m, d) -> (m,)."""
    return np.argmax(predict_proba_tree(node, X), axis=1)


def tree_stats(node: Node) -> tuple[int, int]:
    """(number of leaves, depth) of a tree."""
    if "feature" not in node:
        return 1, node["depth"]
    (l1, d1), (l2, d2) = tree_stats(node["left"]), tree_stats(node["right"])
    return l1 + l2, max(d1, d2)


def show_tree(node: Node, names: list[str], indent: str = "") -> None:
    """Print a tree in the same layout as sklearn.tree.export_text."""
    if "feature" not in node:
        counts = node["counts"].astype(int).tolist()
        print(f"{indent}|--- class: {int(np.argmax(node['counts']))}   counts {counts}")
        return
    name, t = names[node["feature"]], node["threshold"]
    print(f"{indent}|--- {name} <= {t:.2f}")
    show_tree(node["left"], names, indent + "|   ")
    print(f"{indent}|--- {name} >  {t:.2f}")
    show_tree(node["right"], names, indent + "|   ")


tree2 = fit_tree(X_tr, y_tr, 2, max_depth=2)
root_gain = tree2["gain"]
print(f"root: {NAMES[tree2['feature']]} <= {tree2['threshold']:.4f}   Gini {impurity(tree2['counts'][None], 'gini')[0]:.4f}"
      f" -> gain {root_gain:.4f}")
show_tree(tree2, NAMES)
print(f"depth-2 tree: train accuracy {np.mean(predict_tree(tree2, X_tr) == y_tr):.4f}"
      f"   val accuracy {np.mean(predict_tree(tree2, X_va) == y_va):.4f}")
full = fit_tree(X_tr, y_tr, 2)
leaves, depth = tree_stats(full)
print(f"fully grown tree: {leaves} leaves, depth {depth}   train accuracy {np.mean(predict_tree(full, X_tr) == y_tr):.4f}"
      f"   val accuracy {np.mean(predict_tree(full, X_va) == y_va):.4f}")

# %%
section("from scratch: gini versus entropy, and min_samples_leaf")
for crit in ["gini", "entropy"]:
    for depth_limit, leaf in [(3, 1), (None, 1), (None, 5), (None, 20)]:
        t = fit_tree(X_tr, y_tr, 2, max_depth=depth_limit, min_leaf=leaf, criterion=crit)
        lv, dp = tree_stats(t)
        print(f"{crit:<7} max_depth {str(depth_limit):>4}  min_leaf {leaf:>2}: {lv:>2} leaves, depth {dp:>2}   "
              f"train {np.mean(predict_tree(t, X_tr) == y_tr):.4f}  val {np.mean(predict_tree(t, X_va) == y_va):.4f}")

# %%
# With scikit-learn: the same algorithm (CART). Two checks. (1) At every internal node of our tree,
# a scikit-learn stump fitted on that node's rows finds the same impurity decrease: our search finds
# the best split everywhere. (2) The whole trees: same size, and how often they agree. They can
# differ where two splits tie: scikit-learn tries the features in a random order (random_state),
# while our search keeps the lower feature index.
section("with scikit-learn")


def nodes_with_rows(node: Node, X: FloatArray, rows: IntArray) -> list[tuple[Node, IntArray]]:
    """Every internal node with the training row indices that reach it."""
    if "feature" not in node:
        return []
    go_left = X[rows, node["feature"]] <= node["threshold"]
    return [(node, rows)] + nodes_with_rows(node["left"], X, rows[go_left]) + nodes_with_rows(
        node["right"], X, rows[~go_left]
    )


def stump_gain(X: FloatArray, y: IntArray, criterion: str) -> float:
    """Impurity decrease of scikit-learn's best single split. X (m, d), y (m,) -> scalar."""
    t = DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=SEED).fit(X, y).tree_
    n_l, n_r = t.n_node_samples[1], t.n_node_samples[2]
    return float(t.impurity[0] - (n_l * t.impurity[1] + n_r * t.impurity[2]) / (n_l + n_r))


for crit in ["gini", "entropy"]:
    for depth_limit in [1, 2, 3, 4, None]:
        mine = fit_tree(X_tr, y_tr, 2, max_depth=depth_limit, criterion=crit)
        checked = nodes_with_rows(mine, X_tr, np.arange(n))
        for node, rows in checked:
            np.testing.assert_allclose(node["gain"], stump_gain(X_tr[rows], y_tr[rows], crit), atol=1e-9)
        sk = DecisionTreeClassifier(criterion=crit, max_depth=depth_limit, random_state=SEED).fit(X_tr, y_tr)
        same_train = np.allclose(sk.predict_proba(X_tr), predict_proba_tree(mine, X_tr))
        agree_va = np.mean(sk.predict(X_va) == predict_tree(mine, X_va))
        print(f"{crit:<7} max_depth {str(depth_limit):>4}: best split at all {len(checked):>2} nodes OK   "
              f"leaves mine {tree_stats(mine)[0]:>2} sklearn {sk.get_n_leaves():>2}   "
              f"same training leaves {str(same_train):<5}  val predictions agree {agree_va:.4f}")
print("every split matches the best split DecisionTreeClassifier finds on the same rows: OK")
sk1 = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X_tr, y_tr)
print(f"root threshold: sklearn {sk1.tree_.threshold[0]:.6f}  mine {tree2['threshold']:.6f}"
      f"  (|diff| {abs(sk1.tree_.threshold[0] - tree2['threshold']):.0e}: sklearn stores X as float32)")
left_rows = X_tr[:, tree2["feature"]] <= tree2["threshold"]
for name, t in [("worst radius", 17.48), ("worst area", 949.85)]:
    goes = X_tr[left_rows, NAMES.index(name)] <= t
    print(f"tie in the left child: {name:<12} <= {t:<7} sends {int(goes.sum())} rows left, {int((~goes).sum())} right"
          f"   counts {np.bincount(y_tr[left_rows][goes], minlength=2).tolist()} / "
          f"{np.bincount(y_tr[left_rows][~goes], minlength=2).tolist()}")
picks = []
for seed in range(5):
    sk = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(X_tr, y_tr)
    picks.append(NAMES[sk.tree_.feature[sk.tree_.children_left[0]]])
print(f"sklearn's pick there for random_state 0..4: {picks}")

# %%
# Instability: refit on bootstrap resamples of the training rows and look at the top of the tree.
section("instability: the same tree on bootstrap resamples")
boot_rng = np.random.default_rng(SEED)
for b in range(N_BOOT):
    idx = boot_rng.integers(0, n, n)  # n rows drawn with replacement
    t = fit_tree(X_tr[idx], y_tr[idx], 2)
    left_name = NAMES[t["left"]["feature"]] if "feature" in t["left"] else "leaf"
    print(f"resample {b}: root {NAMES[t['feature']]:<22} <= {t['threshold']:>8.4f}   next split on the left: "
          f"{left_name:<22} val accuracy {np.mean(predict_tree(t, X_va) == y_va):.4f}")

# %%
# Key plot: training and validation accuracy and log-loss as the depth limit grows. A leaf's
# probability is its class share, so a pure leaf says 0 or 1; log-loss shows how costly that is.
section("key plot: depth sweep")


def log_loss_clipped(y: IntArray, p1: FloatArray) -> float:
    """Mean binary log-loss with p clipped to [1e-15, 1 - 1e-15], as sklearn.metrics.log_loss does."""
    p1 = np.clip(p1, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1)))


depths = list(range(1, MAX_DEPTH_SWEEP + 1))
curves: dict[str, list[float]] = {"acc_tr": [], "acc_va": [], "ll_tr": [], "ll_va": []}
for dl in depths:
    t = fit_tree(X_tr, y_tr, 2, max_depth=dl)
    p_tr, p_va = predict_proba_tree(t, X_tr)[:, 1], predict_proba_tree(t, X_va)[:, 1]
    curves["acc_tr"].append(float(np.mean((p_tr > 0.5) == y_tr)))
    curves["acc_va"].append(float(np.mean((p_va > 0.5) == y_va)))
    curves["ll_tr"].append(log_loss_clipped(y_tr, p_tr))
    curves["ll_va"].append(log_loss_clipped(y_va, p_va))
    print(f"max_depth {dl:>2}: accuracy train {curves['acc_tr'][-1]:.4f}  val {curves['acc_va'][-1]:.4f}   "
          f"log-loss train {curves['ll_tr'][-1]:.4f}  val {curves['ll_va'][-1]:.4f}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=120)
ax1.plot(depths, curves["acc_tr"], "o-", label="training")
ax1.plot(depths, curves["acc_va"], "s--", label="validation")
ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy: training reaches 1.0, validation stalls")
ax2.plot(depths, curves["ll_tr"], "o-", label="training")
ax2.plot(depths, curves["ll_va"], "s--", label="validation")
ax2.set_ylabel("log-loss (nats)")
ax2.set_title("Log-loss: pure leaves make validation loss explode")
for ax in (ax1, ax2):
    ax.set_xlabel("max_depth (tree depth limit)")
    ax.set_xticks(depths)
    ax.legend(loc="best")
fig.tight_layout()
fig.savefig(IMG / "b3_1_depth_overfitting.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b3_1_depth_overfitting.png")

# %%
# Try this (not graded):
# 1. Replace the criterion with misclassification error, 1 - max(p), and grow a depth-3 tree.
#    How often does the search find no improving split at all?
# 2. Multiply 'mean area' by 1000 or take its log before fitting. Does any split change? Why not?
# 3. Fit on the first 100 training rows only and print the depth-2 tree. How much of it survives?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
