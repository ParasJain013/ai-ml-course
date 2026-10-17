# %% [markdown]
# # B3.1 Decision trees — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b3_1_decision_trees_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Node = dict[str, Any]
SEED = 0
T0 = time.perf_counter()
MAX_DEPTH = 12
OUT = Path(__file__).resolve().parents[3] / "outputs" / "b3_1"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
y8 = np.array([0, 0, 1, 0, 1, 1, 1, 1])


def gini(labels: IntArray) -> float:
    p = np.bincount(labels, minlength=2) / len(labels)
    return float(1 - (p**2).sum())


def entropy_bits(labels: IntArray) -> float:
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def err(labels: IntArray) -> float:
    return float(1 - np.bincount(labels, minlength=2).max() / len(labels))


def weighted(f: Any, labels: IntArray, k: int) -> float:
    """Weighted child impurity for the cut after the first k rows."""
    return (k * f(labels[:k]) + (len(labels) - k) * f(labels[k:])) / len(labels)


print(f"(a) root Gini {gini(y8):.4f}   entropy {entropy_bits(y8):.4f} bits")
print("(b) Gini gain by cut: " + "  ".join(f"{k + 0.5}: {gini(y8) - weighted(gini, y8, k):.4f}" for k in range(1, 8)))
for k in (2, 4):
    print(f"(c,d) cut {k + 0.5}: error {err(y8):.4f} -> {weighted(err, y8, k):.4f}   "
          f"Gini gain {gini(y8) - weighted(gini, y8, k):.4f}   entropy gain {entropy_bits(y8) - weighted(entropy_bits, y8, k):.4f}")
t8 = np.array([2.0, 4, 3, 10, 9, 14, 13, 15])
var_root = t8.var()
cuts = {k + 0.5: (k * t8[:k].var() + (8 - k) * t8[k:].var()) / 8 for k in range(1, 8)}
print(f"(e) root variance {var_root:.4f}; weighted child variance by cut: "
      + "  ".join(f"{c}: {v:.4f}" for c, v in cuts.items()))
best_c = min(cuts, key=cuts.get)
k_best = int(best_c - 0.5)
print(f"    best cut {best_c}: leaves predict {t8[:k_best].mean():.4f} and {t8[k_best:].mean():.4f}")
stump = DecisionTreeRegressor(max_depth=1).fit(np.arange(1, 9).reshape(-1, 1), t8)
print(f"    DecisionTreeRegressor stump: threshold {stump.tree_.threshold[0]}  leaves {stump.tree_.value[1:, 0, 0].tolist()}")

# %%
section("data")
diab = load_diabetes(as_frame=True, scaled=False)
names = list(diab.feature_names)
X_all = diab.data.to_numpy(dtype=np.float64)
y_all = diab.target.to_numpy(dtype=np.float64)
gen = np.random.default_rng(SEED)  # the B1.5 split (train_val_test_split, not stratified)
rows = gen.permutation(len(y_all))
n_te, n_va = round(0.2 * len(y_all)), round(0.2 * len(y_all))
i_tr = gen.permutation(rows[n_te + n_va:])
i_va = gen.permutation(rows[n_te:n_te + n_va])
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
print(f"train {X_tr.shape}  val {X_va.shape}")


# %%
def split_scores_reg(x: FloatArray, y: FloatArray, min_leaf: int) -> tuple[FloatArray, FloatArray]:
    """Weighted child variance for every cut on one feature. x (m,), y (m,) -> (m - 1,), (m - 1,)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    m = len(xs)
    n_left = np.arange(1, m)
    s1, s2 = np.cumsum(ys)[:-1], np.cumsum(ys**2)[:-1]  # running sums of y and y^2
    t1, t2 = ys.sum() - s1, (ys**2).sum() - s2
    sse_left = s2 - s1**2 / n_left  # sum of squared deviations = sum y^2 - (sum y)^2 / count
    sse_right = t2 - t1**2 / (m - n_left)
    score = (sse_left + sse_right) / m  # weighted variance = total SSE / m
    valid = (xs[:-1] < xs[1:]) & (n_left >= min_leaf) & (m - n_left >= min_leaf)
    return np.where(valid, score, np.inf), (xs[:-1] + xs[1:]) / 2


def grow_reg(X: FloatArray, y: FloatArray, depth: int, max_depth: int | None, min_leaf: int) -> Node:
    """Grow a regression tree. X (m, d), y (m,) -> nested dict."""
    node: Node = {"value": float(y.mean()), "n": len(y), "var": float(y.var())}
    if node["var"] <= 0 or (max_depth is not None and depth >= max_depth) or len(y) < 2 * min_leaf:
        return node
    best = (-1, 0.0, np.inf)
    for j in range(X.shape[1]):
        scores, thr = split_scores_reg(X[:, j], y, min_leaf)
        i = int(np.argmin(scores))
        if scores[i] < best[2]:
            best = (j, float(thr[i]), float(scores[i]))
    j, t, score = best
    if j < 0:
        return node
    go_left = X[:, j] <= t
    node.update(feature=j, threshold=t, gain=node["var"] - score,
                left=grow_reg(X[go_left], y[go_left], depth + 1, max_depth, min_leaf),
                right=grow_reg(X[~go_left], y[~go_left], depth + 1, max_depth, min_leaf))
    return node


def predict_reg(node: Node, X: FloatArray) -> FloatArray:
    """Leaf mean for each row. X (m, d) -> (m,)."""
    if "feature" not in node:
        return np.full(len(X), node["value"])
    go_left = X[:, node["feature"]] <= node["threshold"]
    out = np.empty(len(X))
    out[go_left] = predict_reg(node["left"], X[go_left])
    out[~go_left] = predict_reg(node["right"], X[~go_left])
    return out


def internal(node: Node, X: FloatArray, rows: IntArray) -> list[tuple[Node, IntArray]]:
    if "feature" not in node:
        return []
    go_left = X[rows, node["feature"]] <= node["threshold"]
    return [(node, rows)] + internal(node["left"], X, rows[go_left]) + internal(node["right"], X, rows[~go_left])


def rmse(y: FloatArray, y_hat: FloatArray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


section("check against DecisionTreeRegressor")
for depth in [1, 2, 3, 4]:
    mine = grow_reg(X_tr, y_tr, 0, depth, 1)
    nodes = internal(mine, X_tr, np.arange(len(y_tr)))
    for node, rows in nodes:
        t = DecisionTreeRegressor(max_depth=1, random_state=SEED).fit(X_tr[rows], y_tr[rows]).tree_
        n_l, n_r = t.n_node_samples[1], t.n_node_samples[2]
        sk_gain = t.impurity[0] - (n_l * t.impurity[1] + n_r * t.impurity[2]) / (n_l + n_r)
        np.testing.assert_allclose(node["gain"], sk_gain, rtol=1e-9)
    sk = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(X_tr, y_tr)
    np.testing.assert_allclose(predict_reg(mine, X_tr), sk.predict(X_tr), rtol=1e-12)
    print(f"depth {depth}: best split at all {len(nodes):>2} nodes OK, training predictions identical OK   "
          f"val RMSE mine {rmse(y_va, predict_reg(mine, X_va)):.2f}  sklearn {rmse(y_va, sk.predict(X_va)):.2f}")

# %%
section("depth sweep")
depths = list(range(1, MAX_DEPTH + 1))
tr_rmse, va_rmse = [], []
for depth in depths:
    t = grow_reg(X_tr, y_tr, 0, depth, 1)
    tr_rmse.append(rmse(y_tr, predict_reg(t, X_tr)))
    va_rmse.append(rmse(y_va, predict_reg(t, X_va)))
    print(f"depth {depth:>2}: train RMSE {tr_rmse[-1]:6.2f}   val RMSE {va_rmse[-1]:6.2f}")
for leaf in [5, 10, 20]:
    t = grow_reg(X_tr, y_tr, 0, None, leaf)
    print(f"no depth limit, min_leaf {leaf:>2}: train RMSE {rmse(y_tr, predict_reg(t, X_tr)):6.2f}   "
          f"val RMSE {rmse(y_va, predict_reg(t, X_va)):6.2f}")
print("B1.5 linear regression on the same validation rows: RMSE 55.98")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.plot(depths, tr_rmse, "o-", label="training")
ax.plot(depths, va_rmse, "s--", label="validation")
ax.axhline(55.98, color="gray", linestyle=":", label="B1.5 linear regression (val)")
ax.set_xlabel("max_depth")
ax.set_ylabel("RMSE (disease progression units)")
ax.set_title("Regression tree on diabetes: validation RMSE is best at depth 2")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ex2_depth_rmse.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b3_1/ex2_depth_rmse.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
