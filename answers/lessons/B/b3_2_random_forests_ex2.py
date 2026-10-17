# %% [markdown]
# # B3.2 Random forests — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b3_2_random_forests_ex2.py   (CPU, ~40 s)

# %%
from __future__ import annotations

import sys
import time
import warnings
from math import comb
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_TREES = 300
MAX_FEATURES = [1, 2, 4, 8, 16, 32, 64]
OUT = Path(__file__).resolve().parents[3] / "outputs" / "b3_2"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
p_out = (1 - 1 / 4) ** 4
print(f"(a) n = 4: P(out of bag) = (3/4)^4 = {p_out:.4f}   expected distinct rows = 4 (1 - {p_out:.4f}) = {4 * (1 - p_out):.4f}")
maj5 = sum(comb(5, k) * 0.6**k * 0.4 ** (5 - k) for k in (3, 4, 5))
print(f"(b) 5 classifiers at 0.6: P(majority right) = {maj5:.5f}")
s2, rho = 4.0, 0.3
for b in (1, 10):
    print(f"(c) B = {b:>2}: {rho * s2 + (1 - rho) * s2 / b:.4f}")
print(f"(c) B -> inf: {rho * s2:.4f}   within 5% of the floor needs (1-rho) s2 / B <= 0.05 rho s2 -> "
      f"B >= {(1 - rho) / (0.05 * rho):.2f}, so {int(np.ceil((1 - rho) / (0.05 * rho)))} trees")

# %%
section("data")
digits = load_digits()
X_all, y_all = digits.data.astype(np.float64), digits.target.astype(np.int64)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
n = len(y_tr)
print(f"train {X_tr.shape}  val {X_va.shape}")


# %%
def oob_from_samples(model: RandomForestClassifier, X: FloatArray, y: IntArray) -> tuple[float, FloatArray]:
    """OOB accuracy from each tree's in-bag indices. X (n, d), y (n,) -> (accuracy, OOB probabilities (n, K))."""
    m, k = len(y), len(model.classes_)
    summed = np.zeros((m, k))
    counts = np.zeros(m)
    for tree, in_bag_rows in zip(model.estimators_, model.estimators_samples_):
        oob = np.ones(m, dtype=bool)
        oob[in_bag_rows] = False  # rows drawn at least once are in the bag
        summed[oob] += tree.predict_proba(X[oob])
        counts[oob] += 1
    if np.any(counts == 0):
        raise ValueError("some rows were never out of bag; use more trees")
    proba = summed / counts[:, None]
    return float(np.mean(model.classes_[np.argmax(proba, axis=1)] == y)), proba


def tree_stats(model: RandomForestClassifier) -> tuple[float, float]:
    """Mean single-tree validation accuracy and mean pairwise error correlation."""
    pred = np.stack([model.classes_[t.predict(X_va).astype(int)] for t in model.estimators_])  # (B, m)
    wrong = (pred != y_va[None, :]).astype(np.float64)
    C = np.corrcoef(wrong)
    b = len(C)
    return float(np.mean(pred == y_va[None, :])), float((C.sum() - b) / (b * (b - 1)))


section("OOB from estimators_samples_")
rf = RandomForestClassifier(n_estimators=N_TREES, oob_score=True, random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
mine, proba = oob_from_samples(rf, X_tr, y_tr)
np.testing.assert_allclose(proba, rf.oob_decision_function_, rtol=1e-12, atol=1e-12)
assert mine == rf.oob_score_
print(f"my OOB accuracy {mine:.6f}   sklearn oob_score_ {rf.oob_score_:.6f}   exact match: OK")
print(f"trees per row that are out of bag: mean {np.mean([N_TREES - sum(i in set(s) for s in rf.estimators_samples_) for i in range(5)]):.1f}"
      f" (first 5 rows; expect about 0.368 x {N_TREES} = {0.368 * N_TREES:.0f})")

# %%
section("max_features sweep")
rows_out = []
for mf in MAX_FEATURES:
    model = RandomForestClassifier(n_estimators=N_TREES, max_features=mf, oob_score=True, random_state=SEED,
                                   n_jobs=1).fit(X_tr, y_tr)
    acc_tree, err_corr = tree_stats(model)
    rows_out.append((mf, model.oob_score_, model.score(X_va, y_va), acc_tree, err_corr))
    print(f"max_features {mf:>2}: OOB acc {model.oob_score_:.4f}   val acc {rows_out[-1][2]:.4f}   "
          f"mean single-tree acc {acc_tree:.4f}   error correlation {err_corr:.4f}")
best = max(rows_out, key=lambda r: r[1])
print(f"chosen by OOB: max_features {best[0]} (OOB {best[1]:.4f})   its validation accuracy {best[2]:.4f}")
fig, ax1 = plt.subplots(figsize=(6, 4), dpi=120)
mfs = [r[0] for r in rows_out]
ax1.plot(mfs, [r[1] for r in rows_out], "o-", label="OOB accuracy")
ax1.plot(mfs, [r[2] for r in rows_out], "s--", label="validation accuracy")
ax1.plot(mfs, [r[3] for r in rows_out], "^:", label="mean single-tree accuracy")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("max_features (log scale)")
ax1.set_ylabel("accuracy")
ax2 = ax1.twinx()
ax2.plot(mfs, [r[4] for r in rows_out], "d-.", color="tab:red", label="error correlation (right axis)")
ax2.set_ylabel("mean error correlation between trees")
ax1.set_title("Fewer features per split: weaker, less correlated trees")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ex2_max_features.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b3_2/ex2_max_features.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
