# %% [markdown]
# # B2.3 K-nearest neighbours — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b2_3_knn_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_NOISE = 100
K_GRID = list(range(1, 32, 2))


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
P = np.array([[2.0, 3.0], [2.0, 0.0], [3.0, 5.0], [5.0, 6.0], [4.0, 4.0]])
cls = np.array(["A", "A", "B", "B", "B"])
q = np.array([0.0, 5.0])


def vote(d: FloatArray, k: int) -> str:
    """Majority class among the k smallest distances (k odd, two classes: no ties)."""
    near = cls[np.argsort(d)[:k]]
    return "A" if np.sum(near == "A") > k / 2 else "B"


d_euc = np.sqrt(((P - q) ** 2).sum(axis=1))
d_man = np.abs(P - q).sum(axis=1)
print(f"(a) Euclidean {np.round(d_euc, 4).tolist()}   Manhattan {d_man.tolist()}")
print(f"(b) Euclidean order {np.argsort(d_euc).tolist()} -> k=1,3,5: {[vote(d_euc, k) for k in (1, 3, 5)]}")
print(f"    Manhattan order {np.argsort(d_man).tolist()} -> k=1,3,5: {[vote(d_man, k) for k in (1, 3, 5)]}")
scale = np.array([1.0, 0.5])
d_sc = np.sqrt((((P - q) * scale) ** 2).sum(axis=1))
print(f"(c) feature 2 halved: distances {np.round(d_sc, 4).tolist()}   order {np.argsort(d_sc).tolist()} "
      f"-> k=3: {vote(d_sc, 3)}")
for dim in [4, 400]:
    mean, sd = dim / 6, np.sqrt(7 * dim / 180)
    print(f"(d) d = {dim}: mean {mean:.4f}  sd {sd:.4f}  sd/mean {sd / mean:.4f}")

# %%
section("data")
wine = load_wine()
X_all, y_all = wine.data.astype(np.float64), wine.target.astype(np.int64)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
noise = np.random.default_rng(SEED + 1).normal(size=(len(y_all), N_NOISE))  # (178, 100)
X_noisy = np.column_stack([X_all, noise])
print(f"train {len(i_tr)}  val {len(i_va)}  noisy feature count {X_noisy.shape[1]}")

# %%
section("cross-validated search in three settings")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
settings = {
    "13 real features": (X_all, [("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
    "13 real + 100 noise": (X_noisy, [("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
    "13 real + 100 noise, SelectKBest(13) in pipeline": (
        X_noisy,
        [("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=13)), ("knn", KNeighborsClassifier())],
    ),
}
for label, (X, steps) in settings.items():
    grid = GridSearchCV(
        Pipeline(steps),
        {"knn__n_neighbors": K_GRID, "knn__weights": ["uniform", "distance"]},
        cv=cv,
        scoring="accuracy",
    ).fit(X[i_tr], y_all[i_tr])
    best = grid.best_params_
    val_acc = np.mean(grid.predict(X[i_va]) == y_all[i_va])
    print(f"{label:<50} best k {best['knn__n_neighbors']:>2} {best['knn__weights']:<8}  "
          f"CV acc {grid.best_score_:.4f}  val acc {val_acc:.4f}")
    if "select" in grid.best_estimator_.named_steps:
        kept = grid.best_estimator_.named_steps["select"].get_support(indices=True)
        print(f"{'':<50} selector kept {int(np.sum(kept < 13))} of 13 real features and "
              f"{int(np.sum(kept >= 13))} noise columns")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
