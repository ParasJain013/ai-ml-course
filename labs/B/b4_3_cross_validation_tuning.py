# %% [markdown]
# # B4.3 Cross-validation and hyperparameter tuning — lab
# Lesson: lessons/B/b4_3_cross_validation_tuning.md
# Run: uv run python labs/B/b4_3_cross_validation_tuning.py   (CPU, defaults: ~2 min; downloads 0.4 MB once)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import io
import itertools
import sys
import tarfile
import time
import urllib.request
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Split = tuple[IntArray, IntArray]  # (train indices, test indices)
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "b4_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only raises the tuning budget (more trials, more folds).
GPU_SCALE = False
N_CUSTOMERS, ROWS_PER_CUSTOMER = 200, 10  # the synthetic grouped dataset
BUDGET = 60 if GPU_SCALE else 20  # cross-validated configurations per search method
N_FOLDS = 5 if GPU_SCALE else 3
N_ESTIMATORS = 200  # LightGBM trees per fit (fixed, to keep the comparison about the searched settings)
NESTED_ROWS = 1000  # rows used in the nested cross-validation demo
NESTED_TRIALS = 20
TEST_FRAC = 0.2
N_THREADS = 1  # deterministic LightGBM results on every machine

HOUSING_TGZ = DATA / "california_housing" / "housing.tgz"  # as in B3.3 (VERIFY link and licence, as of 2026-10)
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"

# %%
# Worked example by hand: the four splitters on 10 rows.
section("worked example by hand")


def kfold(n: int, k: int) -> list[Split]:
    """Contiguous k-fold without shuffling; the first n % k folds get one extra row."""
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    bounds = np.r_[0, np.cumsum(sizes)]
    idx = np.arange(n)
    return [(np.r_[idx[: bounds[f]], idx[bounds[f + 1]:]], idx[bounds[f]:bounds[f + 1]]) for f in range(k)]


def stratified_kfold(y: IntArray, k: int) -> list[Split]:
    """Stratified k-fold without shuffling, using scikit-learn's allocation rule.

    Number the classes in order of first appearance; sort the codes; fold f's share of class c is the count
    of c among sorted positions f, f+k, f+2k, ... Then each class's rows, in their original order, are dealt
    to folds 0, 0, ..., 1, 1, ... by those shares.
    """
    _, first_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    classes = np.argsort(np.argsort(first_idx))  # class code = rank of its first appearance
    y_enc = classes[y_inv]
    y_sorted = np.sort(y_enc)
    alloc = np.array([np.bincount(y_sorted[f::k], minlength=len(classes)) for f in range(k)])  # (k, n_classes)
    fold_of = np.empty(len(y), dtype=np.int64)
    for c in range(len(classes)):
        fold_of[y_enc == c] = np.repeat(np.arange(k), alloc[:, c])
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def group_kfold(groups: IntArray, k: int) -> list[Split]:
    """Every group's rows land in exactly one test fold: largest groups first, each to the emptiest fold."""
    uniq, counts = np.unique(groups, return_counts=True)
    load = np.zeros(k, dtype=np.int64)
    fold_of_group = {}
    for g in uniq[np.argsort(-counts, kind="stable")]:
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += counts[uniq == g][0]
    fold_of = np.array([fold_of_group[g] for g in groups])
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def time_series_split(n: int, k: int) -> list[Split]:
    """Expanding window: k test blocks of n // (k + 1) rows at the end; train on everything before each block."""
    test_size = n // (k + 1)
    starts = n - k * test_size + test_size * np.arange(k)
    return [(np.arange(s), np.arange(s, s + test_size)) for s in starts]


y10 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
g10 = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 5])
for name, splits in [("k-fold (k=5)", kfold(10, 5)), ("stratified (k=2)", stratified_kfold(y10, 2)),
                     ("group (k=3)", group_kfold(g10, 3)), ("time series (k=3)", time_series_split(10, 3))]:
    print(f"{name}: " + "  ".join(f"test {te.tolist()}" for _, te in splits))
for f, (tr, te) in enumerate(time_series_split(10, 3)):
    print(f"   time-series fold {f}: train {tr.tolist()} -> test {te.tolist()}")

# %%
# The from-scratch splitters agree with scikit-learn's.
section("from scratch versus scikit-learn splitters")
y_chk = np.random.default_rng(SEED).choice([0, 1, 2], size=103, p=[0.6, 0.3, 0.1])
g_chk = np.random.default_rng(SEED + 1).integers(0, 25, size=103)


def same_splits(mine: list[Split], theirs: Iterator[Split]) -> bool:
    theirs_l = list(theirs)
    return len(mine) == len(theirs_l) and all(
        np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(mine, theirs_l))


print(f"KFold(5): {same_splits(kfold(103, 5), KFold(5).split(np.zeros(103)))}")
print(f"StratifiedKFold(5): {same_splits(stratified_kfold(y_chk, 5), StratifiedKFold(5).split(np.zeros(103), y_chk))}")
print(f"TimeSeriesSplit(4): {same_splits(time_series_split(103, 4), TimeSeriesSplit(4).split(np.zeros(103)))}")
mine_g = group_kfold(g_chk, 5)
assert all(not set(g_chk[tr]) & set(g_chk[te]) for tr, te in mine_g)  # no group on both sides
assert np.array_equal(np.sort(np.concatenate([te for _, te in mine_g])), np.arange(103))  # every row tested once
sk_sizes = sorted(len(te) for _, te in GroupKFold(5).split(np.zeros(103), groups=g_chk))
print(f"group k-fold: no group on both sides, every row tested once: OK   fold sizes mine "
      f"{sorted(len(te) for _, te in mine_g)}  sklearn {sk_sizes}")
for f, (tr, te) in enumerate(stratified_kfold(y_chk, 5)):
    if f < 2:
        print(f"   stratified fold {f}: class shares in test {np.round(np.bincount(y_chk[te], minlength=3) / len(te), 3).tolist()} "
              f"(all rows {np.round(np.bincount(y_chk) / len(y_chk), 3).tolist()})")

# %%
# A group leak: 200 customers with 10 rows each. Each customer's rows share a signature (its own feature
# vector, plus a little noise) and a customer-level tendency to be positive that the features cannot see.
section("a group leak")
gen = np.random.default_rng(SEED)
signature = gen.normal(size=(N_CUSTOMERS, 5))  # who the customer is: identifiable, not predictive
tendency = gen.normal(0, 2.5, size=N_CUSTOMERS)  # customer-level effect on the label, unrelated to the signature
groups = np.repeat(np.arange(N_CUSTOMERS), ROWS_PER_CUSTOMER)  # (2000,)
row_feats = gen.normal(size=(len(groups), 3))  # per-row features that do predict the label
X_g = np.column_stack([signature[groups] + 0.1 * gen.normal(size=(len(groups), 5)), row_feats])  # (2000, 8)
logit = tendency[groups] + 1.0 * row_feats[:, 0] - 0.7 * row_feats[:, 1]
y_g = (gen.random(len(groups)) < 1 / (1 + np.exp(-logit))).astype(np.int64)
new_cust = gen.normal(size=(N_CUSTOMERS, 5)), gen.normal(0, 2.5, size=N_CUSTOMERS)  # 200 customers never seen
grp_new = np.repeat(np.arange(N_CUSTOMERS), ROWS_PER_CUSTOMER)
rows_new = gen.normal(size=(len(grp_new), 3))
X_new = np.column_stack([new_cust[0][grp_new] + 0.1 * gen.normal(size=(len(grp_new), 5)), rows_new])
y_new = (gen.random(len(grp_new)) < 1 / (1 + np.exp(-(new_cust[1][grp_new] + 1.0 * rows_new[:, 0] - 0.7 * rows_new[:, 1])))).astype(np.int64)
forest = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=N_THREADS)
leak_scores = {
    "random 5-fold (rows shuffled)": cross_val_score(forest, X_g, y_g, cv=KFold(5, shuffle=True, random_state=SEED)),
    "group 5-fold (by customer)": cross_val_score(forest, X_g, y_g, cv=GroupKFold(5), groups=groups),
}
leak_scores["new customers (truth)"] = np.array([forest.fit(X_g, y_g).score(X_new, y_new)])
for name, sc in leak_scores.items():
    print(f"{name:<31} accuracy {sc.mean():.4f}" + (f"  (folds {np.round(sc, 3).tolist()})" if len(sc) > 1 else ""))
rows_only = cross_val_score(forest, X_g[:, 5:], y_g, cv=GroupKFold(5), groups=groups).mean()
print(f"group 5-fold using only the 3 row features: {rows_only:.4f}")

# %%
# Tuning: LightGBM on California housing, three search methods with the same budget of cross-validated
# configurations, all on the development rows. The test rows are touched once, at the very end.
section("data for tuning")
raw = HOUSING_TGZ.read_bytes() if HOUSING_TGZ.exists() else None
if raw is None:
    HOUSING_TGZ.parent.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(HOUSING_URL, timeout=60).read()
    HOUSING_TGZ.write_bytes(raw)
    print(f"~ downloaded {len(raw):,} bytes from {HOUSING_URL}")
if hashlib.sha256(raw).hexdigest() != HOUSING_SHA256:
    raise ValueError(f"{HOUSING_TGZ} has the wrong sha256; delete it and rerun")
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    member = tar.extractfile("housing.csv")
    assert member is not None
    h = pd.read_csv(io.BytesIO(member.read()))
X_h = np.column_stack([h["median_income"], h["housing_median_age"], h["total_rooms"] / h["households"],
                       h["total_bedrooms"] / h["households"], h["population"], h["population"] / h["households"],
                       h["latitude"], h["longitude"]]).astype(np.float64)
y_h = (h["median_house_value"] / 100_000).to_numpy(dtype=np.float64)
perm = np.random.default_rng(SEED).permutation(len(y_h))
n_test = round(TEST_FRAC * len(y_h))
i_test, i_dev = perm[:n_test], perm[n_test:]  # development = everything we may look at while choosing
X_dev, y_dev = X_h[i_dev], y_h[i_dev]
print(f"development {X_dev.shape}  test {len(i_test)} rows (touched once, at the end)")
folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_dev))


def cv_rmse(params: dict[str, float], X: FloatArray, y: FloatArray, splits: list[Split]) -> float:
    """Mean validation RMSE of LightGBM with these settings over the given folds."""
    scores = []
    for tr, va in splits:
        m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_THREADS, random_state=SEED, deterministic=True,
                              force_row_wise=True, verbose=-1, **params).fit(X[tr], y[tr])
        scores.append(float(np.sqrt(np.mean((y[va] - m.predict(X[va])) ** 2))))
    return float(np.mean(scores))


def to_params(lr: float, leaves: int, min_child: int, colsample: float) -> dict[str, float]:
    return {"learning_rate": lr, "num_leaves": int(leaves), "min_child_samples": int(min_child), "colsample_bytree": colsample}


# %%
section("grid, random and Bayesian search")
history: dict[str, list[float]] = {}
best: dict[str, tuple[float, dict[str, float]]] = {}
t_search = time.perf_counter()
# Grid: 5 learning rates x 4 leaf counts = 20 configurations; the other two settings stay at defaults.
grid = [to_params(lr, nl, 20, 1.0) for lr, nl in itertools.product([0.01, 0.03, 0.1, 0.2, 0.3], [8, 31, 63, 127])]
history["grid"] = [cv_rmse(p, X_dev, y_dev, folds) for p in grid[:BUDGET]]
best["grid"] = (min(history["grid"]), grid[int(np.argmin(history["grid"]))])
# Random: 20 draws, learning rate and leaves on a log scale, all four settings varied.
rs = np.random.default_rng(SEED)
random_cfgs = [to_params(10 ** rs.uniform(-2, np.log10(0.3)), round(2 ** rs.uniform(3, 8)), rs.integers(5, 101),
                         rs.uniform(0.5, 1.0)) for _ in range(BUDGET)]
history["random"] = [cv_rmse(p, X_dev, y_dev, folds) for p in random_cfgs]
best["random"] = (min(history["random"]), random_cfgs[int(np.argmin(history["random"]))])


# Bayesian (Optuna's TPE): each new configuration is proposed from a model of the results so far.
def objective(trial: optuna.Trial) -> float:
    p = to_params(trial.suggest_float("learning_rate", 0.01, 0.3, log=True), trial.suggest_int("num_leaves", 8, 256, log=True),
                  trial.suggest_int("min_child_samples", 5, 100), trial.suggest_float("colsample_bytree", 0.5, 1.0))
    return cv_rmse(p, X_dev, y_dev, folds)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=BUDGET)
history["optuna (TPE)"] = [t.value for t in study.trials]
bp = study.best_params
best["optuna (TPE)"] = (study.best_value, to_params(bp["learning_rate"], bp["num_leaves"], bp["min_child_samples"],
                                                     bp["colsample_bytree"]))
print(f"~ {3 * BUDGET} configurations x {N_FOLDS} folds in {time.perf_counter() - t_search:.0f} s")
default_cv = cv_rmse({}, X_dev, y_dev, folds)
print(f"LightGBM defaults (lr 0.1, 31 leaves): CV RMSE {default_cv:.4f}")
for name, hist in history.items():
    score, params = best[name]
    first5 = min(hist[:5])
    print(f"{name:<13} best CV RMSE {score:.4f} (best of first 5: {first5:.4f})   "
          + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()))

# %%
# The final, single look at the test set: refit each winner on all development rows.
section("one look at the test set")
for name, (score, params) in best.items():
    m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_THREADS, random_state=SEED, deterministic=True,
                          force_row_wise=True, verbose=-1, **params).fit(X_dev, y_dev)
    test_rmse = float(np.sqrt(np.mean((y_h[i_test] - m.predict(X_h[i_test])) ** 2)))
    print(f"{name:<13} CV RMSE {score:.4f}   test RMSE {test_rmse:.4f}")

# %%
# Nested cross-validation on a 1,000-row sample. Non-nested: pick the best of 20 configurations by its
# 3-fold CV score and report that same score. Nested: inside each outer fold, pick the best configuration
# by an inner 3-fold CV on the outer training rows only, then score it on the outer test rows. Both
# report models trained on the same number of rows (two thirds of the sample).
section("nested cross-validation")
sub = np.random.default_rng(SEED).choice(len(y_dev), NESTED_ROWS, replace=False)
X_s, y_s = X_dev[sub], y_dev[sub]
nested_rng = np.random.default_rng(SEED)
cands = [to_params(10 ** nested_rng.uniform(-2, np.log10(0.3)), round(2 ** nested_rng.uniform(3, 8)),
                   nested_rng.integers(5, 101), nested_rng.uniform(0.5, 1.0)) for _ in range(NESTED_TRIALS)]
outer_folds = list(KFold(3, shuffle=True, random_state=SEED).split(X_s))
flat = [cv_rmse(p, X_s, y_s, outer_folds) for p in cands]
print(f"non-nested: best of {NESTED_TRIALS} configurations, scored on the folds that chose it: "
      f"{min(flat):.4f} (configuration {int(np.argmin(flat))})")
outer = []
for o, (tr_o, te_o) in enumerate(outer_folds):
    inner = list(KFold(3, shuffle=True, random_state=SEED).split(X_s[tr_o]))
    scores = [cv_rmse(p, X_s[tr_o], y_s[tr_o], inner) for p in cands]
    k = int(np.argmin(scores))
    m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_THREADS, random_state=SEED, deterministic=True,
                          force_row_wise=True, verbose=-1, **cands[k]).fit(X_s[tr_o], y_s[tr_o])
    outer.append(float(np.sqrt(np.mean((y_s[te_o] - m.predict(X_s[te_o])) ** 2))))
    print(f"outer fold {o}: inner search chose configuration {k:>2}   outer RMSE {outer[-1]:.4f}")
print(f"nested estimate (mean outer RMSE) {np.mean(outer):.4f}   non-nested {min(flat):.4f}   "
      f"optimism of the non-nested number {np.mean(outer) - min(flat):+.4f}")

# %%
# Key plot: the group leak.
section("key plot: group leak")
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=120)
names = list(leak_scores)
vals = [leak_scores[k].mean() for k in names]
bars = ax.barh(names[::-1], vals[::-1], color=["tab:green", "tab:blue", "tab:red"])
bars[2].set_hatch("//")
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1)
ax.set_xlabel("accuracy")
ax.set_title("Shuffled folds share customers and promise too much")
fig.tight_layout()
fig.savefig(IMG / "b4_3_group_leak.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
for (name, hist), style in zip(history.items(), ["-", "--", "-."]):
    ax.plot(np.arange(1, len(hist) + 1), np.minimum.accumulate(hist), style, label=name)
ax.set_xlabel("configurations evaluated")
ax.set_ylabel("best CV RMSE so far ($100,000s)")
ax.set_title("Best score found as the budget is spent")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "search_best_so_far.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved lessons/B/img/b4_3_group_leak.png and outputs/b4_3/search_best_so_far.png")

# %%
# Try this (not graded):
# 1. Score the leak dataset with StratifiedKFold(5, shuffle=True). Does stratification help? Why not?
# 2. Give random search and Optuna 60 configurations instead of 20. Does the gap between them grow or shrink?
# 3. Replace KFold by TimeSeriesSplit on a series with a trend, and compare it with shuffled k-fold.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
