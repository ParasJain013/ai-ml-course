# %% [markdown]
# # B1.1 The ML workflow — lab
# Lesson: lessons/B/b1_1_ml_workflow.md
# Run: uv run python labs/B/b1_1_ml_workflow.py   (CPU, defaults: ~5 s)
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
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

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
OUT = ROOT / "outputs" / "b1_1"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only makes the test-reuse simulation bigger.
GPU_SCALE = False
VAL_FRAC, TEST_FRAC = 0.2, 0.2  # 60 / 20 / 20 split
N_REUSE_TEST = 100  # size of the test set that is reused in the simulation
N_REUSE_MODELS = 1_000  # coin-flip "models" tried on that one test set
N_REUSE_RUNS = 2_000 if GPU_SCALE else 300  # repetitions of the whole simulation

# %%
# Data: two datasets bundled with scikit-learn (no download). The syllabus names California housing
# for the regression baseline; the diabetes set is used instead so the lab runs offline and its
# printed numbers are the same on every machine. California housing is in the "Try this" cell.
section("data")
diab = load_diabetes(as_frame=True, scaled=False)  # raw units: age in years, bmi in kg/m^2, ...
X_reg = diab.data.to_numpy(dtype=np.float64)  # (442, 10)
y_reg = diab.target.to_numpy(dtype=np.float64)  # (442,) disease progression one year later
cancer = load_breast_cancer(as_frame=True)
X_clf = cancer.data.to_numpy(dtype=np.float64)  # (569, 30)
y_clf = cancer.target.to_numpy()  # (569,) 0 = malignant, 1 = benign
print(f"diabetes: X {X_reg.shape}  y {y_reg.shape}  target mean {y_reg.mean():.2f}  "
      f"sd {y_reg.std(ddof=1):.2f}  range [{y_reg.min():.0f}, {y_reg.max():.0f}]")
print(f"breast cancer: X {X_clf.shape}  y {y_clf.shape}  share benign (1) {y_clf.mean():.4f}  "
      f"counts {np.bincount(y_clf).tolist()}")

# %%
# Worked example by hand: a 10-row split, the mean baseline, the majority baseline, a one-feature rule.
section("worked example by hand")
perm_hand = np.array([7, 2, 9, 4, 0, 5, 8, 1, 6, 3])  # a "shuffled" order, written out by hand
n_h = 10
n_test_h, n_val_h = round(0.2 * n_h), round(0.2 * n_h)
test_h, val_h, train_h = perm_hand[:n_test_h], perm_hand[n_test_h:n_test_h + n_val_h], perm_hand[n_test_h + n_val_h:]
print(f"test rows {test_h.tolist()}   val rows {val_h.tolist()}   train rows {sorted(train_h.tolist())}")
y_h = np.array([3.0, 5.0, 4.0, 8.0, 10.0, 6.0, 7.0, 4.0, 7.0, 5.0])  # one target per row 0..9
mean_tr = y_h[train_h].mean()
pred_val = np.full(len(val_h), mean_tr)
print(f"train targets {y_h[np.sort(train_h)].tolist()} -> mean baseline predicts {mean_tr:.4f}")
print(f"val targets {y_h[val_h].tolist()}   MSE {np.mean((y_h[val_h] - pred_val) ** 2):.4f}   "
      f"MAE {np.mean(np.abs(y_h[val_h] - pred_val)):.4f}")
lab_h = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0])  # binary labels for the same rows
feat_h = np.array([1.0, 2.0, 1.5, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0, 0.5])  # one numeric feature
maj = int(np.bincount(lab_h[train_h]).argmax())
print(f"train labels {lab_h[np.sort(train_h)].tolist()} -> majority class {maj}   "
      f"val labels {lab_h[val_h].tolist()} -> accuracy {np.mean(lab_h[val_h] == maj):.2f}")
print(f"train (feature, label) sorted by feature: "
      f"{sorted(zip(feat_h[train_h].tolist(), lab_h[train_h].tolist()))}")
for thr in (1.5, 2.5, 3.5, 5.0, 7.0):  # midpoints between neighbouring training values
    acc_tr = np.mean((feat_h[train_h] > thr).astype(int) == lab_h[train_h])
    print(f"rule 'predict 1 if feature > {thr}': train accuracy {acc_tr:.4f}")
best_thr = 1.5
print(f"chosen threshold {best_thr}: val features {feat_h[val_h].tolist()} -> "
      f"predictions {(feat_h[val_h] > best_thr).astype(int).tolist()}   "
      f"val accuracy {np.mean((feat_h[val_h] > best_thr).astype(int) == lab_h[val_h]):.2f}")

# %%
# From scratch: a seeded three-way split with a stratify option.
section("from scratch: train/validation/test split")


def train_val_test_split(
    n: int,
    val_frac: float,
    test_frac: float,
    gen: np.random.Generator,
    stratify: IntArray | None = None,
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("need 0 < val_frac, test_frac and val_frac + test_frac < 1")
    if stratify is None:
        groups = [np.arange(n)]  # one group: every row
    else:
        groups = [np.flatnonzero(stratify == c) for c in np.unique(stratify)]  # one group per class
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)  # shuffle within the group
        n_test = round(test_frac * len(rows))
        n_val = round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


def check_split(train: IntArray, val: IntArray, test: IntArray, n: int) -> None:
    """Assert the three index sets are disjoint and cover 0..n-1 exactly once."""
    everything = np.concatenate([train, val, test])
    assert len(everything) == n and len(np.unique(everything)) == n, "overlap or missing rows"


n_clf = len(y_clf)
tr, va, te = train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, rng)
check_split(tr, va, te, n_clf)
print(f"plain:      sizes {len(tr)} / {len(va)} / {len(te)}   share benign "
      f"{y_clf[tr].mean():.4f} / {y_clf[va].mean():.4f} / {y_clf[te].mean():.4f}")
tr_s, va_s, te_s = train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, rng, stratify=y_clf)
check_split(tr_s, va_s, te_s, n_clf)
print(f"stratified: sizes {len(tr_s)} / {len(va_s)} / {len(te_s)}   share benign "
      f"{y_clf[tr_s].mean():.4f} / {y_clf[va_s].mean():.4f} / {y_clf[te_s].mean():.4f}")
print("disjoint and complete: OK")

# Same seed, same split; a different seed, a different split.
a1, _, _ = train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
a2, _, _ = train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
a3, _, _ = train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED + 1))
print(f"same seed -> identical train rows: {np.array_equal(a1, a2)}   "
      f"different seed -> identical: {np.array_equal(a1, a3)}")

# How much does an unstratified test set's class mix wander? 1,000 random splits.
shares = np.array([y_clf[train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, rng)[2]].mean()
                   for _ in range(1_000)])
shares_s = np.array([y_clf[train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, rng, stratify=y_clf)[2]].mean()
                     for _ in range(1_000)])
print(f"share benign in the test set over 1000 splits: plain {shares.min():.4f}..{shares.max():.4f} "
      f"(sd {shares.std():.4f})   stratified {shares_s.min():.4f}..{shares_s.max():.4f} (sd {shares_s.std():.4f})")

# %%
# From scratch: the three baselines on both datasets, scored on the validation set only.
section("from scratch: baselines")
n_reg = len(y_reg)
r_tr, r_va, r_te = train_val_test_split(n_reg, VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
c_tr, c_va, c_te = train_val_test_split(n_clf, VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), stratify=y_clf)
print(f"diabetes split {len(r_tr)} / {len(r_va)} / {len(r_te)}   breast cancer split "
      f"{len(c_tr)} / {len(c_va)} / {len(c_te)}")


def mse(y: FloatArray, y_hat: FloatArray) -> float:
    """Mean squared error. y (n,), y_hat (n,) -> scalar."""
    return float(np.mean((y - y_hat) ** 2))


def mae(y: FloatArray, y_hat: FloatArray) -> float:
    """Mean absolute error. y (n,), y_hat (n,) -> scalar."""
    return float(np.mean(np.abs(y - y_hat)))


def r2(y: FloatArray, y_hat: FloatArray) -> float:
    """Share of the variance of y explained: 1 - SSE/SST. y (n,), y_hat (n,) -> scalar."""
    return 1.0 - float(np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


# Baseline 1: predict the training mean for everyone.
mean_pred = y_reg[r_tr].mean()
yv = y_reg[r_va]
base_mean = np.full(len(yv), mean_pred)
print(f"[diabetes] mean baseline predicts {mean_pred:.4f}: val MSE {mse(yv, base_mean):.2f}  "
      f"RMSE {np.sqrt(mse(yv, base_mean)):.2f}  MAE {mae(yv, base_mean):.2f}  R^2 {r2(yv, base_mean):.4f}")
# Median baseline: the constant that minimizes MAE (A5.1), not MSE.
med_pred = np.median(y_reg[r_tr])
print(f"[diabetes] median baseline predicts {med_pred:.4f}: val MSE {mse(yv, np.full(len(yv), med_pred)):.2f}  "
      f"MAE {mae(yv, np.full(len(yv), med_pred)):.2f}")


def fit_one_feature_bins(x: FloatArray, y: FloatArray, n_bins: int) -> tuple[FloatArray, FloatArray]:
    """Cut x at its training quantiles and learn the mean y per bin. x, y (n,) -> (edges (b-1,), means (b,))."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])  # (n_bins - 1,) inner cut points
    bins = np.searchsorted(edges, x, side="right")  # (n,) bin index 0..n_bins-1
    means = np.array([y[bins == b].mean() for b in range(n_bins)])  # (n_bins,)
    return edges, means


def predict_one_feature_bins(x: FloatArray, edges: FloatArray, means: FloatArray) -> FloatArray:
    """Look up the training mean of x's bin. x (m,) -> (m,)."""
    return means[np.searchsorted(edges, x, side="right")]


# Baseline 3 (regression): one feature, bmi, cut into 4 bins at its training quartiles.
j_bmi = list(diab.feature_names).index("bmi")
edges, means = fit_one_feature_bins(X_reg[r_tr, j_bmi], y_reg[r_tr], n_bins=4)
pred_bmi = predict_one_feature_bins(X_reg[r_va, j_bmi], edges, means)
print(f"[diabetes] bmi rule: cut points {show(edges)}   bin means {show(means)}")
print(f"[diabetes] bmi rule: val MSE {mse(yv, pred_bmi):.2f}  RMSE {np.sqrt(mse(yv, pred_bmi)):.2f}  "
      f"MAE {mae(yv, pred_bmi):.2f}  R^2 {r2(yv, pred_bmi):.4f}")

# Baseline 2: predict the majority class of the training set.
yc_tr, yc_va = y_clf[c_tr], y_clf[c_va]
majority = int(np.bincount(yc_tr).argmax())
print(f"[cancer] majority class in train = {majority} (benign)   val accuracy "
      f"{np.mean(yc_va == majority):.4f}   malignant cases caught: 0 of {int(np.sum(yc_va == 0))}")


def fit_threshold_rule(x: FloatArray, y: IntArray) -> tuple[float, int]:
    """Best single cut on one feature by training accuracy. x (n,), y (n,) -> (threshold, class above it)."""
    xs = np.unique(x)
    candidates = (xs[:-1] + xs[1:]) / 2  # midpoints between neighbouring distinct values
    best = (0.0, 0.0, 1)
    for thr in candidates:
        for above in (0, 1):
            pred = np.where(x > thr, above, 1 - above)
            acc = float(np.mean(pred == y))
            if acc > best[0]:
                best = (acc, float(thr), above)
    return best[1], best[2]


# Baseline 3 (classification): the best cut on one feature, "worst concave points", learned on train.
j_feat = list(cancer.feature_names).index("worst concave points")
thr, above = fit_threshold_rule(X_clf[c_tr, j_feat], yc_tr)
pred_rule_tr = np.where(X_clf[c_tr, j_feat] > thr, above, 1 - above)
pred_rule = np.where(X_clf[c_va, j_feat] > thr, above, 1 - above)
print(f"[cancer] rule: predict {above} if worst concave points > {thr:.4f} else {1 - above}   "
      f"train accuracy {np.mean(pred_rule_tr == yc_tr):.4f}   val accuracy {np.mean(pred_rule == yc_va):.4f}")
print(f"[cancer] rule: malignant cases caught on val: {int(np.sum((pred_rule == 0) & (yc_va == 0)))} of "
      f"{int(np.sum(yc_va == 0))}")

# %%
# From scratch: generalization. A 1-nearest-neighbour "memorizer" is perfect on its own training data.
section("from scratch: memorizing is not generalizing")


def one_nn_predict(X_train: FloatArray, y_train: FloatArray, X_query: FloatArray) -> FloatArray:
    """Copy the target of the closest training row. X_train (n, d), X_query (m, d) -> (m,)."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)  # put features on one scale (B1.3)
    A, Q = (X_train - mu) / sd, (X_query - mu) / sd
    d2 = (Q**2).sum(1)[:, None] - 2 * Q @ A.T + (A**2).sum(1)[None, :]  # (m, n) squared distances
    return y_train[np.argmin(d2, axis=1)]


for name, X_, y_, tr_, va_ in [("diabetes", X_reg, y_reg, r_tr, r_va), ("cancer", X_clf, y_clf.astype(float), c_tr, c_va)]:
    p_tr = one_nn_predict(X_[tr_], y_[tr_], X_[tr_])
    p_va = one_nn_predict(X_[tr_], y_[tr_], X_[va_])
    if name == "diabetes":
        print(f"[diabetes] 1-NN: train R^2 {r2(y_[tr_], p_tr):.4f}   val R^2 {r2(y_[va_], p_va):.4f}   "
              f"(mean baseline val R^2 {r2(yv, base_mean):.4f})")
    else:
        print(f"[cancer] 1-NN: train accuracy {np.mean(p_tr == y_[tr_]):.4f}   val accuracy "
              f"{np.mean(p_va == y_[va_]):.4f}   (majority baseline {np.mean(yc_va == majority):.4f})")

# %%
# From scratch: reusing one test set to pick among many models inflates the winner's score.
section("from scratch: reusing the test set")


def best_of_k_on_one_test_set(n_test: int, k: int, gen: np.random.Generator) -> FloatArray:
    """Best test accuracy among the first 1..k coin-flip models on one test set. -> (k,) running max."""
    y_test = gen.integers(0, 2, size=n_test)  # (n_test,) labels
    preds = gen.integers(0, 2, size=(k, n_test))  # (k, n_test) every model guesses at random
    acc = (preds == y_test).mean(axis=1)  # (k,) each model's test accuracy; true accuracy is 0.5
    return np.maximum.accumulate(acc)  # (k,) best seen after trying 1, 2, ..., k models


running = np.array([best_of_k_on_one_test_set(N_REUSE_TEST, N_REUSE_MODELS, rng) for _ in range(N_REUSE_RUNS)])
for k in (1, 10, 100, 1000):
    print(f"best of {k:>4} coin-flip models on one test set of {N_REUSE_TEST}: mean reported accuracy "
          f"{running[:, k - 1].mean():.4f}   (true accuracy of every model: 0.5)")
print(f"standard error of one accuracy on {N_REUSE_TEST} rows at p = 0.5: "
      f"sqrt(0.25 / {N_REUSE_TEST}) = {np.sqrt(0.25 / N_REUSE_TEST):.4f}")

# %%
# With scikit-learn: train_test_split twice, DummyRegressor / DummyClassifier, metrics.
section("with scikit-learn")
idx = np.arange(n_clf)
idx_trval, idx_test = train_test_split(idx, test_size=TEST_FRAC, stratify=y_clf, random_state=SEED)
# The second cut takes val_frac of the WHOLE data out of what is left: 0.2 / 0.8 = 0.25 of it.
idx_train, idx_val = train_test_split(idx_trval, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                      stratify=y_clf[idx_trval], random_state=SEED)
check_split(idx_train, idx_val, idx_test, n_clf)
print(f"sklearn sizes {len(idx_train)} / {len(idx_val)} / {len(idx_test)}   "
      f"mine {len(c_tr)} / {len(c_va)} / {len(c_te)}")
print(f"sklearn share benign {y_clf[idx_train].mean():.4f} / {y_clf[idx_val].mean():.4f} / "
      f"{y_clf[idx_test].mean():.4f}")
# Mine rounds each class separately (212 * 0.2 = 42.4 -> 42, 357 * 0.2 = 71.4 -> 71), scikit-learn
# rounds the test size up once (569 * 0.2 = 113.8 -> 114): sizes may differ by one row per class.
n_classes = len(np.unique(y_clf))
assert abs(len(idx_test) - len(c_te)) <= n_classes and abs(len(idx_train) - len(c_tr)) <= 2 * n_classes
np.testing.assert_allclose(y_clf[idx_test].mean(), y_clf[c_te].mean(), atol=0.01)
print("same sizes (within rounding) and same class shares (within 0.01): OK")
print("(the rows differ: each uses its own random numbers; only the recipe is the same)")

dr = DummyRegressor(strategy="mean").fit(X_reg[r_tr], y_reg[r_tr])
np.testing.assert_allclose(dr.predict(X_reg[r_va]), base_mean, rtol=1e-12)
np.testing.assert_allclose(mean_squared_error(yv, base_mean), mse(yv, base_mean), rtol=1e-12)
np.testing.assert_allclose(mean_absolute_error(yv, base_mean), mae(yv, base_mean), rtol=1e-12)
np.testing.assert_allclose(r2_score(yv, pred_bmi), r2(yv, pred_bmi), rtol=1e-12)
print(f"DummyRegressor(mean) val R^2 {dr.score(X_reg[r_va], yv):.4f}   matches mine: OK")
dc = DummyClassifier(strategy="most_frequent").fit(X_clf[c_tr], yc_tr)
np.testing.assert_allclose(accuracy_score(yc_va, dc.predict(X_clf[c_va])), np.mean(yc_va == majority))
print(f"DummyClassifier(most_frequent) val accuracy {dc.score(X_clf[c_va], yc_va):.4f}   matches mine: OK")

# %%
# Key plot: the best-of-k test accuracy climbs while every model is a coin flip.
section("key plot")
ks = np.arange(1, N_REUSE_MODELS + 1)
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=120)
lo, hi = np.quantile(running, [0.05, 0.95], axis=0)
ax.fill_between(ks, lo, hi, alpha=0.25, label="middle 90% of runs")
ax.plot(ks, running.mean(axis=0), "-", lw=2, label="mean best test accuracy (reported)")
ax.axhline(0.5, color="k", ls="--", lw=1.2, label="true accuracy of every model")
for k in (10, 100, 1000):
    ax.plot(k, running[:, k - 1].mean(), "o", color="tab:red")
    ax.annotate(f"{running[:, k - 1].mean():.2f}", (k, running[:, k - 1].mean()),
                textcoords="offset points", xytext=(-10, 8))
ax.set_xscale("log")
ax.set_ylim(0.4, 0.75)
ax.set(xlabel="number of models compared on the same test set (log scale)",
       ylabel=f"test accuracy (test set of {N_REUSE_TEST})",
       title=f"Best of 100 coin flips on one test set 'scores' {running[:, 99].mean():.2f}")
ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
path_png = IMG / "b1_1_test_reuse.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Run the baselines on California housing (downloads ~400 KB into data/ on first use):
#    from sklearn.datasets import fetch_california_housing
#    h = fetch_california_housing(data_home=DATA / "sklearn")
#    then reuse train_val_test_split, the mean baseline and fit_one_feature_bins on MedInc.
# 2. Set N_REUSE_TEST = 1_000. How much does the best-of-100 inflation shrink, and why (A6.2)?
# 3. Give fit_one_feature_bins 50 bins instead of 4. Compare train and val R^2: what happened?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
