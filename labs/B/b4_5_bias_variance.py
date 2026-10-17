# %% [markdown]
# # B4.5 Bias–variance, learning curves and validation curves — lab
# Lesson: lessons/B/b4_5_bias_variance.md
# Run: uv run python labs/B/b4_5_bias_variance.py   (CPU, defaults: ~30 s; downloads 0.4 MB once)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import time
import urllib.request
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
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor

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
OUT = ROOT / "outputs" / "b4_5"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only runs more simulated training sets.
GPU_SCALE = False
N_SETS = 2000 if GPU_SCALE else 300  # simulated training sets for the decomposition
N_TRAIN = 30  # points per simulated training set
NOISE_SD = 0.3
DEGREES = list(range(0, 11))
TRAIN_SIZES = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
DEPTHS = list(range(1, 21))

HOUSING_TGZ = DATA / "california_housing" / "housing.tgz"  # as in B3.3 (VERIFY link and licence, as of 2026-10)
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"

# %%
# Worked example by hand: one test point, four training sets.
section("worked example by hand")
f_x0 = 4.0  # true value of f at the test point
preds4 = np.array([2.5, 3.5, 3.0, 3.0])  # the model's prediction at x0 after training on four different datasets
noise_var = 1.0
mean_pred = preds4.mean()
bias2, var = (mean_pred - f_x0) ** 2, preds4.var()
print(f"predictions {preds4.tolist()}: mean {mean_pred}  bias^2 ({mean_pred} - {f_x0})^2 = {bias2}  variance {var}")
print(f"expected squared error = bias^2 + variance + noise = {bias2} + {var} + {noise_var} = {bias2 + var + noise_var}")


# %%
# From scratch: the decomposition by simulation. True function sin(2 pi x) on [0, 1], noise sd 0.3,
# 30 training points per set, polynomial fits of degree 0..12, evaluated on a grid of 200 test points.
section("from scratch: bias-variance decomposition")


def f_true(x: FloatArray) -> FloatArray:
    return np.sin(2 * np.pi * x)


def fit_poly(x: FloatArray, y: FloatArray, degree: int) -> FloatArray:
    """Least-squares polynomial coefficients, lowest power first. x, y (n,) -> (degree + 1,)."""
    V = np.vander(x, degree + 1, increasing=True)  # (n, degree + 1): columns 1, x, x^2, ...
    coef, *_ = np.linalg.lstsq(V, y, rcond=None)
    return coef


def predict_poly(coef: FloatArray, x: FloatArray) -> FloatArray:
    return np.vander(x, len(coef), increasing=True) @ coef


x_test = np.linspace(0, 1, 200)
sim = np.random.default_rng(SEED)
train_x = sim.uniform(0, 1, size=(N_SETS, N_TRAIN))  # (sets, n)
train_y = f_true(train_x) + sim.normal(0, NOISE_SD, size=train_x.shape)
y_test_noisy = f_true(x_test)[None, :] + sim.normal(0, NOISE_SD, size=(N_SETS, len(x_test)))  # fresh test noise per set
decomp = {}
for deg in DEGREES:
    P = np.stack([predict_poly(fit_poly(train_x[s], train_y[s], deg), x_test) for s in range(N_SETS)])  # (sets, 200)
    avg = P.mean(axis=0)  # the average model, at each test point
    bias2 = np.mean((avg - f_true(x_test)) ** 2)  # averaged over test points
    variance = np.mean(P.var(axis=0))
    measured = np.mean((y_test_noisy - P) ** 2)  # actual test MSE on fresh noisy points
    train_mse = np.mean([np.mean((train_y[s] - predict_poly(fit_poly(train_x[s], train_y[s], deg), train_x[s])) ** 2)
                         for s in range(min(N_SETS, 100))])
    decomp[deg] = (bias2, variance, measured, train_mse)
print(f"noise variance sigma^2 = {NOISE_SD ** 2:.4f}")
print(f"{'degree':>6} {'bias^2':>8} {'variance':>9} {'sum + noise':>11} {'test MSE':>9} {'train MSE':>9}")
for deg, (b2, v, m, tr) in decomp.items():
    print(f"{deg:>6} {b2:>8.4f} {v:>9.4f} {b2 + v + NOISE_SD ** 2:>11.4f} {m:>9.4f} {tr:>9.4f}")
best_deg = min(decomp, key=lambda d: decomp[d][2])
print(f"lowest test MSE at degree {best_deg}")

# %%
# Data for the curves: California housing, as in B3.3 (NaN bedroom counts kept; trees handle them).
section("data for the curves")
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
X = np.column_stack([h["median_income"], h["housing_median_age"], h["total_rooms"] / h["households"],
                     h["total_bedrooms"] / h["households"], h["population"], h["population"] / h["households"],
                     h["latitude"], h["longitude"]]).astype(np.float64)
y = (h["median_house_value"] / 100_000).to_numpy(dtype=np.float64)
dev = np.random.default_rng(SEED).permutation(len(y))[: round(0.8 * len(y))]  # the last 20% stay aside as a test set
X_dev, y_dev = X[dev], y[dev]
cv = KFold(5, shuffle=True, random_state=SEED)
print(f"development rows {X_dev.shape}   5-fold CV")


def rmse(y_true: FloatArray, y_hat: FloatArray) -> float:
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


# %%
# From scratch: learning curves (train and validation RMSE as the training set grows) for a high-bias
# model (a depth-2 tree: four leaves) and a high-variance one (a fully grown tree).
section("from scratch: learning curves")


def learning_curve_scratch(model: BaseEstimator, X: FloatArray, y: FloatArray, sizes: FloatArray,
                           splits: list[tuple[IntArray, IntArray]]) -> tuple[IntArray, FloatArray, FloatArray]:
    """Train/validation RMSE for the first n training rows of each fold. -> sizes (s,), train (s, k), val (s, k)."""
    n_train = len(splits[0][0])
    counts = (sizes * n_train).astype(int)
    tr_scores = np.zeros((len(counts), len(splits)))
    va_scores = np.zeros_like(tr_scores)
    for j, (tr, va) in enumerate(splits):
        for i, c in enumerate(counts):
            sub = tr[:c]  # scikit-learn's rule: the first c rows of the fold's training indices
            m = clone(model).fit(X[sub], y[sub])
            tr_scores[i, j] = rmse(y[sub], m.predict(X[sub]))
            va_scores[i, j] = rmse(y[va], m.predict(X[va]))
    return counts, tr_scores, va_scores


splits = list(cv.split(X_dev))
models = {"depth-2 tree (high bias)": DecisionTreeRegressor(max_depth=2, random_state=SEED),
          "full-depth tree (high variance)": DecisionTreeRegressor(random_state=SEED)}
lc = {}
for name, model in models.items():
    counts, tr_s, va_s = learning_curve_scratch(model, X_dev, y_dev, TRAIN_SIZES, splits)
    lc[name] = (counts, tr_s, va_s)
    print(f"{name}:")
    for c, t, v in zip(counts, tr_s.mean(1), va_s.mean(1)):
        print(f"   {c:>6} rows: train RMSE {t:.4f}   val RMSE {v:.4f}   gap {v - t:.4f}")

# %%
# From scratch: a validation curve over the tree's max_depth.
section("from scratch: validation curve")
vc_tr = np.zeros((len(DEPTHS), len(splits)))
vc_va = np.zeros_like(vc_tr)
for i, depth in enumerate(DEPTHS):
    for j, (tr, va) in enumerate(splits):
        m = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(X_dev[tr], y_dev[tr])
        vc_tr[i, j] = rmse(y_dev[tr], m.predict(X_dev[tr]))
        vc_va[i, j] = rmse(y_dev[va], m.predict(X_dev[va]))
for i, depth in enumerate(DEPTHS):
    if depth in (1, 2, 4, 6, 8, 10, 12, 15, 20):
        print(f"max_depth {depth:>2}: train RMSE {vc_tr[i].mean():.4f}   val RMSE {vc_va[i].mean():.4f} "
              f"(+- {vc_va[i].std():.4f} across folds)")
best_depth = DEPTHS[int(np.argmin(vc_va.mean(1)))]
print(f"best max_depth by mean validation RMSE: {best_depth}")

# %%
# With scikit-learn: learning_curve and validation_curve give the same numbers on the same folds.
section("with scikit-learn")
for name, model in models.items():
    sizes_sk, tr_sk, va_sk = learning_curve(model, X_dev, y_dev, train_sizes=TRAIN_SIZES, cv=cv,
                                            scoring="neg_root_mean_squared_error")
    np.testing.assert_array_equal(sizes_sk, lc[name][0])
    np.testing.assert_allclose(-tr_sk, lc[name][1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(-va_sk, lc[name][2], rtol=1e-10, atol=1e-12)
print("learning_curve matches for both models: OK")
tr_sk, va_sk = validation_curve(DecisionTreeRegressor(random_state=SEED), X_dev, y_dev, param_name="max_depth",
                                param_range=DEPTHS, cv=cv, scoring="neg_root_mean_squared_error")
np.testing.assert_allclose(-tr_sk, vc_tr, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(-va_sk, vc_va, rtol=1e-10, atol=1e-12)
print("validation_curve matches: OK")

# %%
# Key plot: the decomposition against polynomial degree.
section("key plot: bias-variance decomposition")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
degs = np.array(DEGREES)
b2s = np.array([decomp[d][0] for d in DEGREES])
vs = np.array([decomp[d][1] for d in DEGREES])
ms = np.array([decomp[d][2] for d in DEGREES])
ax.plot(degs, b2s, "o-", label="bias²")
ax.plot(degs, vs, "s--", label="variance")
ax.plot(degs, ms, "^-", label="measured test MSE")
ax.axhline(NOISE_SD**2, color="gray", linestyle=":", label="noise σ² (irreducible)")
ax.set_yscale("log")
ax.set_xlabel("polynomial degree (model capacity)")
ax.set_ylabel("squared error (log scale)")
ax.set_title(f"Bias falls, variance rises; test error is lowest at degree {best_deg}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b4_5_bias_variance.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
for (name, (counts, tr_s, va_s)), style in zip(lc.items(), ["-", "--"]):
    line, = ax1.plot(counts, va_s.mean(1), style, marker="o", label=f"{name}: validation")
    ax1.plot(counts, tr_s.mean(1), style, marker="x", color=line.get_color(), alpha=0.6, label=f"{name}: training")
ax1.set_xscale("log")
ax1.set_xlabel("training rows (log scale)")
ax1.set_ylabel("RMSE ($100,000s)")
ax1.set_title("Learning curves")
ax1.legend(fontsize=7)
ax2.plot(DEPTHS, vc_tr.mean(1), "x-", label="training")
ax2.plot(DEPTHS, vc_va.mean(1), "o-", label="validation")
ax2.set_xlabel("max_depth")
ax2.set_ylabel("RMSE ($100,000s)")
ax2.set_title("Validation curve for a decision tree")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "learning_validation_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved lessons/B/img/b4_5_bias_variance.png and outputs/b4_5/learning_validation_curves.png")

# %%
# Try this (not graded):
# 1. Rerun the decomposition with N_TRAIN = 300. Which degree is best now, and why?
# 2. Add Ridge(alpha) on the degree-12 polynomial features and sweep alpha: a validation curve for regularization.
# 3. Draw the learning curve of a random forest (B3.2). Does its gap close like the single tree's?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
