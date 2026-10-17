# %% [markdown]
# # B3.3 Gradient boosting — lab
# Lesson: lessons/B/b3_3_gradient_boosting.md
# Run: uv run python labs/B/b3_3_gradient_boosting.py   (CPU, defaults: ~35 s; downloads 0.4 MB once)
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
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
OUT = ROOT / "outputs" / "b3_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def rmse(y: FloatArray, y_hat: FloatArray) -> float:
    """Root mean squared error. y (n,), y_hat (n,) -> scalar."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# California housing is small, so the GPU only matters if you scale the number of rounds way up;
# on Colab you would also pass device="cuda" to XGBoost.
GPU_SCALE = False
N_TREES_SCRATCH = 200  # boosting rounds in the from-scratch model and its scikit-learn twin
LR_SCRATCH, DEPTH_SCRATCH = 0.1, 3
LR_GRID = [1.0, 0.3, 0.1, 0.03]  # learning rates for the key plot
N_ROUNDS_CURVE = 3000 if GPU_SCALE else 1000
N_ROUNDS_MAX = 20_000 if GPU_SCALE else 5000  # upper limit for early stopping
EARLY_STOP = 100  # stop after this many rounds without improvement
VAL_FRAC, TEST_FRAC, ES_FRAC = 0.2, 0.2, 0.2  # ES_FRAC: share of the training rows kept for early stopping
N_THREADS = 1  # one thread makes XGBoost and LightGBM results identical across machines

# California housing (1990 US census, one row per block group): the copy from A. Geron's
# handson-ml2 repository (VERIFY link and licence, as of 2026-10). It holds the same 20,640 districts
# as sklearn.datasets.fetch_california_housing, with 207 bedroom counts missing. The checksum
# guarantees everyone runs on the same file.
HOUSING_DIR = DATA / "california_housing"
HOUSING_TGZ = HOUSING_DIR / "housing.tgz"
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# %%
# Data: download once (idempotent), check the checksum, and build scikit-learn's eight features.
section("data")


def load_housing() -> tuple[pd.DataFrame, FloatArray]:
    """Download, verify and parse the housing data. -> features (20640, 8), target in $100,000s (20640,)."""
    if not HOUSING_TGZ.exists():
        HOUSING_DIR.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(HOUSING_URL, timeout=60).read()
        HOUSING_TGZ.write_bytes(raw)
        print(f"~ downloaded {len(raw):,} bytes from {HOUSING_URL}")  # first run only, hence ~
    digest = hashlib.sha256(HOUSING_TGZ.read_bytes()).hexdigest()
    if digest != HOUSING_SHA256:
        raise ValueError(f"{HOUSING_TGZ} has sha256 {digest}, expected {HOUSING_SHA256}; delete it and rerun")
    with tarfile.open(HOUSING_TGZ) as tar:  # read the one CSV member; nothing is extracted to disk
        member = tar.extractfile("housing.csv")
        if member is None:
            raise ValueError("housing.csv not found in the archive")
        df = pd.read_csv(io.BytesIO(member.read()))
    X = pd.DataFrame({  # the same eight features as fetch_california_housing
        "MedInc": df["median_income"],  # median income in the block group, in $10,000s
        "HouseAge": df["housing_median_age"],
        "AveRooms": df["total_rooms"] / df["households"],
        "AveBedrms": df["total_bedrooms"] / df["households"],  # NaN where the bedroom count is missing
        "Population": df["population"],
        "AveOccup": df["population"] / df["households"],
        "Latitude": df["latitude"],
        "Longitude": df["longitude"],
    })
    return X, (df["median_house_value"] / 100_000).to_numpy(dtype=np.float64)


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py (no stratification for a numeric target)
    rows = gen.permutation(n)
    n_test, n_val = round(test_frac * n), round(val_frac * n)
    parts = [rows[n_test + n_val:], rows[n_test:n_test + n_val], rows[:n_test]]
    train, val, test = (gen.permutation(p) for p in parts)
    return train, val, test


X_df, y_all = load_housing()
X_raw = X_df.to_numpy(dtype=np.float64)  # (20640, 8) with NaN in AveBedrms
i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
y_tr, y_va = y_all[i_tr], y_all[i_va]
# scikit-learn's GradientBoostingRegressor refuses NaN: fill AveBedrms with the *training* median (B1.2, B1.4).
bed_median = float(np.nanmedian(X_raw[i_tr, 3]))
X_imp = X_raw.copy()
X_imp[np.isnan(X_imp[:, 3]), 3] = bed_median
X_tr, X_va = X_imp[i_tr], X_imp[i_va]  # imputed, for scikit-learn and the from-scratch model
print(f"{HOUSING_TGZ.relative_to(ROOT)}: checksum OK, {X_raw.shape[0]} districts, {X_raw.shape[1]} features")
print(f"missing AveBedrms: {int(np.isnan(X_raw[:, 3]).sum())} rows (filled with the training median {bed_median:.4f} for sklearn)")
print(f"target: median house value in $100,000s, mean {y_all.mean():.4f}, capped at {y_all.max():.5f} "
      f"({int(np.sum(y_all >= 5.0))} districts at the cap)")
print(f"train {X_tr.shape}  val {X_va.shape}  (test {len(i_te)} rows kept aside)")
print(X_df.iloc[:3].round(4).to_string())

# %%
# Worked example by hand: four points, a stump per round, learning rate 0.5.
section("worked example by hand")
x_toy = np.array([[1.0], [2.0], [3.0], [4.0]])  # (4, 1)
y_toy = np.array([1.0, 3.0, 6.0, 10.0])
F = np.full(4, y_toy.mean())  # F_0: the best constant is the mean
print(f"F0 = mean(y) = {F[0]:.4f}   MSE {np.mean((y_toy - F) ** 2):.4f}")
for m in (1, 2):
    r = y_toy - F  # residuals = negative gradient of (1/2)(y - F)^2
    stump = DecisionTreeRegressor(max_depth=1, random_state=SEED).fit(x_toy, r)
    leaf = stump.predict(x_toy)
    F = F + 0.5 * leaf
    print(f"round {m}: residuals {np.round(r, 4).tolist()}   split x <= {stump.tree_.threshold[0]}   "
          f"leaf values {np.round(stump.tree_.value[1:, 0, 0], 4).tolist()}")
    print(f"         F{m} = {np.round(F, 4).tolist()}   MSE {np.mean((y_toy - F) ** 2):.4f}")
gbr_toy = GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1, random_state=SEED).fit(x_toy, y_toy)
np.testing.assert_allclose(gbr_toy.predict(x_toy), F)
print("GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1) gives the same F2: OK")
for lam in (0.0, 1.0):  # XGBoost's leaf weight -G / (H + lambda); for squared error g = F - y, h = 1
    one = xgb.XGBRegressor(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=lam, base_score=5.0,
                           min_child_weight=0, tree_method="exact", n_jobs=N_THREADS).fit(x_toy, y_toy)
    print(f"XGBoost, one stump, lambda {lam}: leaf weights {np.round(one.predict(x_toy) - 5.0, 4).tolist()}")

# %%
# From scratch: gradient boosting for squared error, with scikit-learn's regression tree as the learner.
section("from scratch: boosting on California housing")


def fit_boosting(X: FloatArray, y: FloatArray, n_trees: int, lr: float, max_depth: int
                 ) -> tuple[float, list[DecisionTreeRegressor]]:
    """Gradient boosting for squared error. X (n, d), y (n,) -> (F0, trees)."""
    f0 = float(y.mean())
    F = np.full(len(y), f0)  # (n,) current predictions
    trees = []
    # Trees break ties between equally good splits with a random feature order (B3.1). scikit-learn's
    # boosting draws every tree's order from one shared legacy RandomState; we share one the same way
    # only so that the check below can demand identical trees. Nothing else uses it.
    tie_breaker = np.random.RandomState(SEED)
    for _ in range(n_trees):
        residual = y - F  # (n,) negative gradient of (1/2)(y - F)^2
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=tie_breaker).fit(X, residual)
        F += lr * tree.predict(X)  # a small step in the direction the tree points
        trees.append(tree)
    return f0, trees


def staged_predict(f0: float, trees: list[DecisionTreeRegressor], lr: float, X: FloatArray) -> FloatArray:
    """Predictions after 1, 2, ..., M trees. X (m, d) -> (M, m)."""
    steps = np.stack([tree.predict(X) for tree in trees])  # (M, m)
    return f0 + lr * np.cumsum(steps, axis=0)


t_fit = time.perf_counter()
f0, trees = fit_boosting(X_tr, y_tr, N_TREES_SCRATCH, LR_SCRATCH, DEPTH_SCRATCH)
print(f"~ fitted {N_TREES_SCRATCH} trees in {time.perf_counter() - t_fit:.1f} s")
stage_tr = staged_predict(f0, trees, LR_SCRATCH, X_tr)
stage_va = staged_predict(f0, trees, LR_SCRATCH, X_va)
print(f"F0 = {f0:.4f}: train RMSE {rmse(y_tr, np.full_like(y_tr, f0)):.4f}  val RMSE {rmse(y_va, np.full_like(y_va, f0)):.4f}")
for m in [1, 10, 50, 100, 200]:
    print(f"after {m:>3} trees: train RMSE {rmse(y_tr, stage_tr[m - 1]):.4f}  val RMSE {rmse(y_va, stage_va[m - 1]):.4f}")

# %%
# With scikit-learn: GradientBoostingRegressor runs the same loop (mean start, residual trees, shrinkage).
section("with scikit-learn: GradientBoostingRegressor")
gbr = GradientBoostingRegressor(n_estimators=N_TREES_SCRATCH, learning_rate=LR_SCRATCH, max_depth=DEPTH_SCRATCH,
                                random_state=SEED).fit(X_tr, y_tr)
np.testing.assert_allclose(gbr.predict(X_va), stage_va[-1], rtol=1e-6, atol=1e-8)
sk_stages = np.stack(list(gbr.staged_predict(X_va)))
np.testing.assert_allclose(sk_stages, stage_va, rtol=1e-6, atol=1e-8)
print(f"val RMSE: mine {rmse(y_va, stage_va[-1]):.4f}   sklearn {rmse(y_va, gbr.predict(X_va)):.4f}")
print("predictions match GradientBoostingRegressor at every one of the 200 stages: OK")

# %%
# Key plot: XGBoost validation curves for four learning rates. Each round adds one depth-6 tree.
section("key plot: learning rate")
curves: dict[float, FloatArray] = {}
for lr in LR_GRID:
    model = xgb.XGBRegressor(n_estimators=N_ROUNDS_CURVE, learning_rate=lr, max_depth=6, n_jobs=N_THREADS,
                             random_state=SEED, eval_metric="rmse")
    model.fit(X_raw[i_tr], y_tr, eval_set=[(X_raw[i_va], y_va)], verbose=False)
    curves[lr] = np.array(model.evals_result()["validation_0"]["rmse"])
    best = int(np.argmin(curves[lr]))
    print(f"learning rate {lr:<4}: best val RMSE {curves[lr][best]:.4f} at round {best + 1:>4}   "
          f"val RMSE at round {N_ROUNDS_CURVE}: {curves[lr][-1]:.4f}")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
for (lr, curve), style in zip(curves.items(), ["-", "--", "-.", ":"]):
    ax.plot(np.arange(1, len(curve) + 1), curve, style, label=f"learning rate {lr}")
ax.set_xscale("log")
ax.set_ylim(0.4, 0.8)
ax.set_xlabel("boosting rounds = trees (log scale)")
ax.set_ylabel("validation RMSE ($100,000s)")
ax.set_title("Too high a learning rate bottoms out early and higher")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(IMG / "b3_3_learning_rate_curves.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b3_3_learning_rate_curves.png")

# %%
# XGBoost and LightGBM in practice: a low learning rate, many rounds, and early stopping on a slice
# of the training rows (so the validation set stays clean for comparing models). Both get the raw
# features, NaN included: they learn which way to send missing values at each split.
section("XGBoost and LightGBM with early stopping")
split_gen = np.random.default_rng(SEED + 1)
perm = split_gen.permutation(len(i_tr))
n_es = round(ES_FRAC * len(i_tr))
i_es, i_fit = i_tr[perm[:n_es]], i_tr[perm[n_es:]]
print(f"training rows split for early stopping: fit {len(i_fit)}  early-stopping {len(i_es)}")
results = {}

t_fit = time.perf_counter()
xgb_model = xgb.XGBRegressor(n_estimators=N_ROUNDS_MAX, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, early_stopping_rounds=EARLY_STOP, eval_metric="rmse",
                             n_jobs=N_THREADS, random_state=SEED)
xgb_model.fit(X_raw[i_fit], y_all[i_fit], eval_set=[(X_raw[i_es], y_all[i_es])], verbose=False)
t_xgb = time.perf_counter() - t_fit
results["XGBoost"] = (rmse(y_va, xgb_model.predict(X_raw[i_va])), xgb_model.best_iteration + 1, t_xgb)

t_fit = time.perf_counter()
lgb_model = lgb.LGBMRegressor(n_estimators=N_ROUNDS_MAX, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, n_jobs=N_THREADS, random_state=SEED,
                              deterministic=True, force_row_wise=True, verbose=-1)
lgb_model.fit(X_raw[i_fit], y_all[i_fit], eval_X=X_raw[i_es], eval_y=y_all[i_es],  # eval_set= is deprecated in 4.7
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
t_lgb = time.perf_counter() - t_fit
results["LightGBM"] = (rmse(y_va, lgb_model.predict(X_raw[i_va])), lgb_model.best_iteration_, t_lgb)

lin = LinearRegression().fit(X_tr, y_tr)
t_fit = time.perf_counter()
forest = RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=2, n_jobs=N_THREADS,
                               random_state=SEED).fit(X_tr, y_tr)
t_rf = time.perf_counter() - t_fit
print(f"{'model':<44} {'val RMSE':>8} {'trees':>6}")
print(f"{'predict the training mean':<44} {rmse(y_va, np.full_like(y_va, y_tr.mean())):>8.4f} {'-':>6}")
print(f"{'linear regression (B1.5)':<44} {rmse(y_va, lin.predict(X_va)):>8.4f} {'-':>6}")
print(f"{'random forest (B3.2), max_features 0.33':<44} {rmse(y_va, forest.predict(X_va)):>8.4f} {200:>6}")
print(f"{'from-scratch boosting, depth 3, lr 0.1':<44} {rmse(y_va, stage_va[-1]):>8.4f} {N_TREES_SCRATCH:>6}")
for name, (score, n_used, _) in results.items():
    print(f"{name + ', lr 0.05, early stopping':<44} {score:>8.4f} {n_used:>6}")
print(f"~ training time: random forest {t_rf:.1f} s   XGBoost {t_xgb:.1f} s   LightGBM {t_lgb:.1f} s (1 thread each)")
xgb_scores = xgb_model.get_booster().get_score(importance_type="total_gain")
print("XGBoost features by total gain: " + ", ".join(
    f"{FEATURES[int(k[1:])]}" for k, _ in sorted(xgb_scores.items(), key=lambda kv: -kv[1])[:4]))

# %%
# Try this (not graded):
# 1. Rerun the from-scratch loop with learning rate 1.0 and depth 3. How fast does the val RMSE bottom out?
# 2. Set max_depth=2 and then 10 in the XGBoost cell. Which needs more rounds, and which overfits?
# 3. Replace the NaN bedroom counts with -1 before XGBoost. Does the score change? Why might it not?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
