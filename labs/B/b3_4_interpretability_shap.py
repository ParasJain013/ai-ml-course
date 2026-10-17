# %% [markdown]
# # B3.4 Interpretability — lab
# Lesson: lessons/B/b3_4_interpretability_shap.md
# Run: uv run python labs/B/b3_4_interpretability_shap.py   (CPU, defaults: ~60 s; downloads 0.4 MB once)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import io
import itertools
import math
import sys
import tarfile
import time
import urllib.request
import warnings
from collections.abc import Callable
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
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

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
OUT = ROOT / "outputs" / "b3_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def r2(y: FloatArray, y_hat: FloatArray) -> float:
    """Coefficient of determination. y (n,), y_hat (n,) -> scalar."""
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only raises the number of repeats and background rows.
GPU_SCALE = False
N_REPEATS = 50 if GPU_SCALE else 20  # shuffles per feature in permutation importance
N_BACKGROUND = 1000 if GPU_SCALE else 200  # background rows for exact Shapley values
GRID_POINTS = 20  # grid of the partial dependence curve
VAL_FRAC, TEST_FRAC, ES_FRAC = 0.2, 0.2, 0.2
N_THREADS = 1

# California housing, the same file and split as B3.3 (VERIFY link and licence, as of 2026-10).
HOUSING_DIR = DATA / "california_housing"
HOUSING_TGZ = HOUSING_DIR / "housing.tgz"
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# %%
section("data")


def load_housing() -> tuple[pd.DataFrame, FloatArray, pd.Series]:
    """Download, verify and parse the housing data. -> features (20640, 8), target in $100,000s, ocean proximity."""
    # from labs/B/b3_3_gradient_boosting.py, also returning the ocean_proximity column (5 categories)
    if not HOUSING_TGZ.exists():
        HOUSING_DIR.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(HOUSING_URL, timeout=60).read()
        HOUSING_TGZ.write_bytes(raw)
        print(f"~ downloaded {len(raw):,} bytes from {HOUSING_URL}")
    digest = hashlib.sha256(HOUSING_TGZ.read_bytes()).hexdigest()
    if digest != HOUSING_SHA256:
        raise ValueError(f"{HOUSING_TGZ} has sha256 {digest}, expected {HOUSING_SHA256}; delete it and rerun")
    with tarfile.open(HOUSING_TGZ) as tar:
        member = tar.extractfile("housing.csv")
        if member is None:
            raise ValueError("housing.csv not found in the archive")
        df = pd.read_csv(io.BytesIO(member.read()))
    X = pd.DataFrame({
        "MedInc": df["median_income"],
        "HouseAge": df["housing_median_age"],
        "AveRooms": df["total_rooms"] / df["households"],
        "AveBedrms": df["total_bedrooms"] / df["households"],
        "Population": df["population"],
        "AveOccup": df["population"] / df["households"],
        "Latitude": df["latitude"],
        "Longitude": df["longitude"],
    })
    return X, (df["median_house_value"] / 100_000).to_numpy(dtype=np.float64), df["ocean_proximity"]


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


X_df, y_all, ocean = load_housing()
X_raw = X_df.to_numpy(dtype=np.float64)  # (20640, 8), NaN in AveBedrms
i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
X_tr, y_tr, X_va, y_va = X_raw[i_tr], y_all[i_tr], X_raw[i_va], y_all[i_va]
print(f"train {X_tr.shape}  val {X_va.shape}  (test {len(i_te)} rows kept aside)")

# %%
# The B3.3 model: XGBoost with early stopping on a slice of the training rows.
section("the B3.3 model")
perm = np.random.default_rng(SEED + 1).permutation(len(i_tr))
n_es = round(ES_FRAC * len(i_tr))
i_es, i_fit = i_tr[perm[:n_es]], i_tr[perm[n_es:]]
model = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=100, eval_metric="rmse", n_jobs=N_THREADS, random_state=SEED)
model.fit(X_raw[i_fit], y_all[i_fit], eval_set=[(X_raw[i_es], y_all[i_es])], verbose=False)
base_r2 = r2(y_va, model.predict(X_va))
print(f"XGBoost: {model.best_iteration + 1} trees   val R^2 {base_r2:.4f}   "
      f"val RMSE {np.sqrt(np.mean((y_va - model.predict(X_va)) ** 2)):.4f}")

# %%
# Worked example by hand: exact Shapley values for f(x) = 10 x1 + 5 x2 + 4 x1 x3 at x = (1, 1, 1),
# with a single background row at 0, so v(S) = f(x with the features outside S set to 0).
section("worked example by hand")


def shapley_exact(value: Callable[[tuple[int, ...]], float], n_feat: int) -> FloatArray:
    """Shapley values from a value function over coalitions (tuples of feature indices) -> (n_feat,)."""
    phi = np.zeros(n_feat)
    for j in range(n_feat):
        others = [k for k in range(n_feat) if k != j]
        for size in range(n_feat):
            weight = math.factorial(size) * math.factorial(n_feat - size - 1) / math.factorial(n_feat)
            for S in itertools.combinations(others, size):
                phi[j] += weight * (value(tuple(sorted(S + (j,)))) - value(S))
    return phi


def f_toy(x: FloatArray) -> float:
    return float(10 * x[0] + 5 * x[1] + 4 * x[0] * x[2])


x_toy = np.array([1.0, 1.0, 1.0])
toy_value = {}
for size in range(4):
    for S in itertools.combinations(range(3), size):
        z = np.zeros(3)
        z[list(S)] = x_toy[list(S)]
        toy_value[S] = f_toy(z)
print("v(S) for every coalition: " + "  ".join(f"{'{' + ','.join(str(k + 1) for k in S) + '}'}: {v:g}"
                                             for S, v in toy_value.items()))
phi_toy = shapley_exact(lambda S: toy_value[S], 3)
print(f"Shapley values {phi_toy.round(4).tolist()}   sum {phi_toy.sum():.4f} = f(x) - v(empty) = "
      f"{f_toy(x_toy) - toy_value[()]:.4f}")

# %%
# From scratch 1: permutation importance on the validation rows (drop in R^2 when one column is shuffled).
section("from scratch: permutation importance")


def perm_importance(predict: Callable[[FloatArray], FloatArray], X: FloatArray, y: FloatArray, n_repeats: int,
                    gen: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Mean and sd of the R^2 drop when each column is shuffled. X (m, d) -> (d,), (d,)."""
    baseline = r2(y, predict(X))
    drops = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        Xp = X.copy()
        for k in range(n_repeats):
            Xp[:, j] = X[gen.permutation(len(X)), j]  # break the link between column j and the target
            drops[j, k] = baseline - r2(y, predict(Xp))
    return drops.mean(axis=1), drops.std(axis=1)


pi_mean, pi_sd = perm_importance(model.predict, X_va, y_va, N_REPEATS, np.random.default_rng(SEED))
order = np.argsort(-pi_mean)
for j in order:
    print(f"{FEATURES[j]:<11} R^2 drop {pi_mean[j]:.4f} +- {pi_sd[j]:.4f}")

# %%
# From scratch 2: one-feature partial dependence for MedInc: set the column to each grid value for every
# row, predict, average.
section("from scratch: partial dependence")
J_PD = FEATURES.index("MedInc")
grid = np.linspace(*np.nanpercentile(X_va[:, J_PD], [5, 95]), GRID_POINTS)


def partial_dependence_1d(predict: Callable[[FloatArray], FloatArray], X: FloatArray, j: int,
                          grid_values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Average (and per-row ICE) prediction with column j set to each grid value. -> (g,), (m, g)."""
    ice = np.zeros((len(X), len(grid_values)))
    for k, v in enumerate(grid_values):
        Xg = X.copy()
        Xg[:, j] = v
        ice[:, k] = predict(Xg)
    return ice.mean(axis=0), ice


pd_mine, ice_mine = partial_dependence_1d(model.predict, X_va, J_PD, grid)
for k in [0, 5, 10, 15, 19]:
    print(f"MedInc = {grid[k]:6.3f} ($10,000s): average prediction {pd_mine[k]:.4f}   "
          f"ICE range {ice_mine[:, k].min():.3f}..{ice_mine[:, k].max():.3f}")

# %%
# From scratch 3: exact Shapley values for a 3-feature model, by enumerating all 8 coalitions.
# v(S) = average prediction when the features in S take this row's values and the others take the
# values of each background row (the "interventional" definition).
section("from scratch: exact Shapley values for a 3-feature model")
F3 = [FEATURES.index(f) for f in ["MedInc", "Latitude", "Longitude"]]
model3 = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, n_jobs=N_THREADS,
                          random_state=SEED).fit(X_tr[:, F3], y_tr)
background = X_tr[np.random.default_rng(SEED).choice(len(X_tr), N_BACKGROUND, replace=False)][:, F3]  # (B, 3)
explain_rows = X_va[:5][:, F3]  # (5, 3)


def coalition_value(x: FloatArray, S: tuple[int, ...]) -> float:
    """Mean prediction over the background with the features in S fixed to x's values."""
    Z = background.copy()
    Z[:, list(S)] = x[list(S)]
    return float(model3.predict(Z).astype(np.float64).mean())


phi_mine = np.stack([shapley_exact(lambda S, x=x: coalition_value(x, S), 3) for x in explain_rows])  # (5, 3)
base_value = coalition_value(explain_rows[0], ())
for x, phi in zip(explain_rows, phi_mine):
    print(f"x = {np.round(x, 2).tolist()}: phi {phi.round(4).tolist()}   base {base_value:.4f} + sum {phi.sum():.4f}"
          f" = {base_value + phi.sum():.4f}   prediction {float(model3.predict(x[None, :])[0]):.4f}")

# %%
# With the libraries: scikit-learn for permutation importance and partial dependence, shap for Shapley values.
section("with scikit-learn and shap")
sk_pi = permutation_importance(model, X_va, y_va, n_repeats=N_REPEATS, random_state=SEED, n_jobs=1)  # default score: R^2
# Different random shuffles, so compare within sampling error: |difference| < 4 standard errors.
se = np.sqrt(pi_sd**2 / N_REPEATS + sk_pi.importances_std**2 / N_REPEATS)
z = np.abs(pi_mean - sk_pi.importances_mean) / se
assert np.all(z < 4), z
print(f"permutation importance: same ranking {bool(np.array_equal(order, np.argsort(-sk_pi.importances_mean)))}   "
      f"largest gap {np.max(np.abs(pi_mean - sk_pi.importances_mean)):.4f} = {np.max(z):.1f} standard errors: OK")
disp = PartialDependenceDisplay.from_estimator(model, X_va, [J_PD], kind="both", grid_resolution=GRID_POINTS,
                                               percentiles=(0.05, 0.95), method="brute", random_state=SEED,
                                               feature_names=FEATURES)
pd_sk = disp.pd_results[0]
# scikit-learn interpolates the 5th and 95th percentiles slightly differently (scipy's mquantiles), so its
# grid is shifted by up to 0.02; evaluate our function on its grid for the check.
pd_on_sk_grid, _ = partial_dependence_1d(model.predict, X_va, J_PD, pd_sk["grid_values"][0])
np.testing.assert_allclose(pd_sk["average"][0], pd_on_sk_grid, rtol=1e-6, atol=1e-6)  # XGBoost predicts in float32
plt.gcf().savefig(OUT / "pdp_ice_medinc.png", dpi=120, bbox_inches="tight")
plt.close("all")
print(f"grid ends: ours {grid[0]:.4f}..{grid[-1]:.4f}   sklearn {pd_sk['grid_values'][0][0]:.4f}..{pd_sk['grid_values'][0][-1]:.4f}")
print("partial dependence matches PartialDependenceDisplay at all 20 of its grid points: OK")
masker = shap.maskers.Independent(background, max_samples=len(background))  # use all background rows (default 100)
explainer3 = shap.TreeExplainer(model3, data=masker, feature_perturbation="interventional")
phi_shap = explainer3.shap_values(explain_rows)
np.testing.assert_allclose(phi_shap, phi_mine, atol=1e-5)  # XGBoost predicts in float32
print(f"exact Shapley values match shap.TreeExplainer(interventional): OK   expected_value {explainer3.expected_value:.4f}")

# %%
# SHAP on the full B3.3 model: a summary over the validation rows and one prediction explained.
section("SHAP on the B3.3 model")
explainer = shap.TreeExplainer(model)  # default for trees: "tree_path_dependent", no background needed
sv = explainer.shap_values(X_va)  # (4128, 8)
pred_margin = model.predict(X_va, output_margin=True)
np.testing.assert_allclose(sv.sum(axis=1) + explainer.expected_value, pred_margin, atol=1e-4)
print(f"SHAP values {sv.shape}; base value {float(explainer.expected_value):.4f} + row sum = prediction for every row: OK")
mean_abs = np.abs(sv).mean(axis=0)
print("mean |SHAP| by feature: " + "  ".join(f"{FEATURES[j]} {mean_abs[j]:.4f}" for j in np.argsort(-mean_abs)))
row = int(np.argmax(model.predict(X_va)))  # explain the most expensive prediction
contrib = sorted(zip(FEATURES, sv[row], X_va[row]), key=lambda t: -abs(t[1]))
print(f"district {row}: prediction {model.predict(X_va[row:row + 1])[0]:.4f} (actual {y_va[row]:.4f}) = "
      f"base {float(explainer.expected_value):.4f}")
for name, value, x in contrib:
    print(f"   {'+' if value >= 0 else '-'} {abs(value):.4f}  {name} = {x:.4f}")
shap.summary_plot(sv, X_va, feature_names=FEATURES, show=False)
plt.gcf().savefig(OUT / "shap_summary.png", dpi=120, bbox_inches="tight")
plt.close("all")
shap.plots.waterfall(shap.Explanation(values=sv[row], base_values=float(explainer.expected_value),
                                      data=X_va[row], feature_names=FEATURES), show=False)
plt.gcf().savefig(OUT / "shap_waterfall.png", dpi=120, bbox_inches="tight")
plt.close("all")
print("saved outputs/b3_4/shap_summary.png and outputs/b3_4/shap_waterfall.png")

# %%
# Key plot: a random forest on the 8 numeric features, the 0/1 indicators of the 5 ocean_proximity
# categories, and a column of pure noise. Impurity importance (MDI) credits the noise; permutation
# importance on held-out rows does not.
section("key plot: a noise feature")
onehot = pd.get_dummies(ocean).astype(np.float64)  # (20640, 5) one 0/1 column per category
noise = np.random.default_rng(SEED + 2).normal(size=len(y_all))  # pure noise, unrelated to the target
names_n = FEATURES + list(onehot.columns) + ["NOISE"]
X_n = np.column_stack([np.nan_to_num(X_raw, nan=float(np.nanmedian(X_tr[:, 3]))), onehot.to_numpy(), noise])
Xn_tr, Xn_va = X_n[i_tr], X_n[i_va]
print(f"features {len(names_n)}: 8 numeric, 5 ocean_proximity indicators {list(onehot.columns)}, NOISE")
forest = RandomForestRegressor(n_estimators=100, max_features=0.33, n_jobs=N_THREADS, random_state=SEED).fit(Xn_tr, y_tr)
mdi = forest.feature_importances_
pi_va = permutation_importance(forest, Xn_va, y_va, n_repeats=5, random_state=SEED, n_jobs=1).importances_mean
pi_tr = permutation_importance(forest, Xn_tr[:4000], y_tr[:4000], n_repeats=5, random_state=SEED,
                               n_jobs=1).importances_mean
print(f"{'feature':<11} {'impurity (MDI)':>14} {'perm. train':>12} {'perm. val':>10}")
for j in np.argsort(-mdi):
    print(f"{names_n[j]:<11} {mdi[j]:>14.4f} {pi_tr[j]:>12.4f} {pi_va[j]:>10.4f}")
k = len(names_n)
ranks = {label: int(np.sum(vals > vals[-1])) + 1 for label, vals in [("mdi", mdi), ("train", pi_tr), ("val", pi_va)]}
print(f"NOISE rank out of {k}: impurity {ranks['mdi']}   permutation on training rows {ranks['train']}   "
      f"permutation on validation rows {ranks['val']}")
beaten = [names_n[j] for j in range(k - 1) if mdi[j] < mdi[-1]]
print(f"real features that impurity importance ranks below NOISE: {beaten}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.4), dpi=120, sharey=True)
order_n = np.argsort(mdi)
colors = ["tab:red" if names_n[j] == "NOISE" else "tab:blue" for j in order_n]
for ax, vals, title in [(ax1, mdi, "Impurity importance (MDI, training data)"),
                        (ax2, pi_va, "Permutation importance (validation rows)")]:
    bars = ax.barh(range(k), vals[order_n], color=colors)
    for bar, j in zip(bars, order_n):
        if names_n[j] == "NOISE":
            bar.set_hatch("//")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)
ax1.set_yticks(range(k), [names_n[j] for j in order_n])
ax1.set_xlabel("share of total impurity decrease")
ax2.set_xlabel("drop in validation R^2 when shuffled")
fig.suptitle(f"Pure noise ranks {ranks['mdi']}th of {k} by impurity, last by permutation")
fig.tight_layout()
fig.savefig(IMG / "b3_4_noise_feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b3_4_noise_feature_importance.png")

# %%
# Try this (not graded):
# 1. Duplicate MedInc as a ninth column and recompute permutation importance. What happens to MedInc's score?
# 2. Draw ICE curves for AveOccup (kind="individual"). Are the curves parallel, or do they cross?
# 3. Explain the same district with TreeExplainer(model, data=background, feature_perturbation="interventional").
#    How different are the two sets of SHAP values?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
