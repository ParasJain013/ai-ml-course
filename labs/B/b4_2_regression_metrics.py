# %% [markdown]
# # B4.2 Regression metrics — lab
# Lesson: lessons/B/b4_2_regression_metrics.md
# Run: uv run python labs/B/b4_2_regression_metrics.py   (CPU, defaults: ~5 s; downloads 5.7 MB once)
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
OUT = ROOT / "outputs" / "b4_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the data is already full size.
GPU_SCALE = False
COSTS = [(1.0, 1.0), (1.0, 5.0)]  # (cost of a false positive, cost of a false negative)
N_STORES_DAYS = 2000  # rows of the synthetic sales data
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# Data files: the same checksummed copies as B3.3 and B4.1 (VERIFY links and licences, as of 2026-10).
HOUSING_TGZ = DATA / "california_housing" / "housing.tgz"
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"
ADULT_FILE = DATA / "adult" / "adult-all.csv"
ADULT_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
ADULT_SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]


def fetch(path: Path, url: str, sha256: str) -> bytes:
    """Download a file once into data/ and verify its checksum. -> the file's bytes."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(url, timeout=60).read()
        path.write_bytes(raw)
        print(f"~ downloaded {len(raw):,} bytes from {url}")  # first run only, hence ~
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{path} has the wrong sha256; delete it and rerun")
    return raw


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator, stratify: IntArray | None = None
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


# %%
# Worked example by hand: five targets, two models.
section("worked example by hand")


def mae(y: FloatArray, y_hat: FloatArray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y: FloatArray, y_hat: FloatArray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mape(y: FloatArray, y_hat: FloatArray) -> float:
    """Mean absolute percentage error, as a fraction (0.1 = 10%). Undefined when any y is 0."""
    return float(np.mean(np.abs(y - y_hat) / np.abs(y)))


def r2(y: FloatArray, y_hat: FloatArray) -> float:
    """1 - SSE / SST: the share of the variance around the mean that the model explains."""
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


y5 = np.array([10.0, 12.0, 14.0, 16.0, 48.0])
model_a = np.array([11.0, 12.0, 14.0, 16.0, 20.0])  # perfect on four, way off on the outlier
model_b = np.array([16.0, 18.0, 20.0, 22.0, 42.0])  # off by 6 everywhere
for name, pred in [("A", model_a), ("B", model_b)]:
    print(f"model {name}: errors {(y5 - pred).tolist()}   MAE {mae(y5, pred):.4f}  RMSE {rmse(y5, pred):.4f}  "
          f"MAPE {mape(y5, pred):.4f}  R^2 {r2(y5, pred):.4f}")
print(f"mean of y {y5.mean():.1f}, median {np.median(y5):.1f}:   constant = mean: MAE {mae(y5, np.full(5, y5.mean())):.4f}  "
      f"RMSE {rmse(y5, np.full(5, y5.mean())):.4f}   constant = median: MAE {mae(y5, np.full(5, np.median(y5))):.4f}  "
      f"RMSE {rmse(y5, np.full(5, np.median(y5))):.4f}")

# %%
# From scratch: the four metrics on California housing, for a model trained on squared error and one
# trained on absolute error. Each wins the metric it was trained for.
section("from scratch: MAE and RMSE pick different winners")
with tarfile.open(fileobj=io.BytesIO(fetch(HOUSING_TGZ, HOUSING_URL, HOUSING_SHA256))) as tar:
    member = tar.extractfile("housing.csv")
    assert member is not None
    housing = pd.read_csv(io.BytesIO(member.read()))
X_h = np.column_stack([housing["median_income"], housing["housing_median_age"],
                       housing["total_rooms"] / housing["households"], housing["total_bedrooms"] / housing["households"],
                       housing["population"], housing["population"] / housing["households"],
                       housing["latitude"], housing["longitude"]]).astype(np.float64)  # B3.3's 8 features, NaN kept
y_h = (housing["median_house_value"] / 100_000).to_numpy(dtype=np.float64)
h_tr, h_va, _ = train_val_test_split(len(y_h), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED))
preds = {}
for loss in ["squared_error", "absolute_error"]:
    model = HistGradientBoostingRegressor(loss=loss, max_iter=300, random_state=SEED).fit(X_h[h_tr], y_h[h_tr])
    preds[loss] = model.predict(X_h[h_va])
    p = preds[loss]
    print(f"trained on {loss:<14}: MAE {mae(y_h[h_va], p):.4f}  RMSE {rmse(y_h[h_va], p):.4f}  "
          f"MAPE {mape(y_h[h_va], p):.4f}  R^2 {r2(y_h[h_va], p):.4f}   mean prediction {p.mean():.4f}")
print(f"validation target: mean {y_h[h_va].mean():.4f}  median {np.median(y_h[h_va]):.4f}")

# %%
# From scratch: MAPE on a target with values near zero (daily unit sales of slow-moving items).
section("from scratch: MAPE and near-zero targets")
sales_gen = np.random.default_rng(SEED)
expected = sales_gen.gamma(shape=1.0, scale=20.0, size=N_STORES_DAYS)  # each item-day's true expected sales
sales = sales_gen.poisson(expected).astype(np.float64) + 0.0  # observed units, many small counts
forecast = expected  # the best possible forecast: the true expected value
print(f"sales: {int(np.sum(sales == 0))} zero days, {int(np.sum(sales <= 2))} days with 2 units or fewer, of {len(sales)}")
nz = sales > 0
print(f"perfect-expectation forecast, nonzero days only: MAE {mae(sales[nz], forecast[nz]):.4f}  "
      f"MAPE {mape(sales[nz], forecast[nz]):.4f}")
worst = np.argsort(-np.abs(sales[nz] - forecast[nz]) / sales[nz])[:3]
for i in worst:
    print(f"   actual {sales[nz][i]:.0f}  forecast {forecast[nz][i]:.2f}  -> absolute percentage error {abs(sales[nz][i] - forecast[nz][i]) / sales[nz][i]:.1f}")
for scale in [0.5, 0.8, 1.0, 1.2]:
    print(f"forecast x {scale}: MAE {mae(sales[nz], scale * forecast[nz]):.4f}  MAPE {mape(sales[nz], scale * forecast[nz]):.4f}")
print(f"sklearn MAPE with zeros included (it divides by max(|y|, eps)): {mean_absolute_percentage_error(sales, forecast):.3e}")

# %%
# A cost-weighted threshold: choose the classification threshold that minimizes expected cost.
section("cost-weighted threshold")
adult = pd.read_csv(io.BytesIO(fetch(ADULT_FILE, ADULT_URL, ADULT_SHA256)), header=None, names=ADULT_COLUMNS,
                    skipinitialspace=True)
X_a, y_a = adult[NUMERIC + CATEGORICAL], (adult["income"] == ">50K").to_numpy(dtype=np.int64)
a_tr, a_va, _ = train_val_test_split(len(y_a), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_a)
pre = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
clf = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))]).fit(X_a.iloc[a_tr], y_a[a_tr])
s_va, y_va = clf.predict_proba(X_a.iloc[a_va])[:, 1], y_a[a_va]  # the B4.1 model's validation scores


def cost_curve(y: IntArray, s: FloatArray, c_fp: float, c_fn: float, thresholds: FloatArray) -> FloatArray:
    """Average cost per row at each threshold (predict 1 when s >= t). -> (len(thresholds),)."""
    flagged = s[None, :] >= thresholds[:, None]  # (T, n)
    fp = np.sum(flagged & (y[None, :] == 0), axis=1)
    fn = np.sum(~flagged & (y[None, :] == 1), axis=1)
    return (c_fp * fp + c_fn * fn) / len(y)


grid = np.linspace(0.0, 1.0, 1001)
cost_curves = {}
for c_fp, c_fn in COSTS:
    curve = cost_curve(y_va, s_va, c_fp, c_fn, grid)
    cost_curves[(c_fp, c_fn)] = curve
    k = int(np.argmin(curve))
    t_theory = c_fp / (c_fp + c_fn)
    k05 = int(np.searchsorted(grid, 0.5))
    print(f"c_FP {c_fp:g}, c_FN {c_fn:g}: best threshold {grid[k]:.3f} (cost {curve[k]:.4f}/row)   theory "
          f"c_FP/(c_FP+c_FN) = {t_theory:.3f} (cost {curve[int(round(t_theory * 1000))]:.4f})   threshold 0.5: {curve[k05]:.4f}")
print(f"flag nobody: cost per row = c_FN x prevalence = {y_va.mean():.4f} x c_FN")

# %%
# With scikit-learn: the library metrics agree with ours.
section("with scikit-learn")
for loss, p in preds.items():
    yv = y_h[h_va]
    np.testing.assert_allclose([mean_absolute_error(yv, p), root_mean_squared_error(yv, p),
                                mean_absolute_percentage_error(yv, p), r2_score(yv, p)],
                               [mae(yv, p), rmse(yv, p), mape(yv, p), r2(yv, p)])
print("mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score match: OK")
print(f"r2_score of predicting the validation mean: {r2_score(y_h[h_va], np.full(len(h_va), y_h[h_va].mean())):.4f};  "
      f"of predicting the training mean: {r2_score(y_h[h_va], np.full(len(h_va), y_h[h_tr].mean())):.4f}")

# %%
# Key plot: expected cost per row against the threshold, for the two cost settings.
section("key plot: cost curves")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
for ((c_fp, c_fn), curve), style in zip(cost_curves.items(), ["-", "--"]):
    k = int(np.argmin(curve))
    line, = ax.plot(grid, curve, style, label=f"FN costs {c_fn:g}x a FP (best t = {grid[k]:.2f})")
    ax.axvline(c_fp / (c_fp + c_fn), color=line.get_color(), linestyle=":", linewidth=1)
    ax.plot(grid[k], curve[k], "o", color=line.get_color())
ax.set_xlabel("decision threshold t (flag if score >= t)")
ax.set_ylabel("average cost per row (cost units)")
ax.set_ylim(0, None)
ax.set_title("The best threshold moves with the cost of a miss")
ax.legend(loc="upper center", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b4_2_cost_threshold.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b4_2_cost_threshold.png (dotted lines: c_FP / (c_FP + c_FN))")

# %%
# Try this (not graded):
# 1. Train HistGradientBoostingRegressor(loss="quantile", quantile=0.9). What share of validation targets fall below its predictions?
# 2. Compute MAPE on house prices in dollars instead of $100,000s. Does it change? Does RMSE?
# 3. Set the cost of a false negative to 20. Where does the best threshold go, and how flat is the curve around it?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
