# %% [markdown]
# # B1.3 Feature engineering — lab
# Lesson: lessons/B/b1_3_feature_engineering.md
# Run: uv run python labs/B/b1_3_feature_engineering.py   (CPU, defaults: ~5 s)
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
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

FloatArray = npt.NDArray[np.float64]
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
OUT = ROOT / "outputs" / "b1_3"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only makes the synthetic table bigger.
GPU_SCALE = False
N_ORDERS = 100_000 if GPU_SCALE else 6_000  # before the night-time thinning
CITIES = ["Bristol", "Leeds", "London", "Manchester"]
VEHICLES = ["bike", "car", "scooter"]

# %%
# Data: synthetic food-delivery orders. The target is delivery time in minutes, built from effects a
# linear model can only use if the features expose them: city, vehicle speed x distance, weekends,
# the evening rush, winter, and new restaurants that are still slow.
section("data")


def make_orders(n: int, gen: np.random.Generator) -> pd.DataFrame:
    """Generate n delivery orders with raw columns and the delivery time in minutes."""
    t = pd.Timestamp("2025-01-01") + pd.to_timedelta(gen.integers(0, 365 * 24 * 60, n), unit="min")
    hour = t.hour.to_numpy()
    keep = gen.random(n) < np.where((hour >= 10) & (hour <= 22), 1.0, 0.05)  # few orders at night
    t, n = t[keep], int(keep.sum())
    opened = t.normalize() - pd.to_timedelta(gen.integers(1, 3 * 365, n), unit="D")
    city = gen.choice(CITIES, n, p=[0.2, 0.2, 0.4, 0.2])
    vehicle = gen.choice(VEHICLES, n, p=[0.5, 0.2, 0.3])
    distance = np.round(gen.lognormal(np.log(3.0), 0.5, n), 2)
    items = gen.poisson(2.0, n) + 1
    base = pd.Series(city).map({"Bristol": 14.0, "Leeds": 12.0, "London": 20.0, "Manchester": 15.0}).to_numpy()
    per_km = pd.Series(vehicle).map({"bike": 4.0, "car": 2.0, "scooter": 2.5}).to_numpy()
    hour = t.hour.to_numpy()
    days_open = (t.normalize() - opened).days.to_numpy()
    minutes = (
        base
        + per_km * distance
        + 1.5 * items
        + 5.0 * (t.dayofweek.to_numpy() >= 5)  # weekends are busier
        + 8.0 * ((hour >= 17) & (hour <= 19))  # the evening rush
        + 3.0 * np.cos(2 * np.pi * (t.month.to_numpy() - 1) / 12)  # slower in winter (January peak)
        + 10.0 * np.exp(-days_open / 60)  # new restaurants are slow for the first months
        + gen.normal(0, 4.0, n)
    )
    return pd.DataFrame({"order_time": t, "opened_date": opened, "city": city, "vehicle": vehicle,
                         "distance_km": distance, "n_items": items, "minutes": np.round(minutes, 1)})


orders = make_orders(N_ORDERS, rng)
cut = int(0.6 * len(orders))
cut2 = int(0.8 * len(orders))
perm = rng.permutation(len(orders))
train, val, test = (orders.iloc[perm[:cut]].reset_index(drop=True), orders.iloc[perm[cut:cut2]].reset_index(drop=True),
                    orders.iloc[perm[cut2:]].reset_index(drop=True))
print(f"orders: {len(orders)} rows   train / val / test: {len(train)} / {len(val)} / {len(test)}")
print(orders.head(4).to_string())
print(f"minutes: mean {orders['minutes'].mean():.2f}  sd {orders['minutes'].std():.2f}")

# %%
# Worked example by hand: scale four numbers, one-hot four colours, turn one date into features.
section("worked example by hand")
x4 = np.array([2.0, 4.0, 6.0, 8.0])
mu4, sd4 = x4.mean(), x4.std()  # ddof=0, like StandardScaler
print(f"x = {x4.tolist()}   mean = {mu4}   deviations = {(x4 - mu4).tolist()}   "
      f"sd (divide by n) = sqrt({np.mean((x4 - mu4) ** 2)}) = {sd4:.4f}")
print(f"standardized = {show((x4 - mu4) / sd4)}   min-max = {show((x4 - x4.min()) / (x4.max() - x4.min()))}")
print(f"new value 10 -> standardized {(10 - mu4) / sd4:.4f}   min-max {(10 - x4.min()) / (x4.max() - x4.min()):.4f} "
      "(outside [0, 1]: the scaler only knows the training range)")
colours = np.array(["red", "blue", "red", "green"])
cats = sorted(set(colours.tolist()))
print(f"categories (sorted) = {cats}")
print(f"one-hot = {[[int(c == k) for k in cats] for c in colours]}   'purple' -> {[int('purple' == k) for k in cats]}")
d0 = pd.Timestamp("2026-10-17 18:30")
print(f"{d0}: day of week = {d0.dayofweek} ({d0.day_name()}, Monday = 0)   weekend = {int(d0.dayofweek >= 5)}   "
      f"hour = {d0.hour}   rush = {int(17 <= d0.hour <= 19)}")
ang0 = 2 * np.pi * (d0.month - 1) / 12
print(f"month = {d0.month}: angle 2*pi*(month - 1)/12 = {ang0:.4f}   (sin, cos) = {show(np.array([np.sin(ang0), np.cos(ang0)]))}")
print(f"December (12) and January (1): cos = {np.cos(2 * np.pi * 11 / 12):.4f} and {np.cos(0.0):.4f}   "
      f"raw month numbers 12 and 1 are 11 apart")
opened0 = pd.Timestamp("2026-09-01")
print(f"days since opening (2026-09-01) = {(d0.normalize() - opened0).days}")
print("interaction: distance 3.0 km on a bike -> [bike_km, car_km, scooter_km] = [3.0, 0.0, 0.0]")

# %%
# From scratch: scalers and a one-hot encoder with fit / transform.
section("from scratch: scalers and one-hot")


class StandardScalerScratch:
    """Learn mean and sd per column on fit; subtract and divide on transform. (n, d) -> (n, d)."""

    def fit(self, X: FloatArray) -> StandardScalerScratch:
        self.mean_ = X.mean(axis=0)  # (d,)
        sd = X.std(axis=0)  # (d,) ddof=0
        self.scale_ = np.where(sd == 0, 1.0, sd)  # a constant column stays 0, not NaN
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        return (X - self.mean_) / self.scale_  # (n, d) broadcast over rows


class MinMaxScalerScratch:
    """Map each column's training range to [0, 1]. (n, d) -> (n, d)."""

    def fit(self, X: FloatArray) -> MinMaxScalerScratch:
        self.min_, rng_ = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
        self.range_ = np.where(rng_ == 0, 1.0, rng_)
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        return (X - self.min_) / self.range_


class OneHotScratch:
    """One 0/1 column per category seen in training; unseen categories -> all zeros. (n, k) -> (n, sum of sizes)."""

    def fit(self, X: np.ndarray) -> OneHotScratch:
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]  # sorted, per column
        return self

    def transform(self, X: np.ndarray) -> FloatArray:
        blocks = [(X[:, [j]] == cats[None, :]).astype(np.float64) for j, cats in enumerate(self.categories_)]
        return np.hstack(blocks)  # (n, sum of len(cats)): (n, 1) == (1, k) broadcasts to (n, k)


num_cols, cat_cols = ["distance_km", "n_items"], ["city", "vehicle"]
Xn_tr, Xn_va = train[num_cols].to_numpy(np.float64), val[num_cols].to_numpy(np.float64)
ss = StandardScalerScratch().fit(Xn_tr)
print(f"standard scaler: mean_ {show(ss.mean_)}   scale_ {show(ss.scale_)}")
print(f"train after scaling: mean {show(ss.transform(Xn_tr).mean(0))}   sd {show(ss.transform(Xn_tr).std(0))}")
print(f"val after scaling (train statistics): mean {show(ss.transform(Xn_va).mean(0))}   sd "
      f"{show(ss.transform(Xn_va).std(0))}")
oh = OneHotScratch().fit(train[cat_cols].to_numpy())
print(f"one-hot categories: {[c.tolist() for c in oh.categories_]}   width {oh.transform(train[cat_cols].to_numpy()).shape[1]}")
new_row = np.array([["York", "bike"]], dtype=object)
print(f"unseen city 'York' on a bike -> {oh.transform(new_row).astype(int).tolist()}")
city_only = OneHotScratch().fit(train[["city"]].to_numpy()).transform(train[["city"]].to_numpy())  # (n, 4)
full = np.hstack([np.ones((len(train), 1)), city_only])  # intercept + 4 city columns
print(f"intercept + all 4 city columns: shape {full.shape}   rank {np.linalg.matrix_rank(full)}   "
      f"(the dummy variable trap: the city columns sum to the ones column)")

# %%
# From scratch: date features, an interaction, and the whole matrix.
section("from scratch: dates, interaction, full matrix")


def date_features(frame: pd.DataFrame) -> FloatArray:
    """Weekend flag, rush-hour flag, month as (sin, cos), log days since opening. -> (n, 5)."""
    t = pd.to_datetime(frame["order_time"])
    angle = 2 * np.pi * (t.dt.month.to_numpy() - 1) / 12
    days = (t.dt.normalize() - pd.to_datetime(frame["opened_date"])).dt.days.to_numpy()
    return np.column_stack([
        (t.dt.dayofweek.to_numpy() >= 5).astype(np.float64),
        ((t.dt.hour.to_numpy() >= 17) & (t.dt.hour.to_numpy() <= 19)).astype(np.float64),
        np.sin(angle),
        np.cos(angle),
        np.exp(-days / 60.0),  # "how new is the restaurant": 1 on opening day, ~0 after a year
    ])


def distance_by_vehicle(frame: pd.DataFrame) -> FloatArray:
    """Distance in its own column per vehicle (km if that vehicle, else 0). -> (n, 3)."""
    km = frame["distance_km"].to_numpy(np.float64)[:, None]  # (n, 1)
    is_v = (frame["vehicle"].to_numpy()[:, None] == np.array(VEHICLES)[None, :])  # (n, 3)
    return km * is_v  # (n, 3)


def build_matrix(frame: pd.DataFrame, scaler: StandardScalerScratch, encoder: OneHotScratch) -> FloatArray:
    """Fitted pieces -> [scaled numeric | one-hot | date features | interaction]. -> (n, 2 + 7 + 5 + 3)."""
    return np.hstack([
        scaler.transform(frame[num_cols].to_numpy(np.float64)),
        encoder.transform(frame[cat_cols].to_numpy()),
        date_features(frame),
        distance_by_vehicle(frame),
    ])


X_tr_mine, X_va_mine = build_matrix(train, ss, oh), build_matrix(val, ss, oh)
print(f"feature matrix: train {X_tr_mine.shape}   val {X_va_mine.shape}")
print(f"first train row: {show(X_tr_mine[0])}")

# Does each group of features help? Fit a linear model on growing subsets, score on validation.
y_tr, y_va = train["minutes"].to_numpy(), val["minutes"].to_numpy()
groups = {"numeric only": slice(0, 2), "+ one-hot city, vehicle": slice(0, 9),
          "+ date features": slice(0, 14), "+ distance x vehicle": slice(0, 17)}
val_mae: dict[str, float] = {}
for name, sl in groups.items():
    m = LinearRegression().fit(X_tr_mine[:, sl], y_tr)
    pred = m.predict(X_va_mine[:, sl])
    val_mae[name] = mean_absolute_error(y_va, pred)
    print(f"{name:<26} {sl.stop:>2} columns   val MAE {val_mae[name]:.3f} min   val R^2 {r2_score(y_va, pred):.4f}")
print(f"mean baseline: val MAE {mean_absolute_error(y_va, np.full(len(y_va), y_tr.mean())):.3f} min")
print(f"noise floor: the target has N(0, 4^2) noise, so even the true formula has MAE about "
      f"4 * sqrt(2 / pi) = {4 * np.sqrt(2 / np.pi):.3f} min")

# %%
# With scikit-learn: the same preprocessing as a ColumnTransformer inside a Pipeline.
section("with scikit-learn")
s_std = StandardScaler().fit(Xn_tr)
np.testing.assert_allclose(s_std.transform(Xn_va), ss.transform(Xn_va), rtol=1e-12)
print("StandardScaler matches StandardScalerScratch: OK")
mm, mm_s = MinMaxScaler().fit(Xn_tr), MinMaxScalerScratch().fit(Xn_tr)
np.testing.assert_allclose(mm.transform(Xn_va), mm_s.transform(Xn_va), rtol=1e-12, atol=1e-15)
print("MinMaxScaler matches MinMaxScalerScratch: OK")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[cat_cols])
np.testing.assert_array_equal(enc.transform(pd.DataFrame(new_row, columns=cat_cols)), oh.transform(new_row))
print("OneHotEncoder(handle_unknown='ignore') matches OneHotScratch, unseen 'York' included: OK")

DATE_NAMES = ["weekend", "rush", "month_sin", "month_cos", "newness"]
INTER_NAMES = [f"{v}_km" for v in VEHICLES]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("date", FunctionTransformer(date_features, feature_names_out=lambda tf, cols: np.array(DATE_NAMES)),
     ["order_time", "opened_date"]),
    ("inter", FunctionTransformer(distance_by_vehicle, feature_names_out=lambda tf, cols: np.array(INTER_NAMES)),
     ["distance_km", "vehicle"]),
])
model = Pipeline([("prep", preprocess), ("reg", LinearRegression())])
model.fit(train, y_tr)  # one call: every step is fitted on the training rows only
X_va_sk = model.named_steps["prep"].transform(val)
np.testing.assert_allclose(X_va_sk, X_va_mine, rtol=1e-12, atol=1e-12)
print(f"ColumnTransformer output {X_va_sk.shape} equals build_matrix: OK")
pred_pipe = model.predict(val)
print(f"pipeline val MAE {mean_absolute_error(y_va, pred_pipe):.3f} min   (from scratch "
      f"{val_mae['+ distance x vehicle']:.3f})")
print(f"feature names: {model.named_steps['prep'].get_feature_names_out().tolist()}")
one_order = pd.DataFrame([{"order_time": pd.Timestamp("2026-10-17 18:30"), "opened_date": pd.Timestamp("2026-09-01"),
                           "city": "York", "vehicle": "bike", "distance_km": 3.0, "n_items": 2}])
print(f"raw order from an unseen city -> predicted {model.predict(one_order)[0]:.2f} min")

# %%
# Key plot: validation error falls as each group of features exposes an effect to the linear model.
section("key plot")
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
names = list(val_mae)
vals = [val_mae[k] for k in names]
bars = ax.barh(names[::-1], vals[::-1], color="tab:blue")
for b_, v in zip(bars, vals[::-1]):
    ax.text(v + 0.1, b_.get_y() + b_.get_height() / 2, f"{v:.2f}", va="center")
ax.axvline(4 * np.sqrt(2 / np.pi), color="k", ls="--", lw=1.2, label="noise floor (MAE of the true formula)")
ax.set_xlim(0, max(vals) * 1.15)
ax.set(xlabel="validation MAE (minutes)",
       title=f"Same linear model, better features: MAE {vals[0]:.1f} -> {vals[-1]:.1f} min")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
path_png = IMG / "b1_3_features_vs_error.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace the (sin, cos) month pair with the raw month number 1..12. What happens to val MAE?
# 2. Replace exp(-days/60) with raw days since opening. Why does the linear model do worse?
# 3. Add OneHotEncoder(drop="first") and check the rank of [ones | city columns] again.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
