# %% [markdown]
# # B1.3 Feature engineering — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b1_3_feature_engineering_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
N_ORDERS = 6_000
CITIES = ["Bristol", "Leeds", "London", "Manchester"]
VEHICLES = ["bike", "car", "scooter"]


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# Exercise 1 check.
section("exercise 1 check")
x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
mu, sd = x.mean(), x.std()
print(f"(a) mean {mu}  sd sqrt({np.mean((x - mu) ** 2)}) = {sd:.4f}  z {show((x - mu) / sd)}  "
      f"min-max {show((x - 1) / 8)}")
for v in (0.0, 12.0):
    print(f"    new {v}: z {(v - mu) / sd:.4f}  min-max {(v - 1) / 8:.4f}")
cats = sorted({"pro", "basic", "team"})
print(f"(b) categories {cats}: " + ", ".join(f"{p} -> {[int(p == c) for c in cats]}"
                                            for p in ["pro", "basic", "team", "basic", "team", "enterprise"]))
t = pd.Timestamp("2026-12-31 23:15")
ang = 2 * np.pi * (t.month - 1) / 12
print(f"(c) {t}: day of week {t.dayofweek} ({t.day_name()})  weekend {int(t.dayofweek >= 5)}  "
      f"rush {int(17 <= t.hour <= 19)}  (sin, cos) {show(np.array([np.sin(ang), np.cos(ang)]))}  "
      f"days since 2026-12-01: {(t.normalize() - pd.Timestamp('2026-12-01')).days}")


def month_point(m: int) -> FloatArray:
    """Month 1..12 -> (sin, cos) on the unit circle. -> (2,)."""
    a = 2 * np.pi * (m - 1) / 12
    return np.array([np.sin(a), np.cos(a)])


print(f"(d) |Dec - Jan| = {np.linalg.norm(month_point(12) - month_point(1)):.4f}   "
      f"|Jun - Jul| = {np.linalg.norm(month_point(6) - month_point(7)):.4f}   2 sin(pi/12) = {2 * np.sin(np.pi / 12):.4f}")
plans = ["pro", "basic", "team", "basic"]
D = np.array([[1] + [int(p == c) for c in cats] for p in plans], dtype=np.float64)
print(f"(e) design matrix {D.astype(int).tolist()}   rank {np.linalg.matrix_rank(D)} of {D.shape[1]} columns   "
      f"col0 - col1 - col2 - col3 = {show(D[:, 0] - D[:, 1:].sum(axis=1))}")

# %%
# Exercise 2 (a): a cyclic encoder that fits in a ColumnTransformer.


class CyclicEncoder(TransformerMixin, BaseEstimator):
    """Integer column(s) with a known period -> (sin, cos) per column. (n, k) -> (n, 2k)."""

    # BaseEstimator gives get_params/set_params (read from __init__'s arguments), which ColumnTransformer
    # needs to clone the transformer; TransformerMixin gives fit_transform.

    def __init__(self, period: float) -> None:
        self.period = period

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> CyclicEncoder:
        self.n_features_in_ = np.asarray(X).shape[1]
        self.feature_names_in_ = np.array(X.columns) if isinstance(X, pd.DataFrame) else None
        return self  # nothing to learn: the period is known in advance

    def transform(self, X: pd.DataFrame | np.ndarray) -> FloatArray:
        a = 2 * np.pi * np.asarray(X, dtype=np.float64) / self.period  # (n, k)
        return np.hstack([np.sin(a), np.cos(a)])  # (n, 2k)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        names = self.feature_names_in_ if self.feature_names_in_ is not None else [f"x{j}" for j in range(self.n_features_in_)]
        return np.array([f"{c}_sin" for c in names] + [f"{c}_cos" for c in names])


section("exercise 2 (a): CyclicEncoder")
hours = pd.DataFrame({"hour": [0, 6, 12, 18, 23]})
enc = CyclicEncoder(24).fit(hours)
print(f"names {enc.get_feature_names_out().tolist()}")
print(f"hours {hours['hour'].tolist()} -> {show(enc.transform(hours))}")
assert np.allclose(enc.transform(pd.DataFrame({"hour": [0]})), enc.transform(pd.DataFrame({"hour": [24]})))
print("hour 0 and hour 24 encode identically: OK")

# %%
# The lab's generator, the same split, and the interaction feature (copied).
# from labs/B/b1_3_feature_engineering.py
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


def distance_by_vehicle(frame: pd.DataFrame) -> FloatArray:
    """Distance in its own column per vehicle (km if that vehicle, else 0). -> (n, 3)."""
    km = frame["distance_km"].to_numpy(np.float64)[:, None]  # (n, 1)
    is_v = (frame["vehicle"].to_numpy()[:, None] == np.array(VEHICLES)[None, :])  # (n, 3)
    return km * is_v  # (n, 3)


orders = make_orders(N_ORDERS, rng)
cut, cut2 = int(0.6 * len(orders)), int(0.8 * len(orders))
perm = rng.permutation(len(orders))
train = orders.iloc[perm[:cut]].reset_index(drop=True)
val = orders.iloc[perm[cut:cut2]].reset_index(drop=True)
for frame in (train, val):
    frame["hour"] = frame["order_time"].dt.hour


def other_date_features(frame: pd.DataFrame) -> FloatArray:
    """The lab's date features without the rush flag: weekend, month sin/cos, newness. -> (n, 4)."""
    t = pd.to_datetime(frame["order_time"])
    angle = 2 * np.pi * (t.dt.month.to_numpy() - 1) / 12
    days = (t.dt.normalize() - pd.to_datetime(frame["opened_date"])).dt.days.to_numpy()
    return np.column_stack([(t.dt.dayofweek.to_numpy() >= 5).astype(np.float64), np.sin(angle), np.cos(angle),
                            np.exp(-days / 60.0)])


def rush_flag(frame: pd.DataFrame) -> FloatArray:
    """1 from 17:00 to 19:59. -> (n, 1)."""
    h = frame["hour"].to_numpy()
    return ((h >= 17) & (h <= 19)).astype(np.float64)[:, None]


HOUR_ENCODINGS = {
    "raw hour (one number)": ("passthrough", ["hour"]),
    "cyclic hour (sin, cos)": (CyclicEncoder(24), ["hour"]),
    "one-hot hour": (OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["hour"]),
    "rush flag (lab)": (FunctionTransformer(rush_flag), ["hour"]),
}

section("exercise 2 (b): four encodings of the hour")
for name, (enc_h, cols) in HOUR_ENCODINGS.items():
    prep = ColumnTransformer([
        ("num", StandardScaler(), ["distance_km", "n_items"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city", "vehicle"]),
        ("date", FunctionTransformer(other_date_features), ["order_time", "opened_date"]),
        ("inter", FunctionTransformer(distance_by_vehicle), ["distance_km", "vehicle"]),
        ("hour", enc_h, cols),
    ])
    pipe = Pipeline([("prep", prep), ("reg", LinearRegression())]).fit(train, train["minutes"])
    width = pipe.named_steps["prep"].transform(val.head(1)).shape[1]
    mae = mean_absolute_error(val["minutes"], pipe.predict(val))
    print(f"{name:<24} {width:>2} columns   val MAE {mae:.3f} min")
print(f"noise floor 4 * sqrt(2 / pi) = {4 * np.sqrt(2 / np.pi):.3f} min")
by_hour = train.groupby("hour")["minutes"].mean().round(1)
print(f"mean minutes by hour (train): {by_hour.to_dict()}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
