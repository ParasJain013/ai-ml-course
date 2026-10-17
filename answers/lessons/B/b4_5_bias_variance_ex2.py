# %% [markdown]
# # B4.5 Bias–variance — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b4_5_bias_variance_ex2.py   (CPU, ~60 s; needs the housing download)

# %%
from __future__ import annotations

import hashlib
import io
import tarfile
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, learning_curve
from sklearn.tree import DecisionTreeRegressor

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_SETS, N_TRAIN, NOISE_SD = 300, 30, 0.3
KS = [1, 2, 5, 10, 20, 30]
TRAIN_SIZES = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
HOUSING_TGZ = Path(__file__).resolve().parents[3] / "data" / "california_housing" / "housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
for name, preds in [("first", np.array([12.0, 8, 14, 6, 10])), ("second", np.array([13.0, 13, 12, 14, 13]))]:
    b2, v = (preds.mean() - 10) ** 2, preds.var()
    print(f"({'a' if name == 'first' else 'b'}) {name} model: mean {preds.mean():g}  bias^2 {b2:g}  variance {v:g}  "
          f"expected error {b2 + v + 4:g}")

# %%
section("(a) k-nearest-neighbour regression: bias and variance by k")
sim = np.random.default_rng(SEED)
train_x = sim.uniform(0, 1, size=(N_SETS, N_TRAIN))
train_y = np.sin(2 * np.pi * train_x) + sim.normal(0, NOISE_SD, size=train_x.shape)
x_test = np.linspace(0, 1, 200)
f_test = np.sin(2 * np.pi * x_test)
print(f"{'k':>3} {'bias^2':>8} {'variance':>9} {'sigma^2/k':>9} {'total':>7}")
for k in KS:
    P = np.empty((N_SETS, len(x_test)))
    for s in range(N_SETS):
        d = np.abs(x_test[:, None] - train_x[s][None, :])  # (200, 30) distances to the training points
        nearest = np.argsort(d, axis=1, kind="stable")[:, :k]  # (200, k)
        P[s] = train_y[s][nearest].mean(axis=1)
    b2 = np.mean((P.mean(0) - f_test) ** 2)
    v = np.mean(P.var(0))
    print(f"{k:>3} {b2:>8.4f} {v:>9.4f} {NOISE_SD ** 2 / k:>9.4f} {b2 + v + NOISE_SD ** 2:>7.4f}")

# %%
section("(b) learning curves: random forest versus full tree")
raw = HOUSING_TGZ.read_bytes()
if hashlib.sha256(raw).hexdigest() != HOUSING_SHA256:
    raise ValueError(f"{HOUSING_TGZ} has the wrong checksum; delete it and rerun the lab")
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    member = tar.extractfile("housing.csv")
    assert member is not None
    h = pd.read_csv(io.BytesIO(member.read()))
X = np.column_stack([h["median_income"], h["housing_median_age"], h["total_rooms"] / h["households"],
                     h["total_bedrooms"] / h["households"], h["population"], h["population"] / h["households"],
                     h["latitude"], h["longitude"]]).astype(np.float64)
y = (h["median_house_value"] / 100_000).to_numpy(dtype=np.float64)
dev = np.random.default_rng(SEED).permutation(len(y))[: round(0.8 * len(y))]  # the lab's development rows
cv = KFold(5, shuffle=True, random_state=SEED)
for name, model in [("full-depth tree", DecisionTreeRegressor(random_state=SEED)),
                    ("random forest", RandomForestRegressor(n_estimators=100, max_features=0.33, random_state=SEED, n_jobs=4))]:
    sizes, tr, va = learning_curve(model, X[dev], y[dev], train_sizes=TRAIN_SIZES, cv=cv, scoring="neg_root_mean_squared_error")
    print(f"{name}:")
    for s, t, v in zip(sizes, -tr.mean(1), -va.mean(1)):
        print(f"   {s:>6} rows: train RMSE {t:.4f}   val RMSE {v:.4f}   gap {v - t:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
