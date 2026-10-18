# %% [markdown]
# # B4.3 Cross-validation and tuning — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b4_3_cross_validation_tuning_ex2.py   (CPU, ~10 s)

# %%
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_DAYS, N_HOLDOUT = 1000, 200


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
y12 = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
g12 = np.array([1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5])
X12 = np.zeros((12, 1))
print("(a) KFold(4):          " + "  ".join(str(te.tolist()) for _, te in KFold(4).split(X12)))
print("(a) StratifiedKFold(3): " + "  ".join(str(te.tolist()) for _, te in StratifiedKFold(3).split(X12, y12)))
print("(a) TimeSeriesSplit(3): " + "  ".join(str(te.tolist()) for _, te in TimeSeriesSplit(3).split(X12)))
# Step 3 by hand: largest group first, each to the emptiest fold
uniq, counts = np.unique(g12, return_counts=True)
load, assign = np.zeros(3, dtype=int), {}
for g in uniq[np.argsort(-counts, kind="stable")]:
    f = int(np.argmin(load))
    assign[int(g)] = f
    load[f] += counts[uniq == g][0]
print(f"(b) Step 3 rule: customer -> fold {assign}   fold sizes {load.tolist()}")
print("    GroupKFold(3) test rows: " + "  ".join(str(te.tolist()) for _, te in GroupKFold(3).split(X12, groups=g12)))
c20 = np.random.default_rng(SEED).normal(size=(200_000, 20)).max(axis=1).mean()
print(f"(c) expected max of 20 standard normals {c20:.3f} -> optimism about {c20 * 0.004:.4f} RMSE")

# %%
section("a daily series with lag features")
gen = np.random.default_rng(SEED)
day = np.arange(N_DAYS)
series = 50 + 0.05 * day + 8 * np.sin(2 * np.pi * day / 7) + gen.normal(0, 3, N_DAYS)  # trend + weekly cycle + noise
df = pd.DataFrame({"y": series})
for lag in (1, 7, 14):
    df[f"lag_{lag}"] = df["y"].shift(lag)
df["mean_7"] = df["y"].shift(1).rolling(7).mean()  # trailing mean of the 7 days before
df["dow"] = day % 7
df = df.dropna()
X = df.drop(columns="y").to_numpy()
y = df["y"].to_numpy()
n_dev = len(y) - N_HOLDOUT
X_dev, y_dev, X_hold, y_hold = X[:n_dev], y[:n_dev], X[n_dev:], y[n_dev:]
print(f"rows after dropping incomplete lags {len(y)}   development {n_dev}   holdout (last {N_HOLDOUT} days)")


def fit_rmse(tr: npt.NDArray[np.int64], te: npt.NDArray[np.int64], X_: FloatArray, y_: FloatArray) -> float:
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=1, random_state=SEED,
                          deterministic=True, force_row_wise=True, verbose=-1).fit(X_[tr], y_[tr])
    return float(np.sqrt(np.mean((y_[te] - m.predict(X_[te])) ** 2)))


section("cross-validated estimates versus the future")
estimates = {
    "shuffled KFold(5)": KFold(5, shuffle=True, random_state=SEED),
    "TimeSeriesSplit(5)": TimeSeriesSplit(5),
    "TimeSeriesSplit(5, gap=7)": TimeSeriesSplit(5, gap=7),
}
for name, cv in estimates.items():
    scores = [fit_rmse(tr, te, X_dev, y_dev) for tr, te in cv.split(X_dev)]
    print(f"{name:<26} CV RMSE {np.mean(scores):.4f}   folds {np.round(scores, 3).tolist()}")
all_dev = np.arange(n_dev)
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=1, random_state=SEED,
                      deterministic=True, force_row_wise=True, verbose=-1).fit(X_dev, y_dev)
hold_rmse = float(np.sqrt(np.mean((y_hold - m.predict(X_hold)) ** 2)))
print(f"{'holdout: last 200 days':<26} RMSE {hold_rmse:.4f}")
print(f"training targets range {y_dev.min():.1f}..{y_dev.max():.1f}   holdout targets range {y_hold.min():.1f}..{y_hold.max():.1f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
