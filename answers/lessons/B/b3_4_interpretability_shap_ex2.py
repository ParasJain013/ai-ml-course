# %% [markdown]
# # B3.4 Interpretability — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b3_4_interpretability_shap_ex2.py   (CPU, ~30 s; needs the B3.3/B3.4 download)

# %%
from __future__ import annotations

import hashlib
import io
import itertools
import math
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
import xgboost as xgb

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_REPEATS = 10
HOUSING_TGZ = Path(__file__).resolve().parents[3] / "data" / "california_housing" / "housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]


def section(name: str) -> None:
    print(f"\n== {name} ==")


def r2(y: FloatArray, y_hat: FloatArray) -> float:
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


# %%
section("exercise 1 check")


def shapley_exact(value: Callable[[tuple[int, ...]], float], n_feat: int) -> FloatArray:
    phi = np.zeros(n_feat)
    for j in range(n_feat):
        others = [k for k in range(n_feat) if k != j]
        for size in range(n_feat):
            w = math.factorial(size) * math.factorial(n_feat - size - 1) / math.factorial(n_feat)
            for S in itertools.combinations(others, size):
                phi[j] += w * (value(tuple(sorted(S + (j,)))) - value(S))
    return phi


x = np.array([2.0, 1.0, 1.0])


def f1(z: FloatArray) -> float:
    return float(3 * z[0] + 2 * z[1] + 6 * z[1] * z[2])


vals = {}
for size in range(4):
    for S in itertools.combinations(range(3), size):
        z = np.zeros(3)
        z[list(S)] = x[list(S)]
        vals[S] = f1(z)
print("(a) v(S): " + "  ".join(f"{'{' + ','.join(str(k + 1) for k in S) + '}'}: {v:g}" for S, v in vals.items()))
phi = shapley_exact(lambda S: vals[S], 3)
print(f"    phi {phi.round(4).tolist()}   sum {phi.sum():g} = f(x) - v(empty) = {f1(x) - vals[()]:g}")
rows = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, -1.0]])
for v in [0.0, 1.0, 2.0]:
    ice = v * rows[:, 1] + 0.0  # + 0.0 turns -0.0 into 0.0
    print(f"(b) v = {v:g}: ICE {ice.tolist()}   PD {ice.mean():g}")

# %%
section("data and the B3.3 model")
if not HOUSING_TGZ.exists():
    raise FileNotFoundError("Run labs/B/b3_3_gradient_boosting.py once first: it downloads the housing data.")
if hashlib.sha256(HOUSING_TGZ.read_bytes()).hexdigest() != HOUSING_SHA256:
    raise ValueError(f"{HOUSING_TGZ} has the wrong checksum; delete it and rerun the B3.3 lab")
with tarfile.open(HOUSING_TGZ) as tar:
    member = tar.extractfile("housing.csv")
    assert member is not None
    df = pd.read_csv(io.BytesIO(member.read()))
X_raw = np.column_stack([df["median_income"], df["housing_median_age"], df["total_rooms"] / df["households"],
                         df["total_bedrooms"] / df["households"], df["population"],
                         df["population"] / df["households"], df["latitude"], df["longitude"]]).astype(np.float64)
y_all = (df["median_house_value"] / 100_000).to_numpy(dtype=np.float64)
gen = np.random.default_rng(SEED)  # the B3.3 split
rows_perm = gen.permutation(len(y_all))
n_te = n_va = round(0.2 * len(y_all))
i_tr = gen.permutation(rows_perm[n_te + n_va:])
i_va = gen.permutation(rows_perm[n_te:n_te + n_va])
perm = np.random.default_rng(SEED + 1).permutation(len(i_tr))
n_es = round(0.2 * len(i_tr))
i_es, i_fit = i_tr[perm[:n_es]], i_tr[perm[n_es:]]


def fit_xgb(X: FloatArray) -> xgb.XGBRegressor:
    """The B3.3 recipe on the given feature matrix (all rows; split inside)."""
    m = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=100, eval_metric="rmse", n_jobs=1, random_state=SEED)
    return m.fit(X[i_fit], y_all[i_fit], eval_set=[(X[i_es], y_all[i_es])], verbose=False)


def group_permutation_importance(predict: Callable[[FloatArray], FloatArray], X: FloatArray, y: FloatArray,
                                 groups: list[list[int]], n_repeats: int, gen: np.random.Generator
                                 ) -> tuple[FloatArray, FloatArray]:
    """R^2 drop when all columns of a group are shuffled with the same permutation. -> mean, sd per group."""
    baseline = r2(y, predict(X))
    drops = np.zeros((len(groups), n_repeats))
    for g, cols in enumerate(groups):
        for k in range(n_repeats):
            Xp = X.copy()
            idx = gen.permutation(len(X))
            Xp[:, cols] = X[idx][:, cols]  # rows keep their (lat, long) pairs, just moved to other districts
            drops[g, k] = baseline - r2(y, predict(Xp))
    return drops.mean(axis=1), drops.std(axis=1)


m8 = fit_xgb(X_raw)
X_va, y_va = X_raw[i_va], y_all[i_va]
print(f"8 features: {m8.best_iteration + 1} trees   val R^2 {r2(y_va, m8.predict(X_va)):.4f}")

# %%
section("(a) a duplicated feature")
X9 = np.column_stack([X_raw, X_raw[:, 0]])  # column 8 = exact copy of MedInc
m9 = fit_xgb(X9)
print(f"9 features (MedInc twice): {m9.best_iteration + 1} trees   val R^2 {r2(y_va, m9.predict(X9[i_va])):.4f}")
singles8, _ = group_permutation_importance(m8.predict, X_va, y_va, [[j] for j in range(8)], N_REPEATS,
                                           np.random.default_rng(SEED))
singles9, _ = group_permutation_importance(m9.predict, X9[i_va], y_va, [[j] for j in range(9)], N_REPEATS,
                                           np.random.default_rng(SEED))
both9, _ = group_permutation_importance(m9.predict, X9[i_va], y_va, [[0, 8]], N_REPEATS, np.random.default_rng(SEED))
sv8 = np.abs(shap.TreeExplainer(m8).shap_values(X_va)).mean(axis=0)
sv9_raw = shap.TreeExplainer(m9).shap_values(X9[i_va])
sv9 = np.abs(sv9_raw).mean(axis=0)
print(f"{'':<26} {'perm. importance':>16} {'mean |SHAP|':>12}")
print(f"{'MedInc, 8-feature model':<26} {singles8[0]:>16.4f} {sv8[0]:>12.4f}")
print(f"{'MedInc, 9-feature model':<26} {singles9[0]:>16.4f} {sv9[0]:>12.4f}")
print(f"{'MedInc_copy':<26} {singles9[8]:>16.4f} {sv9[8]:>12.4f}")
print(f"{'both copies shuffled':<26} {both9[0]:>16.4f} {np.abs(sv9_raw[:, 0] + sv9_raw[:, 8]).mean():>12.4f}")

# %%
section("(b) grouped permutation importance")
lat, lon = FEATURES.index("Latitude"), FEATURES.index("Longitude")
grp, grp_sd = group_permutation_importance(m8.predict, X_va, y_va, [[lat, lon]], N_REPEATS, np.random.default_rng(SEED))
print(f"Latitude alone {singles8[lat]:.4f}   Longitude alone {singles8[lon]:.4f}   sum {singles8[lat] + singles8[lon]:.4f}")
print(f"Latitude + Longitude shuffled together: {grp[0]:.4f} +- {grp_sd[0]:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
