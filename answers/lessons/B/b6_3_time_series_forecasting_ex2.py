# %% [markdown]
# # B6.3 Time-series forecasting — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b6_3_time_series_forecasting_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Split = tuple[IntArray, IntArray]
SEED = 0
PERIOD, HORIZON, N_FOLDS = 12, 12, 5
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
s1 = np.array([20.0, 30, 25, 45, 24, 34, 29, 49])
truth = np.array([28.0, 38, 33, 53])
w = np.r_[0.5, 1, 1, 1, 0.5] / 4
for t in [3, 4]:  # months counted from 1
    tr = w @ s1[t - 3:t + 2]
    print(f"(a) month {t}: trend {tr:.2f}, detrended {s1[t - 1] - tr:+.2f}")
for name, f in [("naive", np.full(4, s1[-1])), ("seasonal naive", s1[-4:]), ("moving average (4)", np.full(4, s1[-4:].mean()))]:
    print(f"(b) {name:<18} {f.tolist()} MAE {np.mean(np.abs(truth - f)):.2f}")
ser = pd.Series(np.arange(1.0, 11.0))  # values = their own month number, so each output shows which months it used
print("(d) row 9 of s.shift(2).rolling(3).mean():", ser.shift(2).rolling(3).mean().iloc[9], "(mean of months 6, 7, 8)")
print("    row 9 of s.rolling(3).mean().shift(1):", ser.rolling(3).mean().shift(1).iloc[9], "(mean of months 7, 8, 9: month 9 is t-1)")

# %%
# from labs/B/b6_3_time_series_forecasting.py
weekly = sm.datasets.co2.load_pandas().data["co2"]
y = weekly.resample("MS").mean().interpolate(method="linear")
x = y.to_numpy()


def walk_forward(n: int, n_folds: int, horizon: int) -> list[Split]:
    starts = n - horizon * np.arange(n_folds, 0, -1)
    return [(np.arange(s), np.arange(s, s + horizon)) for s in starts]


def assert_no_future(features_fn: Callable[[pd.Series], pd.DataFrame], s: pd.Series, min_age: int,
                     n_checks: int = 20) -> None:
    g = np.random.default_rng(SEED)
    base = features_fn(s)
    for t in g.integers(40, len(s), size=n_checks):
        s2 = s.copy()
        s2.iloc[t - min_age + 1:] = g.normal(1000, 100, size=len(s) - (t - min_age + 1))
        row1, row2 = base.iloc[t].to_numpy(float), features_fn(s2).iloc[t].to_numpy(float)
        if not np.allclose(row1, row2, equal_nan=True):
            bad = [c for c, a, b in zip(base.columns, row1, row2) if not np.isclose(a, b, equal_nan=True)]
            raise AssertionError(f"features of {s.index[t]:%Y-%m} use data newer than {min_age} months: {bad}")


def make_features(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"lag_12": s.shift(12), "lag_13": s.shift(13), "lag_24": s.shift(24),
                         "roll_mean_12": s.shift(12).rolling(12).mean(), "month": s.index.month}, index=s.index)


def design(f: pd.DataFrame, lag_cols: list[str]) -> FloatArray:
    """Lag columns plus 11 month dummies (January = reference), built the same way for any row subset."""
    months = np.stack([(f["month"].to_numpy() == mth).astype(float) for mth in range(2, 13)], axis=1)
    return np.column_stack([f[lag_cols].to_numpy(float), months])


splits = walk_forward(len(y), N_FOLDS, HORIZON)


def mae(a: FloatArray, b: FloatArray) -> float:
    return float(np.mean(np.abs(a - b)))


# %%
section("(a) direct versus recursive 12-month forecasts")
DIRECT_COLS = ["lag_12", "lag_13", "lag_24", "roll_mean_12"]
X_dir = design(make_features(y), DIRECT_COLS)
ok_dir = ~np.isnan(X_dir).any(axis=1)


def make_features_1step(s: pd.Series) -> pd.DataFrame:
    """One-step features: the newest input is 1 month old."""
    return pd.DataFrame({"lag_1": s.shift(1), "lag_2": s.shift(2), "lag_12": s.shift(12), "lag_13": s.shift(13),
                         "month": s.index.month}, index=s.index)


ONE_COLS = ["lag_1", "lag_2", "lag_12", "lag_13"]


def recursive_forecast(tr: IntArray, te: IntArray) -> FloatArray:
    """Fit a 1-step model on the training months, then forecast te one month at a time, feeding forecasts back."""
    X1 = design(make_features_1step(y), ONE_COLS)
    rows = tr[~np.isnan(X1[tr]).any(axis=1)]
    model = LinearRegression().fit(X1[rows], x[rows])
    hist = y.iloc[: te[0]].copy()  # what is known at the origin
    out = []
    for t in te:
        ext = pd.concat([hist, pd.Series([np.nan], index=[y.index[t]])])  # add the month to forecast
        feats = design(make_features_1step(ext).iloc[[-1]], ONE_COLS)  # its features use hist only
        out.append(float(model.predict(feats)[0]))
        hist = pd.concat([hist, pd.Series([out[-1]], index=[y.index[t]])])  # the forecast becomes "data"
    return np.array(out)


def direct_forecast(tr: IntArray, te: IntArray) -> FloatArray:
    rows = tr[ok_dir[tr]]
    return LinearRegression().fit(X_dir[rows], x[rows]).predict(X_dir[te])


def snaive(tr: IntArray, te: IntArray) -> FloatArray:
    hist = x[tr]
    return np.array([hist[len(hist) - PERIOD + (j % PERIOD)] for j in range(len(te))])


fc = {"seasonal naive": snaive, "direct (lab)": direct_forecast, "recursive 1-step": recursive_forecast}
preds = {name: [f(tr, te) for tr, te in splits] for name, f in fc.items()}
print(f"{'method':<18}" + "".join(f"{y.index[te[0]].year:>7}" for _, te in splits) + f"{'mean':>8}")
for name, ps in preds.items():
    maes = [mae(x[te], p) for (_, te), p in zip(splits, ps)]
    print(f"{name:<18}" + "".join(f"{v:>7.3f}" for v in maes) + f"{np.mean(maes):>8.3f}")
err_by_h = np.mean([np.abs(x[te] - p) for (_, te), p in zip(splits, preds["recursive 1-step"])], axis=0)
err_by_h_dir = np.mean([np.abs(x[te] - p) for (_, te), p in zip(splits, preds["direct (lab)"])], axis=0)
print("recursive MAE by horizon h=1..12: " + " ".join(f"{v:.2f}" for v in err_by_h))
print("direct    MAE by horizon h=1..12: " + " ".join(f"{v:.2f}" for v in err_by_h_dir))

# %%
section("(b) MASE against in-sample seasonal naive")
for name, ps in preds.items():
    mases = []
    for (tr, te), p in zip(splits, ps):
        scale = np.mean(np.abs(x[tr][PERIOD:] - x[tr][:-PERIOD]))  # in-sample seasonal-naive MAE on the training months
        mases.append(mae(x[te], p) / scale)
    print(f"{name:<18} MASE per fold " + " ".join(f"{v:.3f}" for v in mases) + f"  mean {np.mean(mases):.3f}")

# %%
section("(c) the leakage test on the one-step features")
assert_no_future(make_features_1step, y, 1)
print("min_age=1: OK (every feature is at least 1 month old)")
try:
    assert_no_future(make_features_1step, y, 12)
    print("min_age=12: passed")
except AssertionError as err:
    print(f"min_age=12 fails: {err}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
