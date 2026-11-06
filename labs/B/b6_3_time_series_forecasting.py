# %% [markdown]
# # B6.3 Time-series forecasting: trend, seasonality, lag features, baselines, walk-forward validation — lab
# Lesson: lessons/B/b6_3_time_series_forecasting.md
# Run: uv run python labs/B/b6_3_time_series_forecasting.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
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
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Split = tuple[IntArray, IntArray]
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
OUT = ROOT / "outputs" / "b6_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: nothing here needs a GPU or gets bigger with one; the series is 526 months long.
GPU_SCALE = False
PERIOD = 12  # months in a seasonal cycle (m in the lesson)
HORIZON = 12  # forecast the next 12 months from each origin (h)
N_FOLDS = 5  # walk-forward folds: the years 1997-2001 are forecast one at a time

# %%
# Worked example by hand (lesson): a quarterly toy series, period 4, two years of history, one year to forecast.
section("worked example by hand")
toy = pd.Series([10.0, 16, 6, 12, 12, 18, 8, 14], index=pd.period_range("2022Q1", periods=8, freq="Q"))
truth = np.array([14.0, 20, 10, 16])
w = np.r_[0.5, np.ones(3), 0.5] / 4  # the 2x4 moving average
print(f"2x4 moving average centred on 2022Q3: {np.dot(w, toy.to_numpy()[0:5]):.2f}; on 2022Q4: {np.dot(w, toy.to_numpy()[1:6]):.2f}")
naive = np.full(4, toy.iloc[-1])
snaive = toy.to_numpy()[-4:]
lag_model = 2 * toy.to_numpy()[-4:] - toy.to_numpy()[-8:-4]  # y_t = y_{t-4} + (y_{t-4} - y_{t-8})
for name, f in [("naive", naive), ("seasonal naive", snaive), ("lag model 2*lag4 - lag8", lag_model)]:
    print(f"{name:<24} forecast {f.tolist()}  errors {np.abs(truth - f).tolist()}  MAE {np.mean(np.abs(truth - f)):.2f}")
future = pd.Series(np.nan, index=pd.period_range("2024Q1", periods=4, freq="Q"))  # the quarters to forecast
toy_ext = pd.concat([toy, future])
toy_df = pd.DataFrame({"y": toy_ext, "lag4": toy_ext.shift(4), "lag8": toy_ext.shift(8)})
toy_df["2*lag4 - lag8"] = 2 * toy_df["lag4"] - toy_df["lag8"]
print("lag features with shift() (NaN = not known); for 2024 every input is already known at the end of 2023:")
print(toy_df.to_string())

# %%
# Data: weekly Mauna Loa CO2 (ships with statsmodels, no download), averaged to months.
section("data")
weekly = sm.datasets.co2.load_pandas().data["co2"]
monthly = weekly.resample("MS").mean()
missing = monthly.index[monthly.isna()]
y = monthly.interpolate(method="linear")  # 5 empty months, all before 1965: far from any test fold
print(f"weekly {len(weekly)} rows {weekly.index[0].date()} to {weekly.index[-1].date()}, {int(weekly.isna().sum())} missing weeks")
print(f"monthly series: {len(y)} months, {y.index[0]:%Y-%m} to {y.index[-1]:%Y-%m}; filled by interpolation: "
      + ", ".join(f"{t:%Y-%m}" for t in missing))
print(f"first value {y.iloc[0]:.2f} ppm, last {y.iloc[-1]:.2f} ppm; mean change per year {(y.iloc[-1] - y.iloc[0]) / (len(y) / 12):.3f} ppm")

# %%
# Decomposition from scratch: trend = centred 2x12 moving average, seasonal = mean detrended value per month.
section("from scratch: decomposition")


def centred_ma(x: FloatArray, m: int) -> FloatArray:
    """Centred moving average of even period m (a 2 x m MA), NaN where the window does not fit. (n,) -> (n,)."""
    w = np.r_[0.5, np.ones(m - 1), 0.5] / m  # m + 1 weights, summing to 1
    out = np.full(len(x), np.nan)
    half = m // 2
    out[half:len(x) - half] = np.convolve(x, w, mode="valid")
    return out


def decompose(x: FloatArray, m: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Additive decomposition x = trend + seasonal + residual. (n,) -> three (n,) arrays."""
    trend = centred_ma(x, m)
    detrended = x - trend
    season_means = np.array([np.nanmean(detrended[j::m]) for j in range(m)])  # one value per position in the cycle
    season_means -= season_means.mean()  # seasonal effects sum to 0 over a cycle
    seasonal = np.tile(season_means, len(x) // m + 1)[: len(x)]
    return trend, seasonal, x - trend - seasonal


x = y.to_numpy()
trend, seasonal, resid = decompose(x, PERIOD)
month_names = [f"{t:%b}" for t in y.index[:PERIOD]]
print("seasonal effect by month (ppm): " + ", ".join(f"{mn} {v:+.2f}" for mn, v in zip(month_names, seasonal[:PERIOD])))
print(f"trend from {np.nanmin(trend):.2f} to {np.nanmax(trend):.2f} ppm; residual std {np.nanstd(resid):.3f} ppm")
dec = seasonal_decompose(y, model="additive", period=PERIOD)
np.testing.assert_allclose(trend, dec.trend.to_numpy(), rtol=1e-12, equal_nan=True)
np.testing.assert_allclose(seasonal, dec.seasonal.to_numpy(), rtol=1e-10, atol=1e-12)
print("matches statsmodels.tsa.seasonal.seasonal_decompose (trend and seasonal): OK")
var_y = np.nanvar(x[~np.isnan(trend)])
print(f"share of variance: trend {np.nanvar(trend) / var_y:.4f}, seasonal {np.var(seasonal[~np.isnan(trend)]) / var_y:.4f}, "
      f"residual {np.nanvar(resid) / var_y:.4f}")

# %%
# Walk-forward splits from scratch, checked against scikit-learn's TimeSeriesSplit.
section("from scratch: walk-forward splits")


def walk_forward(n: int, n_folds: int, horizon: int) -> list[Split]:
    """Expanding window. Fold f tests rows [n - (n_folds - f) * horizon, ... + horizon) and trains on all rows before."""
    starts = n - horizon * np.arange(n_folds, 0, -1)
    return [(np.arange(s), np.arange(s, s + horizon)) for s in starts]


splits = walk_forward(len(y), N_FOLDS, HORIZON)
theirs = list(TimeSeriesSplit(n_splits=N_FOLDS, test_size=HORIZON).split(x))
assert all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(splits, theirs))
print("walk_forward matches sklearn TimeSeriesSplit(n_splits=5, test_size=12): OK")
for f, (tr, te) in enumerate(splits):
    print(f"fold {f}: train {y.index[tr[0]]:%Y-%m}..{y.index[tr[-1]]:%Y-%m} ({len(tr)} months) -> test {y.index[te[0]]:%Y-%m}..{y.index[te[-1]]:%Y-%m}")

# %%
# Baselines, each forecasting the 12 months after the origin from data up to the origin only.
section("baselines under walk-forward validation")


def forecast_naive(hist: FloatArray, h: int) -> FloatArray:
    return np.full(h, hist[-1])


def forecast_seasonal_naive(hist: FloatArray, h: int) -> FloatArray:
    return np.array([hist[len(hist) - PERIOD + (j % PERIOD)] for j in range(h)])  # same month last year


def forecast_moving_average(hist: FloatArray, h: int) -> FloatArray:
    return np.full(h, hist[-PERIOD:].mean())  # mean of the last 12 months


def mae(a: FloatArray, b: FloatArray) -> float:
    return float(np.mean(np.abs(a - b)))


def evaluate(forecaster: Callable[[FloatArray, int], FloatArray]) -> list[float]:
    return [mae(x[te], forecaster(x[tr], len(te))) for tr, te in splits]


results: dict[str, list[float]] = {}
for name, fc in [("naive", forecast_naive), ("seasonal naive", forecast_seasonal_naive),
                 ("moving average (12)", forecast_moving_average)]:
    results[name] = evaluate(fc)
    print(f"{name:<20} mean MAE over the {N_FOLDS} folds {np.mean(results[name]):.3f} ppm")
tr0, te0 = splits[0]
print("fold 0 (1997), seasonal naive forecast for Jan-Mar: " + ", ".join(f"{v:.2f}" for v in forecast_seasonal_naive(x[tr0], 3))
      + "; actual: " + ", ".join(f"{v:.2f}" for v in x[te0][:3]))

# %%
# Lag and rolling features with shift(), and a test that fails if any feature can see the future.
section("lag features and the leakage test")


def make_features(s: pd.Series) -> pd.DataFrame:
    """Direct 12-step features: every input is at least HORIZON months older than the target month."""
    return pd.DataFrame({
        "lag_12": s.shift(12), "lag_13": s.shift(13), "lag_24": s.shift(24),
        "roll_mean_12": s.shift(12).rolling(12).mean(),  # mean of months t-23 .. t-12
        "month": s.index.month,
    }, index=s.index)


def make_features_leaky(s: pd.Series) -> pd.DataFrame:
    """The same, with two bugs that look harmless: an off-by-one shift and a centred window."""
    f = make_features(s)
    f["roll_mean_12"] = s.shift(11).rolling(12).mean()  # includes month t-11: only 11 months old
    f["smooth"] = s.rolling(5, center=True).mean().shift(12)  # centred: uses t-10 and t-11 too
    return f


def assert_no_future(features_fn: Callable[[pd.Series], pd.DataFrame], s: pd.Series, min_age: int,
                     n_checks: int = 20) -> None:
    """Scramble every value newer than t - min_age; the feature row of month t must not change."""
    g = np.random.default_rng(SEED)
    base = features_fn(s)
    for t in g.integers(40, len(s), size=n_checks):
        s2 = s.copy()
        s2.iloc[t - min_age + 1:] = g.normal(1000, 100, size=len(s) - (t - min_age + 1))  # garbage
        row1, row2 = base.iloc[t].to_numpy(float), features_fn(s2).iloc[t].to_numpy(float)
        if not np.allclose(row1, row2, equal_nan=True):
            bad = [c for c, a, b in zip(base.columns, row1, row2) if not np.isclose(a, b, equal_nan=True)]
            raise AssertionError(f"features of {s.index[t]:%Y-%m} use data newer than {min_age} months: {bad}")


feats = make_features(y)
print(f"features {list(feats.columns)}; first complete row {feats.dropna().index[0]:%Y-%m} (needs 24 months of history)")
print(feats.loc["1960-03-01":"1960-05-01"].round(3).to_string())
assert_no_future(make_features, y, HORIZON)
print(f"make_features: no feature uses data newer than {HORIZON} months: OK")
try:
    assert_no_future(make_features_leaky, y, HORIZON)
    print("make_features_leaky passed (it should not)")
except AssertionError as err:
    print(f"make_features_leaky fails the test, as it should: {err}")

# %%
# A linear model on the lag features, refitted at every walk-forward origin.
section("lag-feature model under walk-forward validation")


def design(f: pd.DataFrame) -> FloatArray:
    """Numeric features plus one-hot months (11 columns; January is the reference). -> (n, 15)."""
    months = pd.get_dummies(f["month"], prefix="m", drop_first=True, dtype=float)
    return np.column_stack([f[["lag_12", "lag_13", "lag_24", "roll_mean_12"]].to_numpy(), months.to_numpy()])


X_all = design(feats)
ok = ~np.isnan(X_all).any(axis=1)  # rows with complete features (from month 25 on)


def forecast_lag_model(tr: IntArray, te: IntArray) -> FloatArray:
    rows = tr[ok[tr]]
    model = LinearRegression().fit(X_all[rows], x[rows])
    return model.predict(X_all[te])  # test features only use months <= the origin (checked above)


preds_lag = [forecast_lag_model(tr, te) for tr, te in splits]
results["linear model on lag features"] = [mae(x[te], p) for (tr, te), p in zip(splits, preds_lag)]
print(f"{'method':<30}" + "".join(f"{y.index[te[0]].year:>7}" for _, te in splits) + f"{'mean':>8}")
for name, maes in results.items():
    print(f"{name:<30}" + "".join(f"{v:>7.3f}" for v in maes) + f"{np.mean(maes):>8.3f}")
coef = LinearRegression().fit(X_all[splits[-1][0][ok[splits[-1][0]]]], x[splits[-1][0][ok[splits[-1][0]]]]).coef_
print("last fold's coefficients on lag_12, lag_13, lag_24, roll_mean_12: " + ", ".join(f"{c:.3f}" for c in coef[:4]))

# %%
# The same model judged by a shuffled 5-fold split: a random split of a time series is optimistic.
section("random split versus walk-forward")
rows = np.flatnonzero(ok)
shuffled = []
for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(rows):
    m = LinearRegression().fit(X_all[rows[tr]], x[rows[tr]])
    shuffled.append(mae(x[rows[te]], m.predict(X_all[rows[te]])))
print(f"shuffled 5-fold MAE {np.mean(shuffled):.3f} ppm against walk-forward {np.mean(results['linear model on lag features']):.3f} ppm")
seasonal_naive_mae = np.mean(results["seasonal naive"])
print(f"skill vs seasonal naive (1 - MAE / MAE_snaive): {1 - np.mean(results['linear model on lag features']) / seasonal_naive_mae:.3f}")

# %%
# Key plot: the five forecast years, seasonal naive against the lag model.
section("key plot: walk-forward forecasts")
fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
window = y.loc["1995-01-01":]
ax.plot(window.index, window.to_numpy(), color="black", linewidth=1.2, label="actual")
for f, (tr, te) in enumerate(splits):
    ax.axvspan(y.index[te[0]], y.index[te[-1]], color="tab:gray", alpha=0.08 if f % 2 else 0.16)
    ax.plot(y.index[te], forecast_seasonal_naive(x[tr], len(te)), color="tab:orange", linestyle="--",
            marker="v", markersize=3, label="seasonal naive" if f == 0 else None)
    ax.plot(y.index[te], preds_lag[f], color="tab:blue", marker="o", markersize=3,
            label="linear model on lag features" if f == 0 else None)
ax.set_xlabel("month")
ax.set_ylabel("CO2 (ppm)")
ax.set_title("Seasonal naive lags the trend by a year; the lag model does not")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(IMG / "b6_3_walk_forward_forecasts.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b6_3_walk_forward_forecasts.png")

fig, axes = plt.subplots(4, 1, figsize=(8, 7), dpi=120, sharex=True)
for ax, (name, comp) in zip(axes, [("observed", x), ("trend", trend), ("seasonal", seasonal), ("residual", resid)]):
    ax.plot(y.index, comp, linewidth=0.8)
    ax.set_ylabel(f"{name} (ppm)")
axes[0].set_title("Mauna Loa CO2: a smooth trend, a 6 ppm yearly cycle, small residuals")
axes[-1].set_xlabel("month")
fig.tight_layout()
fig.savefig(OUT / "decomposition.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b6_3/decomposition.png")

# %%
# Try this (not graded):
# 1. Forecast 24 months ahead (HORIZON = 24). Which lag features must you drop, and what happens to the MAE?
# 2. Add a time-index feature (months since 1958) to the linear model. Does it help, and what would it do after 2001?
# 3. Replace LinearRegression by a random forest on the same features (B3.2). Why does it do so badly here?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
