# %% [markdown]
# # B6.4 ARIMA and exponential smoothing versus gradient boosting on lag features — lab
# Lesson: lessons/B/b6_4_arima_vs_boosting.md
# Run: uv run python labs/B/b6_4_arima_vs_boosting.py   (CPU, defaults: ~60 s)
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
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Split = tuple[IntArray, IntArray]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
# statsmodels warns about optimizer convergence on some folds; the forecasts are still checked by their MAE
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*(Non-stationary|Non-invertible) starting.*")


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
OUT = ROOT / "outputs" / "b6_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: nothing here needs a GPU; GPU_SCALE only lengthens the synthetic demand series and adds folds.
GPU_SCALE = False
N_FOLDS_MONTHLY, H_MONTHLY, M_MONTHLY = 5, 12, 12  # CO2: forecast 1997..2001 one year at a time (as in B6.3)
N_DAYS = 3 * 365 * (3 if GPU_SCALE else 1)  # synthetic daily demand
N_FOLDS_DAILY, H_DAILY, M_DAILY = (16 if GPU_SCALE else 8), 14, 7
N_THREADS = 1  # deterministic LightGBM results on every machine
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8,
                  subsample_freq=1, random_state=SEED, n_jobs=N_THREADS, deterministic=True, force_row_wise=True,
                  verbose=-1)

# %%
# Worked example by hand (lesson): simple exponential smoothing, and an ARIMA(1,1,0) forecast.
section("worked example by hand")
y_toy = np.array([10.0, 12.0, 11.0, 15.0])
alpha = 0.5
level = y_toy[0]  # initial level = first value
one_step = []
for t, v in enumerate(y_toy, start=1):
    one_step.append(level)  # forecast of y_t made at t-1
    new_level = alpha * v + (1 - alpha) * level
    print(f"t={t}: forecast {level:g}, actual {v:g}, error {v - level:+g}, new level {alpha:g}*{v:g} + {1 - alpha:g}*{level:g} = {new_level:g}")
    level = new_level
print(f"SSE of the one-step errors: {sum((v - f) ** 2 for v, f in zip(y_toy, one_step)):g}; forecast for t=5, 6, ...: {level:g} (flat)")
diffs = np.diff(y_toy)
phi = 0.5
d5 = phi * diffs[-1]
d6 = phi * d5
print(f"differences {diffs.tolist()}; ARIMA(1,1,0) with phi={phi}: d5 = {d5:g} -> y5 = {y_toy[-1] + d5:g}; "
      f"d6 = {d6:g} -> y6 = {y_toy[-1] + d5 + d6:g}")

# %%
# Data: the B6.3 CO2 series (long), the same series from 1992 only (short), and a synthetic daily demand series.
section("data")
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate(method="linear")
co2_short = co2.loc["1992-01-01":]
print(f"CO2 long: {len(co2)} months {co2.index[0]:%Y-%m}..{co2.index[-1]:%Y-%m}; short: {len(co2_short)} months from {co2_short.index[0]:%Y-%m}")

# Synthetic: a flat level, a weekly pattern, promotions planned in advance, and a non-linear heat effect.
g = np.random.default_rng(SEED)
days = pd.date_range("2022-01-01", periods=N_DAYS, freq="D")
doy = days.dayofyear.to_numpy()
temp = 18 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + g.normal(0, 3, N_DAYS)  # degrees C
promo = (g.random(N_DAYS) < 0.10).astype(float)  # about 1 day in 10, decided weeks ahead
weekday_effect = np.array([0, 5, 10, 15, 40, 60, -20])[days.dayofweek.to_numpy()]  # Mon..Sun
heat = 2.0 * np.maximum(temp - 24, 0) ** 2  # nothing below 24 C, then fast growth
demand = 300 + weekday_effect + 80 * promo + heat + g.normal(0, 15, N_DAYS)
daily = pd.DataFrame({"y": demand, "temp": temp, "promo": promo}, index=days)
print(f"synthetic daily demand: {N_DAYS} days {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}; mean {demand.mean():.1f}, "
      f"promo days {int(promo.sum())}, days above 24 C {int((temp > 24).sum())}")
print("(the temperature of a future day is treated as known: in real use it would be the weather forecast)")

# %%
# Simple exponential smoothing from scratch, checked against statsmodels.
section("from scratch: simple exponential smoothing")


def ses_one_step(x: FloatArray, alpha: float, level0: float) -> tuple[FloatArray, float]:
    """One-step-ahead forecasts (n,) of SES and the final level. forecast_t = level_{t-1}."""
    f = np.empty(len(x))
    level = level0
    for t in range(len(x)):
        f[t] = level
        level = alpha * x[t] + (1 - alpha) * level
    return f, level


def fit_ses(x: FloatArray) -> tuple[float, float]:
    """Choose alpha in [0, 1] minimizing the SSE of one-step errors; level0 = x[0]. Returns (alpha, final level)."""
    res = minimize_scalar(lambda a: float(np.sum((x - ses_one_step(x, a, x[0])[0]) ** 2)), bounds=(0, 1), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.x), ses_one_step(x, float(res.x), x[0])[1]


x_co2 = co2.to_numpy()
mine, _ = ses_one_step(x_co2, 0.3, x_co2[0])
sm_ses = SimpleExpSmoothing(co2, initialization_method="known", initial_level=x_co2[0]).fit(smoothing_level=0.3, optimized=False)
np.testing.assert_allclose(mine, sm_ses.fittedvalues.to_numpy(), rtol=1e-12)
print("alpha=0.3: one-step forecasts match statsmodels SimpleExpSmoothing (known initial level): OK")
a_co2, _ = fit_ses(x_co2)
level_only = 50 + g.normal(0, 1, 300).cumsum() * 0.2 + g.normal(0, 2, 300)  # a slowly wandering level plus noise
a_lvl, _ = fit_ses(level_only)
a_sm = SimpleExpSmoothing(level_only, initialization_method="known", initial_level=level_only[0]).fit().params["smoothing_level"]
print(f"fitted alpha on a noisy wandering level: mine {a_lvl:.4f}, statsmodels {a_sm:.4f}")
print(f"fitted alpha on CO2: {a_co2:.4f} (SES has no trend or season, so it copies the last value: naive)")

# %%
# The contenders, each a function (training series, horizon) -> forecast, on the same walk-forward folds as B6.3.
section("the contenders")


def walk_forward(n: int, n_folds: int, horizon: int) -> list[Split]:
    """From labs/B/b6_3_time_series_forecasting.py: expanding window, test blocks of `horizon` at the end."""
    starts = n - horizon * np.arange(n_folds, 0, -1)
    return [(np.arange(s), np.arange(s, s + horizon)) for s in starts]


def seasonal_naive(m: int) -> Callable[[pd.Series, int], FloatArray]:
    return lambda hist, h: np.array([hist.iloc[len(hist) - m + (j % m)] for j in range(h)])


def ses_forecast(hist: pd.Series, h: int) -> FloatArray:
    return np.full(h, fit_ses(hist.to_numpy())[1])


def holt_winters(m: int, damped: bool = False) -> Callable[[pd.Series, int], FloatArray]:
    def fc(hist: pd.Series, h: int) -> FloatArray:
        model = ExponentialSmoothing(hist.to_numpy(), trend="add", damped_trend=damped, seasonal="add",
                                     seasonal_periods=m, initialization_method="estimated").fit()
        return model.forecast(h)
    return fc


def sarima(order: tuple[int, int, int], seasonal: tuple[int, int, int, int]) -> Callable[[pd.Series, int], FloatArray]:
    def fc(hist: pd.Series, h: int) -> FloatArray:
        return SARIMAX(hist.to_numpy(), order=order, seasonal_order=seasonal).fit(disp=False).forecast(h)
    return fc


def lag_table(s: pd.Series, lags: list[int], roll_lag: int, roll_len: int) -> pd.DataFrame:
    """B6.3's lag features: every lag >= the horizon, plus a rolling mean ending roll_lag steps ago."""
    f = pd.DataFrame({f"lag_{k}": s.shift(k) for k in lags}, index=s.index)
    f[f"roll_{roll_len}"] = s.shift(roll_lag).rolling(roll_len).mean()
    return f


def boosting_monthly(difference: bool) -> Callable[[pd.Series, int], FloatArray]:
    """LightGBM on lags >= 12. With difference=True the target is y_t - y_{t-12}, added back to y_{t-12} afterwards."""
    def fc(hist: pd.Series, h: int) -> FloatArray:
        idx = pd.date_range(hist.index[0], periods=len(hist) + h, freq="MS")
        s = pd.concat([hist, pd.Series(np.nan, index=idx[len(hist):])])
        target = s - s.shift(12) if difference else s
        X = lag_table(target, [12, 13, 24], 12, 12)
        X["month"] = s.index.month
        Xv, tv = X.to_numpy(float), target.to_numpy()
        train = ~np.isnan(Xv).any(axis=1) & ~np.isnan(tv)
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xv[train], tv[train])
        pred = model.predict(Xv[-h:])
        return pred + s.shift(12).to_numpy()[-h:] if difference else pred
    return fc


def linear_monthly(hist: pd.Series, h: int) -> FloatArray:
    """B6.3's linear model on lags >= 12 with month dummies."""
    idx = pd.date_range(hist.index[0], periods=len(hist) + h, freq="MS")
    s = pd.concat([hist, pd.Series(np.nan, index=idx[len(hist):])])
    X = lag_table(s, [12, 13, 24], 12, 12)
    months = np.stack([(s.index.month == mth).astype(float) for mth in range(2, 13)], axis=1)
    Xv = np.column_stack([X.to_numpy(float), months])
    train = ~np.isnan(Xv).any(axis=1) & ~np.isnan(s.to_numpy())
    return LinearRegression().fit(Xv[train], s.to_numpy()[train]).predict(Xv[-h:])


MONTHLY = {
    "seasonal naive": seasonal_naive(M_MONTHLY),
    "SES (from scratch)": ses_forecast,
    "Holt-Winters additive": holt_winters(M_MONTHLY),
    "SARIMA(1,1,1)(0,1,1)12": sarima((1, 1, 1), (0, 1, 1, 12)),
    "linear on lags (B6.3)": linear_monthly,
    "LightGBM on lags": boosting_monthly(difference=False),
    "LightGBM on 12-month differences": boosting_monthly(difference=True),
}
print("monthly contenders: " + "; ".join(MONTHLY))


def run(series: pd.Series, methods: dict[str, Callable[[pd.Series, int], FloatArray]], n_folds: int, h: int
        ) -> tuple[dict[str, list[float]], dict[str, list[FloatArray]]]:
    """Walk-forward MAE per fold for every method, and the forecasts themselves."""
    splits = walk_forward(len(series), n_folds, h)
    maes: dict[str, list[float]] = {}
    fcs: dict[str, list[FloatArray]] = {}
    for name, fc in methods.items():
        fcs[name] = [np.asarray(fc(series.iloc[tr], len(te)), dtype=float) for tr, te in splits]
        maes[name] = [float(np.mean(np.abs(series.to_numpy()[te] - f))) for (_, te), f in zip(splits, fcs[name])]
    return maes, fcs


def report(title: str, series: pd.Series, maes: dict[str, list[float]], n_folds: int, h: int, fmt: str = "%Y") -> None:
    splits = walk_forward(len(series), n_folds, h)
    base = np.mean(maes["seasonal naive"])
    print(title)
    print(f"{'method':<34}" + "".join(f"{series.index[te[0]].strftime(fmt):>7}" for _, te in splits) + f"{'mean':>8}{'skill':>7}")
    for name, v in maes.items():
        print(f"{name:<34}" + "".join(f"{e:>7.3f}" for e in v) + f"{np.mean(v):>8.3f}{1 - np.mean(v) / base:>7.3f}")


# %%
section("CO2, long series (466-514 training months)")
maes_long, fcs_long = run(co2, MONTHLY, N_FOLDS_MONTHLY, H_MONTHLY)
report("MAE in ppm per test year; skill = 1 - MAE / MAE(seasonal naive)", co2, maes_long, N_FOLDS_MONTHLY, H_MONTHLY)
lg_last = fcs_long["LightGBM on lags"][-1]
print(f"LightGBM on raw lags, 2001: forecast max {lg_last.max():.2f} ppm, highest value in its training data "
      f"{co2.iloc[:-12].max():.2f}, actual max {co2.iloc[-12:].max():.2f}")

hist_last = co2.iloc[:-12]  # the last fold's training data (to 2000-12)
hw = ExponentialSmoothing(hist_last.to_numpy(), trend="add", seasonal="add", seasonal_periods=12,
                          initialization_method="estimated").fit()
print(f"Holt-Winters fitted on 1958-2000: alpha {hw.params['smoothing_level']:.3f}, beta {hw.params['smoothing_trend']:.3f}, "
      f"gamma {hw.params['smoothing_seasonal']:.3f}; final trend {hw.trend[-1]:.4f} ppm/month")
sar = SARIMAX(hist_last.to_numpy(), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print("SARIMA fitted on 1958-2000: " + ", ".join(f"{k} {v:.3f}" for k, v in zip(sar.model.param_names, sar.params)))

# %%
section("CO2, short series (60-108 training months)")
maes_short, fcs_short = run(co2_short, MONTHLY, N_FOLDS_MONTHLY, H_MONTHLY)
report("MAE in ppm per test year", co2_short, maes_short, N_FOLDS_MONTHLY, H_MONTHLY)

# %%
# The daily demand series: boosting gets the features classical models do not use (promo, temperature).
section("synthetic daily demand with known drivers")


def boosting_daily(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> FloatArray:
    """LightGBM on lags >= 14 days plus weekday, promo and temperature (known for the future days)."""
    df = pd.concat([hist_df, future_df.assign(y=np.nan)])
    X = lag_table(df["y"], [14, 21, 28], 14, 7)
    X["weekday"] = df.index.dayofweek
    X["promo"] = df["promo"].to_numpy()
    X["temp"] = df["temp"].to_numpy()
    Xv, yv = X.to_numpy(float), df["y"].to_numpy()
    train = ~np.isnan(Xv).any(axis=1) & ~np.isnan(yv)
    return lgb.LGBMRegressor(**LGB_PARAMS).fit(Xv[train], yv[train]).predict(Xv[-len(future_df):])


def sarimax_exog(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> FloatArray:
    """Regression on promo and temperature with SARIMA errors: the classical way to use known drivers (linear only)."""
    model = SARIMAX(hist_df["y"].to_numpy(), exog=hist_df[["promo", "temp"]].to_numpy(), order=(1, 0, 0),
                    seasonal_order=(1, 0, 1, 7), trend="c").fit(disp=False)
    return model.forecast(len(future_df), exog=future_df[["promo", "temp"]].to_numpy())


DAILY_SERIES = {
    "seasonal naive": seasonal_naive(M_DAILY),
    "Holt-Winters additive": holt_winters(M_DAILY, damped=True),
    "SARIMA(1,0,0)(1,0,1)7": sarima((1, 0, 0), (1, 0, 1, 7)),
}
splits_d = walk_forward(N_DAYS, N_FOLDS_DAILY, H_DAILY)
maes_daily, _ = run(daily["y"], DAILY_SERIES, N_FOLDS_DAILY, H_DAILY)
for name, fc in [("SARIMAX + promo, temp (linear)", sarimax_exog), ("LightGBM on lags + promo, temp", boosting_daily)]:
    maes_daily[name] = [float(np.mean(np.abs(daily["y"].to_numpy()[te] - fc(daily.iloc[tr], daily.iloc[te]))))
                        for tr, te in splits_d]
report(f"MAE in orders per day, {N_FOLDS_DAILY} folds of {H_DAILY} days (fold start dates)", daily["y"], maes_daily,
       N_FOLDS_DAILY, H_DAILY, fmt="%m-%d")
print(f"(the noise alone has std 15, so even a perfect model has MAE about {15 * np.sqrt(2 / np.pi):.1f})")

# %%
# Key plot: skill against seasonal naive for every method in the three settings.
section("key plot: who wins where")
ML_METHODS = {"linear on lags (B6.3)", "LightGBM on lags", "LightGBM on 12-month differences", "LightGBM on lags + promo, temp"}


def label(name: str) -> str:
    return f"{name}  [{'ML' if name in ML_METHODS else 'classical'}]"


def skills(maes: dict[str, list[float]]) -> dict[str, float]:
    base = np.mean(maes["seasonal naive"])
    return {n: 1 - np.mean(v) / base for n, v in maes.items() if n != "seasonal naive"}


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120, gridspec_kw={"width_ratios": [1.2, 1]})
sk_long, sk_short = skills(maes_long), skills(maes_short)
names = list(sk_long)
yy = np.arange(len(names))
ax1.barh(yy - 0.2, [sk_long[n] for n in names], height=0.4, color="tab:blue", edgecolor="black", label="long: 466-514 months")
ax1.barh(yy + 0.2, [sk_short[n] for n in names], height=0.4, color="tab:orange", edgecolor="black", hatch="//",
         label="short: 60-108 months")
ax1.set_yticks(yy, [label(n) for n in names], fontsize=8)
ax1.set_title("CO2, 12 months ahead: classical models win", fontsize=10)
ax1.legend(loc="upper right", fontsize=8)
sk_daily = skills(maes_daily)
names_d = list(sk_daily)
ax2.barh(np.arange(len(names_d)), [sk_daily[n] for n in names_d], height=0.5, color="tab:green", edgecolor="black")
ax2.set_yticks(np.arange(len(names_d)), [label(n) for n in names_d], fontsize=8)
ax2.set_title("Daily demand with known drivers: boosting wins", fontsize=10)
for ax in (ax1, ax2):
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.35, 1)
    ax.invert_yaxis()
    ax.set_xlabel("skill vs seasonal naive (1 = perfect, 0 = no better)")
fig.tight_layout()
fig.savefig(IMG / "b6_4_who_wins.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b6_4_who_wins.png")

# %%
# Try this (not graded):
# 1. Give LightGBM a time-index feature (months since the start) on the raw CO2 target. Does it extrapolate now?
# 2. Make the heat effect linear (heat = 8 * (temp - 18)). Does SARIMAX with exog catch up with LightGBM?
# 3. Set N_FOLDS_DAILY = 20. How stable is the ranking of the daily methods across folds?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
