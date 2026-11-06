# %% [markdown]
# # B6.4 ARIMA versus boosting — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b6_4_arima_vs_boosting_ex2.py   (CPU, ~60 s)

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*(Non-stationary|Non-invertible) starting.*")


def repo_root() -> Path:
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run from inside the ai-ml-course folder.")


OUT = repo_root() / "outputs" / "b6_4"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
level, sse = 100.0, 0.0
for t, v in enumerate([100.0, 110, 90, 120], start=1):
    err = v - level
    sse += err**2
    new = 0.2 * v + 0.8 * level
    print(f"(a) t={t}: forecast {level:g}, error {err:+g}, level {new:g}")
    level = new
print(f"(a) SSE {sse:g}; next forecast {level:g}")
print("(b) weights " + ", ".join(f"{0.2 * 0.8**j:g}" for j in range(4)) + f"; older {0.8**4:g}")
print(f"(c) {[200 + 3 * h + s for h, s in zip([1, 2, 3], [10, -5, -5])]}; damped h=3: {200 + (0.9 + 0.81 + 0.729) * 3 - 5:g}")
d1 = 1 + 0.6 * (54 - 50)
d2 = 1 + 0.6 * d1
print(f"(d) changes {d1:g}, {d2:g} -> forecasts {54 + d1:g}, {54 + d1 + d2:g}")

# %%
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate(method="linear")
x = co2.to_numpy()
splits = [(np.arange(s), np.arange(s, s + 12)) for s in len(x) - 12 * np.arange(5, 0, -1)]  # B6.3's folds


def wf_mae(fc) -> list[float]:
    return [float(np.mean(np.abs(x[te] - fc(co2.iloc[tr], 12)))) for tr, te in splits]


# %%
section("(a) SARIMA order by AIC on fold 0, then by walk-forward MAE")
aic = {}
for p in range(3):
    for q in range(3):
        res = SARIMAX(co2.iloc[splits[0][0]].to_numpy(), order=(p, 1, q), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        aic[(p, q)] = res.aic
ranked = sorted(aic, key=aic.get)
for p, q in ranked:
    print(f"(p,q)=({p},{q}) AIC {aic[(p, q)]:.1f}")
print("top three by AIC, walk-forward MAE per fold:")
for p, q in ranked[:3]:
    m = wf_mae(lambda h_, n_, p=p, q=q: SARIMAX(h_.to_numpy(), order=(p, 1, q), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(n_))
    print(f"(p,q)=({p},{q}): " + " ".join(f"{v:.3f}" for v in m) + f"  mean {np.mean(m):.3f}")
m = wf_mae(lambda h_, n_: SARIMAX(h_.to_numpy(), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(n_))
print(f"the lab's (1,1): mean {np.mean(m):.3f}")

# %%
section("(b) Holt-Winters: damped versus undamped trend")
fcs_2001 = {}
for damped in [False, True]:
    def hw(h_: pd.Series, n_: int, damped: bool = damped) -> FloatArray:
        return ExponentialSmoothing(h_.to_numpy(), trend="add", damped_trend=damped, seasonal="add", seasonal_periods=12,
                                    initialization_method="estimated").fit().forecast(n_)
    m = wf_mae(hw)
    fcs_2001[f"Holt-Winters{' damped' if damped else ''}"] = hw(co2.iloc[splits[-1][0]], 12)
    print(f"damped={damped!s:<5}: " + " ".join(f"{v:.3f}" for v in m) + f"  mean {np.mean(m):.3f}")

# %%
section("(c) the 2001 forecasts")
hist = co2.iloc[splits[-1][0]]
fcs_2001["SARIMA(1,1,1)(0,1,1)12"] = SARIMAX(hist.to_numpy(), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(12)
params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
              random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
for difference in [False, True]:  # from labs/B/b6_4_arima_vs_boosting.py
    idx = pd.date_range(hist.index[0], periods=len(hist) + 12, freq="MS")
    s = pd.concat([hist, pd.Series(np.nan, index=idx[len(hist):])])
    target = s - s.shift(12) if difference else s
    X = pd.DataFrame({"lag_12": target.shift(12), "lag_13": target.shift(13), "lag_24": target.shift(24),
                      "roll_12": target.shift(12).rolling(12).mean(), "month": s.index.month}).to_numpy(float)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(target.to_numpy())
    pred = lgb.LGBMRegressor(**params).fit(X[ok], target.to_numpy()[ok]).predict(X[-12:])
    fcs_2001["LightGBM on 12-month differences" if difference else "LightGBM on lags"] = (
        pred + s.shift(12).to_numpy()[-12:] if difference else pred)
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
ax.plot(co2.index[-12:], x[-12:], color="black", linewidth=2, label="actual")
for (name, f), mk in zip(fcs_2001.items(), ["o", "s", "^", "v", "D"]):
    ax.plot(co2.index[-12:], f, marker=mk, markersize=4, label=f"{name} (MAE {np.mean(np.abs(x[-12:] - f)):.2f})")
    print(f"{name:<34} MAE {np.mean(np.abs(x[-12:] - f)):.3f}")
ax.set_xlabel("month of 2001")
ax.set_ylabel("CO2 (ppm)")
ax.set_title("2001: LightGBM on the raw level cannot rise above its training maximum")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "ex2_forecasts_2001.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b6_4/ex2_forecasts_2001.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
