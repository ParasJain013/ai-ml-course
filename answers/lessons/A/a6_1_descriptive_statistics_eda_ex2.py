# %% [markdown]
# # A6.1 Descriptive statistics and EDA — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a6_1_descriptive_statistics_eda_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_ROWS = 2_000
MAD_TO_SD = 1.4826


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# Robust building blocks (copied from labs/A/a6_1_descriptive_statistics_eda.py).
def quantile_linear(x: FloatArray, p: float) -> float:
    """Quantile by linear interpolation between order statistics (NumPy's default). x (n,)."""
    xs = np.sort(x)
    h = (len(xs) - 1) * p
    lo = int(np.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (h - lo) * (xs[hi] - xs[lo]))


def median(x: FloatArray) -> float:
    """Middle value of x. x (n,)."""
    return quantile_linear(x, 0.5)


def iqr(x: FloatArray) -> float:
    """Q3 - Q1. x (n,)."""
    return quantile_linear(x, 0.75) - quantile_linear(x, 0.25)


def mad(x: FloatArray, scale: float = 1.0) -> float:
    """Median absolute deviation from the median, times scale. x (n,)."""
    return scale * median(np.abs(x - median(x)))


def trimmed_mean(x: FloatArray, prop: float) -> float:
    """Mean after dropping floor(prop * n) values from each end. x (n,)."""
    xs = np.sort(x)
    k = int(np.floor(prop * len(xs)))
    return float(xs[k : len(xs) - k].mean())


def pearson(x: FloatArray, y: FloatArray) -> float:
    """Pearson correlation. x, y (n,)."""
    dx, dy = x - x.mean(), y - y.mean()
    return float(np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2)))


def rank_average(x: FloatArray) -> FloatArray:
    """Ranks 1..n with ties averaged. x (n,) -> (n,)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks


# %%
section("exercise 1 check")
signups = np.array([12.0, 15, 14, 10, 13, 95, 11, 14])
q1, q3 = quantile_linear(signups, 0.25), quantile_linear(signups, 0.75)
print(f"sorted {np.sort(signups).astype(int).tolist()}   mean {signups.mean():.2f}   median {median(signups):.2f}")
print(f"Q1 {q1:.2f}   Q3 {q3:.2f}   IQR {q3 - q1:.2f}   fences [{q1 - 1.5 * (q3 - q1):.2f}, {q3 + 1.5 * (q3 - q1):.2f}]")
print(f"|x - median| sorted {np.sort(np.abs(signups - median(signups))).tolist()}   MAD {mad(signups):.2f}   "
      f"scaled {mad(signups, MAD_TO_SD):.4f}")
print(f"robust z of 95 = {(95 - median(signups)) / mad(signups, MAD_TO_SD):.2f}")
x = np.array([1.0, 2, 3, 4, 5])
y = np.array([5.0, 4, 3, 2, 20])
print(f"(d) Pearson {pearson(x, y):.4f}   Spearman {pearson(rank_average(x), rank_average(y)):.4f}   "
      f"ranks of y {rank_average(y).tolist()}")


# %%
def describe_robust(df: pd.DataFrame) -> pd.DataFrame:
    """Robust summary per numeric column, missing values ignored. df (n, d) -> (d, 10)."""
    rows = {}
    for col in df.select_dtypes("number").columns:
        raw = df[col].to_numpy(dtype=np.float64)
        x = raw[~np.isnan(raw)]
        med, scale = median(x), mad(x, MAD_TO_SD)
        z = (x - med) / scale if scale > 0 else np.zeros_like(x)
        rows[col] = {
            "count": len(x),
            "missing": 1 - len(x) / len(raw),
            "mean": x.mean(),
            "median": med,
            "sd": x.std(ddof=1),
            "IQR": iqr(x),
            "MAD*1.48": scale,
            "trim10": trimmed_mean(x, 0.1),
            "skew": stats.skew(x),
            "|z|>3.5": int(np.sum(np.abs(z) > 3.5)),
        }
    return pd.DataFrame(rows).T


section("exercise 2: describe_robust")
rng = np.random.default_rng(SEED)
df = pd.DataFrame(
    {
        "gaussian": rng.normal(50, 10, N_ROWS),
        "lognormal": rng.lognormal(np.log(100), 1.0, N_ROWS),
        "unit_error": rng.normal(20, 3, N_ROWS),
        "with_missing": rng.normal(0, 1, N_ROWS),
    }
)
bad = rng.choice(N_ROWS, size=int(0.03 * N_ROWS), replace=False)
df.loc[bad, "unit_error"] *= 1_000  # 3% of the values recorded in the wrong unit
df.loc[rng.choice(N_ROWS, size=int(0.10 * N_ROWS), replace=False), "with_missing"] = np.nan
summary = describe_robust(df)
print(summary.round(3).to_string())

for col in df.columns:  # check every from-scratch number against the libraries
    s = df[col].dropna()
    x = s.to_numpy()
    np.testing.assert_allclose(summary.loc[col, "median"], s.median(), rtol=1e-12)
    np.testing.assert_allclose(summary.loc[col, "IQR"], stats.iqr(x), rtol=1e-12)
    # scale="normal" uses 1/Phi^{-1}(0.75) = 1.482602..., MAD_TO_SD is rounded to 1.4826
    np.testing.assert_allclose(summary.loc[col, "MAD*1.48"], stats.median_abs_deviation(x, scale="normal"), rtol=1e-5)
    np.testing.assert_allclose(summary.loc[col, "trim10"], stats.trim_mean(x, 0.1), rtol=1e-12)
    np.testing.assert_allclose(summary.loc[col, "sd"], s.std(), rtol=1e-12)
print("median, IQR, MAD, trimmed mean and sd match pandas / scipy: OK")

section("exercise 2: Spearman matrix")
complete = df.dropna()  # pairwise-complete would also work; use complete rows for a fair comparison
R = np.column_stack([rank_average(complete[c].to_numpy()) for c in complete.columns])  # (n, d)
mine = np.corrcoef(R, rowvar=False)  # (d, d)
theirs = complete.corr(method="spearman").to_numpy()
np.testing.assert_allclose(mine, theirs, rtol=1e-10, atol=1e-12)
print(pd.DataFrame(mine, index=df.columns, columns=df.columns).round(3).to_string())
print("Spearman matrix matches DataFrame.corr(method='spearman'): OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
