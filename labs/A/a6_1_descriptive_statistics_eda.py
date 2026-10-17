# %% [markdown]
# # A6.1 Descriptive statistics and EDA — lab
# Lesson: lessons/A/a6_1_descriptive_statistics_eda.md
# Run: uv run python labs/A/a6_1_descriptive_statistics_eda.py   (CPU, defaults: ~3 s)
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
from scipy import stats

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
OUT = ROOT / "outputs" / "a6_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the simulations bigger.
GPU_SCALE = False
N_LATENCY = 100_000 if GPU_SCALE else 10_000  # simulated request latencies
CONTAMINATION = [0.0, 0.01, 0.05, 0.20]  # share of requests that time out at 30 s
TIMEOUT_MS = 30_000.0
N_SERVERS = 50_000 if GPU_SCALE else 5_000  # server-minutes in the confounding simulation
MAD_TO_SD = 1.4826  # 1 / Phi^{-1}(0.75): makes the MAD estimate sigma for Gaussian data

# %%
# Data: a tiny latency sample for the hand calculations, Anscombe's quartet typed in from
# Anscombe (1973), and simulated latencies with a heavy right tail.
section("data")
x_small = np.array([2.0, 3.0, 3.0, 4.0, 5.0, 7.0, 40.0])  # latencies in ms, one slow request
print("hand sample (ms):", x_small.tolist())

anscombe = pd.DataFrame(
    {
        "x123": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        "y2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        "x4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        "y4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
    },
    dtype=np.float64,
)
print("Anscombe's quartet:", anscombe.shape, "(11 points per set, 4 sets)")
print(anscombe.head(3).to_string())

base_latency = rng.lognormal(mean=np.log(120.0), sigma=0.5, size=N_LATENCY)  # (N_LATENCY,) ms
print(f"simulated latencies: n = {N_LATENCY:,}   median = {np.median(base_latency):.1f} ms   "
      f"mean = {base_latency.mean():.1f} ms   max = {base_latency.max():.1f} ms")

# %%
# Worked example by hand: summaries of the 7-value sample, two correlations, Simpson's paradox.
section("worked example by hand")
xs = np.sort(x_small)
n_small = len(xs)
mean_s = xs.mean()
sd_s = xs.std(ddof=1)
q1, med, q3 = np.quantile(xs, [0.25, 0.5, 0.75])
dev = np.abs(xs - med)
mad = np.median(dev)
print(f"n = {n_small}   sum = {xs.sum():.0f}   mean = {mean_s:.4f}   sample sd = {sd_s:.4f}")
print(f"median = x[(n+1)/2] = 4th value = {med:.1f}")
print(f"Q1 at position (n-1)*0.25 = {(n_small - 1) * 0.25} -> {q1:.1f}   "
      f"Q3 at position (n-1)*0.75 = {(n_small - 1) * 0.75} -> {q3:.1f}   IQR = {q3 - q1:.1f}")
print(f"|x - median| = {dev.astype(int).tolist()} -> sorted {np.sort(dev).astype(int).tolist()}   "
      f"MAD = {mad:.1f}   scaled MAD = 1.4826 * MAD = {MAD_TO_SD * mad:.4f}")
print(f"20% trimmed mean: drop {int(0.2 * n_small)} from each end -> {xs[1:-1].tolist()} -> "
      f"{xs[1:-1].mean():.1f}")
lo_f, hi_f = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
print(f"Tukey fences: [{lo_f:.1f}, {hi_f:.1f}]   outside: {xs[(xs < lo_f) | (xs > hi_f)].tolist()}")
print(f"without the 40: mean = {xs[:-1].mean():.4f}   median = {np.median(xs[:-1]):.1f}")

xc = np.array([1.0, 2.0, 3.0, 4.0])
yc = np.array([1.0, 2.0, 3.0, 20.0])
dx, dy = xc - xc.mean(), yc - yc.mean()
print(f"x = {xc.tolist()}   y = {yc.tolist()}")
print(f"deviations: dx = {dx.tolist()}   dy = {dy.tolist()}")
print(f"sum dx*dy = {np.sum(dx * dy):.2f}   sum dx^2 = {np.sum(dx**2):.2f}   sum dy^2 = {np.sum(dy**2):.2f}   "
      f"Pearson r = {np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2)):.4f}")
print("ranks of y = [1, 2, 3, 4] equal the ranks of x, so Spearman rho = 1.0")

simpson = pd.DataFrame(
    {
        "treatment": ["A", "A", "B", "B"],
        "stone": ["small", "large", "small", "large"],
        "success": [81, 192, 234, 55],
        "patients": [87, 263, 270, 80],
    }
)
simpson["rate"] = simpson["success"] / simpson["patients"]
print("kidney-stone data (Charig et al., 1986):")
print(simpson.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
overall = simpson.groupby("treatment")[["success", "patients"]].sum()
overall["rate"] = overall["success"] / overall["patients"]
print("pooled over stone size:")
print(overall.to_string(float_format=lambda v: f"{v:.3f}"))

# %%
# From scratch: robust summaries, and how each one reacts when timeouts contaminate the data.
section("from scratch: robust summaries")


def quantile_linear(x: FloatArray, p: float) -> float:
    """Quantile by linear interpolation between order statistics (NumPy's default). x (n,)."""
    xs = np.sort(x)
    h = (len(xs) - 1) * p  # 0-based position
    lo = int(np.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (h - lo) * (xs[hi] - xs[lo]))


def median(x: FloatArray) -> float:
    """Middle value of x (mean of the two middle values when n is even). x (n,)."""
    return quantile_linear(x, 0.5)


def iqr(x: FloatArray) -> float:
    """Interquartile range Q3 - Q1. x (n,)."""
    return quantile_linear(x, 0.75) - quantile_linear(x, 0.25)


def mad(x: FloatArray, scale: float = 1.0) -> float:
    """Median absolute deviation from the median, times scale. x (n,)."""
    return scale * median(np.abs(x - median(x)))


def trimmed_mean(x: FloatArray, prop: float) -> float:
    """Mean after dropping floor(prop * n) values from each end. x (n,)."""
    xs = np.sort(x)
    k = int(np.floor(prop * len(xs)))
    return float(xs[k : len(xs) - k].mean())


def robust_z(x: FloatArray) -> FloatArray:
    """Distance from the median in units of the scaled MAD. x (n,) -> (n,)."""
    return (x - median(x)) / mad(x, MAD_TO_SD)


print(f"hand sample: median {median(x_small):.1f}   IQR {iqr(x_small):.1f}   MAD {mad(x_small):.1f}   "
      f"20% trimmed mean {trimmed_mean(x_small, 0.2):.1f}")
print(f"hand sample robust z-scores: {show(robust_z(x_small))}   (|z| > 3.5 flags an outlier)")

print(f"{'timeouts':>9}{'mean':>10}{'sd':>10}{'median':>9}{'IQR':>8}{'MAD*1.48':>10}{'trim 10%':>10}")
for c in CONTAMINATION:
    lat = base_latency.copy()
    k = int(c * N_LATENCY)
    lat[:k] = TIMEOUT_MS  # the first k requests time out (the draws are i.i.d., so any k will do)
    print(f"{c:>9.0%}{lat.mean():>10.1f}{lat.std(ddof=1):>10.1f}{median(lat):>9.1f}{iqr(lat):>8.1f}"
          f"{mad(lat, MAD_TO_SD):>10.1f}{trimmed_mean(lat, 0.1):>10.1f}")
print("log-scale view: mean of log10(latency) = "
      f"{np.log10(base_latency).mean():.4f} -> geometric mean {10 ** np.log10(base_latency).mean():.1f} ms")

# %%
# From scratch: Pearson and Spearman correlation, and Anscombe's quartet.
section("from scratch: correlation")


def pearson(x: FloatArray, y: FloatArray) -> float:
    """Pearson correlation: covariance over the product of standard deviations. x, y (n,)."""
    dx, dy = x - x.mean(), y - y.mean()
    return float(np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2)))


def rank_average(x: FloatArray) -> FloatArray:
    """Ranks 1..n; tied values share the average of their ranks. x (n,) -> (n,)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # average the ranks within each group of ties
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x: FloatArray, y: FloatArray) -> float:
    """Spearman correlation: Pearson correlation of the ranks. x, y (n,)."""
    return pearson(rank_average(x), rank_average(y))


print(f"hand pair: Pearson {pearson(xc, yc):.4f}   Spearman {spearman(xc, yc):.4f}")
print(f"ties: ranks of [10, 20, 20, 30] = {rank_average(np.array([10.0, 20.0, 20.0, 30.0])).tolist()}")

pairs = {"I": ("x123", "y1"), "II": ("x123", "y2"), "III": ("x123", "y3"), "IV": ("x4", "y4")}
print(f"{'set':>4}{'mean x':>8}{'mean y':>8}{'var x':>8}{'var y':>8}{'Pearson':>9}{'slope':>7}"
      f"{'intercept':>10}{'Spearman':>10}")
fits: dict[str, tuple[float, float]] = {}
for name, (cx, cy) in pairs.items():
    ax_, ay_ = anscombe[cx].to_numpy(), anscombe[cy].to_numpy()
    slope = float(np.sum((ax_ - ax_.mean()) * (ay_ - ay_.mean())) / np.sum((ax_ - ax_.mean()) ** 2))
    intercept = float(ay_.mean() - slope * ax_.mean())
    fits[name] = (slope, intercept)
    print(f"{name:>4}{ax_.mean():>8.2f}{ay_.mean():>8.2f}{ax_.var(ddof=1):>8.2f}{ay_.var(ddof=1):>8.2f}"
          f"{pearson(ax_, ay_):>9.3f}{slope:>7.3f}{intercept:>10.2f}{spearman(ax_, ay_):>10.3f}")

# %%
# From scratch: correlation without causation. Traffic drives both CPU and errors.
section("from scratch: a confounder")
traffic = rng.uniform(100, 1_000, size=N_SERVERS)  # requests per second, the confounder Z
cpu = 0.08 * traffic + rng.normal(0, 8, size=N_SERVERS)  # CPU % depends on traffic only
errors = 0.01 * traffic + rng.normal(0, 1.5, size=N_SERVERS)  # errors/min depend on traffic only
print(f"corr(cpu, errors) = {pearson(cpu, errors):.4f}   (neither causes the other)")
print(f"corr(traffic, cpu) = {pearson(traffic, cpu):.4f}   corr(traffic, errors) = {pearson(traffic, errors):.4f}")


def residual_on(v: FloatArray, z: FloatArray) -> FloatArray:
    """Residuals of v after a least-squares line on z (A2.4). v, z (n,) -> (n,)."""
    Z = np.column_stack([np.ones_like(z), z])  # (n, 2)
    coef, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ coef


partial = pearson(residual_on(cpu, traffic), residual_on(errors, traffic))
print(f"partial correlation of cpu and errors given traffic = {partial:.4f}")
bands = pd.cut(traffic, bins=[100, 250, 400, 550, 700, 850, 1_000], include_lowest=True)
frame = pd.DataFrame({"band": bands, "cpu": cpu, "errors": errors})
within = frame.groupby("band", observed=True).apply(lambda g: pearson(g["cpu"].to_numpy(), g["errors"].to_numpy()),
                                                     include_groups=False)
print("corr(cpu, errors) within traffic bands:", show(within.to_numpy()))
cpu_forced = rng.uniform(cpu.min(), cpu.max(), size=N_SERVERS)  # an experiment: set CPU at random
print(f"if we set CPU at random (an intervention), corr(cpu, errors) = {pearson(cpu_forced, errors):.4f}")

# %%
# With pandas / NumPy / SciPy: the same numbers from the libraries.
section("with pandas and scipy")
lat5 = base_latency.copy()
lat5[: int(0.05 * N_LATENCY)] = TIMEOUT_MS
s = pd.Series(lat5)
checks = {
    "median": (median(lat5), s.median()),
    "quantile 0.9": (quantile_linear(lat5, 0.9), s.quantile(0.9)),
    "IQR": (iqr(lat5), stats.iqr(lat5)),
    "MAD (raw)": (mad(lat5), stats.median_abs_deviation(lat5)),
    "MAD (normal scale)": (mad(lat5, MAD_TO_SD), stats.median_abs_deviation(lat5, scale="normal")),
    "10% trimmed mean": (trimmed_mean(lat5, 0.1), stats.trim_mean(lat5, 0.1)),
}
for name, (mine, theirs) in checks.items():
    # scale="normal" uses 1/Phi^{-1}(0.75) = 1.482602..., our constant is rounded to 1.4826
    rtol = 1e-5 if name == "MAD (normal scale)" else 1e-12
    np.testing.assert_allclose(mine, theirs, rtol=rtol)
    print(f"{name:<20} mine {mine:>10.4f}   library {theirs:>10.4f}")
print("robust summaries match numpy / pandas / scipy: OK")
for name, (cx, cy) in pairs.items():
    ax_, ay_ = anscombe[cx].to_numpy(), anscombe[cy].to_numpy()
    np.testing.assert_allclose(pearson(ax_, ay_), stats.pearsonr(ax_, ay_).statistic, rtol=1e-12)
    np.testing.assert_allclose(spearman(ax_, ay_), stats.spearmanr(ax_, ay_).statistic, rtol=1e-12)
    np.testing.assert_allclose(rank_average(ay_), stats.rankdata(ay_), rtol=1e-12)
print("pearson, spearman and ranks match scipy.stats on all four Anscombe sets: OK")
both = pd.DataFrame({"cpu": cpu, "errors": errors, "traffic": traffic})
np.testing.assert_allclose(both.corr(method="spearman").loc["cpu", "errors"], spearman(cpu, errors), rtol=1e-10)
print("DataFrame.corr(method='spearman') matches: OK")
print("pandas describe() of the 5%-timeout latencies (mean and std are dragged by the timeouts):")
print(s.describe().round(1).to_string())

# %%
# Key plot: Anscombe's quartet, four datasets with the same summary statistics.
section("key plot")
fig, axes = plt.subplots(2, 2, figsize=(8, 6.4), dpi=120, sharex=True, sharey=True)
grid = np.array([3.0, 20.0])
for ax, (name, (cx, cy)) in zip(axes.ravel(), pairs.items()):
    ax.plot(anscombe[cx], anscombe[cy], "o", color="tab:blue", label="data")
    slope, intercept = fits[name]
    ax.plot(grid, intercept + slope * grid, "--", color="tab:orange", label=f"y = {intercept:.2f} + {slope:.3f}x")
    r = pearson(anscombe[cx].to_numpy(), anscombe[cy].to_numpy())
    ax.set_title(f"{name}: r = {r:.3f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
for ax in axes[1]:
    ax.set_xlabel("x (unitless)")
for ax in axes[:, 0]:
    ax.set_ylabel("y (unitless)")
fig.suptitle("Same mean, variance, correlation and line; four different stories: plot first")
fig.tight_layout()
path_png = IMG / "a6_1_anscombe_quartet.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Raise CONTAMINATION to 0.5. Which summary breaks first, the median or the MAD, and why at 50%?
# 2. Add one point (30, 0) to Anscombe's set I. How much do Pearson and Spearman each move?
# 3. In the confounder cell, make errors depend on cpu as well (errors += 0.05 * cpu). What does the
#    partial correlation show now?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
