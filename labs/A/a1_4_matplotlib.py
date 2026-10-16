# %% [markdown]
# # A1.4 matplotlib — lab
# Lesson: lessons/A/a1_4_matplotlib.md
# Run: uv run python labs/A/a1_4_matplotlib.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import math
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

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
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
OUT = ROOT / "outputs" / "a1_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Plotting never needs a GPU; the knob only changes the sample size.
GPU_SCALE = False
N_SAMPLE = 100_000 if GPU_SCALE else 1_000  # skewed sample for the histogram checks
N_DAYS = 730  # two years of a daily metric for the misleading-chart example

# %%
# Data: a skewed sample (like response times) and a daily metric with a flat trend plus noise.
section("data")
latency_ms = rng.exponential(scale=120.0, size=N_SAMPLE) + 20.0  # (N_SAMPLE,)
print("latency sample:", latency_ms.shape, " min", round(float(latency_ms.min()), 4), " max", round(float(latency_ms.max()), 4))
day = np.arange(N_DAYS)  # (730,)
weekly = 3.0 * np.sin(2 * np.pi * day / 7)
signups = 200.0 - 0.02 * day + weekly + rng.normal(0.0, 6.0, size=N_DAYS)  # slowly declining
print("signups first 3 days:", np.round(signups[:3], 4).tolist())

# %%
# Worked example: 7 values, 3 equal-width bins over [1, 10].
section("worked example: histogram")


def histogram(x: FloatArray, n_bins: int, lo: float, hi: float) -> tuple[IntArray, FloatArray]:
    """Equal-width histogram like np.histogram. x (n,) -> counts (n_bins,), edges (n_bins + 1,)."""
    if hi <= lo or n_bins < 1:
        raise ValueError("need hi > lo and n_bins >= 1")
    width = (hi - lo) / n_bins
    edges = lo + width * np.arange(n_bins + 1)  # (n_bins + 1,)
    inside = x[(x >= lo) & (x <= hi)]  # values outside [lo, hi] are ignored
    idx = np.floor((inside - lo) / width).astype(np.int64)  # bin index per value
    idx = np.minimum(idx, n_bins - 1)  # x == hi belongs to the last bin (closed on the right)
    counts = np.bincount(idx, minlength=n_bins)  # (n_bins,)
    return counts, edges


x_small = np.array([1.0, 2.0, 2.0, 3.0, 7.0, 9.0, 10.0])  # (7,)
counts, edges = histogram(x_small, n_bins=3, lo=1.0, hi=10.0)
width = edges[1] - edges[0]
density = counts / (len(x_small) * width)
print("edges:", edges.tolist())
print("bin index per value:", np.minimum(np.floor((x_small - 1.0) / width), 2).astype(int).tolist())
print("counts:", counts.tolist())
print("density:", np.round(density, 4).tolist(), " area:", round(float((density * width).sum()), 4))

# %%
# From scratch vs NumPy and matplotlib on the skewed sample.
section("from scratch vs numpy and matplotlib")
lo, hi = float(latency_ms.min()), float(latency_ms.max())
mine_counts, mine_edges = histogram(latency_ms, 30, lo, hi)
np_counts, np_edges = np.histogram(latency_ms, bins=30, range=(lo, hi))
np.testing.assert_allclose(mine_edges, np_edges, rtol=1e-6, atol=1e-8)
np.testing.assert_array_equal(mine_counts, np_counts)
print("matches np.histogram: OK")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
mpl_counts, mpl_edges, _ = ax.hist(latency_ms, bins=30, range=(lo, hi))
plt.close(fig)
np.testing.assert_array_equal(mine_counts, mpl_counts.astype(np.int64))
print("matches ax.hist: OK")
print("first 5 counts:", mine_counts[:5].tolist(), " total:", int(mine_counts.sum()))


def sturges_bins(n: int) -> int:
    """Sturges' rule for the number of bins: ceil(log2 n) + 1."""
    return math.ceil(math.log2(n)) + 1


k = sturges_bins(len(latency_ms))
print("Sturges bins for n =", len(latency_ms), ":", k, " numpy:", len(np.histogram_bin_edges(latency_ms, bins="sturges")) - 1)
print("numpy 'auto' bins:", len(np.histogram_bin_edges(latency_ms, bins="auto")) - 1)

# %%
# The lie factor of a truncated bar chart.
section("lie factor")
before, after, axis_start = 92.1, 93.4, 91.0
data_change = (after - before) / before
drawn_change = (after - axis_start) / (before - axis_start) - 1.0
print(f"data change:  {data_change:.4f}  ({data_change * 100:.2f} %)")
print(f"drawn change: {drawn_change:.4f}  ({drawn_change * 100:.2f} %)")
print(f"lie factor:   {drawn_change / data_change:.1f}")

# %%
# The four basic plots with the object-oriented API (scratch figure, not committed).
section("basic plots")
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), dpi=120)
axes[0, 0].plot(day, signups, linewidth=0.8)
axes[0, 0].set(xlabel="day", ylabel="signups per day", title="Line: a slow decline under the noise")
axes[0, 1].scatter(latency_ms[:-1][:300], latency_ms[1:][:300], s=6, marker="o")
axes[0, 1].set(xlabel="latency of request i, ms", ylabel="latency of request i+1, ms", title="Scatter: no pattern")
axes[1, 0].hist(latency_ms, bins="auto", edgecolor="black", linewidth=0.3)
axes[1, 0].set(xlabel="latency, ms", ylabel="requests", title="Histogram: right-skewed")
axes[1, 1].hist(np.log10(latency_ms), bins="auto", edgecolor="black", linewidth=0.3)
axes[1, 1].set(xlabel="log10(latency in ms)", ylabel="requests", title="Same data, log x: shape is clearer")
fig.tight_layout()
fig.savefig(OUT / "basic_plots.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved outputs/a1_4/basic_plots.png (scratch)")

# %%
# Key plot: two misleading charts (top) and their honest versions (bottom).
section("key plot")
WINDOW = 60
starts = range(0, N_DAYS - WINDOW + 1)
slopes = [np.polyfit(day[s : s + WINDOW], signups[s : s + WINDOW], 1)[0] for s in starts]
best = int(np.argmax(slopes))  # the 60-day stretch with the steepest rise: the cherry-pick
window = slice(best, best + WINDOW)
print(f"cherry-picked window: days {best}-{best + WINDOW - 1}")
x_win, y_win = day[window], signups[window]
slope_win = np.polyfit(x_win, y_win, 1)[0]
slope_all = np.polyfit(day, signups, 1)[0]
print(f"trend in the cherry-picked 60 days: {slope_win:+.4f} signups/day")
print(f"trend over all {N_DAYS} days:        {slope_all:+.4f} signups/day")

fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), dpi=120)
labels = ["before", "after"]
values = [before, after]
ax = axes[0, 0]
ax.bar(labels, values, color=["tab:gray", "tab:blue"], edgecolor="black")
ax.set_ylim(axis_start, 93.6)
ax.set_ylabel("conversion rate, %")
ax.set_title("Misleading: y-axis starts at 91%\n(the bar looks 2x taller)")
ax = axes[1, 0]
ax.bar(labels, values, color=["tab:gray", "tab:blue"], edgecolor="black")
ax.set_ylim(0, 100)
ax.set_ylabel("conversion rate, %")
ax.set_title(f"Honest: axis from 0 (a {data_change * 100:.1f}% relative change)")
for a in (axes[0, 0], axes[1, 0]):
    for i, v in enumerate(values):
        a.annotate(f"{v}%", (i, v), ha="center", va="bottom")

ax = axes[0, 1]
ax.plot(x_win, y_win, marker="o", markersize=3)
ax.plot(x_win, np.polyval(np.polyfit(x_win, y_win, 1), x_win), linestyle="--", color="tab:red", label="trend")
ax.set_xlabel("day")
ax.set_ylabel("signups per day")
ax.set_title(f"Misleading: 60 hand-picked days\n(trend {slope_win:+.2f}/day)")
ax.legend(loc="best")
ax = axes[1, 1]
ax.plot(day, signups, linewidth=0.6, color="tab:blue", label="daily")
ax.plot(day, np.polyval(np.polyfit(day, signups, 1), day), linestyle="--", color="tab:red", label="trend")
ax.axvspan(x_win[0], x_win[-1], color="tab:orange", alpha=0.25, label="the 60 days above")
ax.set_xlabel("day")
ax.set_ylabel("signups per day")
ax.set_title(f"Honest: all {N_DAYS} days (trend {slope_all:+.2f}/day)")
ax.legend(loc="lower left", fontsize=8)
fig.suptitle("Same data, opposite stories: axis choice and range choice")
fig.tight_layout()
path = IMG / "a1_4_misleading_vs_honest.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Plot latency_ms with bins=5 and bins=300. Which hides the shape, which shows only noise?
# 2. Find the 60-day window with the steepest *falling* trend and make the opposite misleading chart.
# 3. Re-draw the honest signups chart with a 28-day rolling mean (np.convolve) instead of the raw line.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
