# %% [markdown]
# # A1.4 matplotlib — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a1_4_matplotlib_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
OUT = Path(__file__).resolve().parents[3] / "outputs" / "answers_a1_4"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
x = np.array([0.5, 1.5, 1.5, 2.0, 3.9, 4.0, 6.5, 8.0])
counts, edges = np.histogram(x, bins=4, range=(0.5, 8.0))
h = edges[1] - edges[0]
print("h:", h, " edges:", edges.tolist())
print("raw (x - lo) / h:", np.round((x - 0.5) / h, 4).tolist())
print("counts:", counts.tolist(), " density:", np.round(counts / (len(x) * h), 4).tolist())
print("Sturges n=8:", math.ceil(math.log2(8)) + 1, " n=100000:", math.ceil(math.log2(100_000)) + 1)

# %%
section("exercise 2: lie factor")


def draw_bars(values: dict[str, float], y_start: float, title: str, path: Path) -> None:
    """Save a bar chart of values with the y-axis starting at y_start."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    names, heights = list(values), list(values.values())
    ax.bar(names, heights, edgecolor="black")
    for i, v in enumerate(heights):
        ax.annotate(f"{v:g}", (i, v), ha="center", va="bottom")
    ax.set_ylim(y_start, max(heights) * 1.02)
    ax.set_ylabel("validation accuracy, %")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def lie_factor(first: float, last: float, y_start: float) -> float:
    """Tufte's lie factor for two bars drawn from y_start: graphic effect / data effect."""
    data_effect = (last - first) / first
    drawn_effect = (last - y_start) / (first - y_start) - 1.0
    return drawn_effect / data_effect


def misleading_report(values: dict[str, float], y_start: float) -> float:
    """Draw the bars from y_start, save the chart, return the lie factor of first vs last bar."""
    draw_bars(values, y_start, f"Axis from {y_start:g}", OUT / f"bars_from_{y_start:g}.png")
    v = list(values.values())
    return lie_factor(v[0], v[-1], y_start)


def honest_report(values: dict[str, float]) -> float:
    """Draw the bars from zero, save the chart, return the lie factor (1.0 by construction)."""
    return misleading_report(values, 0.0)


acc = {"v1": 87.2, "v2": 88.0}
print(f"misleading (axis from 87): lie factor {misleading_report(acc, 87.0):.1f}")
print(f"honest (axis from 0):      lie factor {honest_report(acc):.1f}")

# %%
section("exercise 2: bin counts")
latency = rng.exponential(scale=120.0, size=1000) + 20.0
fig, axes = plt.subplots(3, 1, figsize=(6, 7), dpi=120, sharex=True)
for ax, bins in zip(axes, [5, 30, 300], strict=True):
    c, _, _ = ax.hist(latency, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_ylabel("requests")
    ax.set_title(f"{bins} bins: {int((c == 0).sum())} empty")
axes[-1].set_xlabel("latency, ms")
fig.tight_layout()
fig.savefig(OUT / "bins_5_30_300.png", dpi=120, bbox_inches="tight")
plt.close(fig)
for bins in [5, 30, 300]:
    c, _ = np.histogram(latency, bins=bins)
    print(f"{bins:>3} bins: max count {int(c.max()):>3}, empty bins {int((c == 0).sum()):>3}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
