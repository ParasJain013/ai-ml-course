# %% [markdown]
# # A4.1 Derivatives — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_1_derivatives_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np

T0 = time.perf_counter()
EPS = np.finfo(np.float64).eps


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run from inside the ai-ml-course folder.")


OUT = repo_root() / "outputs" / "a4_1_answers"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")


def f(x: float) -> float:
    """x^4 - 3x, with multiplications only."""
    return x * x * x * x - 3.0 * x


x, h = 1.0, 0.1
fwd = (f(x + h) - f(x)) / h
ctr = (f(x + h) - f(x - h)) / (2 * h)
print(f"f(1.1) = {f(1.1):.4f}  f(1) = {f(1.0):.4f}  f(0.9) = {f(0.9):.4f}")
print(f"forward = {fwd:.4f} (error {fwd - 1:.4f}, f''h/2 = {12 * h / 2:.4f})")
print(f"central = {ctr:.4f} (error {ctr - 1:.4f}, f'''h^2/6 = {24 * h * h / 6:.4f})")


def g(x1: float, x2: float) -> float:
    """x1^2 x2 + 3 x2 - x1."""
    return x1 * x1 * x2 + 3.0 * x2 - x1


print(f"dg/dx1 at (2, -1) = 2 x1 x2 - 1 = {2 * 2 * -1 - 1}   dg/dx2 = x1^2 + 3 = {2 * 2 + 3}")
print(f"g(2.1, -1) = {g(2.1, -1):.4f}  g(1.9, -1) = {g(1.9, -1):.4f}"
      f"  central = {(g(2.1, -1) - g(1.9, -1)) / 0.2:.4f}")

# %%
section("exercise 2: second derivative")


def second_diff(fn: Callable[[float], float], x: float, h: float) -> float:
    """Central second difference (f(x+h) - 2 f(x) + f(x-h)) / h^2. Truncation error ~ f'''' h^2 / 12."""
    return (fn(x + h) - 2.0 * fn(x) + fn(x - h)) / (h * h)


def quartic(x: float) -> float:
    """x^4 with multiplications only."""
    return x * x * x * x


exact = 12.0  # f''(x) = 12 x^2 at x = 1
hs = np.logspace(-8, -1, 50)
errs = np.array([abs(second_diff(quartic, 1.0, h) - exact) for h in hs])
for k in range(1, 9):
    hk = 10.0**-k
    print(f"h = {hk:.0e}   error = {abs(second_diff(quartic, 1.0, hk) - exact):.2e}")
# total error ~ f'''' h^2 / 12 + 4 EPS |f| / h^2, minimized at h^4 = 48 EPS |f| / f''''
h_best = (48 * EPS * 1.0 / 24.0) ** 0.25
print(f"predicted best h = {h_best:.1e}   error there = {abs(second_diff(quartic, 1.0, h_best) - exact):.1e}")

fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.loglog(hs, np.maximum(errs, 1e-17), "o-", ms=3, label="second difference")
ax.loglog(hs, 24 * hs**2 / 12, "--", label="truncation ≈ f''''h²/12")
ax.loglog(hs, 4 * EPS / hs**2, ":", color="0.3", label="rounding ≈ 4ε|f|/h²")
ax.axvline(h_best, color="0.5", lw=0.8)
ax.set_ylim(1e-10, 1e2)
ax.set_xlabel("step size h (log scale)")
ax.set_ylabel("|estimate − 12| (log scale)")
ax.set_title("Second difference of x⁴ at 1: best step near 1e-4", fontsize=10)
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "second_diff_error.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/a4_1_answers/second_diff_error.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
