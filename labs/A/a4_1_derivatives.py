# %% [markdown]
# # A4.1 Derivatives and partial derivatives — lab
# Lesson: lessons/A/a4_1_derivatives.md
# Run: uv run python labs/A/a4_1_derivatives.py   (CPU, defaults: ~3 s)
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
OUT = ROOT / "outputs" / "a4_1"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knob only sets how many step sizes the error study tries.
GPU_SCALE = False
N_STEPS = 400 if GPU_SCALE else 61  # step sizes h, log-spaced from 1e-16 to 1
EPS64 = float(np.finfo(np.float64).eps)  # 2.22e-16: gap between 1.0 and the next float64
EPS32 = float(np.finfo(np.float32).eps)  # 1.19e-07: the same for float32

# %%
# Data: the functions this lab differentiates.
section("data")


def cube(x: float) -> float:
    """f(x) = x^3; exact derivative 3 x^2."""
    return x**3


def g2(x: FloatArray) -> float:
    """g(x1, x2) = x1^2 x2 + 3 x2. x (2,) -> scalar. Exact partials (2 x1 x2, x1^2 + 3)."""
    return float(x[0] ** 2 * x[1] + 3.0 * x[1])


print("f(x) = x^3            f(2) =", cube(2.0), "  exact f'(2) = 3 * 2^2 =", 3 * 2.0**2)
print("g(x1, x2) = x1^2 x2 + 3 x2   g(1, 2) =", g2(np.array([1.0, 2.0])))
print("exp(x) at x = 1: f = f' = f'' = f''' = e =", round(np.e, 6))
print(f"machine epsilon: float64 {EPS64:.3e}   float32 {EPS32:.3e}")

# %%
# Worked example by hand: forward and central differences of x^3 at x = 2.
section("worked example: x^3 at x = 2")


def forward_diff(f: Callable[[float], float], x: float, h: float) -> float:
    """Forward-difference slope (f(x + h) - f(x)) / h."""
    return (f(x + h) - f(x)) / h


def central_diff(f: Callable[[float], float], x: float, h: float) -> float:
    """Central-difference slope (f(x + h) - f(x - h)) / (2 h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


x0, exact = 2.0, 12.0
for h in (0.1, 0.01):
    fwd, cen = forward_diff(cube, x0, h), central_diff(cube, x0, h)
    print(
        f"h = {h:<5} forward = {fwd:.6f} (error {fwd - exact:.6f}, predicted 3*x*h + h^2 = {3 * x0 * h + h**2:.6f})"
        f"   central = {cen:.6f} (error {cen - exact:.6f}, predicted h^2 = {h**2:.6f})"
    )

# %%
# Worked example by hand: partial derivatives of g at (1, 2).
section("worked example: partial derivatives")
x_pt = np.array([1.0, 2.0])
print("exact: dg/dx1 = 2 x1 x2 =", 2 * x_pt[0] * x_pt[1], "  dg/dx2 = x1^2 + 3 =", x_pt[0] ** 2 + 3)
h = 0.1
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
print(f"g(1.1, 2) = {g2(x_pt + h * e1):.4f}  g(0.9, 2) = {g2(x_pt - h * e1):.4f}"
      f"  -> central dg/dx1 = {(g2(x_pt + h * e1) - g2(x_pt - h * e1)) / (2 * h):.4f}")
print(f"g(1, 2.1) = {g2(x_pt + h * e2):.4f}  g(1, 1.9) = {g2(x_pt - h * e2):.4f}"
      f"  -> central dg/dx2 = {(g2(x_pt + h * e2) - g2(x_pt - h * e2)) / (2 * h):.4f}")

# %%
# From scratch: partial derivatives of any f: R^n -> R by nudging one coordinate at a time.
section("from scratch: partial derivatives")


def partial(f: Callable[[FloatArray], float], x: FloatArray, j: int, h: float = 1e-5) -> float:
    """Central-difference partial derivative of f with respect to x[j]. x (n,) -> scalar."""
    step = np.zeros_like(x)  # (n,)
    step[j] = h
    return (f(x + step) - f(x - step)) / (2.0 * h)


for j in range(2):
    print(f"dg/dx{j + 1} at (1, 2) with h = 1e-5: {partial(g2, x_pt, j):.10f}")


def h3(x: FloatArray) -> float:
    """h(x1, x2, x3) = x1 * sin(x2) + exp(x3). x (3,) -> scalar."""
    return float(x[0] * np.sin(x[1]) + np.exp(x[2]))


x3 = np.array([2.0, 0.5, -1.0])
exact3 = np.array([np.sin(x3[1]), x3[0] * np.cos(x3[1]), np.exp(x3[2])])  # (3,)
approx3 = np.array([partial(h3, x3, j) for j in range(3)])  # (3,)
print("h(x) = x1 sin(x2) + exp(x3) at", show(x3))
print("  exact partials  =", [f"{v:.10f}" for v in exact3])
print("  central (1e-5)  =", [f"{v:.10f}" for v in approx3])
print(f"  largest absolute error = {np.max(np.abs(approx3 - exact3)):.1e}")

# %%
# From scratch: error against step size for exp at x = 1 (the U-shape).
section("from scratch: error against step size")
hs = np.logspace(-16, 0, N_STEPS)  # (N_STEPS,)
err_fwd = np.array([abs(forward_diff(np.exp, 1.0, h) - np.e) for h in hs])
err_cen = np.array([abs(central_diff(np.exp, 1.0, h) - np.e) for h in hs])
for h in (1e-1, 1e-3, 1e-5, 1e-8, 1e-11, 1e-14):
    print(f"h = {h:.0e}   forward error = {abs(forward_diff(np.exp, 1.0, h) - np.e):.2e}"
          f"   central error = {abs(central_diff(np.exp, 1.0, h) - np.e):.2e}")
h_fwd_pred = np.sqrt(2 * EPS64)  # minimizes (h / 2) e + eps e / h
h_cen_pred = (3 * EPS64) ** (1 / 3)  # minimizes (h^2 / 6) e + eps e / h
print(f"forward: predicted best h = {h_fwd_pred:.1e}  predicted error = {np.e * np.sqrt(2 * EPS64):.1e}"
      f"   best on the grid: h = {hs[np.argmin(err_fwd)]:.1e}  error = {err_fwd.min():.1e}")
print(f"central: predicted best h = {h_cen_pred:.1e}  predicted error = {np.e * (h_cen_pred**2 / 6 + EPS64 / h_cen_pred):.1e}"
      f"   best on the grid: h = {hs[np.argmin(err_cen)]:.1e}  error = {err_cen.min():.1e}")

# %%
# From scratch: float32 loses most of the accuracy (PyTorch trains in float32 by default).
section("from scratch: float32 versus float64")


def central_diff_dtype(x: float, h: float, dtype: type) -> float:
    """Central difference of exp at x, with every value stored in the given dtype."""
    xd, hd = dtype(x), dtype(h)
    return float((np.exp(xd + hd) - np.exp(xd - hd)) / (dtype(2.0) * hd))


hs32 = np.logspace(-8, 0, 33)
for dtype, name in ((np.float64, "float64"), (np.float32, "float32")):
    errs = np.array([abs(central_diff_dtype(1.0, h, dtype) - np.e) for h in hs32])
    print(f"{name}: best central error = {errs.min():.1e} at h = {hs32[np.argmin(errs)]:.1e}")

# %%
# From scratch: a kink. |x| has no derivative at 0, and the two formulas disagree.
section("from scratch: a kink")
for h in (1e-2, 1e-6):
    print(f"|x| at 0, h = {h:.0e}: forward = {forward_diff(abs, 0.0, h):.1f}   central = {central_diff(abs, 0.0, h):.1f}")
relu = lambda x: max(x, 0.0)  # noqa: E731
print(f"ReLU at 1e-7, h = 1e-5: central = {central_diff(relu, 1e-7, 1e-5):.4f}  (true slope there is 1)")

# %%
# With SciPy and NumPy: scipy.differentiate.derivative and np.gradient.
section("with scipy and numpy")
from scipy.differentiate import derivative  # noqa: E402

res = derivative(np.exp, 1.0)  # adaptive finite differences with an error estimate
print(f"scipy derivative of exp at 1 = {float(res.df):.12f}  error estimate = {float(res.error):.1e}"
      f"  success = {bool(res.success)}  function calls = {int(res.nfev)}")
np.testing.assert_allclose(res.df, np.e, rtol=1e-10, atol=0)  # scipy extrapolates: far better than one difference
np.testing.assert_allclose(central_diff(np.exp, 1.0, 1e-5), res.df, rtol=1e-9, atol=0)
# rtol 1e-9: a single central difference at its best h is only good to ~1e-10 relative (see the U-shape)
print("central_diff matches scipy: OK")

res3 = derivative(lambda t: t**3, np.array([0.0, 1.0, 2.0]))  # vectorized over many points
print("scipy derivative of x^3 at [0, 1, 2] =", show(res3.df), "  exact 3 x^2 =", show(3 * np.array([0.0, 1, 2]) ** 2))

xs = np.linspace(0.0, np.pi, 11)  # (11,), samples of sin on a grid: all you have for logged data
grad_np = np.gradient(np.sin(xs), xs)  # central differences inside, one-sided at the two ends
err_np = np.abs(grad_np - np.cos(xs))
print(f"np.gradient(sin) on 11 points: max error inside = {err_np[1:-1].max():.4f}  at the ends = {err_np[[0, -1]].max():.4f}")

# %%
# Key plot: the U-shaped error of forward and central differences.
section("key plot")
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
floor = 1e-17  # errors that are exactly 0 cannot go on a log axis
ax.loglog(hs, np.maximum(err_fwd, floor), "o-", ms=3, label="forward difference (measured)")
ax.loglog(hs, np.maximum(err_cen, floor), "s-", ms=3, label="central difference (measured)")
ax.loglog(hs, np.e * hs / 2, ":", color="tab:blue", label="forward truncation  e·h/2")
ax.loglog(hs, np.e * hs**2 / 6, ":", color="tab:orange", label="central truncation  e·h²/6")
ax.loglog(hs, np.e * EPS64 / hs, "--", color="0.4", label="rounding  e·ε/h")
ax.axvline(h_fwd_pred, color="tab:blue", lw=0.8, alpha=0.6)
ax.axvline(h_cen_pred, color="tab:orange", lw=0.8, alpha=0.6)
ax.set_ylim(1e-13, 1e1)
ax.set_xlabel("step size h (log scale)")
ax.set_ylabel("|estimate − true derivative| (log scale)")
ax.set_title("Too large a step: truncation error. Too small: rounding error")
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout()
path = IMG / "a4_1_step_size_error.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace np.exp by np.log and x = 1 by x = 1000. Does the best h move? Why (what scale does log curve on)?
# 2. Repeat the error study with f(x) = x^2. Why is the central error pure rounding for every h?
# 3. Differentiate np.round(x, 3) at x = 0.5 with h = 1e-6 and h = 1e-2. What goes wrong?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
