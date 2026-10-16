# %% [markdown]
# # A4.1 Derivatives and partial derivatives — lab
# Lesson: lessons/A/a4_1_derivatives.md
# Run: uv run python labs/A/a4_1_derivatives.py   (CPU, defaults: ~2 s)
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
from scipy import differentiate, optimize

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


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the knob only sets how many step sizes the error sweep tries.
GPU_SCALE = False
N_STEPS = 400 if GPU_SCALE else 60  # step sizes h in the sweep, log-spaced from 1e-15 to 1e-1
EPS = np.finfo(np.float64).eps  # machine epsilon: the gap between 1.0 and the next float64

# %%
# Data: the functions differentiated in this lab, with their exact derivatives for checking.
section("data")


def cube(x: float) -> float:
    """f(x) = x^3, written as products so every machine rounds it identically."""
    return x * x * x


def cube_prime(x: float) -> float:
    """Exact derivative of x^3: 3 x^2."""
    return 3.0 * x * x


X0 = 2.0  # the point of the worked example
print(f"f(x) = x^3   f({X0}) = {cube(X0)}   exact f'({X0}) = {cube_prime(X0)}")
print(f"machine epsilon for float64 = {EPS:.4e}")

# %%
# Worked example by hand, part 1: forward and central differences of x^3 at x = 2.
section("worked example: slopes of x^3 at x = 2")
for h in (0.1, 0.05):
    fwd = (cube(X0 + h) - cube(X0)) / h
    ctr = (cube(X0 + h) - cube(X0 - h)) / (2 * h)
    # Taylor predictions: forward error = f''(x) h / 2 + f'''(x) h^2 / 6 = 6h + h^2; central = h^2
    print(
        f"h = {h:<5} forward = {fwd:.4f} (error {fwd - 12:.4f}, predicted {6 * h + h * h:.4f})"
        f"   central = {ctr:.4f} (error {ctr - 12:.4f}, predicted {h * h:.4f})"
    )

# %%
# Worked example by hand, part 2: partial derivatives of a one-example squared error.
section("worked example: partial derivatives")
X_EX, Y_EX = 2.0, 3.0  # one example: input x = 2, target y = 3


def sq_error(w: float, b: float) -> float:
    """Squared error of the line w x + b on the single example (X_EX, Y_EX)."""
    r = w * X_EX + b - Y_EX
    return r * r


w0, b0 = 1.0, 0.0
r0 = w0 * X_EX + b0 - Y_EX
print(f"at (w, b) = ({w0}, {b0}): residual r = {r0}   f = {sq_error(w0, b0)}")
print(f"by hand: df/dw = 2 r x = {2 * r0 * X_EX}   df/db = 2 r = {2 * r0}")
h = 0.01
dw = (sq_error(w0 + h, b0) - sq_error(w0 - h, b0)) / (2 * h)
db = (sq_error(w0, b0 + h) - sq_error(w0, b0 - h)) / (2 * h)
print(f"central differences with h = {h}: df/dw = {dw:.4f}   df/db = {db:.4f}")
print(f"f(1.01, 0) = {sq_error(1.01, 0):.4f}   f(0.99, 0) = {sq_error(0.99, 0):.4f}")

# %%
# From scratch: forward and central differences, and their error as the step size shrinks.
section("from scratch: error against step size")


def forward_diff(f: Callable[[float], float], x: float, h: float) -> float:
    """Forward-difference slope (f(x + h) - f(x)) / h. Error shrinks like h."""
    return (f(x + h) - f(x)) / h


def central_diff(f: Callable[[float], float], x: float, h: float) -> float:
    """Central-difference slope (f(x + h) - f(x - h)) / 2h. Error shrinks like h^2."""
    return (f(x + h) - f(x - h)) / (2 * h)


exact = cube_prime(X0)
print(f"{'h':>7}  {'forward error':>14}  {'central error':>14}")
for k in range(1, 14, 2):
    h = 10.0**-k
    e_fwd = abs(forward_diff(cube, X0, h) - exact)
    e_ctr = abs(central_diff(cube, X0, h) - exact)
    print(f"{h:>7.0e}  {e_fwd:>14.2e}  {e_ctr:>14.2e}")

hs = np.logspace(-15, -1, N_STEPS)  # (N_STEPS,) step sizes
err_fwd = np.array([abs(forward_diff(cube, X0, h) - exact) for h in hs])  # (N_STEPS,)
err_ctr = np.array([abs(central_diff(cube, X0, h) - exact) for h in hs])  # (N_STEPS,)
# Error model: truncation + rounding. Rounding: f(x) is stored with relative error ~EPS/2, so each
# f value is off by about EPS * |f(x)| / 2; dividing a difference of two such values by h (or 2h)
# gives about EPS * |f(x)| / h.
f0, f2, f3 = cube(X0), 6.0 * X0, 6.0  # f, f'', f''' at x = 2
h_fwd = np.sqrt(2 * EPS * f0 / f2)  # minimizes f'' h / 2 + EPS |f| / h
h_ctr = np.cbrt(3 * EPS * f0 / f3)  # minimizes f''' h^2 / 6 + EPS |f| / h
e_fwd_best = abs(forward_diff(cube, X0, h_fwd) - exact)
e_ctr_best = abs(central_diff(cube, X0, h_ctr) - exact)
print(f"predicted best h: forward {h_fwd:.1e} (error there {e_fwd_best:.1e})"
      f"   central {h_ctr:.1e} (error there {e_ctr_best:.1e})")

# %%
# From scratch: a partial derivative of a function of a vector, by nudging one coordinate.
section("from scratch: partial derivatives of f(w, b)")


def partial(f: Callable[[FloatArray], float], x: FloatArray, j: int, h: float = 1e-6) -> float:
    """Central-difference partial derivative of f along coordinate j. x (n,) -> scalar."""
    e_j = np.zeros_like(x)  # (n,), unit vector along coordinate j
    e_j[j] = 1.0
    return (f(x + h * e_j) - f(x - h * e_j)) / (2 * h)


def sq_error_vec(theta: FloatArray) -> float:
    """The same squared error with theta = (w, b) packed into one array. theta (2,) -> scalar."""
    return sq_error(float(theta[0]), float(theta[1]))


theta0 = np.array([1.0, 0.0])
for j, name, by_hand in ((0, "w", 2 * r0 * X_EX), (1, "b", 2 * r0)):
    num = partial(sq_error_vec, theta0, j)
    print(f"df/d{name}: numerical = {num:.6f}   by hand = {by_hand}   match: {abs(num - by_hand) < 1e-6}")

# Partial derivatives change from point to point: at the minimum both are zero.
theta_star = np.array([1.5, 0.0])  # w x + b = 3 = y, so the residual is zero
print("at (w, b) = (1.5, 0):", [round(partial(sq_error_vec, theta_star, j), 6) + 0.0 for j in (0, 1)])

# %%
# With SciPy and NumPy: library derivatives, and checking that they agree with ours.
section("with scipy and numpy")
res = differentiate.derivative(np.exp, 1.0)  # adaptive central differences with error control
print(f"scipy.differentiate.derivative(exp, 1): {res.df:.10f}   exact e = {np.e:.10f}"
      f"   success = {bool(res.success)}")
np.testing.assert_allclose(res.df, np.e, rtol=1e-9)
np.testing.assert_allclose(central_diff(np.exp, 1.0, 1e-5), res.df, rtol=1e-9)
# rtol 1e-9: our fixed-step central difference is only accurate to about 1e-10 (see the sweep)
print("central_diff(exp, 1, h=1e-5) matches scipy.differentiate.derivative: OK")

fprime = optimize.approx_fprime(np.array([1.0, 0.0]), sq_error_vec)  # FORWARD differences
print(f"scipy.optimize.approx_fprime at (1, 0) = {np.round(fprime, 6).tolist()}"
      f"   (default step {np.sqrt(EPS):.2e} = sqrt(eps))")
np.testing.assert_allclose(fprime, [-4.0, -2.0], rtol=1e-6)
print("approx_fprime matches the hand-derived partials (rtol 1e-6): OK")

grid = np.linspace(0.5, 2.5, 201)  # (201,) samples of sin on [0.5, 2.5], spacing 0.01
spacing = grid[1] - grid[0]
slope = np.gradient(np.sin(grid), spacing)  # central inside, one-sided at the two ends
err = np.abs(slope - np.cos(grid))
print(f"np.gradient on sampled sin (spacing {spacing:.4f}): max error inside = {err[1:-1].max():.1e}"
      f"   at the ends = {max(err[0], err[-1]):.1e}")

# %%
# Key plot: the U-shaped error of finite differences.
section("key plot")
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120)
floor = 1e-17  # errors that are exactly 0 cannot go on a log axis
ax.loglog(hs, np.maximum(err_fwd, floor), "o-", ms=3, lw=1, label="forward difference")
ax.loglog(hs, np.maximum(err_ctr, floor), "s-", ms=3, lw=1, label="central difference")
ax.loglog(hs, f2 * hs / 2, "--", color="tab:blue", lw=1, alpha=0.7, label="forward truncation ≈ f''h/2")
ax.loglog(hs, f3 * hs**2 / 6, "--", color="tab:orange", lw=1, alpha=0.7, label="central truncation ≈ f'''h²/6")
ax.loglog(hs, EPS * f0 / hs, ":", color="0.3", lw=1.5, label="rounding ≈ ε|f|/h")
ax.axvline(h_fwd, color="tab:blue", lw=0.8, alpha=0.5)
ax.axvline(h_ctr, color="tab:orange", lw=0.8, alpha=0.5)
ax.set_ylim(1e-12, 1e2)
ax.set_xlabel("step size h (log scale)")
ax.set_ylabel("|estimate − exact f'(2)| (log scale)")
ax.set_title("Derivative of x³ at x = 2: too large or too small a step both hurt", fontsize=10)
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
path = IMG / "a4_1_error_vs_step.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Repeat the sweep in float32: x = np.float32(2.0) and h = np.float32(h). Where is the best h now?
# 2. Differentiate abs(x) at x = 0 with both methods and several h. What does each return, and why?
# 3. Replace cube with np.exp at x = 1. Does the best central h move? (It depends on f and f'''.)

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
