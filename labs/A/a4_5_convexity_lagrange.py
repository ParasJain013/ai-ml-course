# %% [markdown]
# # A4.5 Convexity, constrained optimization and Lagrange multipliers — lab
# Lesson: lessons/A/a4_5_convexity_lagrange.md
# Run: uv run python labs/A/a4_5_convexity_lagrange.py   (CPU, defaults: ~3 s)
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
from scipy import optimize

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
OUT = ROOT / "outputs" / "a4_5"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knobs size the random convexity test and the number of GD starts.
GPU_SCALE = False
N_CHORDS = 100_000 if GPU_SCALE else 2_000  # random chords per function in the convexity test
N_STARTS = 41  # starting points for gradient descent, evenly spaced on [-2, 2]
GD_LR = 0.01
GD_STEPS = 2_000

# %%
# Data: the 1-D functions used for convexity, with their derivatives.
section("data")
FUNCS: dict[str, tuple[Callable[[FloatArray], FloatArray], Callable[[FloatArray], FloatArray]]] = {
    "x^2": (lambda x: x**2, lambda x: 2 * x),
    "|x|": (lambda x: np.abs(x), lambda x: np.sign(x)),
    "exp(x)": (lambda x: np.exp(x), lambda x: np.exp(x)),
    "x^4 - 3x^2 + x": (lambda x: x**4 - 3 * x**2 + x, lambda x: 4 * x**3 - 6 * x + 1),
    "sin(x)": (lambda x: np.sin(x), lambda x: np.cos(x)),
}
print("functions:", list(FUNCS))

# %%
# Worked example by hand, part 1: min x^2 + y^2 subject to x + y = c.
section("worked example: closest point on a line")
c = 1.0
lam = c  # from grad f = lam grad h: (2x, 2y) = lam (1, 1) -> x = y = lam/2; x + y = c -> lam = c
x_opt = np.array([lam / 2, lam / 2])
grad_f = 2 * x_opt
grad_h = np.array([1.0, 1.0])
print(f"solution (x, y) = {show(x_opt)}   lambda = {lam}   f* = {x_opt @ x_opt}")
print(f"grad f = {show(grad_f)}   grad h = {show(grad_h)}   grad f = lambda * grad h: {np.allclose(grad_f, lam * grad_h)}")
for c_new in (1.01, 1.1):
    f_new = c_new**2 / 2  # x = y = c/2 for any c
    print(f"c = {c_new}: f* = {f_new:.5f}   (f*(c) - f*(1)) / (c - 1) = {(f_new - 0.5) / (c_new - 1):.4f}   lambda = {lam}")

# %%
# Worked example by hand, part 2: the shortest w with a . w = c (a minimum-norm solution).
section("worked example: minimum-norm w")
a = np.array([1.0, 2.0])
c2 = 5.0
w_mn = c2 * a / (a @ a)  # from 2w = lam a and a . w = c
lam2 = 2 * c2 / (a @ a)
print(f"w = c a / |a|^2 = {show(w_mn)}   |w|^2 = {w_mn @ w_mn}   lambda = 2c/|a|^2 = {lam2}")
print(f"np.linalg.pinv gives the same w: {show(np.linalg.pinv(a.reshape(1, 2)) @ np.array([c2]))}")

# %%
# Worked example by hand, part 3: an inequality constraint, active and inactive.
section("worked example: inequality constraint")
for bound in (1.0, 3.0):
    x_star = min(2.0, bound)  # unconstrained minimum of (x - 2)^2 is 2; clip to x <= bound
    mu = max(0.0, -2 * (x_star - 2))  # KKT: 2(x - 2) + mu = 0 with mu >= 0, and mu = 0 if inactive
    state = "active" if np.isclose(x_star, bound) else "inactive"
    print(f"min (x - 2)^2 s.t. x <= {bound}: x* = {x_star}   constraint {state}   multiplier mu = {mu}")

# %%
# From scratch: a chord test for convexity. It can prove a function is NOT convex, never that it is.
section("from scratch: chord test")


def chord_violations(f: Callable[[FloatArray], FloatArray], lo: float, hi: float, n: int, rng: np.random.Generator) -> float:
    """Share of random chords on [lo, hi] that dip below f. 0 means no counterexample found."""
    x1 = rng.uniform(lo, hi, size=n)  # (n,)
    x2 = rng.uniform(lo, hi, size=n)  # (n,)
    t = rng.uniform(0.0, 1.0, size=n)  # (n,)
    left = f(t * x1 + (1 - t) * x2)  # f at a point between x1 and x2
    right = t * f(x1) + (1 - t) * f(x2)  # the chord at the same point
    return float(np.mean(left > right + 1e-12))  # 1e-12: ignore rounding on straight pieces


for name, (f, _) in FUNCS.items():
    share = chord_violations(f, -2.0, 2.0, N_CHORDS, rng)
    verdict = "no violation found" if share == 0 else "NOT convex"
    print(f"{name:<16} chords below the curve: {share:6.1%}   {verdict}")

# %%
# From scratch: second derivatives and the Hessian test.
section("from scratch: curvature tests")
grid = np.linspace(-2.0, 2.0, 401)
f2 = 12 * grid**2 - 6  # second derivative of x^4 - 3x^2 + x
neg = grid[f2 < 0]
print(f"x^4 - 3x^2 + x: f'' = 12x^2 - 6 < 0 on ({neg.min():.2f}, {neg.max():.2f}) -> not convex there")
x_data = np.random.default_rng(SEED).uniform(0.0, 10.0, size=50)  # the A2.4 inputs
X = np.column_stack([np.ones(50), x_data])  # (50, 2)
H = 2.0 / 50 * X.T @ X  # (2, 2) Hessian of the MSE (A4.4)
print(f"MSE Hessian (2/n) X^T X eigenvalues = {show(np.linalg.eigvalsh(H))}  -> all >= 0: convex")
X_dup = np.column_stack([X, X[:, 1]])  # (50, 3): the same feature twice
ev_dup = np.linalg.eigvalsh(2.0 / 50 * X_dup.T @ X_dup)
print(f"with a duplicated column: eigenvalues = {show(ev_dup)}  -> convex but not strictly (a flat valley)")

# %%
# From scratch: gradient descent from many starting points, convex versus non-convex.
section("from scratch: where gradient descent ends up")


def gd_1d(grad: Callable[[FloatArray], FloatArray], starts: FloatArray, lr: float, steps: int) -> FloatArray:
    """Run 1-D gradient descent from every start at once. starts (k,) -> endpoints (k,)."""
    x = starts.copy()
    for _ in range(steps):
        x = x - lr * grad(x)
    return x


starts = np.linspace(-2.0, 2.0, N_STARTS)  # (41,)
ends_convex = gd_1d(lambda x: 2 * (x - 0.5), starts, GD_LR, GD_STEPS)  # f = (x - 0.5)^2
print(f"convex (x - 0.5)^2: distinct endpoints = {np.unique(np.round(ends_convex, 4)).tolist()}")
f_nc, g_nc = FUNCS["x^4 - 3x^2 + x"]
ends_nc = gd_1d(g_nc, starts, GD_LR, GD_STEPS)
for m in np.unique(np.round(ends_nc, 4)):
    from_starts = starts[np.isclose(np.round(ends_nc, 4), m)]
    print(f"non-convex: endpoint x = {m:.4f}  f = {f_nc(np.array(m)):.4f}  reached from starts in "
          f"[{from_starts.min():.1f}, {from_starts.max():.1f}]  ({len(from_starts)} of {N_STARTS})")
crit = np.sort(np.roots([4.0, 0.0, -6.0, 1.0]).real)  # f'(x) = 4x^3 - 6x + 1 = 0
print(f"critical points (roots of f') = {show(crit)}  f'' there = {show(12 * crit**2 - 6)}")

# %%
# With SciPy: constrained minimization with SLSQP, which also returns the multipliers.
section("with scipy")
r1 = optimize.minimize(lambda v: v @ v, np.zeros(2), method="SLSQP",
                       constraints=[{"type": "eq", "fun": lambda v: v[0] + v[1] - 1.0}])
print(f"line:    x = {show(r1.x)}   f* = {r1.fun:.4f}   multiplier = {r1.multipliers[0]:.4f}")
np.testing.assert_allclose(r1.x, x_opt, atol=1e-6)
np.testing.assert_allclose(r1.multipliers[0], lam, atol=1e-6)
r2 = optimize.minimize(lambda v: v @ v, np.zeros(2), method="SLSQP",
                       constraints=[{"type": "eq", "fun": lambda v: a @ v - c2}])
print(f"min-norm: w = {show(r2.x)}   f* = {r2.fun:.4f}   multiplier = {r2.multipliers[0]:.4f}")
np.testing.assert_allclose(r2.x, w_mn, atol=1e-6)
for bound in (1.0, 3.0):
    r3 = optimize.minimize(lambda v: (v[0] - 2.0) ** 2, np.zeros(1), method="SLSQP",
                           constraints=[{"type": "ineq", "fun": lambda v, b=bound: b - v[0]}])  # b - x >= 0
    print(f"x <= {bound}:  x = {show(r3.x)}   multiplier = {show(r3.multipliers)}")
print("SLSQP matches every hand solution (atol 1e-6): OK")
# atol 1e-6: SLSQP stops at its own tolerance (default ftol 1e-6), not at machine precision
r4 = optimize.minimize(lambda v: f_nc(v[0]), np.array([1.5]), method="BFGS")
r5 = optimize.minimize(lambda v: f_nc(v[0]), np.array([-1.5]), method="BFGS")
print(f"BFGS on the non-convex function from 1.5 -> {r4.x[0]:.4f}, from -1.5 -> {r5.x[0]:.4f}: "
      "a library optimizer is still local")

# %%
# Key plot: convex versus non-convex endpoints, and the Lagrange tangency picture.
section("key plot")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)
xs = np.linspace(-2.0, 2.0, 400)
ax = axes[0]
ax.plot(xs, (xs - 0.5) ** 2, color="tab:blue", label="f(x) = (x − 0.5)²")
ax.plot(starts, (starts - 0.5) ** 2, "o", ms=3, color="0.6", label="41 starting points")
ax.plot(ends_convex, (ends_convex - 0.5) ** 2, "*", ms=14, color="tab:green", label="every GD run ends here")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Convex: every start reaches the one minimum", fontsize=10)
ax.legend(loc="upper center", fontsize=7)
ax = axes[1]
ax.plot(xs, f_nc(xs), color="tab:blue", label="f(x) = x⁴ − 3x² + x")
left = starts < crit[1]
ax.plot(starts[left], f_nc(starts[left]), "o", ms=3, color="tab:orange", label="starts that end at the global min")
ax.plot(starts[~left], f_nc(starts[~left]), "s", ms=3, color="tab:purple", label="starts that end at the local min")
for m, mk in zip(np.unique(np.round(ends_nc, 4)), ("*", "P")):
    ax.plot(m, f_nc(np.array(m)), mk, ms=13, color="k")
ax.axvline(crit[1], color="0.6", ls=":", lw=1)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Non-convex: the start decides the answer", fontsize=10)
ax.legend(loc="upper center", fontsize=7)
ax = axes[2]
g0, g1 = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
cs = ax.contour(g0, g1, g0**2 + g1**2, levels=[0.125, 0.5, 1.125, 2.0], colors="0.6", linewidths=0.8)
ax.clabel(cs, fontsize=7, fmt="%.2f")
line = np.linspace(-0.5, 1.5, 10)
ax.plot(line, 1 - line, "-", color="tab:red", lw=2, label="constraint x + y = 1")
ax.plot(*x_opt, "k*", ms=12, label="optimum (0.5, 0.5), f* = 0.5")
ax.annotate("", xy=x_opt + 0.3 * grad_f / np.linalg.norm(grad_f), xytext=x_opt,
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
ax.annotate("", xy=x_opt + 0.5 * grad_h / np.linalg.norm(grad_h), xytext=x_opt,
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2, ls="--"))
ax.plot([], [], "-", color="tab:blue", lw=2, label="∇f (thick) ∥ ∇h (dashed)")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Constrained optimum: contour touches the line", fontsize=10)
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
path = IMG / "a4_5_convex_nonconvex_lagrange.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Run the chord test on x^3 over [0, 2] and over [-2, 2]. Is convexity a property of f alone?
# 2. Change GD_LR to 0.2 for the non-convex function. Do the endpoints still sort by starting side?
# 3. Solve min x + y subject to x^2 + y^2 = 1 with SLSQP from several starts. Why two answers?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
