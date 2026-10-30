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
# Nothing here needs a GPU; the knobs set how many random chords and starting points are tried.
GPU_SCALE = False
N_CHORDS = 100_000 if GPU_SCALE else 2_000  # random pairs for the chord test
N_STARTS = 41  # evenly spaced gradient-descent starting points in [-2, 2]
D_PROJ = 50 if GPU_SCALE else 5  # dimension of the random projection problem

# %%
# Data: the one-dimensional functions this lab tests.
section("data")


def convex_f(x: FloatArray) -> FloatArray:
    """(x - 0.5)^2 + 0.25 x^4: convex, since f'' = 2 + 3 x^2 > 0."""
    return (x - 0.5) ** 2 + 0.25 * x**4


def convex_df(x: FloatArray) -> FloatArray:
    """Derivative of convex_f."""
    return 2 * (x - 0.5) + x**3


def wiggly_f(x: FloatArray) -> FloatArray:
    """x^4 - 3 x^2 + x: not convex, since f'' = 12 x^2 - 6 < 0 for |x| < 0.707."""
    return x**4 - 3 * x**2 + x


def wiggly_df(x: FloatArray) -> FloatArray:
    """Derivative of wiggly_f."""
    return 4 * x**3 - 6 * x + 1


stationary = np.sort(np.roots([4.0, 0.0, -6.0, 1.0]).real)  # where 4x^3 - 6x + 1 = 0
print("x^4 - 3x^2 + x: stationary points =", show(stationary), "  f there =", show(wiggly_f(stationary)),
      "  f'' there =", show(12 * stationary**2 - 6))

# %%
# Worked example by hand: convexity by the chord test and by the second derivative.
section("worked example: chord test")


def worst_chord_gap(f: Callable[[FloatArray], FloatArray], lo: float, hi: float, n: int = N_CHORDS) -> float:
    """Largest f(t a + (1-t) b) - (t f(a) + (1-t) f(b)) over random a, b, t; > 0 means not convex."""
    a, b, t = rng.uniform(lo, hi, n), rng.uniform(lo, hi, n), rng.uniform(0, 1, n)
    return float(np.max(f(t * a + (1 - t) * b) - (t * f(a) + (1 - t) * f(b))))


x_a, x_b, t = -1.0, 1.0, 0.5
for name, f in (("x^2", lambda x: x**2), ("x^4 - 3x^2 + x", wiggly_f)):
    mid, chord = f(t * x_a + (1 - t) * x_b), t * f(x_a) + (1 - t) * f(x_b)
    print(f"{name:<15} between -1 and 1: f(midpoint) = {mid:+.4f}   chord midpoint = {chord:+.4f}   "
          f"{'below the chord' if mid <= chord else 'ABOVE the chord: not convex'}")
for name, f in (("x^2", lambda x: x**2), ("exp(x)", np.exp), ("x^3", lambda x: x**3),
                ("(x - 0.5)^2 + 0.25 x^4", convex_f), ("x^4 - 3x^2 + x", wiggly_f)):
    gap = round(worst_chord_gap(f, -2.0, 2.0), 4) + 0.0  # + 0.0 turns -0.0 into 0.0
    print(f"{name:<24} worst chord gap on [-2, 2] = {gap:+.4f}   -> {'convex' if gap <= 0 else 'NOT convex'}")

# %%
# Worked example by hand: two quadratics, tested with the eigenvalues of the Hessian.
section("worked example: Hessian test")
for name, H in (("x^2 + xy + y^2", np.array([[2.0, 1.0], [1.0, 2.0]])),
                ("x^2 + 3xy + y^2", np.array([[2.0, 3.0], [3.0, 2.0]]))):
    eig = np.linalg.eigvalsh(H)
    print(f"{name:<16} Hessian = {show(H)}  eigenvalues = {show(eig)}  -> {'convex' if eig.min() >= 0 else 'NOT convex (a saddle)'}")
v = np.array([1.0, -1.0])
print("x^2 + 3xy + y^2 along (t, -t): f = 2t^2 - 3t^2 = -t^2, e.g. at t = 1:", 1 + 3 * 1 * (-1) + 1)
X_any = rng.normal(size=(30, 3))
print("MSE Hessian (2/n) X^T X for a random X (30, 3): smallest eigenvalue =",
      f"{np.linalg.eigvalsh(2 / 30 * X_any.T @ X_any).min():.4f}  (>= 0: the MSE is always convex)")

# %%
# Worked example by hand: a Lagrange multiplier. Minimize x^2 + y^2 subject to x + y = c.
section("worked example: Lagrange multiplier")
c = 1.0
x_star = np.array([c / 2, c / 2])
grad_f, grad_g = 2 * x_star, np.array([1.0, 1.0])
lam = grad_f[0] / grad_g[0]
print(f"solution (x, y) = {show(x_star)}  f* = {x_star @ x_star:.4f}  grad f = {show(grad_f)}  grad g = {show(grad_g)}  lambda = {lam:.4f}")
for dc in (0.1, 0.01):
    f_new = 2 * ((c + dc) / 2) ** 2  # optimum for the constraint x + y = c + dc
    print(f"  constraint x + y = {c + dc}: f* = {f_new:.4f}   (f* change) / (c change) = {(f_new - 0.5) / dc:.4f}  -> lambda = {lam}")

# %%
# Worked example by hand: an inequality constraint, active and inactive.
section("worked example: inequality constraint")
p = np.array([2.0, 2.0])  # f(x) = ||x - p||^2, unconstrained minimum at p
for c_ in (1.0, 5.0):
    if p.sum() <= c_:  # the unconstrained minimum is feasible: constraint inactive
        sol, mu = p.copy(), 0.0
    else:  # active: the answer lies on the line x + y = c_, nearest point to p
        sol = p - (p.sum() - c_) / 2 * np.array([1.0, 1.0])
        mu = -2 * (sol - p)[0]  # stationarity: grad f + mu grad g = 0, grad g = (1, 1)
    print(f"x + y <= {c_}: solution = {show(sol)}  x + y = {sol.sum():.1f}  mu = {mu:.1f}  "
          f"({'active' if mu > 0 else 'inactive'}; mu * slack = {mu * (c_ - sol.sum()):.1f})")

# %%
# From scratch: an equality-constrained quadratic as one linear system (the KKT system).
section("from scratch: KKT linear system")


def project_onto_hyperplane(p_: FloatArray, a: FloatArray, c_: float) -> tuple[FloatArray, float]:
    """Minimize ||x - p||^2 subject to a . x = c by solving the KKT system. p, a (d,) -> (x (d,), lambda)."""
    d = len(p_)
    K = np.zeros((d + 1, d + 1))  # (d+1, d+1): [[2 I, -a], [a^T, 0]]
    K[:d, :d] = 2 * np.eye(d)
    K[:d, d] = -a
    K[d, :d] = a
    rhs = np.concatenate([2 * p_, [c_]])  # (d+1,)
    sol = np.linalg.solve(K, rhs)
    return sol[:d], float(sol[d])


x_kkt, lam_kkt = project_onto_hyperplane(np.zeros(2), np.array([1.0, 1.0]), 1.0)
print("min x^2 + y^2 s.t. x + y = 1 via KKT: x =", show(x_kkt), f"  lambda = {lam_kkt:.4f}")
p_r, a_r, c_r = rng.normal(size=D_PROJ), rng.normal(size=D_PROJ), 1.0
x_r, lam_r = project_onto_hyperplane(p_r, a_r, c_r)
x_closed = p_r + (c_r - a_r @ p_r) / (a_r @ a_r) * a_r  # projection formula (A2.4 style)
np.testing.assert_allclose(x_r, x_closed, rtol=1e-6, atol=1e-8)
print(f"random d = {D_PROJ}: a . x = {a_r @ x_r:.6f}   matches the closed-form projection: OK   lambda = {lam_r:.4f}")
print("residual x - p is parallel to a (the constraint's normal): cos angle =",
      f"{abs((x_r - p_r) @ a_r) / (np.linalg.norm(x_r - p_r) * np.linalg.norm(a_r)):.6f}")

# %%
# From scratch: gradient descent on a convex and a non-convex function from many starting points.
section("from scratch: where gradient descent ends up")


def gd_1d(df: Callable[[FloatArray], FloatArray], x0: FloatArray, lr: float = 0.02, steps: int = 2_000) -> FloatArray:
    """Run 1-D gradient descent from many starting points at once. x0 (k,) -> final points (k,)."""
    x = x0.copy()
    for _ in range(steps):
        x -= lr * df(x)
    return x


starts = np.linspace(-2.0, 2.0, N_STARTS)  # (k,)
ends_convex = gd_1d(convex_df, starts)
ends_wiggly = gd_1d(wiggly_df, starts)
print("convex (x - 0.5)^2 + 0.25 x^4: distinct end points =", show(np.unique(np.round(ends_convex, 4))))
uniq, counts = np.unique(np.round(ends_wiggly, 4), return_counts=True)
for u, k in zip(uniq, counts, strict=True):
    print(f"x^4 - 3x^2 + x: {k:>2} of {N_STARTS} starts end at x = {u:+.4f}  f = {wiggly_f(u):+.4f}")
boundary = stationary[1]
print(f"the dividing line is the local maximum at x = {boundary:.4f}: starts left of it go left, right of it go right")

# %%
# With SciPy: constrained minimization (SLSQP) returns the same points and multipliers.
section("with scipy")
from scipy.optimize import minimize  # noqa: E402

res_eq = minimize(lambda v: v @ v, x0=np.zeros(2), method="SLSQP",
                  constraints=[{"type": "eq", "fun": lambda v: v[0] + v[1] - 1.0}])
np.testing.assert_allclose(res_eq.x, x_kkt, rtol=1e-6, atol=1e-6)  # SLSQP stops at ~1e-7 accuracy
print(f"equality:   x = {show(res_eq.x)}  multiplier = {res_eq.multipliers[0]:.4f}  (lambda by hand = 1)  matches: OK")
res_in = minimize(lambda v: (v - p) @ (v - p), x0=np.zeros(2), method="SLSQP",
                  constraints=[{"type": "ineq", "fun": lambda v: 1.0 - v[0] - v[1]}])  # scipy: fun(v) >= 0
np.testing.assert_allclose(res_in.x, [0.5, 0.5], rtol=1e-6, atol=1e-6)
print(f"inequality: x = {show(res_in.x)}  multiplier = {res_in.multipliers[0]:.4f}  (mu by hand = 3)  matches: OK")
for x0 in (-1.0, 0.5):
    r = minimize(lambda v: float(wiggly_f(v[0])), x0=np.array([x0]), method="BFGS")
    print(f"BFGS on x^4 - 3x^2 + x from x0 = {x0:+.1f}: x = {r.x[0]:+.4f}  f = {r.fun:+.4f}  success = {r.success}")

# %%
# Key plot: the convex function sends every start to one minimum; the non-convex one splits them.
section("key plot")
xs = np.linspace(-2.1, 2.1, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
for ax, f, ends, title in ((axes[0], convex_f, ends_convex, "Convex: every start reaches the same minimum"),
                           (axes[1], wiggly_f, ends_wiggly, "Not convex: the start decides the minimum")):
    ax.plot(xs, f(xs), "-", color="0.3", lw=2, label="f(x)")
    goes_left = ends < 0
    if len(np.unique(np.round(ends, 4))) == 1:
        ax.plot(starts, f(starts), "o", color="tab:blue", ms=5, label="starting points (all end at the star)")
    else:
        ax.plot(starts[goes_left], f(starts[goes_left]), "o", color="tab:blue", ms=5, label="start that ends at the left star")
        ax.plot(starts[~goes_left], f(starts[~goes_left]), "s", color="tab:orange", ms=5, label="start that ends at the right star")
    for e in np.unique(np.round(ends, 4)):
        ax.plot(e, f(e), "*", color="tab:red", ms=16, zorder=5, markeredgecolor="k")
    ax.plot([], [], "*", color="tab:red", ms=10, markeredgecolor="k", label="where gradient descent ends")
    ax.set_xlabel("x (unitless)")
    ax.set_ylabel("f(x) (unitless)")
    ax.set_title(title, fontsize=10)
axes[0].legend(loc="upper center", fontsize=8)
axes[1].legend(loc="upper left", bbox_to_anchor=(0.18, 1.0), fontsize=8)
axes[1].plot([stationary[1]] * 2, [-4, 7], ":", color="0.5")
axes[1].text(stationary[1] + 0.05, -3.7, "local max:\nthe dividing line", fontsize=8, color="0.3")
axes[0].set_ylim(-0.5, 11)
axes[1].set_ylim(-4, 7)
fig.tight_layout()
path = IMG / "a4_5_convex_vs_nonconvex.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# A second figure (not committed): the geometry of the Lagrange condition.
fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
g1, g2 = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
cs = ax.contour(g1, g2, g1**2 + g2**2, levels=[0.125, 0.5, 1.0, 2.0], colors="0.6")
ax.clabel(cs, fontsize=7)
ax.plot([-0.5, 1.5], [1.5, -0.5], "-", color="tab:blue", lw=2, label="constraint x + y = 1")
ax.annotate("", xy=(0.5 + 0.35, 0.5 + 0.35), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
ax.plot(0.5, 0.5, "k*", ms=12, label="optimum: contour tangent to the line")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("At the optimum ∇f is parallel to ∇g")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "lagrange_geometry.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/a4_5/lagrange_geometry.png")

# %%
# Try this (not graded):
# 1. Change the inequality to x + y <= 3.5 and to x + y <= 4. When does mu become 0, and why?
# 2. Run gd_1d on wiggly_f with lr = 0.2. Do the end points still split at the local maximum?
# 3. Add a tiny L2 penalty eps * x^2 to x^3 and run the chord test again for eps = 0.1, 1, 10.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
